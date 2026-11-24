"""Generate a Vitest regression test from a unified diff on stdin."""
import argparse
import json
import re
import sys
from pathlib import Path

from fixtures import messages

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
REVISION = "b3252a2f97102b1fb1571fec2c9b27219a8536be"
MAX_TOKENS = 192
TEMPERATURE = 0.0
PROMPTS = ("original", "distinguish")


def model_path(cache_only=False):
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=cache_only))
    except Exception as error:
        raise RuntimeError(
            f"Could not load {MODEL}@{REVISION}. "
            "Connect to Hugging Face and run uv run --locked python generate.py --download. "
            f"Download details: {error}"
        ) from error
    config = path / "config.json"
    if path.name != REVISION or not config.is_file() or not any(file.is_file() for file in path.glob("*.safetensors")):
        raise RuntimeError(f"Snapshot {path} lacks the pinned model files. Retry --download online.")
    if json.loads(config.read_text()).get("quantization", {}).get("bits") != 4:
        raise RuntimeError(f"Snapshot {path} does not have the expected 4-bit model config.")
    return str(path)


def prompt_messages(patch, module="./subject", variant="original"):
    chat = messages(patch, module)
    if variant == "distinguish":
        original = chat[0]["content"].removesuffix("Choose an input that fails before the fix.")
        chat[0]["content"] = original + (
            "Choose a numeric input where the old and fixed functions return different values. "
            "Trace both versions for that input, then assert the value returned by the fixed version."
        )
    elif variant != "original":
        raise ValueError(f"Unknown prompt: {variant}")
    return chat


def normalized_answer(answer):
    answer = answer.strip()
    fence = re.fullmatch(r"```(?:typescript|ts|javascript|js)?\n(.*?)\n```", answer, re.S)
    return fence[1] if fence else answer


def load_model(adapter=None, cache_only=False):
    from mlx_lm import load

    if adapter is not None:
        folder = Path(adapter)
        for name in ("adapter_config.json", "adapters.safetensors"):
            if not (folder / name).is_file():
                raise ValueError(f"Missing {folder / name}. Use a trained checkpoint directory with --adapter.")
    return load(model_path(cache_only), adapter_path=adapter)


def raw_prediction(model, tokenizer, patch, module="./subject", variant="original", *, sampler=None):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(
        prompt_messages(patch, module, variant), tokenize=True,
        add_generation_prompt=True, return_dict=False,
    )
    if len(prompt) > 7000:
        raise ValueError("Select a smaller single-function diff (prompt limit 7,000 tokens).")
    return generate(
        model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
        sampler=make_sampler(temp=TEMPERATURE) if sampler is None else sampler, verbose=False,
    )


def predictor(adapter=None, *, variant="original", cache_only=False):
    model, tokenizer = load_model(adapter, cache_only)

    def predict(patch, module="./subject"):
        return normalized_answer(raw_prediction(model, tokenizer, patch, module, variant))

    return predict


def diff_module(patch):
    after = re.findall(r"^\+\+\+ (?:b/)?([^\t\r\n]+)(?:\t[^\r\n]*)?\r?$", patch, re.M)
    before = re.findall(r"^--- (?:a/)?([^\t\r\n]+)(?:\t[^\r\n]*)?\r?$", patch, re.M)
    if len(after) == 1 and before == after and re.search(r"^@@ ", patch, re.M):
        name = Path(after[0]).name
        if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.ts", name) and not name.endswith(".d.ts"):
            return "./" + Path(name).stem
    raise ValueError("Supply a textual unified diff that modifies one existing, unquoted .ts source path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--download", action="store_true", help="Resolve the pinned snapshot and print its path.")
    model.add_argument("--adapter", help="Directory containing a trained adapter. Omit for baseline.")
    parser.add_argument("--cache-only", action="store_true", help="Use cached weights without contacting Hugging Face.")
    parser.add_argument("--prompt", choices=PROMPTS, default="original")
    args = parser.parse_args()
    try:
        if args.download:
            print(model_path(args.cache_only))
            return
        patch = sys.stdin.read()
        module = diff_module(patch)
        print(predictor(args.adapter, variant=args.prompt, cache_only=args.cache_only)(patch, module))
    except ImportError as error:
        parser.exit(1, f"Install the local model dependencies with uv sync --locked --python 3.12. {error}\n")
    except Exception as error:
        parser.exit(1, f"Generation setup or inference failed: {error}\n")


if __name__ == "__main__":
    main()
