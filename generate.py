"""Generate a test from a unified diff on stdin."""
import argparse
import re
import sys

from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from fixtures import messages

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
REVISION = "b3252a2f97102b1fb1571fec2c9b27219a8536be"

def model_path():
    return snapshot_download(MODEL, revision=REVISION)

def predictor(adapter=None):
    model, tokenizer = load(model_path(), adapter_path=adapter)
    def predict(patch, module="./subject"):
        prompt = tokenizer.apply_chat_template(messages(patch, module), tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(prompt)) > 7000:
            raise ValueError("Select a smaller single-function diff (prompt limit 7,000 tokens).")
        answer = generate(model, tokenizer, prompt=prompt, max_tokens=192, verbose=False)
        fence = re.fullmatch(r"```(?:typescript|ts|javascript|js)?\n(.*?)\n```", answer.strip(), re.S)
        return fence[1] if fence else answer.strip()
    return predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="Path to trained adapters. Omit for baseline.")
    args = parser.parse_args()
    patch = sys.stdin.read()
    if not patch.strip():
        parser.error("Supply a unified diff on stdin.")
    paths = re.findall(r"^\+\+\+ (?:b/)?([^\t\n]+)", patch, re.M)
    if len(paths) != 1 or not paths[0].endswith(".ts"):
        parser.error("Supply a unified diff for exactly one TypeScript source file.")
    from pathlib import Path
    print(predictor(args.adapter)(patch, "./" + Path(paths[0]).stem))
