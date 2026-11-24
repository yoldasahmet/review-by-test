"""Compare the original prompt and explicit retry instruction on historical validation."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from evaluate import Trial, score_trials
from fixtures import cases, diff, messages
from generate import MAX_TOKENS, MODEL, REVISION, load_model, normalized_answer

OLD = "Choose an input that fails before the fix."
NEW = ("Choose an input. Compute the old and fixed functions' return values for that input. "
       "If the values are the same, choose another input. If they differ, assert the fixed "
       "function's return value using toBe(...). Output only the Vitest test.")
adapter = ROOT / "adapters"
records, trials = [], []
for mode in ("original", "fine-tuned"):
    model, tokenizer = load_model(str(adapter) if mode == "fine-tuned" else None, cache_only=True)
    for variant in ("old", "new"):
        for name, before, after, _ in cases("valid"):
            patch = diff(before, after)
            chat = messages(patch)
            assert chat[0]["content"].endswith(OLD)
            if variant == "new":
                chat[0]["content"] = chat[0]["content"].removesuffix(OLD) + NEW
            prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False)
            raw = generate(model, tokenizer, prompt=prompt, sampler=make_sampler(temp=0),
                           max_tokens=MAX_TOKENS, verbose=False)
            trials.append(Trial(name, before, after, normalized_answer(raw)))
            records.append(dict(model=mode, prompt=variant, case=name, diff=patch, raw=raw))
        print(mode, variant, "generated", flush=True)
    del model, tokenizer
for row, outcome in zip(records, score_trials(trials), strict=True):
    row.update(outcome)
    print(row["model"], row["prompt"], row["case"], outcome)
report = dict(model=MODEL, revision=REVISION, adapter_sha256=hashlib.sha256(
    (adapter / "adapters.safetensors").read_bytes()).hexdigest(), old=OLD, new=NEW,
    temperature=0, max_tokens=MAX_TOKENS, split="fixtures.cases('valid')", records=records)
Path(__file__).with_name("results.json").write_text(json.dumps(report, indent=2) + "\n")
