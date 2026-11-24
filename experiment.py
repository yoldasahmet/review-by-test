"""Run or replay the locked benchmark. Only run and train require MLX."""
import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from benchmark_fixtures import ROOT, fingerprint, locked_cases
from evaluate import Trial, score_trials
from generate import (
    MAX_TOKENS, MODEL, PROMPTS, REVISION, TEMPERATURE, load_model, model_path,
    normalized_answer, prompt_messages, raw_prediction,
)

ARCHIVE = ROOT / "results/benchmark"
SMOKE_STEPS = 8
CHECKPOINTS = (24, 48)
STATUSES = ("red_green", "passes_both", "fails_on_fix", "outside_format")
SETTINGS = dict(model=MODEL, revision=REVISION, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, attempts=1)


def write_json(path, data):
    with path.open("x") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")


def read_json(path):
    return json.loads(path.read_text())


def scored_rows(cases, outputs):
    trials = [
        Trial(case.name, case.before, case.after, normalized_answer(output))
        for case, output in zip(cases, outputs, strict=True)
    ]
    return [
        dict(name=case.name, domain=case.domain, split=case.split, diff=case.patch,
             raw=output, **result)
        for case, output, result in zip(cases, outputs, score_trials(trials), strict=True)
    ]


def verify_data(cases):
    rows = scored_rows(cases, [case.reference for case in cases])
    failures = [row["name"] for row in rows if row["status"] != "red_green"]
    if failures:
        raise ValueError(f"Reference tests failed verification: {', '.join(failures)}")
    counts = Counter(case.split for case in cases)
    print("Verified red/green references:", dict(counts), flush=True)
    return rows


def training_rows(cases, split, prompt):
    return [
        {"messages": prompt_messages(case.patch, variant=prompt)
         + [{"role": "assistant", "content": case.reference}]}
        for case in cases if case.split == split
    ]


def train_adapter(cases, directory, prompt, steps):
    import mlx.core as mx
    import numpy as np
    import yaml
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_local_dataset

    directory.mkdir(parents=True)
    data = directory / "data"
    data.mkdir()
    for split in ("train", "valid"):
        with (data / f"{split}.jsonl").open("x") as out:
            for row in training_rows(cases, split, prompt):
                print(json.dumps(row), file=out)

    # Reuse the historical LoRA settings. Only the schedule and dataset change.
    config = yaml.safe_load((ROOT / "train.yaml").read_text())
    interval = SMOKE_STEPS if steps == SMOKE_STEPS else CHECKPOINTS[0]
    config.update(
        iters=steps, steps_per_report=interval, steps_per_eval=interval,
        save_every=interval, val_batches=-1,
    )
    write_json(directory / "config.json", dict(
        prompt=prompt, dataset_sha256=fingerprint(), model_revision=REVISION, mlx_lm=config,
    ))
    args = SimpleNamespace(**(CONFIG_DEFAULTS | config | dict(
        data=str(data), adapter_path=str(directory / "adapters"), model=model_path(cache_only=True),
    )))
    np.random.seed(args.seed)
    started = perf_counter()
    model, tokenizer = load(args.model, trust_remote_code=False)
    train_set, valid_set, _ = load_local_dataset(data, tokenizer, args)
    lengths = [len(dataset.process(row)[0]) for dataset in (train_set, valid_set) for row in dataset]
    if max(lengths) > args.max_seq_length:
        raise ValueError(f"Training chat length {max(lengths)} exceeds the {args.max_seq_length}-token limit.")
    train_model(args, model, train_set, valid_set)
    elapsed = perf_counter() - started
    metrics = dict(
        elapsed_seconds=elapsed, steps_per_second=steps / elapsed, peak_mlx_bytes=mx.get_peak_memory(),
        longest_chat_tokens=max(lengths),
    )
    write_json(directory / "metrics.json", metrics)
    print(f"Training {steps} steps: {elapsed:.2f}s, {metrics['peak_mlx_bytes'] / 1e9:.2f} GB MLX peak", flush=True)

    for step in range(interval, steps + 1, interval):
        checkpoint = directory / f"step-{step}"
        checkpoint.mkdir()
        shutil.copyfile(directory / "adapters/adapter_config.json", checkpoint / "adapter_config.json")
        (checkpoint / "adapters.safetensors").hardlink_to(directory / f"adapters/{step:07d}_adapters.safetensors")


def infer(cases, prompt, split, adapter=None):
    import mlx.core as mx

    cases = [case for case in cases if case.split == split]
    mx.clear_cache()
    mx.reset_peak_memory()
    started = perf_counter()
    model, tokenizer = load_model(str(adapter) if adapter else None, cache_only=True)
    load_seconds = perf_counter() - started
    raw, latency = [], []
    for case in cases:
        started = perf_counter()
        raw.append(raw_prediction(model, tokenizer, case.patch, variant=prompt))
        latency.append(perf_counter() - started)
    peak = mx.get_peak_memory()
    rows = scored_rows(cases, raw)
    for row, seconds in zip(rows, latency, strict=True):
        row["generation_seconds"] = seconds
    result = dict(
        split=split, prompt=prompt, settings=SETTINGS, rows=rows,
        model_load_seconds=load_seconds, median_generation_seconds=statistics.median(latency),
        peak_mlx_bytes=peak,
    )
    if adapter and split == "valid":
        from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
        from mlx_lm.tuner.trainer import evaluate

        dataset = ChatDataset(training_rows(cases, split, prompt), tokenizer, mask_prompt=True)
        result["validation_loss"] = evaluate(model, CacheDataset(dataset), 1, -1, max_seq_length=512)
    return result


def counts(result):
    found = Counter(row["status"] for row in result["rows"])
    return {status: found[status] for status in STATUSES}


def select_prompt(results):
    return max(PROMPTS, key=lambda prompt: counts(results[prompt])["red_green"])


def select_checkpoint(results):
    return max(CHECKPOINTS, key=lambda step: (counts(results[step])["red_green"], -results[step]["validation_loss"]))


def record(directory, label, result):
    write_json(directory / f"{label}.json", result)
    print(label, counts(result), flush=True)
    return result


def run_experiment(cases, directory):
    directory.mkdir(parents=True)
    write_json(directory / "references.json", verify_data(cases))
    hardware = dict(
        chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        macos=platform.mac_ver()[0], architecture=platform.machine(), python=platform.python_version(),
        node=subprocess.check_output(["node", "--version"], text=True).strip(),
        vitest=read_json(ROOT / "node_modules/vitest/package.json")["version"],
        **{package: version(package) for package in ("mlx", "mlx-lm", "huggingface-hub")},
    )
    write_json(directory / "environment.json", hardware)
    prompts = {
        prompt: record(directory, f"valid-{prompt}", infer(cases, prompt, "valid"))
        for prompt in PROMPTS
    }
    prompt = select_prompt(prompts)
    print(f"Selected prompt on validation: {prompt}", flush=True)

    for label, steps in (("smoke", SMOKE_STEPS), ("fit", CHECKPOINTS[-1])):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "train", str(directory / label),
             "--prompt", prompt, "--steps", str(steps)],
            check=True,
        )

    checkpoints = {
        step: record(directory, f"valid-step-{step}", infer(cases, prompt, "valid", directory / f"fit/step-{step}"))
        for step in CHECKPOINTS
    }
    step = select_checkpoint(checkpoints)
    write_json(directory / "selection.json", dict(
        dataset_sha256=fingerprint(), prompt=prompt, checkpoint_step=step, settings=SETTINGS,
    ))
    print(f"Selected step {step} on validation. Starting the locked test comparison.", flush=True)
    record(directory, "test-baseline", infer(cases, prompt, "test"))
    record(directory, "test-tuned", infer(cases, prompt, "test", directory / f"fit/step-{step}"))


def replay(cases, directory):
    selection = read_json(directory / "selection.json")
    if selection["dataset_sha256"] != fingerprint() or selection["settings"] != SETTINGS:
        raise ValueError("Saved experiment does not match the dataset or inference settings.")
    if read_json(directory / "references.json") != verify_data(cases):
        raise ValueError("Saved references differ from replay.")

    def check(label, split, prompt):
        result = read_json(directory / f"{label}.json")
        saved = result["rows"]
        subset = [case for case in cases if case.split == split]
        if (result["settings"], result["split"], result["prompt"]) != (SETTINGS, split, prompt):
            raise ValueError(f"Saved settings differ for {label}.")
        expected = [{key: value for key, value in row.items() if key != "generation_seconds"} for row in saved]
        if expected != scored_rows(subset, [row["raw"] for row in saved]):
            raise ValueError(f"Saved scores differ for {label}.")
        print(label, counts(result), flush=True)
        return result

    prompts = {prompt: check(f"valid-{prompt}", "valid", prompt) for prompt in PROMPTS}
    prompt = select_prompt(prompts)
    checkpoints = {step: check(f"valid-step-{step}", "valid", prompt) for step in CHECKPOINTS}
    if prompt != selection["prompt"] or select_checkpoint(checkpoints) != selection["checkpoint_step"]:
        raise ValueError("Saved prompt or checkpoint selection differs.")
    check("test-baseline", "test", prompt)
    check("test-tuned", "test", prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-data", help="Recheck every frozen reference with Vitest.")
    run = commands.add_parser("run", help="Select on validation, then generate the two test comparisons.")
    run.add_argument("directory", type=Path)
    train = commands.add_parser("train", help="Train a local adapter on the frozen training split.")
    train.add_argument("directory", type=Path)
    train.add_argument("--prompt", choices=PROMPTS, required=True)
    train.add_argument("--steps", type=int, choices=(SMOKE_STEPS, CHECKPOINTS[-1]), default=CHECKPOINTS[-1])
    saved = commands.add_parser("replay", help="Verify saved generations with Vitest.")
    saved.add_argument("directory", type=Path, nargs="?", default=ARCHIVE)
    args = parser.parse_args()
    try:
        cases = locked_cases()
        if args.command == "verify-data":
            verify_data(cases)
        elif args.command == "replay":
            replay(cases, args.directory.resolve())
        elif args.command == "train":
            verify_data([case for case in cases if case.split != "test"])
            train_adapter(cases, args.directory.resolve(), args.prompt, args.steps)
        else:
            run_experiment(cases, args.directory.resolve())
    except ImportError as error:
        parser.exit(1, f"Install the local model dependencies with uv sync --locked --python 3.12. {error}\n")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")


if __name__ == "__main__":
    main()
