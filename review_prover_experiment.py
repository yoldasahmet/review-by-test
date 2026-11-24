"""Run or replay the prover experiment, the third round of the review work.

Each training bug PR carries candidate inputs that Vitest has confirmed expose the
bug. The base model scores the canonical test at every candidate, and the most
likely one becomes the training answer, so the adapter keeps the model's own habits
where they already work and only moves where they fail. Two adapter sizes train
on those answers. Validation and the earlier development cases choose between them
and the untuned model, and the choice is scored once on a fresh locked split
against acceptance checks fixed in advance. Only run and train require MLX.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from ablation_experiment import expected_literal
from experiment import read_json, write_json
from generate import MAX_TOKENS, MODEL, REVISION, model_path, prompt_messages
from grammar_experiment import assertion_parts, environment, grammar_settings
from review_prover_fixtures import ROOT, fingerprint, locked_cases, verify_review_data
from review_tuning_experiment import generate_proofs, proof_count, prover_summary, scored_split, trained

ARCHIVE = ROOT / "results/review-prover"
VARIANTS = {
    "gentle": dict(num_layers=8, lora_parameters=dict(rank=8, scale=16.0, dropout=0.0), learning_rate=5e-5),
    "wide": dict(num_layers=16, lora_parameters=dict(rank=16, scale=16.0, dropout=0.0), learning_rate=1e-4),
}
CHECKPOINTS = (100, 200, 300)
SELECTION_SPLITS = ("valid", "dev")
ACCEPTANCE = (
    "The chosen prover proves more bug PRs than the untuned model on validation, on the "
    "development cases and on the fresh test split, and proves no refactor on any of them."
)


def design():
    return dict(
        dataset_sha256=fingerprint(), model=MODEL, revision=REVISION,
        prompt="original", patch="the pull request reversed into the revert a reviewer would suggest",
        grammar=grammar_settings(), max_tokens=MAX_TOKENS,
        training_answer="the candidate test the untuned model scores highest, among inputs that expose the bug",
        variants=VARIANTS, checkpoints=list(CHECKPOINTS),
        selection="most proofs on validation and development bug PRs together; ties keep the untuned "
                  "model, then the earlier candidate",
        expected_value="main's return at the model's input, measured with Vitest",
        acceptance=ACCEPTANCE,
    )


def answer(name, argument, expected):
    prefix, middle, suffix = assertion_parts(name)
    return prefix + expected_literal(argument) + middle + expected_literal(expected) + suffix


def of(cases, split, verdict=None):
    return [item.case for item in cases if item.case.split == split and verdict in (None, item.case.verdict)]


def candidate_order():
    return [("base", 0)] + [(variant, step) for variant in VARIANTS for step in CHECKPOINTS]


def label(variant, step):
    return "base" if variant == "base" else f"{variant}-{step}"


# Training answers chosen by the untuned model.

def candidate_scores(model, tokenizer, item):
    """Log-probability of the canonical test at each candidate input, reusing the prompt cache."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    chat = prompt_messages(item.case.reversed_patch)
    prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False)
    cache = make_prompt_cache(model)
    last = model(mx.array(prompt)[None], cache=cache)[0, -1]
    scores = []
    for argument, expected in item.candidates:
        full = tokenizer.apply_chat_template(
            chat + [{"role": "assistant", "content": answer(item.case.name, argument, expected)}],
            tokenize=True, return_dict=False,
        )
        if full[:len(prompt)] != prompt:
            raise RuntimeError("The chat template does not extend the generation prompt.")
        tokens = full[len(prompt):]
        out = model(mx.array(tokens)[None], cache=cache)[0]
        steps = mx.concatenate([last[None], out[:-1]]).astype(mx.float32)
        logprobs = steps - mx.logsumexp(steps, axis=-1, keepdims=True)
        scores.append(mx.take_along_axis(logprobs, mx.array(tokens)[:, None], axis=-1).sum().item())
        trim_prompt_cache(cache, len(tokens))
    return scores


def score_targets(cases):
    import mlx.core as mx

    from review_experiment import loaded

    model, tokenizer, load_seconds = loaded()
    started = perf_counter()
    rows = []
    for item in cases:
        if item.case.split in ("train", "valid") and item.candidates:
            rows.append(dict(split=item.case.split, name=item.case.name, change=item.case.change,
                             scores=candidate_scores(model, tokenizer, item)))
    return dict(rows=rows, seconds=perf_counter() - started, model_load_seconds=load_seconds,
                peak_mlx_bytes=mx.get_peak_memory())


def training_rows(cases, targets, split):
    items = [item for item in cases if item.case.split in ("train", "valid") and item.candidates]
    if [(t["split"], t["name"], t["change"]) for t in targets["rows"]] != [
            (i.case.split, i.case.name, i.case.change) for i in items]:
        raise ValueError("Saved candidate scores do not match the locked fixtures.")
    rows = []
    for item, target in zip(items, targets["rows"], strict=True):
        if item.case.split != split:
            continue
        best = max(range(len(target["scores"])), key=lambda i: (target["scores"][i], -i))
        argument, expected = item.candidates[best]
        rows.append({"messages": prompt_messages(item.case.reversed_patch) + [
            {"role": "assistant", "content": answer(item.case.name, argument, expected)}]})
    return rows


def rows_sha256(cases, targets):
    text = json.dumps([training_rows(cases, targets, split) for split in ("train", "valid")], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def target_summary(cases, targets):
    """How often the chosen answer differs from the fixture's own witness."""
    items = [item for item in cases if item.case.split in ("train", "valid") and item.candidates]
    moved = sum(
        max(range(len(t["scores"])), key=lambda i: (t["scores"][i], -i)) != 0
        for item, t in zip(items, targets["rows"], strict=True)
    )
    return dict(bug_prs=len(items), single_candidate=sum(len(item.candidates) == 1 for item in items),
                answer_differs_from_fixture_witness=moved)


# Training.

def train(cases, variant, directory, targets_path):
    import mlx.core as mx
    import numpy as np
    import yaml
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_local_dataset

    targets = read_json(targets_path)
    directory.mkdir(parents=True)
    data = directory / "data"
    data.mkdir()
    for split in ("train", "valid"):
        with (data / f"{split}.jsonl").open("x") as out:
            for row in training_rows(cases, targets, split):
                print(json.dumps(row), file=out)
    config = yaml.safe_load((ROOT / "train.yaml").read_text())
    interval, steps = CHECKPOINTS[0], CHECKPOINTS[-1]
    config.update(VARIANTS[variant], iters=steps, steps_per_report=interval, steps_per_eval=interval,
                  save_every=interval, val_batches=-1)
    write_json(directory / "config.json", dict(
        task=f"prover, {variant}", dataset_sha256=fingerprint(), model_revision=REVISION,
        training_rows_sha256=rows_sha256(cases, targets), mlx_lm=config,
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
    metrics = dict(elapsed_seconds=elapsed, steps_per_second=steps / elapsed,
                   peak_mlx_bytes=mx.get_peak_memory(), longest_chat_tokens=max(lengths),
                   train_rows=len(train_set), valid_rows=len(valid_set))
    write_json(directory / "metrics.json", metrics)
    print(f"{variant} prover training, {steps} steps: {elapsed:.1f}s", flush=True)
    for step in CHECKPOINTS:
        checkpoint = directory / f"step-{step}"
        checkpoint.mkdir()
        shutil.copyfile(directory / "adapters/adapter_config.json", checkpoint / "adapter_config.json")
        (checkpoint / "adapters.safetensors").hardlink_to(directory / f"adapters/{step:07d}_adapters.safetensors")


def train_in_subprocess(variant, directory, targets_path):
    # A fresh process keeps training memory out of the inference measurements.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "train", variant, str(directory),
                    str(targets_path)], check=True)


# Selection, acceptance and summaries. Everything below runs without MLX.

def select(selection_proofs):
    order = candidate_order()
    best = max(order, key=lambda c: (selection_proofs[label(*c)], c == ("base", 0), -order.index(c)))
    return dict(variant=best[0], step=best[1], label=label(*best), selection_proofs=selection_proofs[label(*best)])


def acceptance(chosen, proofs):
    base, picked = "base", chosen["label"]
    checks = {f"beats_base_on_{split}": proof_count(proofs[split][picked]) > proof_count(proofs[split][base])
              for split in ("valid", "dev", "test")}
    checks["no_refactor_proofs"] = all(
        row["proof"]["status"] != "red_green"
        for split in ("dev", "test") for rows in proofs[split].values() for row in rows if row["gold"] == "ok"
    )
    return dict(checks=checks, acceptable=all(checks.values()))


def scored(cases, outputs):
    return {split: {name: scored_split(of(cases, split, "bug" if split == "valid" else None), value["outputs"])
                    for name, value in by_name.items()} for split, by_name in outputs.items()}


def selection_counts(proofs):
    return {name: sum(proof_count(proofs[split][name]) for split in SELECTION_SPLITS) for name in proofs["valid"]}


def summarize(cases, references, targets, proofs, chosen):
    return dict(
        targets=target_summary(cases, targets), training_rows_sha256=rows_sha256(cases, targets),
        selection_proofs=selection_counts(proofs), chosen=chosen,
        provers={split: {name: prover_summary(rows, references) for name, rows in by_name.items()}
                 for split, by_name in proofs.items()},
        acceptance=acceptance(chosen, proofs),
    )


# The full run. Each stage is saved as it finishes, so --resume continues after a crash.

def run(cases, directory, resume=False):
    if resume:
        if read_json(directory / "design.json") != design():
            raise ValueError(f"{directory} was started with different fixtures or settings.")
    else:
        if directory.exists():
            raise ValueError(f"{directory} already exists. Add --resume to continue that run, "
                             "or choose a new run directory.")
        directory.mkdir(parents=True)
        write_json(directory / "design.json", design())
    loaded_stages = []

    def stage(name, compute):
        path = directory / f"{name}.json"
        if path.exists():
            loaded_stages.append(name)
            return read_json(path)
        value = compute()
        write_json(path, value)
        return value

    references = verify_review_data([item.case for item in cases])
    targets = stage("targets", lambda: score_targets(cases))
    print("targets", target_summary(cases, targets), flush=True)

    fits = {}
    for variant in VARIANTS:
        folder = directory / f"fit-{variant}"
        if trained(folder, CHECKPOINTS):
            loaded_stages.append(f"fit-{variant}")
        elif folder.exists():
            raise ValueError(f"{folder} holds an unfinished training run. Start a new run directory.")
        else:
            train_in_subprocess(variant, folder, directory / "targets.json")
        if read_json(folder / "config.json")["training_rows_sha256"] != rows_sha256(cases, targets):
            raise ValueError(f"{folder} trained on different answers.")
        fits[variant] = folder

    outputs = {"valid": {}, "dev": {}}
    for variant, step in candidate_order():
        name = label(variant, step)
        adapter = None if variant == "base" else fits[variant] / f"step-{step}"
        outputs["valid"][name] = stage(f"valid-{name}", lambda a=adapter: generate_proofs(of(cases, "valid", "bug"), a))
        outputs["dev"][name] = stage(f"dev-{name}", lambda a=adapter: generate_proofs(of(cases, "dev"), a))
    proofs = scored(cases, outputs)
    selection_proofs = selection_counts(proofs)
    print("selection proofs", selection_proofs, flush=True)
    chosen = select(selection_proofs)
    if not (directory / "selection.json").exists():
        write_json(directory / "selection.json", chosen)
    elif read_json(directory / "selection.json") != chosen:
        raise ValueError("Saved selection differs from the validation results.")
    print("selection", chosen, flush=True)

    test = of(cases, "test")
    outputs["test"] = {"base": stage("test-base", lambda: generate_proofs(test))}
    if chosen["label"] != "base":
        adapter = fits[chosen["variant"]] / f"step-{chosen['step']}"
        outputs["test"][chosen["label"]] = stage(f"test-{chosen['label']}", lambda: generate_proofs(test, adapter))

    proofs |= scored(cases, {"test": outputs["test"]})
    summary = summarize(cases, references, targets, proofs, chosen)
    result = dict(
        design=design(), references=references, targets=targets, outputs=outputs, summary=summary,
        training={variant: read_json(folder / "metrics.json") for variant, folder in fits.items()},
        resumed_stages=loaded_stages, environment=environment(),
    )
    saved = directory / "comparison.json"
    if not saved.exists():
        write_json(saved, result)
    else:
        # Resuming a finished run recomputes everything; the saved result must match.
        ignored = ("resumed_stages", "environment")
        previous = read_json(saved)
        if {k: v for k, v in previous.items() if k not in ignored} != {k: v for k, v in result.items() if k not in ignored}:
            raise ValueError(f"{saved} differs from the resumed run.")
        print(f"{saved} already holds this result.", flush=True)
    print_summary(summary)


def print_summary(summary):
    print("targets", summary["targets"], flush=True)
    print("selection proofs (validation and development)", summary["selection_proofs"], flush=True)
    print("chosen", summary["chosen"]["label"], flush=True)
    for split, by_name in summary["provers"].items():
        for name, counts in by_name.items():
            print(f"{split} {name}: {counts['proofs']}/{counts['bug_prs']} proofs, "
                  f"{counts['proofs_by_difference_density']}", flush=True)
    print("acceptance", summary["acceptance"], flush=True)


def replay(cases, path):
    saved = read_json(path)
    if saved["design"] != design():
        raise ValueError("Saved comparison does not match the frozen fixtures and settings.")
    references = verify_review_data([item.case for item in cases])
    if saved["references"] != references:
        raise ValueError("Saved fixture checks differ from replay.")
    proofs = scored(cases, saved["outputs"])
    chosen = select(selection_counts(proofs))
    summary = summarize(cases, references, saved["targets"], proofs, chosen)
    if summary != saved["summary"]:
        raise ValueError("Saved summary differs from replay.")
    print_summary(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-data", help="Recheck every witness and refactor with Vitest.")
    experiment = commands.add_parser("run", help="Score answers, train, select, then test once.")
    experiment.add_argument("directory", type=Path)
    experiment.add_argument("--resume", action="store_true", help="Continue a run directory after a crash.")
    training = commands.add_parser("train", help="Train one adapter variant.")
    training.add_argument("variant", choices=tuple(VARIANTS))
    training.add_argument("directory", type=Path)
    training.add_argument("targets", type=Path)
    saved = commands.add_parser("replay", help="Rescore the saved comparison without loading MLX.")
    saved.add_argument("path", type=Path, nargs="?", default=ARCHIVE / "comparison.json")
    args = parser.parse_args()
    try:
        cases = locked_cases()
        if args.command == "verify-data":
            verify_review_data([item.case for item in cases])
            print("All witnesses and refactors check out.", flush=True)
        elif args.command == "train":
            train(cases, args.variant, args.directory.resolve(), args.targets.resolve())
        elif args.command == "run":
            run(cases, args.directory.resolve(), args.resume)
        else:
            replay(cases, args.path)
    except ImportError as error:
        parser.exit(1, f"Install the local model dependencies with uv sync --locked --python 3.12. {error}\n")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Prover experiment failed: {error}\n")


if __name__ == "__main__":
    main()
