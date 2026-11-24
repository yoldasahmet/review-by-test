"""Run or replay the second review experiment.

Round two changes the training, not the task. The reviewer adapter trains on
structural refactors and slipped rewrites, with more trainable layers and a higher
LoRA rank, and its threshold is chosen on validation. A second adapter trains the
prover to pick witness inputs. Both are measured on the money split from round one,
which is now seen, and on a fresh locked split. Only run and train require MLX.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from experiment import read_json, write_json
from generate import MAX_TOKENS, MODEL, REVISION, model_path, prompt_messages
from grammar_experiment import LITERAL, assertion_parts, environment, grammar_settings
from review_experiment import (
    ANSWERS, REVIEW_PROMPT, balanced_accuracy, calibrate, confusion, density_band, flagged, gate,
    loaded, pairwise_ranking, prove, review, review_messages, score_proofs,
)
from review_tuning_fixtures import (
    ROOT, TRAINING_LIKE_REFACTORS, fingerprint, locked_cases, verify_review_data, witness_literals,
)

ARCHIVE = ROOT / "results/review-tuning"
ADAPTER = dict(num_layers=16, lora_parameters=dict(rank=16, scale=16.0, dropout=0.0))
REVIEWER_CHECKPOINTS = (120, 240, 360, 480)
PROVER_CHECKPOINTS = (80, 160, 240)
SPLITS = ("seen", "test")
PIPELINES = {
    "base_reviewer_base_prover": ("base", "base"),
    "tuned_reviewer_base_prover": ("tuned", "base"),
    "base_reviewer_tuned_prover": ("base", "tuned"),
    "tuned_reviewer_tuned_prover": ("tuned", "tuned"),
    "tuned_prover_only": ("everything", "tuned"),
}


def design():
    return dict(
        dataset_sha256=fingerprint(), model=MODEL, revision=REVISION,
        adapter=ADAPTER,
        reviewer=dict(
            prompt=REVIEW_PROMPT, answers=ANSWERS, checkpoints=list(REVIEWER_CHECKPOINTS),
            selection="checkpoint and threshold with the best validation balanced accuracy; "
                      "ties keep fewer steps, then the threshold nearest 0",
        ),
        prover=dict(
            patch="the pull request reversed into the revert a reviewer would suggest",
            prompt="original", grammar=grammar_settings(), max_tokens=MAX_TOKENS,
            training_answer="the canonical grammar test at the witness, asserting main's return",
            checkpoints=list(PROVER_CHECKPOINTS),
            selection="most validation proofs, with the base model as step 0; ties keep fewer steps",
            expected_value="main's return at the model's input, measured with Vitest",
        ),
        gate="comment only when the reviewer flags a bug and the proof fails on the pull request and passes on main",
    )


def of(cases, split, verdict=None):
    return [case for case in cases if case.split == split and verdict in (None, case.verdict)]


# Training data.

def reviewer_rows(cases, split):
    return [
        {"messages": review_messages(case.patch) + [{"role": "assistant", "content": ANSWERS[case.verdict]}]}
        for case in of(cases, split)
    ]


def prover_rows(cases, split):
    rows = []
    for case in of(cases, split, "bug"):
        argument, expected = witness_literals(case)
        if argument is None or expected is None or not re.fullmatch(LITERAL, argument):
            raise ValueError(f"Witness for {case.split}/{case.name} is outside the grammar.")
        prefix, middle, suffix = assertion_parts(case.name)
        answer = prefix + argument + middle + expected + suffix
        rows.append({"messages": prompt_messages(case.reversed_patch) + [{"role": "assistant", "content": answer}]})
    return rows


def train(cases, task, directory):
    import mlx.core as mx
    import numpy as np
    import yaml
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_local_dataset

    rows, checkpoints = {"reviewer": (reviewer_rows, REVIEWER_CHECKPOINTS),
                         "prover": (prover_rows, PROVER_CHECKPOINTS)}[task]
    directory.mkdir(parents=True)
    data = directory / "data"
    data.mkdir()
    for split in ("train", "valid"):
        with (data / f"{split}.jsonl").open("x") as out:
            for row in rows(cases, split):
                print(json.dumps(row), file=out)

    config = yaml.safe_load((ROOT / "train.yaml").read_text())
    interval, steps = checkpoints[0], checkpoints[-1]
    config.update(ADAPTER, iters=steps, steps_per_report=interval, steps_per_eval=interval,
                  save_every=interval, val_batches=-1)
    write_json(directory / "config.json", dict(
        task=task, dataset_sha256=fingerprint(), model_revision=REVISION, mlx_lm=config,
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
    print(f"{task} training, {steps} steps: {elapsed:.1f}s, {metrics['peak_mlx_bytes'] / 1e9:.2f} GB MLX peak",
          flush=True)
    for step in checkpoints:
        checkpoint = directory / f"step-{step}"
        checkpoint.mkdir()
        shutil.copyfile(directory / "adapters/adapter_config.json", checkpoint / "adapter_config.json")
        (checkpoint / "adapters.safetensors").hardlink_to(directory / f"adapters/{step:07d}_adapters.safetensors")


def train_in_subprocess(task, directory):
    # A fresh process keeps training memory out of the inference measurements.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "train", task, str(directory)], check=True)


def trained(directory, checkpoints):
    return (directory / "metrics.json").exists() and all(
        (directory / f"step-{step}/adapters.safetensors").exists() for step in checkpoints
    )


def generate_proofs(cases, adapter=None):
    import mlx.core as mx

    model, tokenizer, load_seconds = loaded(adapter)
    outputs, seconds = prove(model, tokenizer, cases)
    return dict(outputs=outputs, generation_seconds=seconds, model_load_seconds=load_seconds,
                peak_mlx_bytes=mx.get_peak_memory())


# Decisions and metrics. Everything below runs without MLX.

def select(base_valid, reviewer_valid, prover_valid_scores):
    candidates = []
    for step in REVIEWER_CHECKPOINTS:
        rows = reviewer_valid[str(step)]["rows"]
        threshold = calibrate(rows)
        candidates.append((balanced_accuracy(rows, threshold), -step, -abs(threshold), step, threshold))
    accuracy, _, _, reviewer_step, reviewer_threshold = max(candidates)
    prover_step = max((0, *PROVER_CHECKPOINTS), key=lambda step: (prover_valid_scores[str(step)], -step))
    return dict(
        dataset_sha256=fingerprint(), reviewer_step=reviewer_step, reviewer_threshold=reviewer_threshold,
        reviewer_validation_balanced_accuracy=str(accuracy),
        base_threshold=calibrate(base_valid["rows"]), prover_step=prover_step,
        prover_validation_proofs=prover_valid_scores[str(prover_step)],
    )


def proof_count(rows):
    return sum(row["proof"]["status"] == "red_green" for row in rows)


def prover_summary(rows, references):
    bands = {(row["split"], row["name"]): density_band(row) for row in references}
    by_band = {band: dict(bug_prs=0, proofs=0) for band in ("narrow", "medium", "wide")}
    bugs = [row for row in rows if row["gold"] == "bug"]
    for row in bugs:
        band = by_band[bands[row["split"], row["name"]]]
        band["bug_prs"] += 1
        band["proofs"] += row["proof"]["status"] == "red_green"
    return dict(
        bug_prs=len(bugs), proofs=proof_count(bugs),
        own_assertion_red_green=sum(row["model_assertion"]["status"] == "red_green" for row in bugs),
        distinguishing_inputs=sum(row["model_assertion"].get("distinguishes") is True for row in bugs),
        proofs_on_refactor_prs=proof_count([row for row in rows if row["gold"] == "ok"]),
        proofs_by_difference_density=by_band,
    )


def split_summary(cases, reviewers, proofs, chosen, references):
    decisions = {
        "base": [flagged(row) for row in reviewers["base"]["rows"]],
        "base_calibrated": [flagged(row, chosen["base_threshold"]) for row in reviewers["base"]["rows"]],
        "tuned": [flagged(row, chosen["reviewer_threshold"]) for row in reviewers["tuned"]["rows"]],
        "tuned_threshold_0": [flagged(row) for row in reviewers["tuned"]["rows"]],
        "everything": [True for _ in cases],
    }
    return dict(
        reviewers={arm: gate(cases, decisions[arm], proofs["base"], TRAINING_LIKE_REFACTORS)
                   for arm in ("base", "base_calibrated", "tuned", "tuned_threshold_0")},
        ranking={arm: str(pairwise_ranking(reviewers[arm]["rows"])) for arm in ("base", "tuned")},
        provers={arm: prover_summary(proofs[arm], references) for arm in ("base", "tuned")},
        pipelines={name: gate(cases, decisions[reviewer], proofs[prover], TRAINING_LIKE_REFACTORS)
                   for name, (reviewer, prover) in PIPELINES.items()},
    )


def validation_summary(base_valid, reviewer_valid, prover_valid_scores):
    rows = {"base": base_valid["rows"], **{f"step_{s}": reviewer_valid[str(s)]["rows"] for s in REVIEWER_CHECKPOINTS}}
    return dict(
        reviewers={name: dict(
            confusion(r), balanced_accuracy=str(balanced_accuracy(r)), ranking=str(pairwise_ranking(r)),
            calibrated_balanced_accuracy=str(balanced_accuracy(r, calibrate(r))),
        ) for name, r in rows.items()},
        prover_proofs=prover_valid_scores,
    )


def scored_split(cases, outputs):
    proofs = score_proofs(cases, outputs)
    for case, row in zip(cases, proofs, strict=True):
        row["split"] = case.split
    return proofs


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

    def fit(task, checkpoints):
        folder = directory / f"fit-{task}"
        if trained(folder, checkpoints):
            loaded_stages.append(f"fit-{task}")
        elif folder.exists():
            raise ValueError(f"{folder} holds an unfinished training run. Start a new run directory.")
        else:
            train_in_subprocess(task, folder)
        return folder

    references = verify_review_data(cases)
    valid, valid_bugs = of(cases, "valid"), of(cases, "valid", "bug")

    base_valid = stage("valid-base-reviewer", lambda: review(valid))
    reviewer_fit = fit("reviewer", REVIEWER_CHECKPOINTS)
    reviewer_valid = {str(step): stage(f"valid-reviewer-step-{step}",
                                       lambda step=step: review(valid, reviewer_fit / f"step-{step}"))
                      for step in REVIEWER_CHECKPOINTS}
    print("reviewer validation", {k: confusion(v["rows"]) for k, v in reviewer_valid.items()}, flush=True)

    prover_valid = {"0": stage("valid-base-prover", lambda: generate_proofs(valid_bugs))}
    prover_fit = fit("prover", PROVER_CHECKPOINTS)
    for step in PROVER_CHECKPOINTS:
        prover_valid[str(step)] = stage(f"valid-prover-step-{step}",
                                        lambda step=step: generate_proofs(valid_bugs, prover_fit / f"step-{step}"))
    prover_valid_scores = {step: proof_count(scored_split(valid_bugs, value["outputs"]))
                           for step, value in prover_valid.items()}
    print("prover validation proofs", prover_valid_scores, flush=True)

    chosen = select(base_valid, reviewer_valid, prover_valid_scores)
    if not (directory / "selection.json").exists():
        write_json(directory / "selection.json", chosen)
    elif read_json(directory / "selection.json") != chosen:
        raise ValueError("Saved selection differs from the validation results.")
    print("selection", chosen, flush=True)

    evaluations = {}
    for split in SPLITS:
        split_cases = of(cases, split)
        reviewer_step, prover_step = chosen["reviewer_step"], chosen["prover_step"]
        base = stage(f"{split}-base-reviewer", lambda: review(split_cases))
        base_prover = stage(f"{split}-base-prover", lambda: generate_proofs(split_cases))
        tuned = stage(f"{split}-tuned-reviewer", lambda: review(split_cases, reviewer_fit / f"step-{reviewer_step}"))
        tuned_prover = base_prover if prover_step == 0 else stage(
            f"{split}-tuned-prover", lambda: generate_proofs(split_cases, prover_fit / f"step-{prover_step}"))
        evaluations[split] = dict(reviewers=dict(base=base, tuned=tuned),
                                  provers=dict(base=base_prover, tuned=tuned_prover))

    result = assemble(cases, references, chosen, base_valid, reviewer_valid, prover_valid,
                      prover_valid_scores, evaluations)
    result.update(
        training=dict(reviewer=read_json(reviewer_fit / "metrics.json"),
                      prover=read_json(prover_fit / "metrics.json")),
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
    print_summary(result)


def assemble(cases, references, chosen, base_valid, reviewer_valid, prover_valid, prover_valid_scores,
             evaluations):
    splits = {}
    for split, evaluation in evaluations.items():
        split_cases = of(cases, split)
        proofs = {arm: scored_split(split_cases, value["outputs"]) for arm, value in evaluation["provers"].items()}
        splits[split] = dict(
            reviewers=evaluation["reviewers"], provers=evaluation["provers"], proofs=proofs,
            summary=split_summary(split_cases, evaluation["reviewers"], proofs, chosen, references),
        )
    return dict(
        design=design(), references=references, selection=chosen,
        validation=dict(base_reviewer=base_valid, reviewer=reviewer_valid, prover=prover_valid,
                        summary=validation_summary(base_valid, reviewer_valid, prover_valid_scores)),
        splits=splits,
    )


def print_summary(result):
    for split, value in result["splits"].items():
        summary = value["summary"]
        print(f"== {split}", flush=True)
        for arm, counts in summary["reviewers"].items():
            print(f"reviewer {arm}: flagged {counts['flagged_bug_prs']}/{counts['bug_prs']} bugs, "
                  f"{counts['flagged_refactor_prs']}/{counts['refactor_prs']} refactors", flush=True)
        print("ranking", summary["ranking"], flush=True)
        for arm, counts in summary["provers"].items():
            print(f"prover {arm}: {counts['proofs']}/{counts['bug_prs']} proofs, "
                  f"{counts['proofs_by_difference_density']}", flush=True)
        for name, counts in summary["pipelines"].items():
            print(f"pipeline {name}: {counts['comments_on_bug_prs']}/{counts['bug_prs']} proven comments, "
                  f"{counts['comments_on_refactor_prs']}/{counts['refactor_prs']} on refactors", flush=True)


def replay(cases, path):
    saved = read_json(path)
    if saved["design"] != design():
        raise ValueError("Saved comparison does not match the frozen fixtures and settings.")
    references = verify_review_data(cases)
    if saved["references"] != references:
        raise ValueError("Saved fixture checks differ from replay.")
    for label, rows, split in [("base validation", saved["validation"]["base_reviewer"]["rows"], "valid"),
                               *[(f"step {s} validation", saved["validation"]["reviewer"][str(s)]["rows"], "valid")
                                 for s in REVIEWER_CHECKPOINTS],
                               *[(f"{split} {arm}", saved["splits"][split]["reviewers"][arm]["rows"], split)
                                 for split in SPLITS for arm in ("base", "tuned")]]:
        expected = [(c.name, c.split, c.verdict, c.change) for c in of(cases, split)]
        if [(r["name"], r["split"], r["gold"], r["change"]) for r in rows] != expected:
            raise ValueError(f"Saved {label} rows do not match the locked fixtures.")
    valid_bugs = of(cases, "valid", "bug")
    scores = {step: proof_count(scored_split(valid_bugs, value["outputs"]))
              for step, value in saved["validation"]["prover"].items()}
    chosen = select(saved["validation"]["base_reviewer"], saved["validation"]["reviewer"], scores)
    if chosen != saved["selection"]:
        raise ValueError("Saved selection differs from replay.")
    evaluations = {split: dict(reviewers=saved["splits"][split]["reviewers"],
                               provers=saved["splits"][split]["provers"]) for split in SPLITS}
    result = assemble(cases, references, chosen, saved["validation"]["base_reviewer"],
                      saved["validation"]["reviewer"], saved["validation"]["prover"], scores, evaluations)
    for key in ("validation", "splits"):
        if result[key] != saved[key]:
            raise ValueError(f"Saved {key} differ from replay.")
    print_summary(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-data", help="Recheck every witness and refactor with Vitest.")
    experiment = commands.add_parser("run", help="Train, select on validation, then evaluate.")
    experiment.add_argument("directory", type=Path)
    experiment.add_argument("--resume", action="store_true", help="Continue a run directory after a crash.")
    training = commands.add_parser("train", help="Train one adapter.")
    training.add_argument("task", choices=("reviewer", "prover"))
    training.add_argument("directory", type=Path)
    saved = commands.add_parser("replay", help="Rescore the saved comparison without loading MLX.")
    saved.add_argument("path", type=Path, nargs="?", default=ARCHIVE / "comparison.json")
    args = parser.parse_args()
    try:
        cases = locked_cases()
        if args.command == "verify-data":
            rows = verify_review_data(cases)
            print(Counter((row["split"], row["verdict"]) for row in rows), flush=True)
        elif args.command == "train":
            train(cases, args.task, args.directory.resolve())
        elif args.command == "run":
            run(cases, args.directory.resolve(), args.resume)
        else:
            replay(cases, args.path)
    except ImportError as error:
        parser.exit(1, f"Install the local model dependencies with uv sync --locked --python 3.12. {error}\n")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Review tuning experiment failed: {error}\n")


if __name__ == "__main__":
    main()
