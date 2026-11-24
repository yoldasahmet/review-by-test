"""Run or replay the pull-request review experiment.

A reviewer model gives each pull request a verdict. A prover writes a Vitest test for
the suspected regression, and a comment is posted only when that test fails on the
pull request and passes on main. Only run and train require MLX.
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from ablation_experiment import expected_literal
from evaluate import Trial, score_trials
from experiment import read_json, write_json
from generate import MAX_TOKENS, MODEL, REVISION, load_model, model_path, normalized_answer
from grammar_experiment import assertion_parts, environment, grammar_settings
from review_fixtures import ROOT, TRAINING_LIKE_REFACTORS, fingerprint, locked_cases, verify_review_data

ARCHIVE = ROOT / "results/review"
CHECKPOINTS = (60, 120, 180, 240)
ANSWERS = {"bug": '{"verdict": "bug"}', "ok": '{"verdict": "ok"}'}
REVIEW_PROMPT = (
    "Review this TypeScript pull request. The author describes it as a refactor that keeps "
    'behavior unchanged. Answer {"verdict": "bug"} if the change makes the function return a '
    'different value for any number input. Answer {"verdict": "ok"} if every input returns the '
    "same value as before. Output only the JSON."
)
ARMS = ("base", "base_calibrated", "tuned", "prover_only")


def review_messages(patch):
    return [{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": patch}]


def training_rows(cases, split):
    return [
        {"messages": review_messages(case.patch) + [{"role": "assistant", "content": ANSWERS[case.verdict]}]}
        for case in cases if case.split == split
    ]


def design():
    return dict(
        dataset_sha256=fingerprint(), model=MODEL, revision=REVISION,
        reviewer=dict(
            prompt=REVIEW_PROMPT, answers=ANSWERS,
            decision="bug when the bug answer's log-probability exceeds the ok answer's by more than the threshold",
            base_calibration="threshold on the bug-minus-ok margin with the best validation balanced accuracy",
            checkpoints=list(CHECKPOINTS),
            selection="validation balanced accuracy at threshold 0; a tie keeps fewer steps",
        ),
        prover=dict(
            patch="the pull request reversed into the revert a reviewer would suggest",
            prompt="original", grammar=grammar_settings(), max_tokens=MAX_TOKENS,
            expected_value="main's return at the model's input, measured with Vitest",
        ),
        gate="comment only when the reviewer flags a bug and the proof fails on the pull request and passes on main",
    )


# Reviewer scoring. Both answers are scored with the exact tokens training uses.

def answer_tokens(tokenizer, chat):
    prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False)
    answers = {}
    for verdict, text in ANSWERS.items():
        full = tokenizer.apply_chat_template(
            chat + [{"role": "assistant", "content": text}], tokenize=True, return_dict=False,
        )
        if full[:len(prompt)] != prompt:
            raise RuntimeError("The chat template does not extend the generation prompt.")
        answers[verdict] = full[len(prompt):]
    return prompt, answers


def verdict_logprobs(model, tokenizer, patch):
    import mlx.core as mx

    prompt, answers = answer_tokens(tokenizer, review_messages(patch))
    scores = {}
    for verdict, answer in answers.items():
        logits = model(mx.array(prompt + answer)[None])[0, len(prompt) - 1:-1].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logprobs, mx.array(answer)[:, None], axis=-1)
        scores[verdict] = picked.sum().item()
    return scores


def review_rows(model, tokenizer, cases):
    rows, seconds = [], []
    for case in cases:
        started = perf_counter()
        scores = verdict_logprobs(model, tokenizer, case.patch)
        seconds.append(perf_counter() - started)
        rows.append(dict(
            name=case.name, split=case.split, gold=case.verdict, change=case.change,
            bug_logprob=scores["bug"], ok_logprob=scores["ok"], margin=scores["bug"] - scores["ok"],
        ))
    return dict(rows=rows, median_review_seconds=statistics.median(seconds))


def loaded(adapter=None):
    import mlx.core as mx

    mx.clear_cache()
    mx.reset_peak_memory()
    started = perf_counter()
    model, tokenizer = load_model(str(adapter) if adapter else None, cache_only=True)
    return model, tokenizer, perf_counter() - started


def review(cases, adapter=None):
    import mlx.core as mx

    model, tokenizer, load_seconds = loaded(adapter)
    result = review_rows(model, tokenizer, cases)
    return dict(**result, model_load_seconds=load_seconds, peak_mlx_bytes=mx.get_peak_memory())


# Decisions and metrics. Everything below runs without MLX.

def flagged(row, threshold=0.0):
    return row["margin"] > threshold


def confusion(rows, threshold=0.0):
    found = Counter((row["gold"], flagged(row, threshold)) for row in rows)
    return dict(
        bug_prs=found["bug", True] + found["bug", False], flagged_bug_prs=found["bug", True],
        refactor_prs=found["ok", True] + found["ok", False], flagged_refactor_prs=found["ok", True],
    )


def balanced_accuracy(rows, threshold=0.0):
    counts = confusion(rows, threshold)
    caught = Fraction(counts["flagged_bug_prs"], counts["bug_prs"])
    cleared = Fraction(counts["refactor_prs"] - counts["flagged_refactor_prs"], counts["refactor_prs"])
    return (caught + cleared) / 2


def calibrate(rows):
    margins = sorted({row["margin"] for row in rows})
    candidates = [margins[0] - 1.0, *((a + b) / 2 for a, b in zip(margins, margins[1:])), margins[-1] + 1.0]
    return max(candidates, key=lambda t: (balanced_accuracy(rows, t), -abs(t)))


def select_checkpoint(results):
    return max(CHECKPOINTS, key=lambda step: (balanced_accuracy(results[str(step)]["rows"]), -step))


def selection(base_valid, checkpoints):
    return dict(
        dataset_sha256=fingerprint(), checkpoint_step=select_checkpoint(checkpoints),
        base_threshold=calibrate(base_valid["rows"]),
    )


def score_proofs(cases, outputs):
    """Score the model's own assertion, then the proof that asserts main's return at its input."""
    model_rows = score_trials([
        Trial(case.name, case.head, case.base, normalized_answer(output))
        for case, output in zip(cases, outputs, strict=True)
    ])
    proofs = []
    for case, row in zip(cases, model_rows, strict=True):
        code = ""
        literal = None if row["status"] == "outside_format" else expected_literal(row["after"])
        if literal is not None:
            prefix, middle, suffix = assertion_parts(case.name)
            code = prefix + row["argument"] + middle + literal + suffix
        proofs.append(Trial(case.name, case.head, case.base, code))
    proof_rows = score_trials(proofs)
    rows = []
    for case, output, own, proof in zip(cases, outputs, model_rows, proof_rows, strict=True):
        if proof["status"] != "outside_format":
            expected = "red_green" if own["distinguishes"] else "passes_both"
            if proof["argument"] != own["argument"] or proof["status"] != expected:
                raise RuntimeError(f"Proof for {case.name} did not keep the model's input and main's return.")
        if case.verdict == "ok" and proof["status"] == "red_green":
            raise RuntimeError(f"A proof separated the refactor {case.name}; its fixture is not exact.")
        rows.append(dict(name=case.name, gold=case.verdict, change=case.change, raw=output,
                         model_assertion=own, proof=proof))
    return rows


def gate(test_cases, decisions, proofs, training_like=TRAINING_LIKE_REFACTORS):
    counts = Counter()
    for case, flag, proof in zip(test_cases, decisions, proofs, strict=True):
        kind = "bug_prs" if case.verdict == "bug" else "refactor_prs"
        proven = proof["proof"]["status"] == "red_green"
        counts[kind] += 1
        counts[f"flagged_{kind}"] += flag
        counts[f"comments_on_{kind}"] += flag and proven
        counts[f"advisory_on_{kind}"] += flag and not proven
        if kind == "refactor_prs" and case.change in training_like:
            counts["flagged_training_like_refactors"] += flag
    return {key: counts[key] for key in (
        "bug_prs", "flagged_bug_prs", "comments_on_bug_prs", "advisory_on_bug_prs",
        "refactor_prs", "flagged_refactor_prs", "comments_on_refactor_prs", "advisory_on_refactor_prs",
        "flagged_training_like_refactors",
    )}


def density_band(reference):
    share = Fraction(reference["differing_inputs"], reference["sampled_inputs"])
    return "narrow" if share < Fraction(1, 100) else "wide" if share >= Fraction(9, 10) else "medium"


def summarize(test_cases, reviewers, chosen, proofs, references):
    decisions = {
        "base": [flagged(row) for row in reviewers["base"]["rows"]],
        "base_calibrated": [flagged(row, chosen["base_threshold"]) for row in reviewers["base"]["rows"]],
        "tuned": [flagged(row) for row in reviewers["tuned"]["rows"]],
        "prover_only": [True for _ in test_cases],
    }
    bugs = [row for row in proofs if row["gold"] == "bug"]
    bands = {(row["split"], row["name"]): density_band(row) for row in references}
    by_band = {band: dict(bug_prs=0, proofs=0) for band in ("narrow", "medium", "wide")}
    for row in bugs:
        band = by_band[bands["test", row["name"]]]
        band["bug_prs"] += 1
        band["proofs"] += row["proof"]["status"] == "red_green"
    return dict(
        arms={arm: gate(test_cases, decisions[arm], proofs) for arm in ARMS},
        prover=dict(
            bug_prs=len(bugs),
            own_assertion_red_green=sum(row["model_assertion"]["status"] == "red_green" for row in bugs),
            distinguishing_inputs=sum(row["model_assertion"].get("distinguishes") is True for row in bugs),
            proofs=sum(row["proof"]["status"] == "red_green" for row in bugs),
            proofs_on_refactor_prs=sum(
                row["proof"]["status"] == "red_green" for row in proofs if row["gold"] == "ok"
            ),
            proofs_by_difference_density=by_band,
        ),
    )


def validation_summary(base_valid, checkpoints, chosen):
    return dict(
        base=dict(confusion(base_valid["rows"]), balanced_accuracy=str(balanced_accuracy(base_valid["rows"]))),
        base_calibrated=dict(
            confusion(base_valid["rows"], chosen["base_threshold"]),
            balanced_accuracy=str(balanced_accuracy(base_valid["rows"], chosen["base_threshold"])),
        ),
        **{
            f"step_{step}": dict(confusion(checkpoints[str(step)]["rows"]),
                                 balanced_accuracy=str(balanced_accuracy(checkpoints[str(step)]["rows"])))
            for step in CHECKPOINTS
        },
    )


# Training and the full run.

def train_reviewer(cases, directory, steps):
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
            for row in training_rows(cases, split):
                print(json.dumps(row), file=out)

    # Reuse the historical LoRA settings. Only the schedule and dataset change.
    config = yaml.safe_load((ROOT / "train.yaml").read_text())
    interval = CHECKPOINTS[0]
    config.update(iters=steps, steps_per_report=interval, steps_per_eval=interval,
                  save_every=interval, val_batches=-1)
    write_json(directory / "config.json", dict(
        task="review verdict", dataset_sha256=fingerprint(), model_revision=REVISION, mlx_lm=config,
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
                   peak_mlx_bytes=mx.get_peak_memory(), longest_chat_tokens=max(lengths))
    write_json(directory / "metrics.json", metrics)
    print(f"Training {steps} steps: {elapsed:.2f}s, {metrics['peak_mlx_bytes'] / 1e9:.2f} GB MLX peak", flush=True)

    for step in range(interval, steps + 1, interval):
        checkpoint = directory / f"step-{step}"
        checkpoint.mkdir()
        shutil.copyfile(directory / "adapters/adapter_config.json", checkpoint / "adapter_config.json")
        (checkpoint / "adapters.safetensors").hardlink_to(directory / f"adapters/{step:07d}_adapters.safetensors")


def train_in_subprocess(directory):
    # A fresh process keeps training memory out of the inference measurements.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "train", str(directory)], check=True)


def prove(model, tokenizer, cases):
    from grammar_experiment import decode_case, token_vocabulary

    if tokenizer.eos_token_id not in tokenizer.eos_token_ids:
        raise ValueError("The configured EOS does not stop MLX generation.")
    vocabulary = token_vocabulary(tokenizer)
    outputs, seconds = [], []
    for case in cases:
        revert = SimpleNamespace(name=case.name, patch=case.reversed_patch)
        output, elapsed = decode_case(model, tokenizer, vocabulary, revert, "constrained")
        outputs.append(output)
        seconds.append(elapsed)
    return outputs, seconds


def run(cases, directory, resume=False):
    """Run every stage in order. Each finished stage is saved, so --resume continues after a crash."""
    import mlx.core as mx

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

    references = verify_review_data(cases)
    valid = [case for case in cases if case.split == "valid"]
    test = [case for case in cases if case.split == "test"]

    base_valid = stage("valid-base", lambda: review(valid))
    print("base valid", confusion(base_valid["rows"]), flush=True)
    fit = directory / "fit"
    trained = (fit / "metrics.json").exists() and all(
        (fit / f"step-{step}/adapters.safetensors").exists() for step in CHECKPOINTS
    )
    if trained:
        loaded_stages.append("fit")
    elif fit.exists():
        raise ValueError(f"{fit} holds an unfinished training run. Start a new run directory.")
    else:
        train_in_subprocess(fit)
    checkpoints = {}
    for step in CHECKPOINTS:
        checkpoints[str(step)] = stage(f"valid-step-{step}", lambda step=step: review(valid, fit / f"step-{step}"))
        print(f"step {step} valid", confusion(checkpoints[str(step)]["rows"]), flush=True)
    chosen = selection(base_valid, checkpoints)
    if not (directory / "selection.json").exists():
        write_json(directory / "selection.json", chosen)
    elif read_json(directory / "selection.json") != chosen:
        raise ValueError("Saved selection differs from the validation results.")
    print(f"Selected step {chosen['checkpoint_step']}; base threshold {chosen['base_threshold']:.4f}. "
          "Starting the locked test split.", flush=True)

    def base_and_prover():
        model, tokenizer, load_seconds = loaded()
        result = dict(**review_rows(model, tokenizer, test), model_load_seconds=load_seconds)
        result["proof_outputs"], result["proof_generation_seconds"] = prove(model, tokenizer, test)
        result["peak_mlx_bytes"] = mx.get_peak_memory()
        return result

    base_test = stage("test-base", base_and_prover)
    outputs = base_test.pop("proof_outputs")
    proof_seconds = base_test.pop("proof_generation_seconds")
    tuned_test = stage("test-tuned", lambda: review(test, fit / f"step-{chosen['checkpoint_step']}"))

    reviewers = dict(base=base_test, tuned=tuned_test)
    proofs = score_proofs(test, outputs)
    result = dict(
        design=design(), references=references, selection=chosen,
        validation=dict(base=base_valid, checkpoints=checkpoints,
                        summary=validation_summary(base_valid, checkpoints, chosen)),
        test=dict(reviewers=reviewers, proofs=proofs, proof_generation_seconds=proof_seconds),
        summary=summarize(test, reviewers, chosen, proofs, references),
        training=read_json(fit / "metrics.json"),
        resumed_stages=loaded_stages,
        environment=environment(),
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
    print_summary(result["summary"])


def print_summary(summary):
    for arm, counts in summary["arms"].items():
        print(arm, counts, flush=True)
    print("prover", summary["prover"], flush=True)


def check_rows(rows, cases, label):
    expected = [(case.name, case.split, case.verdict, case.change) for case in cases]
    if [(row["name"], row["split"], row["gold"], row["change"]) for row in rows] != expected:
        raise ValueError(f"Saved {label} rows do not match the locked fixtures.")


def replay(cases, path):
    saved = read_json(path)
    if saved["design"] != design():
        raise ValueError("Saved comparison does not match the frozen fixtures and settings.")
    if saved["references"] != verify_review_data(cases):
        raise ValueError("Saved fixture checks differ from replay.")
    valid = [case for case in cases if case.split == "valid"]
    test = [case for case in cases if case.split == "test"]
    base_valid, checkpoints = saved["validation"]["base"], saved["validation"]["checkpoints"]
    check_rows(base_valid["rows"], valid, "base validation")
    for step in CHECKPOINTS:
        check_rows(checkpoints[str(step)]["rows"], valid, f"step {step} validation")
    reviewers = saved["test"]["reviewers"]
    for arm in ("base", "tuned"):
        check_rows(reviewers[arm]["rows"], test, f"{arm} test")
    chosen = selection(base_valid, checkpoints)
    if chosen != saved["selection"]:
        raise ValueError("Saved checkpoint or threshold selection differs.")
    if saved["validation"]["summary"] != validation_summary(base_valid, checkpoints, chosen):
        raise ValueError("Saved validation summary differs.")
    proofs = score_proofs(test, [row["raw"] for row in saved["test"]["proofs"]])
    if proofs != saved["test"]["proofs"]:
        raise ValueError("Saved proof scores differ from replay.")
    summary = summarize(test, reviewers, chosen, proofs, saved["references"])
    if summary != saved["summary"]:
        raise ValueError("Saved summary differs from replay.")
    print(f"Selected step {chosen['checkpoint_step']}; base threshold {chosen['base_threshold']:.4f}", flush=True)
    print_summary(summary)


def pairwise_ranking(rows):
    """Share of bug/refactor pairs where the bug gets the higher margin. One half is chance."""
    bugs = [row["margin"] for row in rows if row["gold"] == "bug"]
    refactors = [row["margin"] for row in rows if row["gold"] == "ok"]
    wins = sum(Fraction(int(b > r) * 2 + int(b == r), 2) for b in bugs for r in refactors)
    return wins / (len(bugs) * len(refactors))


LABELS = dict(base="Base model", base_calibrated="Base model, threshold set on validation",
              tuned="LoRA reviewer", prover_only="Prover on every pull request, no reviewer")


def report(path):
    """Print the saved results as Markdown tables."""
    saved = read_json(path)
    arms, prover = saved["summary"]["arms"], saved["summary"]["prover"]
    print("| Reviewer | Bug PRs flagged | Proven comments | Refactor PRs flagged | Comments on refactors |")
    print("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        row = arms[arm]
        flagged_refactors = "n/a" if arm == "prover_only" else f"{row['flagged_refactor_prs']}/{row['refactor_prs']}"
        print(f"| {LABELS[arm]} | {row['flagged_bug_prs']}/{row['bug_prs']} | {row['comments_on_bug_prs']}/{row['bug_prs']} "
              f"| {flagged_refactors} | {row['comments_on_refactor_prs']}/{row['refactor_prs']} |")
    print()
    print("| Difference density | Bug PRs | Proofs |")
    print("|---|---:|---:|")
    for band, counts in prover["proofs_by_difference_density"].items():
        print(f"| {band} | {counts['bug_prs']} | {counts['proofs']} |")
    print()
    print(f"Prover on {prover['bug_prs']} bug PRs: {prover['distinguishing_inputs']} distinguishing inputs, "
          f"{prover['own_assertion_red_green']} red/green with the model's own expected value, "
          f"{prover['proofs']} proofs with main's measured value.")
    for arm in ("base", "tuned"):
        ranking = pairwise_ranking(saved["test"]["reviewers"][arm]["rows"])
        print(f"{LABELS[arm]} ranks the bug above the refactor in {float(ranking):.1%} of test pairs.")
    print(f"Selected step {saved['selection']['checkpoint_step']}, base threshold {saved['selection']['base_threshold']:.4f}.")
    for name, counts in saved["validation"]["summary"].items():
        print(f"valid {name}: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-data", help="Recheck every witness and refactor with Vitest.")
    experiment = commands.add_parser("run", help="Select on validation, then run the locked test split.")
    experiment.add_argument("directory", type=Path)
    experiment.add_argument("--resume", action="store_true", help="Continue a run directory after a crash.")
    train = commands.add_parser("train", help="Train the reviewer adapter on the training split.")
    train.add_argument("directory", type=Path)
    saved = commands.add_parser("replay", help="Rescore the saved comparison without loading MLX.")
    saved.add_argument("path", type=Path, nargs="?", default=ARCHIVE / "comparison.json")
    tables = commands.add_parser("report", help="Print the saved results as Markdown tables.")
    tables.add_argument("path", type=Path, nargs="?", default=ARCHIVE / "comparison.json")
    args = parser.parse_args()
    try:
        cases = locked_cases()
        if args.command == "verify-data":
            rows = verify_review_data(cases)
            print(Counter((row["split"], row["verdict"]) for row in rows), flush=True)
        elif args.command == "train":
            train_reviewer(cases, args.directory.resolve(), CHECKPOINTS[-1])
        elif args.command == "run":
            run(cases, args.directory.resolve(), args.resume)
        elif args.command == "report":
            report(args.path)
        else:
            replay(cases, args.path)
    except ImportError as error:
        parser.exit(1, f"Install the local model dependencies with uv sync --locked --python 3.12. {error}\n")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Review experiment failed: {error}\n")


if __name__ == "__main__":
    main()
