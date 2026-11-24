"""Run or replay the verified-input and computed-expected-value ablation."""
import argparse
from decimal import Decimal
from math import isfinite
from pathlib import Path
import subprocess

from ablation_fixtures import ROOT, fingerprint, locked_cases
from experiment import SETTINGS, counts, read_json, scored_rows, verify_data, write_json
from generate import load_model
from grammar_experiment import assertion_parts, decode_case, environment, grammar_settings, token_vocabulary

ARCHIVE = ROOT / "results/ablation"
MODEL_ARMS = ("grammar_only", "verified_input")


def design():
    return dict(
        dataset_sha256=fingerprint(), settings=SETTINGS, prompt="original",
        grammar=grammar_settings(),
        verified_input="force the validated reference argument during generation",
        computed_expected="reuse the grammar-only argument and assert the measured fixed return",
        generation_order="alternate the two model arms; computed-expected reuses grammar-only output",
    )


def expected_literal(value):
    """Encode fixed return values in the evaluator's finite literal syntax."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value == "-0":
        return "-0"
    if isinstance(value, (int, float)) and isfinite(value):
        # Float repr round-trips to the same binary64 value. Expand exponent notation.
        return format(Decimal(str(value)), "f")
    return None


def computed_assertion(case, row):
    if row["status"] == "outside_format":
        return ""
    expected = expected_literal(row["after"])
    if expected is None:
        return ""
    prefix, middle, suffix = assertion_parts(case.name)
    return prefix + row["argument"] + middle + expected + suffix


def score_ablation(cases, witnesses, outputs):
    grammar = scored_rows(cases, outputs["grammar_only"])
    verified = scored_rows(cases, outputs["verified_input"])
    for row, reference in zip(verified, witnesses, strict=True):
        if row["status"] != "outside_format" and (
            row["argument"] != reference["argument"] or not row["distinguishes"]
        ):
            raise RuntimeError(f"Verified input was not retained for {row['name']}.")
    corrected = scored_rows(cases, [
        computed_assertion(case, row) for case, row in zip(cases, grammar, strict=True)
    ])
    for control, oracle in zip(grammar, corrected, strict=True):
        if oracle["status"] != "outside_format":
            expected_status = "red_green" if control["distinguishes"] else "passes_both"
            if oracle["argument"] != control["argument"] or oracle["status"] != expected_status:
                raise RuntimeError(f"Computed assertion did not preserve the input or fixed return for {oracle['name']}.")
    return dict(grammar_only=grammar, verified_input=verified, computed_expected=corrected)


def summary(rows):
    return {
        name: {
            **counts(dict(rows=entries)),
            "distinguishing_inputs": sum(row.get("distinguishes") is True for row in entries),
        }
        for name, entries in rows.items()
    }


def print_scores(scores):
    for name, outcomes in scores.items():
        print(name, outcomes, flush=True)


def run(cases, directory):
    directory.mkdir(parents=True)
    witnesses = verify_data(cases)
    write_json(directory / "design.json", design())
    model, tokenizer = load_model(cache_only=True)
    if tokenizer.eos_token_id not in tokenizer.eos_token_ids:
        raise ValueError("The configured EOS does not stop MLX generation.")
    vocabulary = token_vocabulary(tokenizer)
    outputs = {arm: [] for arm in MODEL_ARMS}
    for index, (case, witness) in enumerate(zip(cases, witnesses, strict=True)):
        order = MODEL_ARMS if index % 2 == 0 else MODEL_ARMS[::-1]
        for arm in order:
            argument = witness["argument"] if arm == "verified_input" else None
            output, _ = decode_case(model, tokenizer, vocabulary, case, "constrained", argument=argument)
            outputs[arm].append(output)
        print(f"{index + 1}/{len(cases)} {case.name}", flush=True)
    rows = score_ablation(cases, witnesses, outputs)
    scores = summary(rows)
    result = dict(design=design(), rows=rows, summary=scores, environment=environment())
    write_json(directory / "comparison.json", result)
    print_scores(scores)


def replay(cases, path):
    saved = read_json(path)
    if saved["design"] != design():
        raise ValueError("Saved comparison does not match the frozen dataset and generation settings.")
    witnesses = verify_data(cases)
    outputs = {arm: [row["raw"] for row in saved["rows"][arm]] for arm in MODEL_ARMS}
    rows = score_ablation(cases, witnesses, outputs)
    scores = summary(rows)
    if rows != saved["rows"] or scores != saved["summary"]:
        raise ValueError("Saved scores differ from replay.")
    print_scores(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("run", help="Generate both model arms offline and run the ablation.")
    generate.add_argument("directory", type=Path)
    saved = commands.add_parser("replay", help="Verify all saved outputs without loading MLX.")
    saved.add_argument("path", type=Path, nargs="?", default=ARCHIVE / "comparison.json")
    args = parser.parse_args()
    try:
        cases = locked_cases()
        if args.command == "run":
            run(cases, args.directory)
        else:
            replay(cases, args.path)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, ImportError) as error:
        parser.exit(1, f"Ablation failed: {error}\n")


if __name__ == "__main__":
    main()
