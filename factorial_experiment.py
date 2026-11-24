"""Run or replay a paired base/adapter by unconstrained/constrained experiment."""
import argparse
from hashlib import file_digest
from itertools import product
from pathlib import Path
from statistics import median
from string import hexdigits
import subprocess

from experiment import SETTINGS, counts, read_json, scored_rows, verify_data, write_json
from factorial_fixtures import ROOT, fingerprint, locked_cases
from generate import REVISION, load_model
from grammar_experiment import MODES, decode_case, environment, grammar_settings, token_vocabulary

ARCHIVE = ROOT / "results/factorial"
ADAPTER = ROOT / "runs/fresh-v1/fit/step-48"
CONDITIONS = tuple(product(("base", "tuned"), MODES))
TRAINING_KEYS = ("seed", "num_layers", "batch_size", "learning_rate", "mask_prompt", "lora_parameters")


def checkpoint_recipe():
    selection = read_json(ROOT / "results/benchmark/selection.json")
    fit = read_json(ROOT / "results/benchmark/fit/config.json")
    return dict(
        selection_dataset_sha256=selection["dataset_sha256"],
        prompt=selection["prompt"], step=selection["checkpoint_step"],
        **{key: fit["mlx_lm"][key] for key in TRAINING_KEYS},
    )


def identify_adapter(path):
    recipe = checkpoint_recipe()
    config = read_json(path / "adapter_config.json")
    if (config["iters"] != recipe["step"] or Path(config["model"]).name != REVISION
            or any(config[key] != recipe[key] for key in TRAINING_KEYS)):
        raise ValueError("The adapter does not match the previously selected checkpoint recipe.")
    with (path / "adapters.safetensors").open("rb") as weights:
        return file_digest(weights, "sha256").hexdigest()


def design(weights_sha256):
    return dict(
        dataset_sha256=fingerprint(), settings=SETTINGS, prompt="original",
        grammar=grammar_settings(),
        adapter=dict(**checkpoint_recipe(), weights_sha256=weights_sha256),
        generation_order="rotate the four conditions one position per consecutive case",
    )


def label(condition):
    return "-".join(condition)


def summary(rows):
    return {
        name: {**counts(dict(rows=entries)),
               "format_compliant": sum(row["status"] != "outside_format" for row in entries)}
        for name, entries in rows.items()
    }


def contrasts(summary_by_condition, metric):
    base, base_grammar, tuned, tuned_grammar = (
        summary_by_condition[label(condition)][metric] for condition in CONDITIONS
    )
    return dict(
        grammar_on_base=base_grammar - base,
        grammar_on_tuned=tuned_grammar - tuned,
        tuning_unconstrained=tuned - base,
        tuning_constrained=tuned_grammar - base_grammar,
        combined_vs_base=tuned_grammar - base,
        interaction=tuned_grammar - tuned - base_grammar + base,
    )


def print_results(result):
    for name, scores in result["summary"].items():
        elapsed = median(result["generation_seconds"][name])
        print(f"{name} {scores} median {elapsed:.3f}s", flush=True)
    for metric, comparison in result["contrasts"].items():
        print(metric, comparison, flush=True)


def run(cases, directory, adapter):
    experiment = design(identify_adapter(adapter))
    directory.mkdir(parents=True)
    verify_data(cases)
    write_json(directory / "design.json", experiment)
    base, tokenizer = load_model(cache_only=True)
    tuned, _ = load_model(adapter=str(adapter), cache_only=True)
    models = {"base": base, "tuned": tuned}
    if tokenizer.eos_token_id not in tokenizer.eos_token_ids:
        raise ValueError("The configured EOS does not stop MLX generation.")
    vocabulary = token_vocabulary(tokenizer)
    outputs = {label(condition): [] for condition in CONDITIONS}
    timings = {name: [] for name in outputs}
    for index, case in enumerate(cases):
        offset = index % len(CONDITIONS)
        for condition in CONDITIONS[offset:] + CONDITIONS[:offset]:
            model, mode = condition
            name = label(condition)
            output, seconds = decode_case(models[model], tokenizer, vocabulary, case, mode)
            outputs[name].append(output)
            timings[name].append(seconds)
        print(f"{index + 1}/{len(cases)} {case.name}", flush=True)

    rows = {name: scored_rows(cases, entries) for name, entries in outputs.items()}
    scores = summary(rows)
    result = dict(
        design=experiment, rows=rows, summary=scores, generation_seconds=timings,
        contrasts={metric: contrasts(scores, metric) for metric in ("red_green", "format_compliant")},
        environment=environment(),
    )
    write_json(directory / "comparison.json", result)
    print_results(result)


def replay(cases, path):
    saved = read_json(path)
    digest = saved["design"]["adapter"]["weights_sha256"]
    if len(digest) != 64 or any(char not in hexdigits for char in digest) or saved["design"] != design(digest):
        raise ValueError("Saved design does not match the frozen dataset and comparison settings.")
    verify_data(cases)
    rows = saved["rows"]
    if rows.keys() != {label(condition) for condition in CONDITIONS}:
        raise ValueError("Saved comparison is missing an experimental condition.")
    for name, entries in rows.items():
        if scored_rows(cases, [row["raw"] for row in entries]) != entries:
            raise ValueError(f"Saved scores differ for {name}.")
    scores = summary(rows)
    if saved["summary"] != scores or any(
        saved["contrasts"][metric] != contrasts(scores, metric) for metric in ("red_green", "format_compliant")
    ):
        raise ValueError("Saved aggregate scores differ from replay.")
    print_results(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("run", help="Run the four arms locally using cached weights.")
    generate.add_argument("directory", type=Path)
    generate.add_argument("--adapter", type=Path, default=ADAPTER)
    saved = commands.add_parser("replay", help="Verify saved outputs without loading MLX.")
    saved.add_argument("path", type=Path, nargs="?", default=ARCHIVE / "comparison.json")
    args = parser.parse_args()
    try:
        cases = locked_cases()
        if args.command == "run":
            run(cases, args.directory, args.adapter)
        else:
            replay(cases, args.path)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, ImportError) as error:
        parser.exit(1, f"Factorial comparison failed: {error}\n")


if __name__ == "__main__":
    main()
