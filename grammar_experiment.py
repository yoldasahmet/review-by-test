"""Compare greedy decoding with and without a Vitest output grammar."""
import argparse
import platform
import subprocess
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from experiment import SETTINGS, counts, read_json, scored_rows, verify_data, write_json
from generate import load_model, raw_prediction
from grammar_fixtures import ROOT, fingerprint, locked_cases

ARCHIVE = ROOT / "results/grammar"
LITERAL = r"-?(?:0|[1-9][0-9]{0,14})(?:\.[0-9]{1,12})?"
MODES = ("unconstrained", "constrained")


def assertion_parts(name):
    prefix = (
        f"import {{ expect, test }} from 'vitest';\nimport {{ {name} }} from './subject';\n\n"
        f"test('regression', () => {{\n  expect({name}("
    )
    return prefix, ")).toBe(", ");\n});"


def assertion_grammar(name, *, argument=None):
    import regex

    prefix, middle, suffix = assertion_parts(name)
    if argument is not None and not regex.fullmatch(LITERAL, argument):
        raise ValueError(f"Supplied argument is outside the decoding grammar: {argument!r}")
    input_pattern = LITERAL if argument is None else regex.escape(argument)
    return regex.compile(
        regex.escape(prefix) + input_pattern + regex.escape(middle) +
        f"(?:{LITERAL}|true|false)" + regex.escape(suffix)
    )


@dataclass
class TokenTrie:
    children: dict[str, "TokenTrie"] = field(default_factory=dict)
    ids: list[int] = field(default_factory=list)

    def insert(self, piece, token_id):
        node = self
        for char in piece:
            if char not in node.children:
                node.children[char] = TokenTrie()
            node = node.children[char]
        node.ids.append(token_id)

    def continuations(self, grammar, prefix):
        for char, child in self.children.items():
            extended = prefix + char
            if grammar.fullmatch(extended, partial=True):
                yield from child.ids
                yield from child.continuations(grammar, extended)


def token_vocabulary(tokenizer):
    backend = tokenizer.backend_tokenizer
    if type(backend.decoder).__name__ != "ByteLevel":
        raise ValueError("The experiment expects the pinned Qwen byte-level tokenizer.")
    trie = TokenTrie()
    pieces = {}
    for token, token_id in backend.get_vocab(with_added_tokens=False).items():
        # Each ASCII piece can be concatenated directly for this byte-level tokenizer.
        piece = backend.decoder.decode([token])
        if piece and piece.isascii() and token_id not in tokenizer.eos_token_ids:
            trie.insert(piece, token_id)
            pieces[token_id] = piece
    return trie, pieces


class GrammarSampler:
    def __init__(self, name, end_token, vocabulary, *, argument=None):
        self.grammar = assertion_grammar(name, argument=argument)
        self.end_token = end_token
        self.trie, self.pieces = vocabulary
        self.output = ""

    def __call__(self, logprobs):
        import mlx.core as mx

        if self.grammar.fullmatch(self.output):
            # MLX may sample ahead of the consumed EOS, so keep EOS allowed.
            return mx.full(logprobs.shape[:-1], self.end_token, dtype=mx.uint32)
        allowed = sorted(self.trie.continuations(self.grammar, self.output))
        if not allowed:
            raise RuntimeError(f"No legal next token after {self.output!r}.")
        ids = mx.array(allowed, dtype=mx.uint32)
        winner = ids[mx.argmax(mx.take(logprobs, ids, axis=-1), axis=-1)]
        self.output += self.pieces[winner.item()]
        return winner


def decode_case(model, tokenizer, vocabulary, case, mode, *, argument=None):
    sampler = None
    if mode == "constrained":
        sampler = GrammarSampler(case.name, tokenizer.eos_token_id, vocabulary, argument=argument)
    start = perf_counter()
    output = raw_prediction(model, tokenizer, case.patch, sampler=sampler)
    if sampler and not sampler.output.startswith(output):
        raise RuntimeError("The tokenizer output disagrees with the grammar's decoded pieces.")
    return output, perf_counter() - start


def grammar_settings():
    return dict(
        template="canonical Vitest test with the changed export and two literal slots",
        number_pattern=LITERAL, decoder="greedy over grammar-valid Qwen tokens",
        name_source="exported signature in the diff",
    )


def environment():
    return dict(
        chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        macos=platform.mac_ver()[0], python=platform.python_version(),
        node=subprocess.check_output(["node", "--version"], text=True).strip(),
        vitest=read_json(ROOT / "node_modules/vitest/package.json")["version"],
        **{package: version(package) for package in ("mlx", "mlx-lm", "regex")},
    )


def design():
    return dict(
        dataset_sha256=fingerprint(), prompt="original", settings=SETTINGS,
        grammar=grammar_settings(),
        generation_order="alternate the first mode between consecutive cases",
    )


def paired(rows):
    control, treatment = rows["unconstrained"], rows["constrained"]
    pair_counts = dict(both=0, constrained_only=0, unconstrained_only=0, neither=0)
    added_from_unsupported = 0
    for before, after in zip(control, treatment, strict=True):
        before_hit = before["status"] == "red_green"
        after_hit = after["status"] == "red_green"
        if before_hit and after_hit:
            pair_counts["both"] += 1
        elif after_hit:
            pair_counts["constrained_only"] += 1
            added_from_unsupported += before["status"] == "outside_format"
        elif before_hit:
            pair_counts["unconstrained_only"] += 1
        else:
            pair_counts["neither"] += 1
    return dict(**pair_counts, constrained_wins_from_unsupported=added_from_unsupported)


def compare(cases, directory):
    directory.mkdir(parents=True)
    verify_data(cases)
    write_json(directory / "design.json", design())

    model, tokenizer = load_model(cache_only=True)
    if tokenizer.eos_token_id not in tokenizer.eos_token_ids:
        raise ValueError("The configured EOS does not stop MLX generation.")
    vocabulary = token_vocabulary(tokenizer)
    outputs = {mode: [] for mode in MODES}
    timings = {mode: [] for mode in MODES}
    for index, case in enumerate(cases):
        order = MODES if index % 2 == 0 else MODES[::-1]
        # Both arms use the same diff and prompt. The constraint gets only the
        # exported name visible in that diff, plus the fixed output grammar.
        for mode in order:
            output, seconds = decode_case(model, tokenizer, vocabulary, case, mode)
            outputs[mode].append(output)
            timings[mode].append(seconds)
        print(f"{index + 1}/{len(cases)} {case.name}", flush=True)

    rows = {mode: scored_rows(cases, outputs[mode]) for mode in MODES}
    for mode in MODES:
        print(mode, counts(dict(rows=rows[mode])), flush=True)
    result = dict(
        design=design(), rows=rows, paired=paired(rows), generation_seconds=timings,
        environment=environment(),
    )
    write_json(directory / "comparison.json", result)
    print(result["paired"], flush=True)


def replay(cases, path):
    result = read_json(path)
    if result["design"] != design():
        raise ValueError("Saved comparison does not match the frozen data and decoding settings.")
    verify_data(cases)
    rows = result["rows"]
    for mode in MODES:
        saved = rows[mode]
        if saved != scored_rows(cases, [row["raw"] for row in saved]):
            raise ValueError(f"Saved scores differ for {mode}.")
        print(mode, counts(dict(rows=saved)), flush=True)
        print(f"Median generation: {median(result['generation_seconds'][mode]):.3f}s", flush=True)
    if result["paired"] != paired(rows):
        raise ValueError("Saved paired counts differ.")
    print(result["paired"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Compare both decoding modes with cached weights.")
    run.add_argument("directory", type=Path)
    saved = commands.add_parser("replay", help="Rescore the saved comparison without loading MLX.")
    saved.add_argument("path", type=Path, nargs="?", default=ARCHIVE / "comparison.json")
    args = parser.parse_args()
    cases = locked_cases()
    if args.command == "run":
        compare(cases, args.directory)
    else:
        replay(cases, args.path)


if __name__ == "__main__":
    main()
