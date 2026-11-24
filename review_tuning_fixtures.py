"""Training pairs and a fresh locked test split for the second review experiment.

The first review adapter learned a shortcut: pull requests that only renamed or
extracted code were safe, and anything that touched an operator was a bug. This
dataset is built to break that shortcut. Each fixed function from the earlier
fixtures gets two exact refactors, preferring structural rewrites, and two bug
pull requests: the reverted fix and a rewrite with one slip. Vitest samples every
candidate on main and on the pull request. It keeps exact refactors and real
behavior changes, and picks each slip's witness near a literal in the code.

The money split from the first review experiment is kept as a seen split. The
locked test split is new: 24 authored pull requests in payroll, insurance
premiums, loyalty points and savings accounts.

Build the candidates once with `python review_tuning_fixtures.py build`. The
result is frozen in data/review-tuning-cases.json and data/review-tuning.sha256.
"""
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from ablation_experiment import expected_literal
from review_fixtures import (
    COMPARISON, INPUTS, ROOT, SOURCES, ReviewCase, assemble, authored, extract_result,
    flip_comparison, parts, rename_parameter, swap_arguments, verify_review_data,
)
from review_fixtures import TEST as MONEY

CASES_FILE = ROOT / "data/review-tuning-cases.json"
CHECKSUM_FILE = ROOT / "data/review-tuning.sha256"
LITERAL = re.compile(r"-?(?:0|[1-9][0-9]{0,14})(?:\.[0-9]{1,12})?")

# Exact rewrites. Each is an identity for every finite number input, including -0.
NEGATED = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "===": "!==", "!==": "==="}
BARE = r"([A-Za-z_]\w*) (<=|>=|===|!==|<|>) (-?\d+(?:\.\d+)?)"


def negate_comparison(text):
    # x >= 5 and !(x < 5) agree for every input that is not NaN.
    name, parameter, lines = parts(text)
    body = "\n".join(lines)
    negated = COMPARISON.sub(lambda m: f"!({m[1]} {NEGATED[m[2]]} {m[3]})", body, count=1)
    return None if negated == body else assemble(name, parameter, negated.split("\n"))


def de_morgan(text):
    name, parameter, lines = parts(text)
    match = re.fullmatch(rf"  return {BARE} (&&|\|\|) {BARE};", lines[-1])
    if not match:
        return None
    left, first, x, joiner, right, second, y = match.groups()
    other = "||" if joiner == "&&" else "&&"
    rewritten = f"  return !({left} {NEGATED[first]} {x} {other} {right} {NEGATED[second]} {y});"
    return assemble(name, parameter, lines[:-1] + [rewritten])


def ternary_to_if(text):
    name, parameter, lines = parts(text)
    match = re.fullmatch(r"  return ([^?:]+) \? ([^?:]+) : ([^?:]+);", lines[-1])
    if not match:
        return None
    condition, yes, no = match.groups()
    return assemble(name, parameter, lines[:-1] + [
        f"  if ({condition}) {{", f"    return {yes};", "  }", f"  return {no};",
    ])


def if_to_ternary(text):
    name, parameter, lines = parts(text)
    if len(lines) < 2:
        return None
    guard = re.fullmatch(r"  if \((.+)\) return ([^;?:]+);", lines[-2])
    final = re.fullmatch(r"  return ([^;?:]+);", lines[-1])
    if not guard or not final or "?" in guard[1]:
        return None
    return assemble(name, parameter, lines[:-2] + [f"  return {guard[1]} ? {guard[2]} : {final[1]};"])


# A bare identifier added to or multiplied by a literal, alone inside a return or
# parentheses. IEEE 754 addition and multiplication are commutative.
OPERANDS = re.compile(r"(?:(?<=return )|(?<=\())([A-Za-z_]\w*) ([+*]) (-?\d+(?:\.\d+)?)(?=[;),])")


def commute_operands(text):
    name, parameter, lines = parts(text)
    body = "\n".join(lines)
    commuted = OPERANDS.sub(lambda m: f"{m[3]} {m[2]} {m[1]}", body, count=1)
    return None if commuted == body else assemble(name, parameter, commuted.split("\n"))


def name_constant(text):
    name, parameter, lines = parts(text)
    last = lines[-1]
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", last):
        if float(match[1]) in (0, 1):
            continue
        label = "limit" if re.search(r"(<=|>=|===|!==|<|>)\s*-?$", last[:match.start()]) else "factor"
        if re.search(rf"\b{label}\b", text):
            return None
        named = last[:match.start()] + label + last[match.end():]
        return assemble(name, parameter, [f"  const {label} = {match[1]};"] + lines[:-1] + [named])
    return None


def inline_constant(text):
    name, parameter, lines = parts(text)
    for index, line in enumerate(lines):
        match = re.fullmatch(r"  const (\w+) = (-?\d+(?:\.\d+)?);", line)
        if not match:
            continue
        rest = "\n".join(lines[:index] + lines[index + 1:])
        if not re.search(rf"\b{match[1]}\b", rest):
            continue
        literal = f"({match[2]})" if match[2].startswith("-") else match[2]
        return assemble(name, parameter, re.sub(rf"\b{match[1]}\b", literal, rest).split("\n"))
    return None


STRUCTURAL = (
    ("negate-comparison", negate_comparison), ("de-morgan", de_morgan),
    ("ternary-to-if", ternary_to_if), ("if-to-ternary", if_to_ternary),
    ("commute-operands", commute_operands), ("name-constant", name_constant),
    ("inline-constant", inline_constant), ("flip-comparison", flip_comparison),
    ("swap-min-max-arguments", swap_arguments),
)
TEXTUAL = (("extract-result", extract_result), ("rename-parameter", rename_parameter))
# Rarer rewrites first, so the two refactors per function lean toward structure.
PRIORITY = (
    "de-morgan", "ternary-to-if", "if-to-ternary", "commute-operands", "inline-constant",
    "negate-comparison", "swap-min-max-arguments", "flip-comparison", "name-constant",
    "extract-result", "rename-parameter",
)


# One-token slips applied after a structural rewrite. Sampling keeps only slips
# that change at least one return value.

def first(pattern, text, change):
    match = re.search(pattern, text)
    return None if not match else text[:match.start()] + change(match) + text[match.end():]


SLIPS = (
    ("boundary", lambda t: first(r" (<=|>=|<|>) ", t, lambda m: {
        "<": " <= ", "<=": " < ", ">": " >= ", ">=": " > "}[m[1]])),
    ("off-by-one", lambda t: first(r"(<=|>=|<|>|===|!==) (-?\d+)(?![\d.])", t,
                                   lambda m: f"{m[1]} {int(m[2]) + 1}")),
    ("min-max", lambda t: first(r"Math\.(min|max)\(", t,
                                lambda m: "Math.max(" if m[1] == "min" else "Math.min(")),
    ("rounding", lambda t: first(r"Math\.(floor|ceil|round|trunc)\(", t, lambda m: {
        "floor": "Math.ceil(", "ceil": "Math.floor(", "round": "Math.floor(", "trunc": "Math.floor("}[m[1]])),
    ("sign", lambda t: first(r" ([+-]) (?=[\w(])", t, lambda m: " - " if m[1] == "+" else " + ")),
    ("scale", lambda t: first(r"(?<![\w.])\d+\.\d+(?![\w.])", t,
                              lambda m: format((Decimal(m[0]) * 10).normalize(), "f"))),
)


def literals(*texts):
    values = set()
    for text in texts:
        for literal in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text):
            values.update((float(literal), -float(literal)))
    return sorted(values)


SAMPLER = """import {{ writeFileSync }} from 'node:fs';
import {{ test }} from 'vitest';
import {{ inputs }} from '../inputs';
import {{ {name} as onMain }} from './main';
import {{ {name} as inPullRequest }} from './pull';
const literals = {literals};
const encodable = (value) => typeof value === 'boolean' || (typeof value === 'number'
  && !Object.is(value, -0) && /^-?(?:0|[1-9][0-9]{{0,14}})(?:\\.[0-9]{{1,12}})?$/.test(String(value)));
test('sample', () => {{
  let differing = 0;
  let best = null;
  for (const input of inputs) {{
    const main = onMain(input);
    if (Object.is(main, inPullRequest(input))) continue;
    differing += 1;
    if (!encodable(input) || !encodable(main)) continue;
    const distance = Math.min(...literals.map((value) => Math.abs(input - value)));
    // Prefer integers, then inputs nearest a literal in the code, then small inputs.
    const key = [Number.isInteger(input) ? 0 : 1, distance, Math.abs(input), input];
    const index = best ? key.findIndex((value, i) => value !== best.key[i]) : -1;
    if (!best || (index >= 0 && key[index] < best.key[index])) best = {{ key, input, main }};
  }}
  writeFileSync(new URL('./result.json', import.meta.url), JSON.stringify({{
    checked: inputs.length, differing, witness: best && {{ input: best.input, main: best.main }},
  }}));
}});
"""


def sample(candidates, batch=150):
    """Run (base, head, name) pairs through the sampled inputs. Pairs that fail to run return None."""
    results = []
    for start in range(0, len(candidates), batch):
        chunk = candidates[start:start + batch]
        with TemporaryDirectory(dir=ROOT) as tmp:
            folder = Path(tmp).resolve()
            (folder / "inputs.ts").write_text(INPUTS)
            (folder / "vitest.config.mjs").write_text("export default {};\n")
            for index, (base, head, name) in enumerate(chunk):
                trial = folder / str(index)
                trial.mkdir()
                (trial / "main.ts").write_text(base)
                (trial / "pull.ts").write_text(head)
                (trial / "sample.test.ts").write_text(SAMPLER.format(
                    name=name, literals=json.dumps(literals(base, head) or [0.0]),
                ))
            subprocess.run(
                [str(ROOT / "node_modules/.bin/vitest"), "run", "--root", str(folder),
                 "--config", str(folder / "vitest.config.mjs"), "--maxWorkers", "2"],
                cwd=folder, capture_output=True, text=True, timeout=900,
            )
            for index in range(len(chunk)):
                result = folder / str(index) / "result.json"
                results.append(json.loads(result.read_text()) if result.exists() else None)
    return results


def candidates_for(case, index):
    """Every rewrite and every slip-after-rewrite that applies to one fixed function."""
    refactors, slips = [], []
    for label, transform in STRUCTURAL + TEXTUAL:
        rewritten = transform(case.after)
        if rewritten is None or rewritten == case.after:
            continue
        refactors.append((label, rewritten))
        if (label, transform) in STRUCTURAL:
            for slip_label, slip in SLIPS:
                slipped = slip(rewritten)
                if slipped is not None and slipped != rewritten:
                    slips.append((f"{label}+{slip_label}", slipped))
    # Rotate the slip families across functions and take them round-robin, so no
    # single family fills the six candidates.
    names = [name for name, _ in SLIPS]
    families = [[item for item in slips if item[0].endswith("+" + name)]
                for name in names[index % len(names):] + names[:index % len(names)]]
    ordered = [family[depth] for depth in range(max(map(len, families), default=0))
               for family in families if depth < len(family)]
    return refactors, ordered[:6]


def build():
    sources = {split: [case for module in SOURCES[split] for case in module.locked_cases()]
               for split in SOURCES}
    plans = []
    for split, cases in sources.items():
        for index, case in enumerate(cases):
            refactors, slips = candidates_for(case, index)
            plans.append((split, index, case, refactors, slips))
    flat = [(case.after, head, case.name) for _, _, case, refactors, slips in plans for _, head in refactors + slips]
    print(f"Sampling {len(flat)} candidates", flush=True)
    outcomes = iter(sample(flat))
    built, counts = [], {"exact": 0, "inexact": 0, "slips": 0, "equivalent_slips": 0, "failed": 0}
    for split, index, case, refactors, slips in plans:
        exact, changed = [], []
        for label, head in refactors:
            result = next(outcomes)
            if result is None:
                counts["failed"] += 1
            elif result["differing"] == 0:
                counts["exact"] += 1
                exact.append((label, head))
            else:
                counts["inexact"] += 1
        for label, head in slips:
            result = next(outcomes)
            if result is None:
                counts["failed"] += 1
            elif result["differing"] == 0:
                counts["equivalent_slips"] += 1
            elif result["witness"] is not None:
                counts["slips"] += 1
                changed.append((label, head, result["witness"]))
        chosen = sorted(exact, key=lambda item: PRIORITY.index(item[0]))[:2]
        built.append(ReviewCase(split, case.domain, case.name, case.after, case.before, "bug",
                                "revert-fix", case.argument, case.expected))
        if changed:
            label, head, witness = changed[0]
            built.append(ReviewCase(split, case.domain, case.name, case.after, head, "bug",
                                    label, witness["input"], witness["main"]))
        for label, head in chosen:
            built.append(ReviewCase(split, case.domain, case.name, case.after, head, "ok", label))
    print(counts, flush=True)
    CASES_FILE.write_text(json.dumps([asdict(case) for case in built], indent=1) + "\n")
    cases = all_cases()
    verify_review_data(cases)
    CHECKSUM_FILE.write_text(fingerprint(cases) + "\n")
    return cases


# Rows: name, parameter, main body, pull request body, change, then the witness
# argument and main's return value for behavior changes.
TEST = (
    *authored("payroll", [
        ("qualifiesForOvertime", "hoursWorked",
         "return hoursWorked > 40;",
         "return !(hoursWorked < 40);",
         "negation-slip", 40, False),
        ("withholdingCents", "grossCents",
         "return Math.floor(grossCents * 0.37);",
         "return Math.round(grossCents * 0.37);",
         "floor-to-round", 2, 0),
        ("netPayCents", "grossCents",
         "return grossCents - Math.round(grossCents * 0.37);",
         "return grossCents + Math.round(grossCents * 0.37);",
         "flipped-sign", 100, 63),
        ("overtimePayCents", "hoursWorked",
         "if (hoursWorked <= 40) return 0;\nreturn (hoursWorked - 40) * 3000;",
         "return hoursWorked <= 40 ? 0 : (hoursWorked - 40) * 3000;",
         "guard-to-ternary"),
        ("isFullTime", "hoursWorked",
         "return hoursWorked >= 32;",
         "return !(hoursWorked < 32);",
         "negate-comparison"),
        ("holidayAllowanceCents", "grossCents",
         "return Math.round(grossCents * 0.08);",
         "const allowanceRate = 0.08;\nreturn Math.round(grossCents * allowanceRate);",
         "name-constant"),
    ]),
    *authored("insurance premiums", [
        ("isSeniorRate", "age",
         "return age >= 65;",
         "return age >= 66;",
         "off-by-one-limit", 65, True),
        ("deductibleCents", "claimCents",
         "return Math.min(Math.max(claimCents, 0), 50000);",
         "return claimCents > 50000 ? 50000 : claimCents;",
         "clamp-rewrite-slip", -100, 0),
        ("monthlyPremiumCents", "annualCents",
         "return Math.round(annualCents / 12);",
         "return Math.round(annualCents / 10);",
         "wrong-divisor", 1200, 100),
        ("noClaimDiscountPercent", "claimFreeYears",
         "return Math.min(claimFreeYears * 10, 50);",
         "return Math.min(50, claimFreeYears * 10);",
         "swap-min-arguments"),
        ("riskBand", "age",
         "if (age < 25) return 3;\nif (age < 65) return 1;\nreturn 2;",
         "if (age >= 65) return 2;\nif (age >= 25) return 1;\nreturn 3;",
         "reordered-guards"),
        ("excessCents", "claimCents",
         "return claimCents > 25000 ? claimCents - 25000 : 0;",
         "return Math.max(claimCents - 25000, 0);",
         "ternary-to-max"),
    ]),
    *authored("loyalty points", [
        ("bonusTierReached", "points",
         "return points >= 1000;",
         "if (points > 1000) {\n  return true;\n}\nreturn false;",
         "if-rewrite-slip", 1000, True),
        ("pointsEarned", "spendCents",
         "return Math.floor(spendCents / 100);",
         "return Math.round(spendCents / 100);",
         "floor-to-round", 150, 1),
        ("pointsToCents", "points",
         "return points * 0.5;",
         "return points * 5;",
         "decimal-slip", 10, 5),
        ("pointsAfterRedemption", "points",
         "return Math.max(0, points - 2500);",
         "const redemptionCost = 2500;\nreturn Math.max(0, points - redemptionCost);",
         "name-constant"),
        ("isEliteMember", "points",
         "return points >= 50000 && points < 1000000;",
         "return !(points < 50000 || points >= 1000000);",
         "de-morgan"),
        ("tierMultiplier", "points",
         "return points >= 10000 ? 2 : 1;",
         "if (points >= 10000) {\n  return 2;\n}\nreturn 1;",
         "ternary-to-if"),
    ]),
    *authored("savings accounts", [
        ("earnsBonusRate", "balanceCents",
         "return balanceCents >= 100000;",
         "return balanceCents >= 100001;",
         "off-by-one-limit", 100000, True),
        ("dailyInterestCents", "balanceCents",
         "return Math.floor(balanceCents * 0.0001);",
         "return Math.trunc(balanceCents * 0.0001);",
         "floor-to-trunc", -5000, -1),
        ("annualToMonthlyRateBps", "annualBps",
         "return annualBps / 12;",
         "const monthsPerYear = 12;\nreturn annualBps * monthsPerYear;",
         "name-constant-slip", 120, 10),
        ("cappedDepositCents", "amountCents",
         "return Math.min(amountCents, 1000000);",
         "return amountCents > 1000000 ? 1000000 : amountCents;",
         "min-to-ternary"),
        ("isOverdrawn", "balanceCents",
         "return balanceCents < 0;",
         "return 0 > balanceCents;",
         "flip-comparison"),
        ("interestTierIndex", "balanceCents",
         "return balanceCents < 100000 ? 0 : balanceCents < 1000000 ? 1 : 2;",
         "if (balanceCents < 100000) return 0;\nif (balanceCents < 1000000) return 1;\nreturn 2;",
         "ternary-chain-to-guards"),
    ]),
)

# Test refactors whose transformation family appears in the round-two training data.
TRAINING_LIKE_REFACTORS = (
    "guard-to-ternary", "negate-comparison", "name-constant", "swap-min-arguments",
    "de-morgan", "ternary-to-if", "flip-comparison",
)


def all_cases():
    generated = [ReviewCase(**row) for row in json.loads(CASES_FILE.read_text())]
    seen = [replace(case, split="seen") for case in MONEY]
    return (*generated, *seen, *TEST)


def fingerprint(cases=None):
    contents = json.dumps([asdict(case) for case in (cases or all_cases())], sort_keys=True)
    return hashlib.sha256(contents.encode()).hexdigest()


def locked_cases():
    cases = all_cases()
    if fingerprint(cases) != CHECKSUM_FILE.read_text().strip():
        raise ValueError("Review tuning fixtures differ from data/review-tuning.sha256.")
    return cases


def witness_literals(case):
    """The proof a prover should write: the witness input and main's return there."""
    return expected_literal(case.argument), expected_literal(case.expected)


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        sys.exit("Usage: python review_tuning_fixtures.py build")
    from collections import Counter
    cases = build()
    print(Counter((case.split, case.verdict) for case in cases))
    print(Counter(case.change for case in cases if case.split in ("train", "valid")).most_common())
    print(fingerprint(cases))
