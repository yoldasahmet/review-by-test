"""Pull-request cases for the review experiment.

Every pull request is presented to the reviewer as a refactor that keeps behavior
unchanged. Its gold verdict is "bug" when some number input returns a different
value and "ok" when every input returns the same value.

Training and validation cases come from the earlier locked fixtures. A bug case
reverts an authored fix, so main holds the fixed function and the pull request
restores the old one. Its authored witness already proves the regression. A
refactor case rewrites the same fixed function with one exact transformation.

The locked test split is new: 24 authored money-handling pull requests, half of
them behavior changes and half exact refactors.
"""
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import ablation_fixtures
import benchmark_fixtures
import factorial_fixtures
import grammar_fixtures
from benchmark_fixtures import ROOT
from evaluate import Trial, score_trials, vitest_report
from fixtures import ROWS, diff, reference

SIGNATURE = re.compile(r"export function (\w+)\((\w+): number\) \{\n(.*)\n\}\n", re.S)
SOURCES = {
    "train": (benchmark_fixtures, grammar_fixtures, factorial_fixtures),
    "valid": (ablation_fixtures,),
}


@dataclass(frozen=True)
class ReviewCase:
    split: str
    domain: str
    name: str
    base: str
    head: str
    verdict: str
    change: str
    argument: int | float | None = None
    expected: int | float | bool | None = None

    @property
    def patch(self):
        """The pull request: main to head."""
        return diff(self.base, self.head)

    @property
    def reversed_patch(self):
        """The revert a reviewer would suggest: head back to main."""
        return diff(self.head, self.base)

    @property
    def proof(self):
        """Authored witness that passes on main and fails on the pull request."""
        if self.verdict != "bug":
            return None
        return reference(self.name, self.argument, self.expected)


def source(name, parameter, body):
    lines = "\n".join(f"  {line}" for line in body.splitlines())
    return f"export function {name}({parameter}: number) {{\n{lines}\n}}\n"


def parts(text):
    match = SIGNATURE.fullmatch(text)
    if not match:
        raise ValueError(f"Unexpected fixture source:\n{text}")
    name, parameter, body = match.groups()
    return name, parameter, body.split("\n")


def assemble(name, parameter, lines):
    return f"export function {name}({parameter}: number) {{\n" + "\n".join(lines) + "\n}\n"


# Each transformation is an identity for every finite number input, including -0.
# verify_review_data samples that claim on about 38,000 inputs per case.

def extract_result(text):
    name, parameter, lines = parts(text)
    match = re.fullmatch(r"  return (.+);", lines[-1])
    if not match or re.search(r"\bresult\b", text):
        return None
    return assemble(name, parameter, lines[:-1] + [f"  const result = {match[1]};", "  return result;"])


FLIPPED = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "===": "===", "!==": "!=="}
# A bare identifier compared with a literal, bounded by tokens that bind more
# loosely than the comparison, so swapping the operands keeps the same parse.
COMPARISON = re.compile(
    r"(?:(?<=return )|(?<=\()|(?<=&& )|(?<=\|\| ))([A-Za-z_]\w*) (<=|>=|===|!==|<|>) (-?\d+(?:\.\d+)?)(?=;|\)| &&| \|\|| \?)"
)


def flip_comparison(text):
    name, parameter, lines = parts(text)
    body = "\n".join(lines)
    flipped = COMPARISON.sub(lambda m: f"{m[3]} {FLIPPED[m[2]]} {m[1]}", body)
    return None if flipped == body else assemble(name, parameter, flipped.split("\n"))


CALL = re.compile(r"Math\.(min|max)\(([^(),]+), ([^(),]+)\)")


def swap_arguments(text):
    # Math.min and Math.max are symmetric, including their handling of -0 and NaN.
    match = CALL.search(text)
    if not match or match[2].strip() == match[3].strip():
        return None
    return text[:match.start()] + f"Math.{match[1]}({match[3]}, {match[2]})" + text[match.end():]


def rename_parameter(text):
    _, parameter, _ = parts(text)
    for new in ("value", "input", "quantity", "reading"):
        if new != parameter and not re.search(rf"\b{new}\b", text):
            return re.sub(rf"\b{re.escape(parameter)}\b", new, text)
    return None


TRANSFORMS = (
    ("extract-result", extract_result),
    ("flip-comparison", flip_comparison),
    ("swap-min-max-arguments", swap_arguments),
    ("rename-parameter", rename_parameter),
)


def refactor(text, index):
    """Prefer refactors that touch operators, then alternate the two textual ones."""
    for label, transform in TRANSFORMS[1:3]:
        changed = transform(text)
        if changed is not None:
            return label, changed
    textual = (TRANSFORMS[0], TRANSFORMS[3]) if index % 2 == 0 else (TRANSFORMS[3], TRANSFORMS[0])
    for label, transform in textual:
        changed = transform(text)
        if changed is not None:
            return label, changed
    raise ValueError(f"No refactor applies to:\n{text}")


def derived(split):
    index = 0
    for module in SOURCES[split]:
        for case in module.locked_cases():
            label, head = refactor(case.after, index)
            index += 1
            yield ReviewCase(split, case.domain, case.name, case.after, case.before, "bug",
                             "revert-fix", case.argument, case.expected)
            yield ReviewCase(split, case.domain, case.name, case.after, head, "ok", label)


def authored(domain, parameter_rows):
    for name, parameter, base, head, change, *witness in parameter_rows:
        verdict = "bug" if witness else "ok"
        argument, expected = witness if witness else (None, None)
        yield ReviewCase("test", domain, name, source(name, parameter, base), source(name, parameter, head),
                         verdict, change, argument, expected)


# Rows: name, parameter, main body, pull request body, change, then the witness
# argument and main's return value for behavior changes.
TEST = (
    *authored("card fees", [
        ("interchangeFeeCents", "amountCents",
         "return Math.round(amountCents * 0.015);",
         "return Math.floor(amountCents * 0.015);",
         "round-to-floor", 50, 1),
        ("authorizationFeeCents", "amountCents",
         "return amountCents === 0 ? 0 : Math.round(amountCents * 0.029) + 30;",
         "return Math.round(amountCents * 0.029) + 30;",
         "dropped-zero-guard", 0, 0),
        ("meetsMinimumCharge", "amountCents",
         "return amountCents >= 50;",
         "return amountCents > 50;",
         "inclusive-to-exclusive", 50, True),
        ("refundableFeeCents", "amountCents",
         "return amountCents <= 0 ? 0 : 30;",
         "if (amountCents <= 0) {\n  return 0;\n}\nreturn 30;",
         "ternary-to-if"),
        ("processingFeeCents", "amountCents",
         "return Math.round(amountCents * 0.014) + 25;",
         "const percentRate = 0.014;\nconst fixedFeeCents = 25;\n"
         "return Math.round(amountCents * percentRate) + fixedFeeCents;",
         "named-constants"),
        ("isMicroPayment", "amountCents",
         "return amountCents < 100;",
         "return 100 > amountCents;",
         "flip-comparison"),
    ]),
    *authored("VAT invoicing", [
        ("vatCents", "netCents",
         "return Math.round(netCents * 0.21);",
         "return Math.round(netCents * 21) / 100;",
         "fractional-cents", 3, 1),
        ("isSmallInvoice", "netCents",
         "return netCents <= 4000;",
         "return netCents <= 3999;",
         "off-by-one-limit", 4000, True),
        ("creditNoteVatCents", "creditCents",
         "return -Math.round(creditCents * 0.21);",
         "return Math.round(creditCents * 0.21);",
         "dropped-sign", 100, -21),
        ("invoiceTotalCents", "netCents",
         "return netCents + Math.round(netCents * 0.21);",
         "const vat = Math.round(netCents * 0.21);\nreturn netCents + vat;",
         "extract-variable"),
        ("vatBandIndex", "netCents",
         "if (netCents < 0) return 0;\nif (netCents < 100000) return 1;\nreturn 2;",
         "return netCents < 0 ? 0 : netCents < 100000 ? 1 : 2;",
         "guards-to-ternary"),
        ("invoiceLineCents", "netCents",
         "return Math.max(0, Math.round(netCents));",
         "return Math.max(Math.round(netCents), 0);",
         "swap-max-arguments"),
    ]),
    *authored("SME loan repayments", [
        ("monthlyInterestCents", "principalCents",
         "return Math.round(principalCents * 0.06 / 12);",
         "return Math.round(principalCents * 0.06);",
         "dropped-monthly-division", 120000, 600),
        ("lateFeeCents", "daysLate",
         "if (daysLate <= 0) return 0;\nreturn Math.min(daysLate * 500, 5000);",
         "if (daysLate <= 0) return 0;\nreturn daysLate * 500;",
         "dropped-cap", 30, 5000),
        ("remainingTermMonths", "paidMonths",
         "return Math.max(0, 36 - paidMonths);",
         "return 36 - paidMonths;",
         "dropped-clamp", 40, 0),
        ("balanceAfterPaymentCents", "balanceCents",
         "return Math.max(0, balanceCents - 25000);",
         "const monthlyPaymentCents = 25000;\nreturn Math.max(0, balanceCents - monthlyPaymentCents);",
         "named-constant"),
        ("isInGracePeriod", "daysLate",
         "return daysLate >= 0 && daysLate <= 5;",
         "return !(daysLate < 0 || daysLate > 5);",
         "de-morgan"),
        ("arrearsCategory", "daysLate",
         "if (daysLate > 90) return 3;\nif (daysLate > 30) return 2;\nif (daysLate > 0) return 1;\nreturn 0;",
         "if (daysLate <= 0) return 0;\nif (daysLate <= 30) return 1;\nif (daysLate <= 90) return 2;\nreturn 3;",
         "reordered-guards"),
    ]),
    *authored("payouts and FX", [
        ("payoutAfterReserveCents", "balanceCents",
         "return Math.floor(balanceCents * 0.9);",
         "return Math.round(balanceCents * 0.9);",
         "floor-to-round", 15, 13),
        ("eurToUsdCents", "eurCents",
         "return Math.round(eurCents * 1.08);",
         "return Math.round(eurCents / 1.08);",
         "inverted-rate", 100, 108),
        ("canInstantPayout", "balanceCents",
         "return balanceCents >= 1000 && balanceCents <= 500000;",
         "return balanceCents >= 1000 && balanceCents < 500000;",
         "inclusive-to-exclusive-limit", 500000, True),
        ("settlementWeekday", "dayOfWeek",
         "return (dayOfWeek + 2) % 7;",
         "return (2 + dayOfWeek) % 7;",
         "commuted-addition"),
        ("payoutFeeCents", "payouts",
         "return payouts * 2 * 25;",
         "return payouts * 50;",
         "folded-constant"),
        ("isWeekendSettlement", "dayOfWeek",
         "return dayOfWeek === 0 || dayOfWeek === 6;",
         "return dayOfWeek === 6 || dayOfWeek === 0;",
         "reordered-condition"),
    ]),
)

# Test refactors that reuse a transformation family seen in training.
TRAINING_LIKE_REFACTORS = ("flip-comparison", "extract-variable", "swap-max-arguments")


def all_cases():
    return (*derived("train"), *derived("valid"), *TEST)


def fingerprint(cases=None):
    contents = json.dumps([asdict(case) for case in (cases or all_cases())], sort_keys=True)
    return hashlib.sha256(contents.encode()).hexdigest()


def locked_cases():
    cases = all_cases()
    expected = (ROOT / "data/review.sha256").read_text().strip()
    if fingerprint(cases) != expected:
        raise ValueError("Review fixtures differ from data/review.sha256.")
    return cases


INPUTS = """export const inputs: number[] = [-0];
for (let i = -5000; i <= 5000; i += 1) inputs.push(i);
for (let i = -4000; i <= 4000; i += 1) inputs.push(i / 4);
for (let i = -10000; i <= 10000; i += 1) inputs.push(i / 100);
inputs.push(1 / 3, 2 / 3, 0.1 + 0.2, 86399, 86400, 99999.5, 100000, 100000.5, 120000,
  499999, 500000, 500000.5, 500001, 1e6, -1e6, 1e9, 1e12, 1e15, -1e15);
"""


def sampled_differences(cases, batch=100):
    """Count the sampled inputs where main and the pull request return different values."""
    if len(cases) > batch:
        results = {}
        for start in range(0, len(cases), batch):
            results |= sampled_differences(cases[start:start + batch], batch)
        return results
    results = {}
    with TemporaryDirectory(dir=ROOT) as tmp:
        folder = Path(tmp).resolve()
        (folder / "inputs.ts").write_text(INPUTS)
        tests = {}
        for index, case in enumerate(cases):
            trial = folder / str(index)
            trial.mkdir()
            (trial / "main.ts").write_text(case.base)
            (trial / "pull.ts").write_text(case.head)
            test = trial / "same.test.ts"
            test.write_text(f"""import {{ writeFileSync }} from 'node:fs';
import {{ expect, test }} from 'vitest';
import {{ inputs }} from '../inputs';
import {{ {case.name} as onMain }} from './main';
import {{ {case.name} as inPullRequest }} from './pull';
test('same returns', () => {{
  const differences = inputs.filter((input) => !Object.is(onMain(input), inPullRequest(input)));
  writeFileSync(new URL('./result.json', import.meta.url), JSON.stringify({{
    checked: inputs.length, differing: differences.length,
    examples: differences.slice(0, 3).map((input) => Object.is(input, -0) ? '-0' : String(input)),
  }}));
  expect(differences.length).toBe(0);
}});
""")
            tests[test] = case
        outcomes = vitest_report(folder, set(tests))
        for test, case in tests.items():
            summary = json.loads(test.with_name("result.json").read_text())
            results[(case.split, case.name, case.change)] = dict(outcome=outcomes[test], **summary)
    return results


def verify_review_data(cases):
    """Prove every behavior change with its witness and sample every refactor for equality."""
    bugs = [case for case in cases if case.verdict == "bug"]
    trials = [Trial(case.name, case.head, case.base, case.proof) for case in bugs]
    proofs = [row for start in range(0, len(trials), 100) for row in score_trials(trials[start:start + 100])]
    for case, outcome in zip(bugs, proofs, strict=True):
        if outcome["status"] != "red_green":
            raise ValueError(f"Witness does not prove the {case.split} pull request {case.name}: {outcome}")
    sampled = sampled_differences(cases)
    rows = []
    for case in cases:
        result = sampled[(case.split, case.name, case.change)]
        if case.verdict == "ok" and result["differing"]:
            raise ValueError(f"Refactor {case.split}/{case.name} changed returns at {result['examples']}.")
        rows.append(dict(split=case.split, name=case.name, verdict=case.verdict, change=case.change,
                         witness_red_green=case.verdict == "bug",
                         sampled_inputs=result["checked"], differing_inputs=result["differing"]))
    names = [case.name for case in cases if case.split == "test"]
    earlier = {case.name for case in cases if case.split != "test"} | {
        name for domain in ROWS.values() for name, *_ in domain
    }
    if len(names) != len(set(names)) or earlier & set(names):
        raise ValueError("Test pull requests must use fresh function names.")
    return rows


if __name__ == "__main__":
    cases = all_cases()
    counts = {}
    for case in cases:
        key = (case.split, case.verdict)
        counts[key] = counts.get(key, 0) + 1
    print(counts)
    verify_review_data(cases)
    print(fingerprint(cases))
