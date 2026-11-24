"""Bug pull requests with candidate witnesses for the prover experiment.

The second-round prover trained on witnesses picked by a rule: integers nearest a
literal in the code. That rule pulled the model toward 0 and 1, where many
revisions agree. Here every training bug PR carries a list of candidate inputs:
round numbers, literals from the code and their neighbours, and the authored or
sampled witness. Vitest keeps the candidates where main and the pull request
differ and records main's return. The experiment then asks the base model which
of those correct tests it finds most likely and trains on that one.

Training and validation bugs are the reverted fixes and every working slip from
the second-round generator. The second round's seen and fresh splits become one
development split. The locked test split is new: 24 authored bug PRs and 12 exact
refactors in ride fares, parking fees, event ticketing and gym memberships.

Build the candidates once with `python review_prover_fixtures.py build`.
"""
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import review_tuning_fixtures as tuning
from ablation_experiment import expected_literal
from review_fixtures import ROOT, SOURCES, ReviewCase, authored, verify_review_data
from review_tuning_fixtures import candidates_for, literals, sample

CASES_FILE = ROOT / "data/review-prover-cases.json"
CHECKSUM_FILE = ROOT / "data/review-prover.sha256"
LITERAL = re.compile(r"-?(?:0|[1-9][0-9]{0,14})(?:\.[0-9]{1,12})?")
ROUND_NUMBERS = (0, 1, -1, 2, 5, 10, -10, 50, 100, -100, 500, 1000, -1000, 5000, 10000, 100000, 1000000)
MAX_CANDIDATES = 20


@dataclass(frozen=True)
class ProverCase:
    case: ReviewCase
    # Inputs where main and the pull request differ, with main's return at each.
    candidates: tuple


def encodable(value):
    literal = expected_literal(value)
    return literal is not None and literal != "-0" and (isinstance(value, bool) or LITERAL.fullmatch(literal))


def candidate_inputs(case):
    """Round numbers, then the code's literals and their neighbours, then the known witness."""
    values = [*ROUND_NUMBERS]
    for literal in literals(case.base, case.head):
        values += [literal, literal + 1, literal - 1, literal + 0.5]
    if case.argument is not None:
        values.insert(0, case.argument)
    seen, ordered = set(), []
    for value in values:
        value = int(value) if float(value).is_integer() else value
        if value not in seen and not isinstance(value, bool) and encodable(value):
            seen.add(value)
            ordered.append(value)
    return ordered


CHECKER = """import {{ writeFileSync }} from 'node:fs';
import {{ test }} from 'vitest';
import {{ {name} as onMain }} from './main';
import {{ {name} as inPullRequest }} from './pull';
const candidates = {candidates};
test('candidates', () => {{
  const found = [];
  for (const input of candidates) {{
    const main = onMain(input);
    if (!Object.is(main, inPullRequest(input)) && (typeof main === 'boolean'
        || (Number.isFinite(main) && !Object.is(main, -0)))) found.push([input, main]);
  }}
  writeFileSync(new URL('./result.json', import.meta.url), JSON.stringify(found));
}});
"""


def distinguishing(cases, batch=150):
    """For each bug PR, the candidate inputs where main and the pull request differ."""
    results = []
    for start in range(0, len(cases), batch):
        chunk = cases[start:start + batch]
        with TemporaryDirectory(dir=ROOT) as tmp:
            folder = Path(tmp).resolve()
            (folder / "vitest.config.mjs").write_text("export default {};\n")
            for index, case in enumerate(chunk):
                trial = folder / str(index)
                trial.mkdir()
                (trial / "main.ts").write_text(case.base)
                (trial / "pull.ts").write_text(case.head)
                (trial / "candidates.test.ts").write_text(CHECKER.format(
                    name=case.name, candidates=json.dumps(candidate_inputs(case))))
            subprocess.run(
                [str(ROOT / "node_modules/.bin/vitest"), "run", "--root", str(folder),
                 "--config", str(folder / "vitest.config.mjs"), "--maxWorkers", "2"],
                cwd=folder, capture_output=True, text=True, timeout=900,
            )
            for index in range(len(chunk)):
                result = folder / str(index) / "result.json"
                found = json.loads(result.read_text()) if result.exists() else []
                results.append(tuple((x, y) for x, y in found if encodable(y))[:MAX_CANDIDATES])
    return results


def build():
    bugs = []
    for split in SOURCES:
        sources = [case for module in SOURCES[split] for case in module.locked_cases()]
        slips = []
        for index, case in enumerate(sources):
            bugs.append(ReviewCase(split, case.domain, case.name, case.after, case.before, "bug",
                                   "revert-fix", case.argument, case.expected))
            _, candidates = candidates_for(case, index)
            heads = {case.before, case.after}
            for label, head in candidates:
                if head not in heads:
                    heads.add(head)
                    slips.append(ReviewCase(split, case.domain, case.name, case.after, head, "bug", label))
        print(f"Sampling {len(slips)} {split} slips", flush=True)
        for slip, result in zip(slips, sample([(s.base, s.head, s.name) for s in slips]), strict=True):
            if result and result["differing"] and result["witness"]:
                witness = result["witness"]
                bugs.append(replace(slip, argument=witness["input"], expected=witness["main"]))
    print(f"Checking candidates for {len(bugs)} bug PRs", flush=True)
    rows = []
    for case, found in zip(bugs, distinguishing(bugs), strict=True):
        if found:
            rows.append(dict(asdict(case), candidates=[list(pair) for pair in found]))
    CASES_FILE.write_text(json.dumps(rows, indent=1) + "\n")
    cases = all_cases()
    verify_review_data([item.case for item in cases])
    CHECKSUM_FILE.write_text(fingerprint(cases) + "\n")
    return cases


# Rows: name, parameter, main body, pull request body, change, then the witness
# argument and main's return value for behavior changes.
TEST = (
    *authored("ride fares", [
        ("isLongTrip", "distanceKm",
         "return distanceKm > 25;",
         "return distanceKm >= 25;",
         "exclusive-to-inclusive", 25, False),
        ("isSurgeHour", "hour",
         "return hour >= 17 && hour < 19;",
         "return !(hour < 17 || hour > 19);",
         "de-morgan-slip", 19, False),
        ("roundedFareCents", "fareCents",
         "return Math.round(fareCents / 10) * 10;",
         "return Math.floor(fareCents / 10) * 10;",
         "round-to-floor", 15, 20),
        ("tipCents", "fareCents",
         "return Math.max(0, Math.round(fareCents * 0.15));",
         "return Math.round(fareCents * 0.15);",
         "dropped-clamp", -100, 0),
        ("distanceFareCents", "distanceKm",
         "return Math.round(distanceKm * 125);",
         "return Math.round(distanceKm * 12.5);",
         "decimal-slip", 2, 250),
        ("driverShareCents", "fareCents",
         "return fareCents - Math.round(fareCents * 0.25);",
         "const commission = 0.25;\nreturn Math.round(fareCents * commission);",
         "name-constant-slip", 100, 75),
        ("isAirportTrip", "zone",
         "return zone === 9;",
         "return 9 === zone;",
         "flip-comparison"),
        ("nightSurchargeCents", "hour",
         "return hour >= 22 || hour < 6 ? 150 : 0;",
         "if (hour >= 22 || hour < 6) {\n  return 150;\n}\nreturn 0;",
         "ternary-to-if"),
        ("bookingFeeCents", "distanceKm",
         "return Math.min(200 + distanceKm * 10, 500);",
         "return Math.min(10 * distanceKm + 200, 500);",
         "reordered-sum"),
    ]),
    *authored("parking fees", [
        ("isFreeParking", "minutes",
         "return minutes <= 15;",
         "return minutes < 15;",
         "inclusive-to-exclusive", 15, True),
        ("dailyCapReached", "feeCents",
         "return feeCents >= 2400;",
         "return feeCents >= 2500;",
         "wrong-limit", 2400, True),
        ("billedHours", "minutes",
         "return Math.ceil(minutes / 60);",
         "return Math.round(minutes / 60);",
         "ceil-to-round", 70, 2),
        ("overstayFeeCents", "minutes",
         "if (minutes <= 120) return 0;\nreturn (minutes - 120) * 5;",
         "return (minutes - 120) * 5;",
         "dropped-guard", 60, 0),
        ("parkingFeeCents", "minutes",
         "return Math.ceil(minutes / 60) * 300;",
         "return Math.ceil(minutes / 60) * 30;",
         "dropped-digit", 60, 300),
        ("meterMinutes", "coinsCents",
         "return coinsCents / 2;",
         "return coinsCents * 2;",
         "divide-to-multiply", 10, 5),
        ("isOversized", "vehicleLengthCm",
         "return vehicleLengthCm > 520;",
         "return !(vehicleLengthCm <= 520);",
         "negate-comparison"),
        ("graceMinutes", "minutes",
         "return minutes < 10 ? minutes : 10;",
         "return Math.min(minutes, 10);",
         "ternary-to-min"),
        ("zoneRateCents", "zone",
         "if (zone === 1) return 400;\nif (zone === 2) return 250;\nreturn 150;",
         "if (zone === 2) return 250;\nif (zone === 1) return 400;\nreturn 150;",
         "reordered-guards"),
    ]),
    *authored("event ticketing", [
        ("isChildTicket", "age",
         "return age < 12;",
         "return age <= 12;",
         "exclusive-to-inclusive", 12, False),
        ("soldOut", "ticketsSold",
         "return ticketsSold >= 500;",
         "if (ticketsSold > 500) {\n  return true;\n}\nreturn false;",
         "if-rewrite-slip", 500, True),
        ("serviceFeeCents", "priceCents",
         "return Math.ceil(priceCents * 0.08);",
         "return Math.round(priceCents * 0.08);",
         "ceil-to-round", 5, 1),
        ("refundCents", "priceCents",
         "return Math.max(0, priceCents - 250);",
         "return priceCents - 250;",
         "dropped-clamp", 100, 0),
        ("groupDiscountCents", "priceCents",
         "return Math.round(priceCents * 0.1);",
         "return Math.round(priceCents * 0.01);",
         "decimal-slip", 1000, 100),
        ("totalWithFeeCents", "priceCents",
         "return priceCents + Math.ceil(priceCents * 0.08);",
         "return priceCents - Math.ceil(priceCents * 0.08);",
         "flipped-sign", 100, 108),
        ("isEarlyBird", "daysBefore",
         "return daysBefore >= 30;",
         "const earlyBirdDays = 30;\nreturn daysBefore >= earlyBirdDays;",
         "name-constant"),
        ("seatsLeft", "ticketsSold",
         "return Math.max(500 - ticketsSold, 0);",
         "return Math.max(0, 500 - ticketsSold);",
         "swap-max-arguments"),
        ("priceTier", "daysBefore",
         "return daysBefore >= 30 ? 1 : daysBefore >= 7 ? 2 : 3;",
         "if (daysBefore >= 30) return 1;\nif (daysBefore >= 7) return 2;\nreturn 3;",
         "ternary-chain-to-guards"),
    ]),
    *authored("gym memberships", [
        ("qualifiesForLoyalty", "months",
         "return months >= 24;",
         "return months > 24;",
         "inclusive-to-exclusive", 24, True),
        ("isTeenRate", "ageYears",
         "return ageYears >= 13 && ageYears <= 17;",
         "return !(ageYears < 13 || ageYears >= 17);",
         "de-morgan-slip", 17, True),
        ("perVisitCents", "visits",
         "return visits <= 0 ? 0 : Math.round(3000 / visits);",
         "return Math.round(3000 / visits);",
         "dropped-guard", -10, 0),
        ("freezeDaysUsed", "days",
         "return Math.min(Math.max(days, 0), 30);",
         "return Math.min(days, 30);",
         "dropped-clamp", -5, 0),
        ("annualFeeCents", "monthlyCents",
         "return monthlyCents * 12;",
         "return monthlyCents * 10;",
         "wrong-multiplier", 100, 1200),
        ("personalTrainingCents", "sessions",
         "return sessions * 4500;",
         "const sessionRate = 450;\nreturn sessions * sessionRate;",
         "name-constant-slip", 2, 9000),
        ("isPeakHour", "hour",
         "return hour >= 17 && hour <= 20;",
         "return !(hour < 17 || hour > 20);",
         "de-morgan"),
        ("lockerFeeCents", "months",
         "return months * 300;",
         "return 300 * months;",
         "commute-operands"),
        ("membershipLevel", "visits",
         "if (visits >= 100) return 3;\nif (visits >= 40) return 2;\nreturn 1;",
         "return visits >= 100 ? 3 : visits >= 40 ? 2 : 1;",
         "guards-to-ternary"),
    ]),
)


def all_cases():
    generated = [ProverCase(ReviewCase(**{k: v for k, v in row.items() if k != "candidates"}),
                            tuple(tuple(pair) for pair in row["candidates"]))
                 for row in json.loads(CASES_FILE.read_text())]
    development = [ProverCase(replace(case, split="dev"), ()) for case in tuning.all_cases()
                   if case.split in ("seen", "test")]
    test = [ProverCase(case, ()) for case in TEST]
    return (*generated, *development, *test)


def fingerprint(cases=None):
    contents = json.dumps([dict(asdict(item.case), candidates=[list(p) for p in item.candidates])
                           for item in (cases or all_cases())], sort_keys=True)
    return hashlib.sha256(contents.encode()).hexdigest()


def locked_cases():
    cases = all_cases()
    if fingerprint(cases) != CHECKSUM_FILE.read_text().strip():
        raise ValueError("Prover fixtures differ from data/review-prover.sha256.")
    return cases


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        sys.exit("Usage: python review_prover_fixtures.py build")
    from collections import Counter
    cases = build()
    print(Counter((item.case.split, item.case.verdict) for item in cases))
    print("candidates per bug PR", Counter(len(item.candidates) for item in cases if item.candidates).most_common(6))
    print(fingerprint(cases))
