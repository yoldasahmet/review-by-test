"""Score one generation per fix. Only the documented literal-assertion grammar is executed."""
import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from fixtures import cases, diff

ROOT = Path(__file__).resolve().parent
NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
TITLE = r"""(?:'[^'\\\r\n\u2028\u2029]{1,180}'|"[^"\\\r\n\u2028\u2029]{1,180}")"""


def literal_assertion(code, name):
    """Return the two validated source literals, or None for unsupported output."""
    name = re.escape(name)
    code = code.strip()

    imports = [
        r"""import\s*\{\s*(?:test\s*,\s*expect|expect\s*,\s*test)\s*\}\s*from\s*(['"])vitest\1\s*;?""",
        rf"""import\s*\{{\s*{name}\s*\}}\s*from\s*(['"])\./subject\1\s*;?""",
    ]
    for pattern in sorted(imports, key=lambda p: not re.match(p, code)):
        match = re.match(pattern, code)
        if not match:
            return None
        code = code[match.end():].strip()
    body = (
        rf"test\s*\(\s*{TITLE}\s*,\s*\(\s*\)\s*=>\s*\{{"
        rf"\s*expect\s*\(\s*{name}\s*\(\s*({NUMBER})\s*\)\s*\)"
        rf"\s*\.toBe\s*\(\s*({NUMBER}|true|false)\s*\)\s*;?\s*\}}\s*\)\s*;?"
    )
    match = re.fullmatch(body, code)
    if match and all(value in ("true", "false") or isfinite(float(value)) for value in match.groups()):
        return match.groups()
    return None


def supported(code, name):
    return literal_assertion(code, name) is not None


@dataclass(frozen=True)
class Trial:
    name: str
    before: str
    after: str
    code: str


def vitest_report(folder, expected_files):
    runner = ROOT / "node_modules/.bin/vitest"
    if not runner.is_file():
        raise RuntimeError("Vitest is missing. Run npm ci in the repository root.")
    config = folder / "vitest.config.mjs"
    config.write_text("export default {};\n")
    report = folder / "vitest.json"
    command = [
        str(runner), "run", "--root", str(folder), "--config", str(config),
        "--maxWorkers", "2", "--reporter=json", "--outputFile", str(report),
    ]
    proc = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=120)
    if proc.returncode not in (0, 1) or not report.exists():
        raise RuntimeError(f"Vitest did not produce a usable report:\n{proc.stderr or proc.stdout}")
    outcomes = {}
    for suite in json.loads(report.read_text())["testResults"]:
        results = suite["assertionResults"]
        if len(results) == 1 and results[0]["status"] == "passed":
            outcomes[Path(suite["name"])] = "pass"
        elif (len(results) == 1 and results[0]["status"] == "failed"
              and any(message.startswith("AssertionError:") for message in results[0]["failureMessages"])):
            outcomes[Path(suite["name"])] = "assertion_failure"
        else:
            raise RuntimeError(f"Unexpected Vitest outcome: {suite}")
    if outcomes.keys() != expected_files or (proc.returncode == 0) != all(
        outcome == "pass" for outcome in outcomes.values()
    ):
        raise RuntimeError("Vitest did not report the expected tests and exit status.")
    return outcomes


def write_trial(folder, name, subject, argument, expected):
    folder.mkdir()
    (folder / "subject.ts").write_text(subject)
    test = folder / "subject.test.ts"
    # Reconstruct the assertion from literals. No model-provided code is executed.
    test.write_text(f"""import {{ writeFileSync }} from 'node:fs';
import {{ expect, test }} from 'vitest';
import {{ {name} }} from './subject';
test('regression', () => {{
  const actual = {name}({argument});
  const value = typeof actual === 'boolean' ? actual :
    Object.is(actual, -0) ? '-0' : Number.isFinite(actual) ? actual : String(actual);
  writeFileSync(new URL('./actual.json', import.meta.url), JSON.stringify(value));
  expect(actual).toBe({expected});
}});
""")
    return test


def score_trials(trials):
    """Batch controlled fixtures through Vitest and capture returns on both revisions."""
    results = [{"status": "outside_format"} for _ in trials]
    with TemporaryDirectory(dir=ROOT) as tmp:
        folder = Path(tmp).resolve()
        accepted = {}
        for index, trial in enumerate(trials):
            literals = literal_assertion(trial.code, trial.name)
            if literals is None:
                continue
            argument, expected = literals
            before = write_trial(folder / f"{index}-before", trial.name, trial.before, argument, expected)
            after = write_trial(folder / f"{index}-after", trial.name, trial.after, argument, expected)
            accepted[index] = (argument, expected, before, after)
        if not accepted:
            return results
        outcomes = vitest_report(folder, {path for entry in accepted.values() for path in entry[2:]})
        for index, (argument, expected, before, after) in accepted.items():
            actual_before = json.loads(before.with_name("actual.json").read_text())
            actual_after = json.loads(after.with_name("actual.json").read_text())
            if outcomes[after] == "assertion_failure":
                status = "fails_on_fix"
            elif outcomes[before] == "assertion_failure":
                status = "red_green"
            else:
                status = "passes_both"
            results[index] = dict(
                status=status, argument=argument, expected=expected,
                before=actual_before, after=actual_after,
                distinguishes=type(actual_before) is not type(actual_after) or actual_before != actual_after,
            )
    return results

def score(name, code, old, new):
    return score_trials([Trial(name, old, new, code)])[0]["status"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "baseline", "tuned"])
    parser.add_argument("--output", type=Path, help="New JSON file for the historical rerun.")
    args = parser.parse_args()
    dest = args.output or ROOT / "runs/historical" / f"{args.mode}.json"
    if dest.exists():
        parser.error(f"{dest} already exists. Choose a new --output path.")
    if args.mode != "reference":
        from generate import predictor
        predict = predictor("adapters" if args.mode == "tuned" else None)
    rows, trials = [], []
    for name, old, new, reference in cases("test"):
        patch = diff(old, new)
        code = reference if args.mode == "reference" else predict(patch)
        trials.append(Trial(name, old, new, code))
        rows.append(dict(name=name, diff=patch, generated=code))
    for row, outcome in zip(rows, score_trials(trials), strict=True):
        row["status"] = outcome["status"]
        print(row["name"], row["status"], flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as out:
        out.write(json.dumps(rows, indent=2) + "\n")
    print(f"{sum(row['status'] == 'red_green' for row in rows)}/{len(rows)} red/green")
