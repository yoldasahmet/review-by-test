"""Score one generation per fix. Only the documented literal-assertion grammar is executed."""
import argparse
import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from fixtures import cases, diff

ROOT = Path(__file__).resolve().parent
NUMBER = r"-?(?:0|[1-9]\d*)(?:\.\d+)?"
TITLE = r"""(?:'[^'\\\n]{1,180}'|"[^"\\\n]{1,180}")"""

def supported(code, name):
    imports = [
        r"""import\s*\{\s*(?:test\s*,\s*expect|expect\s*,\s*test)\s*\}\s*from\s*(['"])vitest\1\s*;?""",
        rf"""import\s*\{{\s*{name}\s*\}}\s*from\s*(['"])\./subject\1\s*;?""",
    ]
    for pattern in sorted(imports, key=lambda p: not re.match(p, code)):
        match = re.match(pattern, code)
        if not match:
            return False
        code = code[match.end():].strip()
    body = rf"test\s*\(\s*{TITLE}\s*,\s*\(\s*\)\s*=>\s*\{{\s*expect\s*\(\s*{name}\s*\(\s*{NUMBER}\s*\)\s*\)\s*\.toBe\s*\(\s*(?:{NUMBER}|true|false)\s*\)\s*;?\s*\}}\s*\)\s*;?"
    return bool(re.fullmatch(body, code))

def run_revision(folder, subject):
    (folder / "subject.ts").write_text(subject)
    report = folder / "vitest.json"
    report.unlink(missing_ok=True)
    command = [str(ROOT / "node_modules/.bin/vitest"), "run", "--root", str(folder),
               "--reporter=json", "--outputFile", str(report)]
    proc = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=45)
    if not report.exists():
        raise RuntimeError(proc.stderr or proc.stdout)
    suites = json.loads(report.read_text())["testResults"]
    assertions = [a for s in suites for a in s["assertionResults"]]
    if len(assertions) == 1:
        result = assertions[0]
        if proc.returncode == 0 and result["status"] == "passed":
            return "pass"
        if result["status"] == "failed" and "AssertionError" in str(result["failureMessages"]):
            return "assertion_failure"
    raise RuntimeError(f"Unexpected Vitest outcome: {suites}")

def score(name, code, old, new):
    if not supported(code, name):
        return "outside_format"
    with TemporaryDirectory(dir=ROOT) as tmp:
        folder = Path(tmp)
        (folder / "subject.test.ts").write_text(code)
        after = run_revision(folder, new)
        if after != "pass":
            return "fails_on_fix"
        before = run_revision(folder, old)
        return "red_green" if before == "assertion_failure" else "passes_both"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "baseline", "tuned"])
    args = parser.parse_args()
    if args.mode != "reference":
        from generate import predictor
        predict = predictor("adapters" if args.mode == "tuned" else None)
    rows = []
    for name, old, new, reference in cases("test"):
        patch = diff(old, new)
        code = reference if args.mode == "reference" else predict(patch)
        status = score(name, code, old, new)
        rows.append(dict(name=name, diff=patch, generated=code, status=status))
        print(name, status, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.mode}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{sum(row['status'] == 'red_green' for row in rows)}/{len(rows)} red/green")
