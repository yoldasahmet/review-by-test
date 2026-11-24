"""Verify the committed generations with Vitest, without downloading a model."""
import json
from pathlib import Path
from evaluate import Trial, score_trials
from fixtures import cases, diff

for mode in ("baseline", "tuned", "reference"):
    rows = json.loads(Path(f"results/{mode}.json").read_text())
    trials = []
    for row, (name, old, new, _) in zip(rows, cases("test"), strict=True):
        if row["name"] != name or row["diff"] != diff(old, new):
            raise ValueError(f"Fixture mismatch in {mode}: {name}")
        trials.append(Trial(name, old, new, row["generated"]))
    hits = 0
    for row, result in zip(rows, score_trials(trials), strict=True):
        status = result["status"]
        if status != row["status"] or (mode == "reference" and status != "red_green"):
            raise ValueError(f"{mode}/{row['name']}: {status}")
        hits += status == "red_green"
    print(f"{mode}: {hits}/{len(rows)} red/green", flush=True)
