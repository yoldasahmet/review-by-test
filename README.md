# Review by Test

Review by Test asks a local 1.5B coding model to turn a TypeScript bug-fix diff into one Vitest regression test. A test earns credit when its assertion fails on the buggy revision and passes on the fixed revision.

```diff
 export function freeShipping(n: number) {
-  return n > 50;
+  return n >= 50;
 }
```

Qwen2.5-Coder generated `expect(freeShipping(50)).toBe(true)` from the [shipping diff](examples/shipping.diff), with an import from `./shipping`. Vitest verified that the old function returns `false` and the fixed function returns `true`. The first adapter chose 51, where both functions return `true`. [All six cases from the first experiment](docs/first-experiment.md)

## Results

Each model had one attempt per diff. We chose the prompt and adapter checkpoint on validation before generating the test outputs.

| Locked test set | Red then green | Passes both | Fails on fix | Unsupported |
|---|---:|---:|---:|---:|
| Original 4-bit model | 3/10 (30%) | 1/10 (10%) | 0/10 (0%) | 6/10 (60%) |
| Same model with 48-step QLoRA adapter | 5/10 (50%) | 0/10 (0%) | 4/10 (40%) | 1/10 (10%) |
| Authored reference tests | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |

The models both caught two cases. The adapter alone caught three, the original alone caught one, and both missed four. All three adapter-only catches came from cases where the original produced unsupported output. The adapter also asserted `clampEmptyCursor(0)` should be `1`, which is the old return value. The original caught that case using `clampEmptyCursor(-1)` and expecting `1`. [Paired results and timings](docs/benchmark.md) · [Raw generations and run settings](results/benchmark)

The earlier, separate pricing benchmark scored **3/6 original, 1/6 tuned, and 6/6 reference**. The fresh dataset contains 46 original synthetic fixes across eight domains, with 28 training, 8 validation and 10 test cases. The source includes each authored revision and witness. [Fixture source](benchmark_fixtures.py) · [Split, selection rules and measurement methods](docs/benchmark.md)

A separate, 24-case experiment compared the base model with and without constrained decoding. The constraint raised format compliance from 17/24 to 24/24 and red/green accuracy from 5/24 to 7/24. [Method and paired results](docs/grammar-experiment.md)

A fresh 2×2 comparison found that base plus grammar and tuned plus grammar both caught 5/24. Either grammar or tuning alone brought format compliance to 24/24. [Full 2×2 results](docs/factorial-experiment.md)

An input and expected-value ablation on another fresh 24-case split scored 3/24 with grammar alone, 12/24 with a verified input, and 11/24 when the fixed implementation supplied the expected value for the model's input. [Ablation results](docs/ablation-experiment.md)

A pull-request review experiment on a fresh split of 12 bug PRs and 12 exact refactors posted proven comments on 5 of the bugs and on none of the refactors. A LoRA-tuned reviewer did not beat the base model there: both flagged all 24 pull requests. [Review results](docs/review-experiment.md)

A second round trained on structural refactors and on rewrites that slip, and added a prover adapter. On another fresh split, the reviewer adapter flagged 11 of 12 bugs and cleared 4 of 12 refactors, but only refactor families it had trained on, and validation did not confirm the gain. The prover adapter proved 4 of 12 bugs, against 8 for the base prover. No refactor received a comment in either round. [Second-round results](docs/review-tuning-experiment.md)

A third round trained the prover on the most likely correct test the untuned model could write for each training bug. On a fresh split of 24 bug PRs, the selected adapter proved 17 against 11 for the untuned model, including 5 of 8 bugs that change a single boundary value against 1, and proved none of the 12 refactors. It met all four acceptance checks written into the run before it started. [Third-round results](docs/review-prover-experiment.md)

## Reviewing refactor pull requests

The review experiment points the same model at pull requests whose authors call them refactors. A reviewer, either the base model or a LoRA-tuned adapter, reads the diff and answers `bug` or `ok`. For each suspected bug, a prover writes a Vitest test, and a comment is posted only when that test fails on the pull request and passes on main. A wrong verdict can still waste a proof attempt, but it can never post an unproven comment. [Fixtures](review_fixtures.py) · [Runner](review_experiment.py)

The five pull requests below are cases from the locked money-handling test split. Each proof shown is the authored witness, checked with Vitest. Each result line shows what the first round's archived run did.

### 1. VAT that stops rounding to whole cents

```diff
 export function vatCents(netCents: number) {
-  return Math.round(netCents * 0.21);
+  return Math.round(netCents * 21) / 100;
 }
```

The cleanup looks like it avoids floating-point error. A spot check on a €100.00 line returns 2100 cents on both revisions. On a €19.99 line, main returns 420 and the pull request returns 419.79, so invoices start carrying fractions of a cent. Proof: `expect(vatCents(3)).toBe(1)` fails on the pull request, which returns 0.63.

Result: both reviewers flagged it. The prover chose a round €1.00 line, 100 cents, where both revisions return 21. Without a proof, the run posted only an advisory note.

### 2. A late-fee cap lost in a tidy-up

```diff
 export function lateFeeCents(daysLate: number) {
   if (daysLate <= 0) return 0;
-  return Math.min(daysLate * 500, 5000);
+  return daysLate * 500;
 }
```

At 10 days late, both revisions charge €50.00. At 30 days, the pull request charges a small business €150.00 instead of the €50.00 cap. Proof: `expect(lateFeeCents(30)).toBe(5000)`.

Result: both reviewers flagged it. The prover chose −1, below the guard both revisions share, so it found no proof and the run posted an advisory note.

### 3. An instant-payout limit that drops exactly €5,000

```diff
 export function canInstantPayout(balanceCents: number) {
-  return balanceCents >= 1000 && balanceCents <= 500000;
+  return balanceCents >= 1000 && balanceCents < 500000;
 }
```

Only 1 of the 38,023 sampled inputs changes: a merchant holding exactly €5,000.00 can no longer pay out instantly. Random inputs almost never find it, so the proof has to pick the limit itself. Proof: `expect(canInstantPayout(500000)).toBe(true)`.

Result: both reviewers flagged it. The prover chose the lower limit, 1000, where both revisions return `true`, so the run posted an advisory note.

### 4. A fee on a zero-amount card check

```diff
 export function authorizationFeeCents(amountCents: number) {
-  return amountCents === 0 ? 0 : Math.round(amountCents * 0.029) + 30;
+  return Math.round(amountCents * 0.029) + 30;
 }
```

Card checks that authorize €0.00 were free. After this simplification each one costs 30 cents, while every paid amount still returns the same fee. Proof: `expect(authorizationFeeCents(0)).toBe(0)`.

Result: both reviewers flagged it. The prover chose 0 and asserted main's return of 0. The test failed on the pull request, which returns 30, so the run posted a proven comment.

### 5. A refactor that looks risky and is safe

```diff
 export function isInGracePeriod(daysLate: number) {
-  return daysLate >= 0 && daysLate <= 5;
+  return !(daysLate < 0 || daysLate > 5);
 }
```

Every comparison changed, so the diff reads like a boundary bug. Both revisions return the same value on all 38,023 sampled inputs, so no test can separate them, and the gate cannot post a comment. A reviewer that flags it still wastes the author's time.

Result: both reviewers flagged it as a bug. The prover chose −1, where both revisions return `false`, and no comment was posted.

Telling example 5 apart from examples 1 to 4 is the reviewer's job, and the fine-tuning did not learn it. The adapter trained on 94 reversed fixes and 94 exact refactors of the same functions. On validation it cleared every rename and extracted variable but flagged every refactor that touched an operator, and on the money cases it flagged all 24 pull requests, as the base model did. What protected the author was the proof requirement: across every reviewer, 5 of the 12 bug PRs received a proven comment and none of the 12 refactors did. [Method and results](docs/review-experiment.md)

A second round trained on structural refactors, such as the one in example 5, and on rewrites that slip, such as `!(weightKg <= 70)` in place of `!(weightKg < 70)`. Its reviewer cleared a third of the refactors on a fresh split while flagging 11 of 12 bugs, but only for refactor families it had seen in training. It still flagged example 5 and missed example 2. [Second-round results](docs/review-tuning-experiment.md)

### 6. A boundary the untuned prover missed by one

```diff
 export function soldOut(ticketsSold: number) {
-  return ticketsSold >= 500;
+  if (ticketsSold > 500) {
+    return true;
+  }
+  return false;
 }
```

This case comes from the third round's fresh test split. The rewrite reads like a harmless move to an if statement, but an event with exactly 500 tickets sold no longer counts as sold out. The untuned prover chose 501, where both revisions return `true`. The third-round prover, fine-tuned on the model's own most likely correct tests, chose 500 and asserted main's `true`, so the test fails on the pull request and the comment is proven. It found the exact boundary on four more bugs in that split that the untuned prover missed, and it proved 17 of the 24 bug PRs against 11. On examples 1 to 5 it made the same choices as the untuned prover. [Third-round results](docs/review-prover-experiment.md)

## Replay in a clean checkout

With Node 22 and Python 3.11 or 3.12 installed, run from the repository root. Replay uses saved generations and requires no model download or Python dependencies.

```sh
npm ci
python3 -m unittest
python3 replay.py
python3 experiment.py replay
python3 review_experiment.py replay
python3 review_tuning_experiment.py replay
python3 review_prover_experiment.py replay
```

GitHub Actions runs the same checks. Replay checks each raw generation against the output grammar, runs accepted assertions on both controlled revisions, and verifies the frozen dataset, reference outcomes, and validation-based selection.

## Run locally on Apple Silicon

Measured on an **Apple M1 Pro with 16 GB unified memory**, macOS 26.5.1, Python 3.12.10, uv 0.13.9, Node 22.22.2, MLX 0.32.4, MLX-LM 0.32.2, and Vitest 3.2.7. The 48-step run took **22.84 s** and peaked at **1.54 GB** of MLX-managed allocations. Median test generation time was **0.76 s** for the original and **0.66 s** for the tuned model on this run. [Timing details](docs/benchmark.md)

From the repo root, install locked dependencies and download the pinned snapshot on first use. The snapshot occupies about 880 MB.

```sh
npm ci
uv sync --locked --python 3.12
uv run --locked python generate.py --download
uv run --locked python generate.py --cache-only --download
```

Both download commands print the resolved snapshot path. Its last component must be `b3252a2f97102b1fb1571fec2c9b27219a8536be`. The cached snapshot's `config.json` specifies 4-bit weights with group size 64. Use `--cache-only` for subsequent generations. If the cache check fails, rerun `--download` with internet access. [Pinned Hugging Face snapshot](https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit/tree/b3252a2f97102b1fb1571fec2c9b27219a8536be)

```sh
uv run --locked python generate.py --cache-only < examples/shipping.diff

# Train an adapter on the fresh training split.
uv run --locked python experiment.py train runs/my-adapter --prompt original --steps 48
uv run --locked python generate.py --cache-only --adapter runs/my-adapter/step-48 < examples/shipping.diff

# Reproduce validation, training, selection, and the locked test comparison.
uv run --locked python experiment.py run runs/my-comparison
python3 experiment.py replay runs/my-comparison

# Reproduce the review experiment. Add --resume to continue after a crash.
uv run --locked python review_experiment.py run runs/my-review
python3 review_experiment.py replay runs/my-review/comparison.json

# Reproduce the second round, which trains a reviewer and a prover adapter.
uv run --locked python review_tuning_experiment.py run runs/my-review-tuning
python3 review_tuning_experiment.py replay runs/my-review-tuning/comparison.json

# Reproduce the third round, which trains the prover on its own most likely correct tests.
uv run --locked python review_prover_experiment.py run runs/my-review-prover
python3 review_prover_experiment.py replay runs/my-review-prover/comparison.json
```

Use a new run directory for each experiment. Each run retains raw generations, timing, configuration, train and validation JSONL, and checkpoints under `runs/`. Git ignores that directory, the original adapter directory, model cache files, and local dependency installs.

### Use a Git diff

From the root of the project containing your patch, save a textual diff for one existing TypeScript source file, for example `git diff main...HEAD -- src/shipping.ts > /tmp/shipping.diff`. From the Review by Test root, run `uv run --locked python generate.py --cache-only < /tmp/shipping.diff`.

Generation prints TypeScript for a test beside the modified module, such as `./shipping`. It expects one unquoted `.ts` source path with a simple basename and a change to one exported function that takes one number and returns a number or boolean. Review the generated test before using it. Automated scoring only uses the repository's authored fixtures. It accepts two named imports, one `test(...)` block, and one direct `expect(fn(number)).toBe(numberOrBoolean)` assertion using decimal literals. See the [evaluator](evaluate.py) for the complete grammar.

## License

[MIT](LICENSE)
