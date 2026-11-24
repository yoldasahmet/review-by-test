# Verified-input and computed-expected-value ablation

The test-only split contains 24 synthetic TypeScript fixes across survey processing, acoustic measurement, ceramics, and nautical calculations. Each domain contributes six cases. All 24 references failed by assertion before the fix and passed afterward. The fixture checksum was recorded before model inference. [Fixture source](../ablation_fixtures.py) · [Frozen checksum](../data/ablation-test.sha256)

All cases use the cached, pinned 4-bit Qwen2.5-Coder-1.5B-Instruct base model. The two model calls per case use the original prompt, the same canonical Vitest grammar, greedy decoding, and a 192-token cap. Their order alternates across cases. Both patches and test arguments remain within the existing single-function, numeric-argument benchmark.

- **Grammar alone.** The model chooses the argument and the expected value.
- **Verified input.** The grammar forces the reference argument, which Vitest has already verified distinguishes the two revisions. The model chooses the expected value from that generated context. The reference expected value is used for witness verification and is kept out of the model call.
- **Computed expected value.** The evaluator runs the fixed fixture at the grammar-only argument. It builds a new assertion with that observed return value and reruns it against both revisions. This arm reuses the original model generation, and each argument is preserved exactly. The computed value is encoded as a finite decimal or boolean in the evaluator's syntax, including signed zero.

Red/green requires an assertion failure on the old fixture and a pass on the fixed fixture. The computed-value arm uses an execution oracle on controlled fixtures, so it measures how many of the model's selected inputs can expose the fix with a correct assertion. The verified-input arm gives the model one particular authored witness per case. Witnesses can vary in how difficult their expected values are to compute.

## Results

| Arm | Red/green | Distinguishing inputs | Passes both | Fails on fix |
|---|---:|---:|---:|---:|
| Grammar alone | 3/24 (12.5%) | 11/24 | 11 | 10 |
| Verified input supplied | 12/24 (50.0%) | 24/24 | 0 | 12 |
| Expected value computed for the grammar-only input | 11/24 (45.8%) | 11/24 | 13 | 0 |

All 72 evaluated outputs passed the format check. The grammar-only model chose 11 inputs that exposed the fix. It asserted the correct fixed value for three of those inputs, and computing the value rescued the other eight. The remaining 13 model-chosen inputs gave the same result on both revisions. Supplying one verified input per case added ten catches, lost one, and still left 12 wrong expected values. Both input selection and expected-value accuracy limited red/green on this split.

For example, the model chose `isClipping(2)`, which returns `true` in both revisions. The verified input of `-1` let it correctly assert the fixed value of `true` while the old function returns `false`. On `watchRotationCrew`, the model already chose 0, where the old function returns 1 and the fixed function returns 3. Computing the expected value corrected its assertion from 1 to 3. The lost catch was `followUpCredits`. The model correctly asserted 0 at its original input of 0, then repeated that answer when the verified input of 2 required 3.

The interventions test different conditions. The verified-input arm fixes the argument before the model computes its answer. The computed-value arm executes the fixed code after the baseline model chooses its argument. Their one-case score difference is descriptive for this locked split. [Raw assertions and revision returns](../results/ablation/comparison.json)

## Reproduce

Replay the scored outputs with Node 22 and Python 3.11 or 3.12.

```sh
npm ci
python3 ablation_experiment.py replay
```

With the pinned model cached and the locked Python environment installed, generate and score both model arms locally.

```sh
HF_HUB_OFFLINE=1 uv run --locked python ablation_experiment.py run runs/ablation-reproduction
python3 ablation_experiment.py replay runs/ablation-reproduction/comparison.json
```

The archive includes the generated assertions, the corrected assertions, and both return values for every supported output. Replay recomputes the expected values on the fixed fixtures. [Saved comparison](../results/ablation/comparison.json)
