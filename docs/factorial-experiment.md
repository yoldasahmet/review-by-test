# Base model, fine-tuning, and grammar constraints

This paired experiment tests four conditions on the same 24 synthetic TypeScript fixes. The test-only split has six cases each from typesetting, weather, tabletop scoring, and household energy. The cases have distinct function names from the prior benchmarks. Vitest verified all 24 authored witnesses before the split checksum was recorded. [Fixtures](../factorial_fixtures.py) · [Frozen checksum](../data/factorial-test.sha256)

Both models use the pinned 4-bit Qwen2.5-Coder-1.5B-Instruct revision. The tuned arms use the existing 48-step QLoRA checkpoint, selected on the earlier validation set. The run records the exact adapter weight checksum. All four arms use the original prompt, greedy decoding, a 192-token maximum, and one attempt per case. The order rotates across cases so each condition occupies each position six times. [Earlier checkpoint selection](../results/benchmark/selection.json)

The constrained arms reuse the previous experiment's canonical Vitest grammar. The syntax supplies the import, export name visible in the diff, and assertion structure. The model chooses the numeric input and the numeric or boolean expected value. The same evaluator scores all four arms. Format compliance means the evaluator accepted the structure. Red/green means the assertion failed on the old implementation and passed on the fixed implementation. [Grammar details](grammar-experiment.md)

## Results

| Model | Decoding | Red/green | Format compliant | Passes both | Fails on fix | Unsupported |
|---|---|---:|---:|---:|---:|---:|
| Base | Unconstrained | 3/24 (12.5%) | 9/24 (37.5%) | 4 | 2 | 15 |
| Base | Constrained | 5/24 (20.8%) | 24/24 (100%) | 11 | 8 | 0 |
| 48-step adapter | Unconstrained | 4/24 (16.7%) | 24/24 (100%) | 6 | 14 | 0 |
| 48-step adapter | Constrained | 5/24 (20.8%) | 24/24 (100%) | 7 | 12 | 0 |

Grammar added two red/green catches on net to the base model and one to the tuned model. Tuning added one catch on net with unconstrained decoding and zero with constrained decoding. The combination tied the grammar-only arm, one catch below the six implied by adding both standalone gains to the baseline. Either change alone reached full format compliance.

The two constrained arms shared two catches and each caught three that the other missed. For example, `glyphAdvancePx(0.5)` returns 16 before the fix and 8 afterward. The constrained base model chose 0.5 and correctly expected 8. The constrained adapter chose 2.5 and expected 800, while the fixed function returns 40. On `rankAfterMatch`, the adapter chose 10 in both modes and changed its expected value from 18 to the correct 1 under the grammar. Across all cases, 12 of the combined arm's 24 compliant tests failed on the fixed revision.

These are descriptive results from one synthetic split and one adapter. The benchmark accepts any finite numeric argument, so a distinguishing test can exercise a value outside a realistic application range. The archive records the inputs and observed revision returns for inspection. [Saved comparison](../results/factorial/comparison.json)

## Reproduce

Replay the saved outputs with Node 22 and Python 3.11 or 3.12.

```sh
npm ci
python3 factorial_experiment.py replay
```

On Apple Silicon with the locked Python environment and pinned model already cached, use the existing selected adapter to rerun in a new directory.

```sh
HF_HUB_OFFLINE=1 uv run --locked python factorial_experiment.py run runs/factorial-reproduction
python3 factorial_experiment.py replay runs/factorial-reproduction/comparison.json
```

If the original adapter is unavailable, recreate the same training recipe. The run records the new adapter checksum so it can be compared with the original archive.

```sh
HF_HUB_OFFLINE=1 uv run --locked python experiment.py train runs/factorial-adapter --prompt original --steps 48
HF_HUB_OFFLINE=1 uv run --locked python factorial_experiment.py run runs/factorial-from-retrained --adapter runs/factorial-adapter/step-48
```

The archive contains all 96 raw outputs, the observed return values for accepted assertions, the comparison settings, and per-generation timings. [Saved comparison](../results/factorial/comparison.json)
