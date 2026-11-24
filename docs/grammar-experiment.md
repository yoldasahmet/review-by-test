# Grammar-constrained decoding experiment

This paired comparison uses the same pinned 4-bit Qwen2.5-Coder-1.5B-Instruct model for both arms. The test-only split contains 24 original synthetic TypeScript fixes across rail transit, greenhouse controls, cartography, and materials testing. Each domain contributes six cases. Vitest verified all 24 authored references before the split was frozen, and the run verified them again before generating any outputs. [Fixture source](../grammar_fixtures.py) · [Frozen checksum](../data/grammar-test.sha256)

Both arms use the original prompt, greedy decoding, a 192-token cap, and one attempt per case on the same loaded base model. The run alternates which arm goes first. The constrained sampler filters the Qwen vocabulary to tokens that extend a valid prefix of one canonical Vitest test and chooses the highest-scoring token among those choices. It allows EOS after the assertion is complete. The template supplies the imports, a fixed test title, the export name visible in the diff, and the assertion syntax. The model picks the numeric input and the numeric or boolean expected value. The grammar permits signed decimals with up to 15 integer digits and 12 fractional digits.

Both outputs go through the same evaluator. It records accepted literals and the actual values returned by each revision, so a test can conform to the grammar while choosing the wrong input or asserting the wrong value. A red/green result requires the assertion to fail on the old fixture and pass on the fixed fixture.

## Results

| Outcome | Unconstrained | Constrained |
|---|---:|---:|
| Red/green | 5/24 (20.8%) | 7/24 (29.2%) |
| Format compliant | 17/24 (70.8%) | 24/24 (100%) |
| Passes both revisions | 8 | 7 |
| Fails on the fixed revision | 4 | 10 |
| Unsupported output | 7 | 0 |
| Median generation time | 0.644 s | 0.584 s |

The constraint added two net red/green cases, an increase of 8.3 percentage points. It brought seven additional outputs into the accepted format. Both arms caught four cases, only the constrained arm caught three, only the unconstrained arm caught one, and both missed 16.

| Changed catch | Unconstrained | Constrained |
|---|---|---|
| `tileSpan` | Started with a correct assertion but emitted a long series of assertions and hit the token cap | One assertion, `tileSpan(0) === 1`, which distinguishes old 2 from fixed 1 |
| `longitude360` | Chose 0, where both return 0 | Chose −180 and correctly expected 180 |
| `terrainColorBand` | Chose 499, where both return 0 | Chose −1000 and correctly expected 0 |
| `compressionDeltaMm` | Chose 0 and correctly expected 25 | Chose 25 and expected −25, while both return 0 |

The three wins include one format recovery and two changes to input selection. Only one of the seven previously unsupported cases became red/green. Among compliant outputs, the success rates were 5/17 (29.4%) and 7/24 (29.2%). Those denominators contain different sets of cases. On the 17 paired cases where the control was already compliant, red/green went from five to six. The strongest observed effect was reliable formatting, with a smaller end-to-end gain on this split. The forced template also changes the generated prefix the model sees before choosing each literal, so this experiment measures the effect of this specific grammar.

Run locally on an Apple M1 Pro with Python 3.12.10, MLX 0.32.4, MLX-LM 0.32.2, and Vitest 3.2.7. The archive contains the full versions and all paired outputs. [Saved comparison](../results/grammar/comparison.json)

## Reproduce

Replay the archive with Node 22, a regular Python 3.11 or 3.12 installation, and no model weights.

```sh
npm ci
python3 grammar_experiment.py replay
```

On Apple Silicon with the locked Python environment and pinned model already cached, run another comparison in a new directory.

```sh
HF_HUB_OFFLINE=1 uv run --locked python grammar_experiment.py run runs/grammar-reproduction
python3 grammar_experiment.py replay runs/grammar-reproduction/comparison.json
```

The result file includes raw generations, actual returns, scoring decisions, settings, and per-case generation times. Timing covers prompt formatting and decoding with the model loaded, including token filtering during generation. Token-vocabulary preparation is outside the per-case timer. [Saved comparison](../results/grammar/comparison.json)
