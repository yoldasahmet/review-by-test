# Second review experiment: changing the training

In the [first review experiment](review-experiment.md), the adapter flagged every test pull request. Its validation results showed why. It cleared refactors that only renamed or extracted code and flagged every refactor that touched an operator. This round keeps the task, prompt, model and comment gate, and changes what the adapters train on. [Fixtures](../review_tuning_fixtures.py) · [Runner](../review_tuning_experiment.py) · [Frozen cases](../data/review-tuning-cases.json) · [Frozen checksum](../data/review-tuning.sha256)

## What changed

| | First round | Second round |
|---|---|---|
| Training refactors | 94, from four simple edits | 188, two per function, preferring structural rewrites |
| Training bug PRs | 94 reverted fixes | 94 reverted fixes plus 83 rewrites with one slip |
| Reviewer adapter | 4 layers, rank 8, 240 steps | 16 layers, rank 16, 480 steps |
| Reviewer threshold | 0 | Chosen with the checkpoint on validation |
| Prover | Base model | Adapter trained on 177 witness tests, selected against the base model on validation |
| Test split | Money handling | Fresh: payroll, insurance premiums, loyalty points, savings accounts |

The structural rewrites are negated comparisons, De Morgan's law, ternary to if and back, commuted operands, and named or inlined constants, alongside the first round's flipped comparisons, swapped arguments and extracted results. A slip applies one change after a rewrite: a boundary, an off-by-one limit, `Math.min` for `Math.max`, a rounding mode, a sign or a decimal scale. For example, `!(weightKg < 70)` is an exact rewrite of `weightKg >= 70`, and `!(weightKg <= 70)` is a slip that changes only 70.

Vitest ran 769 candidates on 38,023 sampled inputs each. All 464 candidate rewrites returned the same values as main. Of 305 candidate slips, 298 changed at least one return and 7 changed nothing, so they were dropped. Each function kept its two rarest exact rewrites and its first working slip, rotating through the slip types across functions. A slip's witness is a sampled input where the revisions differ, preferring integers, then inputs nearest a literal in the code, then small inputs. Before inference, the run checked every witness on both revisions and every refactor on the sampled inputs again.

| Split | Bug PRs | Refactor PRs | Use |
|---|---:|---:|---|
| Train | 177 | 188 | Both adapters |
| Validation | 46 | 48 | Checkpoints, reviewer threshold, prover choice |
| Seen | 12 | 12 | The first round's money split, whose results were known when this round was designed |
| Test | 12 | 12 | Fresh, frozen before inference |

The fresh test split is synthetic and assistant-authored, and it was written after the first round's results. Four of its bugs are rewrites with a slip. Eight of its refactors use a family that appears in this round's training. The other four use new ones: reordered guards, a ternary rewritten with `Math.max`, `Math.min` rewritten as a ternary, and a ternary chain rewritten as guards. By the share of sampled inputs that change, four bugs are narrow, five are medium and three are wide.

The rest of the recipe matches the first round: scale 16, dropout 0, Adam at 0.0001, answer-only loss, batch size 1, seed 42 and a 512-token limit. The reviewer saves checkpoints at 120, 240, 360 and 480 steps, and the prover at 80, 160 and 240. Validation balanced accuracy picks the reviewer checkpoint together with its threshold. Validation proofs pick the prover checkpoint, with the base model as step 0. The run writes `selection.json` before evaluating the seen or test split.

## Validation

| Reviewer | Ranking | Best balanced accuracy with a threshold |
|---|---:|---:|
| Base model | 69.3% | 66.1% |
| Adapter, step 120 | 54.1% | 56.8% |
| Adapter, step 240 | 53.1% | 57.2% |
| Adapter, step 360 | 55.6% | 59.3% |
| Adapter, step 480 (selected) | 59.5% | 60.5% |

Ranking is the share of bug and refactor pairs where the bug gets the higher margin, and 50% is chance. At threshold 0, the adapter checkpoints answered `ok` on almost every pull request. With its threshold, step 480 flagged 25 of 46 bugs and 16 of 48 refactors. The selection rule compared checkpoints only, and the base model with its own threshold scored higher on validation than any checkpoint.

The prover produced 18 proofs on the 46 validation bug PRs without an adapter, and 24, 24 and 20 at steps 80, 160 and 240. Validation selected step 80.

## Results

| Reviewer | Fresh: bugs flagged | Fresh: refactors flagged | Fresh: balanced accuracy | Seen: bugs flagged | Seen: refactors flagged | Seen: balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Base model | 12/12 | 12/12 | 50.0% | 12/12 | 12/12 | 50.0% |
| Base model, threshold set on validation | 7/12 | 8/12 | 45.8% | 3/12 | 6/12 | 37.5% |
| Adapter, threshold set on validation | 11/12 | 8/12 | 62.5% | 11/12 | 9/12 | 58.3% |

On the fresh split, the adapter's ranking was 72.9% against the base model's 42.4%. On the seen split, it was 45.5% against 21.5%. The adapter cleared 4 fresh refactors, all from families in its training: two named constants, De Morgan's law, and a ternary rewritten as an if. It cleared none of the four with new shapes. It missed one fresh bug, `annualToMonthlyRateBps`, which introduces a named constant and also switches division to multiplication. On the seen split, it cleared 3 refactors and missed `lateFeeCents`. Validation did not show an advantage over the calibrated base model, so this gain rests on 24 cases.

| Prover | Fresh proofs | Narrow | Medium | Wide | Seen proofs |
|---|---:|---:|---:|---:|---:|
| Base model | 8/12 | 2/4 | 3/5 | 3/3 | 5/12 |
| Adapter, step 80 | 4/12 | 2/4 | 0/5 | 2/3 | 3/12 |

The prover adapter learned to pick boundaries. It chose 40, 50 and 1000 for three single-value bugs. On medium and wide bugs it usually chose 0 or another small input where both revisions agree. It chose 0 for 9 of the 24 fresh pull requests and 14 of the 24 seen ones. The training witnesses explain the pull toward 0: the sampler preferred integers nearest a literal, which for rates such as 0.37 means 0 or 1. Its validation gain did not carry over to either split.

| Reviewer and prover | Fresh: proven comments | Fresh: comments on refactors | Fresh: advisory notes on refactors | Seen: proven comments | Seen: comments on refactors |
|---|---:|---:|---:|---:|---:|
| Base and base | 8/12 | 0/12 | 12 | 5/12 | 0/12 |
| Adapter and base | 7/12 | 0/12 | 8 | 5/12 | 0/12 |
| Base and adapter | 4/12 | 0/12 | 12 | 3/12 | 0/12 |
| Adapter and adapter | 3/12 | 0/12 | 8 | 3/12 | 0/12 |
| Prover adapter on every pull request | 4/12 | 0/12 | n/a | 3/12 | 0/12 |

No refactor received a comment in any of the ten combinations. With the base prover, the reviewer adapter cut advisory notes on fresh refactors from 12 to 8 and cost one proven comment.

## Interpretation

Structural refactors and slipped rewrites improved the reviewer's ranking on both evaluated splits. On the fresh split, it cleared a third of the refactors at the cost of one missed bug. The gain covered only refactor families in its training, and validation did not confirm it. The prover adapter learned boundary inputs and lost more proofs than it gained. The proof requirement still decided which comments were correct.

For a third round, choose prover witnesses that represent the whole input range instead of the nearest literal, add more refactor families, let reviewer selection keep the base model, and freeze another test split, since this one has now been seen. The [third round](review-prover-experiment.md) tried the prover change. These are descriptive results from one seed per adapter and 24 cases per split.

## Local measurements

| Step | Measurement |
|---|---|
| Reviewer training, 480 steps | 326.9 s, 1.47 updates/s, 2.07 GB MLX peak |
| Prover training, 240 steps | 181.1 s, 1.33 updates/s, 2.21 GB MLX peak |
| Review, fresh split | 0.45 s median for both the base model and the adapter |
| Proof generation, fresh split | 0.76 s median for the base model, 1.07 s for the adapter |

Measured on an Apple M1 Pro with macOS 26.5.1, Python 3.12.10, Node 22.22.2, MLX 0.32.4, MLX-LM 0.32.2 and Vitest 3.2.7.

## Reproduce

Replay the archive with Node 22 and Python 3.11 or 3.12, without model weights.

```sh
npm ci
python3 review_tuning_experiment.py replay
```

On Apple Silicon with the locked Python environment and pinned model already cached, run the experiment in a new directory. Each stage is saved as it finishes, so `--resume` continues an interrupted run. Regenerating the frozen cases with `python3 review_tuning_fixtures.py build` is only needed after changing the generator.

```sh
HF_HUB_OFFLINE=1 uv run --locked python review_tuning_experiment.py run runs/review-tuning-reproduction
python3 review_tuning_experiment.py replay runs/review-tuning-reproduction/comparison.json
```

The archive contains every reviewer margin, raw prover output, scored assertion, proof and fixture check. [Saved comparison](../results/review-tuning/comparison.json)
