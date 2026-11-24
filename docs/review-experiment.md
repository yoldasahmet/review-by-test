# Pull-request review experiment

This experiment asks whether the pinned 4-bit Qwen2.5-Coder-1.5B-Instruct model can review pull requests that their authors describe as refactors, and whether a LoRA adapter improves that review. Each pull request changes one exported function that takes one number. Its gold verdict is `bug` when some number input returns a different value and `ok` when every input returns the same value. A comment is posted only when the prover's Vitest test fails on the pull request and passes on main. [Fixtures](../review_fixtures.py) · [Runner](../review_experiment.py) · [Frozen checksum](../data/review.sha256)

## Data

| Split | Bug PRs | Refactor PRs | Source |
|---|---:|---:|---|
| Train | 94 | 94 | Reversed fixes from the benchmark, grammar and factorial fixtures |
| Validation | 24 | 24 | Reversed fixes from the ablation fixtures |
| Test | 12 | 12 | New money-handling pull requests: card fees, VAT invoicing, SME loan repayments, payouts and FX |

A training or validation bug PR reverts an earlier authored fix. Main holds the fixed function, the pull request restores the old one, and the earlier witness proves the regression. Each fixed function also gets one exact refactor: swapping the operands of a bare comparison, swapping `Math.min` or `Math.max` arguments, extracting the returned expression into `result`, or renaming the parameter. The historical pricing cases stay out of training because they overlap the money domain.

The 24 test pull requests are synthetic and assistant-authored, like the earlier fixtures. Three test refactors reuse a transformation family seen in training: a flipped comparison, an extracted variable and swapped `Math.max` arguments. The other nine use new ones, such as De Morgan's law, reordered guards and a folded constant. Before any model inference, Vitest ran every witness on both revisions and compared every refactor with main on 38,023 sampled inputs, including `-0`, quarters, cents and large values.

The test bugs vary in how easy they are to prove. Four are narrow, changing at most 4 of the 38,023 sampled inputs. Four are medium, changing 40 to 50 percent. Four are wide, changing at least 95 percent.

## Method

- **Reviewer.** The model scores two fixed answers, `{"verdict": "bug"}` and `{"verdict": "ok"}`, by total log-probability, using the exact tokens training uses. It flags a bug when the bug answer's log-probability exceeds the ok answer's by more than a threshold. The threshold is 0, except for one base-model arm whose threshold maximizes validation balanced accuracy.
- **Adapter.** Training reuses the earlier recipe: rank 8, four layers, scale 16, dropout 0, Adam at 0.0001, answer-only loss, batch size 1, seed 42, and a 512-token limit. It runs 240 steps on the 188 training pull requests and saves checkpoints at 60, 120, 180 and 240. Validation balanced accuracy at threshold 0 picks the checkpoint, and a tie keeps fewer steps. The run writes `selection.json` before any test inference.
- **Prover.** The base model writes one test with the canonical Vitest grammar from the [grammar experiment](grammar-experiment.md), the original prompt and greedy decoding. It sees the pull request reversed, as the revert a reviewer would suggest. The model chooses the input. Vitest measures main's return at that input, and the proof asserts it, as in the ablation's computed-value arm. In a real review, main's current behavior is observable. The model's own expected value is scored as well.
- **Gate.** A comment is posted when the reviewer flags a bug and the proof fails on the pull request and passes on main. A flagged pull request without a proof gets an advisory note.

## Validation

| Reviewer | Bug PRs flagged | Refactor PRs flagged | Balanced accuracy |
|---|---:|---:|---:|
| Base model | 23/24 | 24/24 | 47.9% |
| Base model, threshold set on validation | 23/24 | 17/24 | 62.5% |
| Adapter, step 60 | 0/24 | 0/24 | 50.0% |
| Adapter, step 120 | 0/24 | 0/24 | 50.0% |
| Adapter, step 180 | 4/24 | 5/24 | 47.9% |
| Adapter, step 240 (selected) | 24/24 | 12/24 | 75.0% |

Step 240 cleared all 12 validation refactors that only changed text, six extracted results and six renamed parameters. It flagged all 12 that touched an operator, eight flipped comparisons and four swapped arguments. The adapter had learned to tell renames and extractions from edited expressions, which is not the same as comparing behavior.

## Locked test results

| Reviewer | Bug PRs flagged | Proven comments | Refactor PRs flagged | Comments on refactors |
|---|---:|---:|---:|---:|
| Base model | 12/12 | 5/12 | 12/12 | 0/12 |
| Base model, threshold set on validation | 7/12 | 2/12 | 12/12 | 0/12 |
| Adapter, step 240 | 12/12 | 5/12 | 12/12 | 0/12 |
| Prover on every pull request, no reviewer | 12/12 | 5/12 | n/a | 0/12 |

The adapter did not transfer to the test split. Like the base model, it flagged all 24 pull requests, including the three refactors from families it saw in training. The margins tell the same story. Across the 144 bug and refactor pairs, the base model gave the bug the higher margin in 21.5% of pairs and the adapter in 48.6%, where 50% is chance. The calibrated threshold lost five bug flags and cleared no refactors, because every test refactor's margin was above it.

The proof requirement held in every arm. No refactor received a comment, and the same five bug PRs received proven comments wherever they were flagged.

| Difference density | Bug PRs | Proofs |
|---|---:|---:|
| Narrow, at most 4 sampled inputs | 4 | 1 |
| Medium, 40 to 50 percent | 4 | 1 |
| Wide, at least 95 percent | 4 | 3 |

The prover's own expected values were red/green on 3 of 12 bug PRs. Asserting main's measured value raised that to 5 of 12, every case where the model's input distinguished the revisions. On `authorizationFeeCents`, it chose 0, the only input that changes, and the proof failed on the pull request's 30. On `vatCents`, it chose 100, where both revisions return 21. On `meetsMinimumCharge`, it chose 49, one below the boundary, and on `canInstantPayout`, it chose the lower limit instead of the upper one. On `lateFeeCents`, it chose −1, below the guard both revisions share.

## Interpretation

The proof requirement is what kept comments trustworthy: zero false comments on 12 refactors, at a recall of 5 of 12. Neither reviewer helped choose which pull requests to prove, because both flagged everything on the test split. The adapter learned a surface rule that held on validation and failed on authored refactors with new shapes. For a follow-up, train on structural refactors, such as guard reordering and if/else rewrites, and freeze another test split before training. The [second round](review-tuning-experiment.md) tried this. These are descriptive results from one adapter seed and one 24-case split.

## Local measurements

| Step | Measurement |
|---|---|
| Training, 240 steps | 119.0 s, 2.02 updates/s, 1.54 GB MLX peak |
| Base reviewer, test | 0.72 s model load, 0.37 s median per pull request, 1.40 GB MLX peak with the prover |
| Adapter reviewer, test | 0.58 s model load, 0.42 s median per pull request, 1.44 GB MLX peak |
| Prover | 0.62 s median generation |

A review time covers scoring both answers. Measured on an Apple M1 Pro with macOS 26.5.1, Python 3.12.10, Node 22.22.2, MLX 0.32.4, MLX-LM 0.32.2 and Vitest 3.2.7.

An earlier attempt stopped during validation scoring when the computer restarted, before any test inference. One test-case label was then corrected, which changed the fixture checksum, so the archived run started from scratch.

## Reproduce

Replay the archive with Node 22 and Python 3.11 or 3.12, without model weights. `report` prints the tables above.

```sh
npm ci
python3 review_experiment.py replay
python3 review_experiment.py report
```

On Apple Silicon with the locked Python environment and pinned model already cached, run the experiment in a new directory. Each stage is saved as it finishes, so `--resume` continues an interrupted run.

```sh
HF_HUB_OFFLINE=1 uv run --locked python review_experiment.py run runs/review-reproduction
python3 review_experiment.py replay runs/review-reproduction/comparison.json
```

The archive contains every reviewer margin, raw prover output, scored assertion, proof and fixture check. [Saved comparison](../results/review/comparison.json)
