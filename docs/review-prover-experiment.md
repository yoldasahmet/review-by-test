# Third review experiment: training the prover on its own answers

The [second round's](review-tuning-experiment.md) prover adapter proved fewer bugs than the untuned model. It learned to pick boundaries but fell back to 0 elsewhere, because a rule had chosen its training witnesses: integers nearest a literal in the code. This round changes only how the training answers are chosen. The prover's task, prompt, grammar and scoring stay the same, and the reviewer is not part of this round. [Fixtures](../review_prover_fixtures.py) · [Runner](../review_prover_experiment.py) · [Frozen cases](../data/review-prover-cases.json) · [Frozen checksum](../data/review-prover.sha256)

## Training answers

Each training and validation bug PR carries up to 20 candidate inputs: round numbers from −1,000 to 1,000,000, every literal in either revision with its neighbours, and the fixture's own witness. Vitest keeps the candidates where main and the pull request return different values and records main's return at each. The untuned model then scores the full canonical test at every remaining candidate, and its highest-scoring test becomes the training answer. The adapter is trained toward a correct test the model already finds plausible, not toward one a rule picked.

Of the 416 training and validation bug PRs, 95 had a single distinguishing candidate, which fixes the answer, usually a boundary. In 214 the chosen answer differs from the fixture's witness.

## Data

| Split | Bug PRs | Refactor PRs | Use |
|---|---:|---:|---|
| Train | 318 | 0 | Training answers: 94 reverted fixes and 224 slipped rewrites |
| Validation | 98 | 0 | Selection: 24 reverted fixes and 74 slipped rewrites |
| Development | 24 | 24 | Selection: the second round's seen and fresh splits |
| Test | 24 | 12 | Fresh, frozen before inference |

The slipped rewrites come from the second round's generator, now keeping every slip that changes a return instead of one per function. The fresh test split is synthetic and assistant-authored, with six bug PRs and three exact refactors each from ride fares, parking fees, event ticketing and gym memberships. Five of its bugs are rewrites with a slip. By the share of sampled inputs that change, 8 bugs are narrow, 9 medium and 7 wide. Before inference, Vitest checked every witness on both revisions and every refactor on 38,023 sampled inputs.

## Method

| Variant | Trainable layers | LoRA rank | Learning rate |
|---|---:|---:|---:|
| Gentle | 8 | 8 | 0.00005 |
| Wide | 16 | 16 | 0.0001 |

Both variants use scale 16, dropout 0, Adam, answer-only loss, batch size 1, seed 42 and a 512-token limit, and save checkpoints at 100, 200 and 300 steps. The untuned model and the six checkpoints all wrote proofs for the validation and development pull requests. The candidate with the most proofs across both was chosen, ties keeping the untuned model and then the earlier checkpoint. Only the chosen prover and the untuned model saw the fresh test split.

The acceptance checks were written into the run's design before it started: the chosen prover must prove more bug PRs than the untuned model on validation, on development and on the fresh test split, and must prove no refactor.

## Selection

| Prover | Validation | Development | Total |
|---|---:|---:|---:|
| Untuned | 39/98 | 13/24 | 52 |
| Gentle, step 100 | 50/98 | 13/24 | 63 |
| Gentle, step 200 (selected) | 50/98 | 16/24 | 66 |
| Gentle, step 300 | 50/98 | 16/24 | 66 |
| Wide, steps 100, 200 and 300 | 32/98 | 4/24 | 36 |

The wide variant collapsed. At every checkpoint it wrote `-0` as the input and the expected value for all 98 validation pull requests. Gentle steps 200 and 300 tied, and the rule kept step 200.

## Results

| Prover | Fresh proofs | Narrow | Medium | Wide | Own expected value red/green | Proofs on refactors |
|---|---:|---:|---:|---:|---:|---:|
| Untuned | 11/24 | 1/8 | 5/9 | 5/7 | 8/24 | 0/12 |
| Gentle, step 200 | 17/24 | 5/8 | 6/9 | 6/7 | 11/24 | 0/12 |

All four acceptance checks passed. Because a comment needs a proof, a prover-only pipeline would post 17 proven comments on the 24 bug PRs and none on the 12 refactors.

The adapter gained eight proofs and lost two. Five gains were boundaries the untuned model missed: it chose 24, 14, 501 and 23 where the limits were 25, 15, 500 and 24, and 2501 where only 2400 to 2499 separate the revisions. The adapter chose the exact boundary each time. The other three gains moved off an input where both revisions agree: 100 instead of 120 below an overstay guard, 10 instead of 0 for a parking fee, and 1 instead of 0 for a training fee. It lost `roundedFareCents`, where it chose 10 and both revisions round to 10, and `isChildTicket`, where the untuned model had chosen the boundary 12 and the adapter chose 10.

Its remaining misses show what it has not learned. On both De Morgan slips it chose an input at or just below the range's lower bound, while the slip moved the upper bound. On two rounding-mode changes it chose 0, where both rounding modes agree. On validation it chose 0 for 62 of 98 bug PRs, against 42 for the untuned model, so the pull toward 0 seen in the second round remains on that split. On the fresh split it chose 0 for 4 of 24 bug PRs, against 3.

## Interpretation

Training on the model's own most likely correct test fixed the second round's failure. The gain is concentrated where the untuned model already came close, on boundaries it missed by one, while medium and wide bugs moved by one proof each. Update size mattered: the 16-layer variant at the higher learning rate collapsed to a single answer by its first checkpoint, and the 8-layer variant at half that rate improved. These are descriptive results from one seed per variant and 24 fresh bug PRs, so the six-proof difference is clear on this split but not a precise estimate.

## Local measurements

| Step | Measurement |
|---|---|
| Candidate scoring, 416 bug PRs | 880 s with the untuned model |
| Gentle training, 300 steps | 240.6 s, 1.25 updates/s, 1.78 GB MLX peak |
| Wide training, 300 steps | 282.2 s, 1.06 updates/s, 2.21 GB MLX peak |
| Proof generation, fresh split | 0.66 s median untuned, 0.80 s with the adapter |

Measured on an Apple M1 Pro with macOS 26.5.1, Python 3.12.10, Node 22.22.2, MLX 0.32.4, MLX-LM 0.32.2 and Vitest 3.2.7.

## Reproduce

Replay the archive with Node 22 and Python 3.11 or 3.12, without model weights.

```sh
npm ci
python3 review_prover_experiment.py replay
```

On Apple Silicon with the locked Python environment and pinned model already cached, run the experiment in a new directory. Each stage is saved as it finishes, so `--resume` continues an interrupted run. Regenerating the frozen cases with `python3 review_prover_fixtures.py build` is only needed after changing the generator.

```sh
HF_HUB_OFFLINE=1 uv run --locked python review_prover_experiment.py run runs/review-prover-reproduction
python3 review_prover_experiment.py replay runs/review-prover-reproduction/comparison.json
```

The archive contains every candidate score, raw prover output, scored assertion, proof and fixture check. [Saved comparison](../results/review-prover/comparison.json)
