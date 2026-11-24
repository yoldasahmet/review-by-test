# Locked benchmark

The second experiment uses 46 original, assistant-authored TypeScript fixes with handwritten witnesses. The fixtures use synthetic business rules. Each exports a pure function with one numeric argument and a numeric or boolean return. Empty inputs are represented by a zero count or zero cursor. Before collecting model outputs, we checked every reference with Vitest and recorded the checksum of the case data in [benchmark.sha256](../data/benchmark.sha256). The checksum covers split and domain labels, function names, both sources, and reference inputs and expected values.

| Split | Cases | Domains | Use |
|---|---:|---|---|
| Train | 28 | Fulfillment, calendar, telemetry, library | Adapter updates on verified answers |
| Validation | 8 | Subscriptions, analytics | Prompt choice, checkpoint choice and loss |
| Test | 10 | Cursor pagination, media processing | One generation per selected model and case |

Related functions stay within their domain split. Across domains, familiar bug patterns recur in different code, such as rounding, bounds, and changes to arithmetic. The fresh run uses the locked fixtures throughout and exports only its train and validation JSONL for MLX-LM. Gold witnesses for the test set are public for replay. The experiment reserves them from prompt and checkpoint selection by writing `selection.json` before final inference.

## Comparison rules

- Both untuned prompt candidates use the pinned 4-bit Qwen snapshot, greedy decoding, 192 new tokens, and one generation per diff. The original prompt asks for an input that fails before the fix. The alternative asks for different return values and the fixed result. Red/green validation count chooses the prompt. A tie keeps the original prompt.
- The smoke run uses eight updates. The full run starts from the base snapshot with the same seed, rank 8, four trainable layers, batch size 1, scale 16, dropout 0, Adam at 0.0001, answer-only loss, and a 512-token limit. It compares checkpoints at 24 and 48 updates. Red/green validation count ranks these checkpoints, then answer-token validation loss breaks a tie. The loss is measured on the saved checkpoint across all eight validation cases. The historical run's 60 updates and four validation batches belong to the original experiment.
- A generation is allowed one complete outer code fence. The evaluator accepts only two named imports and one direct literal `expect(fn(number)).toBe(literal)` assertion. It constructs its own Vitest test from parsed literals and runs the two authored revisions. Missing reports and unexpected runtime failures stop the experiment.
- Latency is wall-clock time around one generation with the model loaded, including chat formatting, decoding and detokenization. The training time includes local model load, validation within the training loop and adapter saves. MLX peak measures allocations tracked by `mlx.core.get_peak_memory()` and includes model load. It measures MLX-managed memory.

## Validation results

| Candidate | Red/green | Passes both | Fails on fix | Unsupported | Answer-token loss |
|---|---:|---:|---:|---:|---:|
| Original prompt, base model | 2/8 | 3 | 0 | 3 | |
| Distinguishing-input prompt, base model | 0/8 | 0 | 0 | 8 | |
| QLoRA at step 24, original prompt | 1/8 | 3 | 4 | 0 | 0.131983 |
| QLoRA at step 48, original prompt | 2/8 | 0 | 5 | 1 | 0.131524 |

The alternative prompt often elicited intermediate variables or multiple assertions. Its loss on this benchmark came through the format gate, which leaves the underlying input and expected-value quality unmeasured for that candidate. On supported outputs, step 48 chose distinguishing inputs on 4/7 cases, compared with 2/5 for the untuned model. It got the fixed value right on two of those four cases. Validation red/green selected the original prompt and step 48.

## Paired test results

| Case | Original | Tuned |
|---|---|---|
| `inclusiveRowsLeft` | Unsupported | Red/green |
| `cursorExpiresAt` | Unsupported | Unsupported |
| `shouldFetchAnotherBatch` | Unsupported | Red/green |
| `clampEmptyCursor` | Red/green | Fails on fix |
| `resumeIndex` | Red/green | Red/green |
| `padToEvenWidth` | Unsupported | Fails on fix |
| `hasRenderableFrames` | Unsupported | Red/green |
| `isSupportedAudioRate` | Unsupported | Fails on fix |
| `previewWidth` | Passes both | Fails on fix |
| `transparentPixelAlpha` | Red/green | Red/green |

The original produced four supported tests. Three chose distinguishing inputs and asserted the fixed value. The adapter produced nine supported tests. Six chose distinguishing inputs and five asserted the fixed value. The four supported misses included one distinguishing input with the old expected value, and three non-distinguishing inputs with wrong expected values. The main gain in this run was producing more scorable outputs. The held-out result covers ten authored cases and one adapter seed.

For a follow-up, freeze another test split and compare the same base model with and without grammar-constrained decoding before training again. That isolates how much of the gain comes from producing a scorable test, while the return-value diagnostics measure boundary selection and arithmetic separately.

## Local measurements

| Training run | Steps | Wall time | End-to-end updates/s | MLX peak |
|---|---:|---:|---:|---:|
| Smoke | 8 | 12.41 s | 0.64 | 1.53 GB |
| Full | 48 | 22.84 s | 2.10 | 1.54 GB |

| Test inference | Model load | Median generation | MLX peak |
|---|---:|---:|---:|
| Original | 0.62 s | 0.76 s | 1.27 GB |
| Step 48 | 0.56 s | 0.66 s | 1.29 GB |

These are measurements from one local run with cached weights. The smoke and full training jobs each include startup and validation overhead. The archive includes per-case latencies, both training configurations, and package versions. Local adapter weights and raw training inputs generated from the fixture source live in the individual run directories. [Experiment archive](../results/benchmark)
