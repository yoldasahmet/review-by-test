# Review by Test

Can a 1.5B coding model learn to pick the exact input that exposes a TypeScript regression? This local experiment gives Qwen2.5-Coder a one-function PR diff and asks for one runnable Vitest test. A test earns credit when its assertion fails on the original function and passes on the fix.

## One change, two generated tests

The shipping threshold changed from `n > 50` to `n >= 50`. Both models generated valid tests. Only one exercised the changed behavior.

| Model | Generated assertion | Before fix | After fix |
|---|---|---|---|
| Original | `expect(freeShipping(50)).toBe(true)` | Fail | Pass |
| Fine-tuned | `expect(freeShipping(51)).toBe(true)` | Pass | Pass |

## Results on this Mac

| Model | Red then green | Passes both | Fails on fix | Outside requested format |
|---|---:|---:|---:|---:|
| Original 4-bit Qwen2.5-Coder-1.5B-Instruct | 3/6 | 1 | 0 | 2 |
| Same model with trained adapter | 1/6 | 2 | 2 | 1 |
| Handwritten reference tests | 6/6 | 0 | 0 | 0 |

Fine-tuning improved adherence to the requested format on one additional example. It also lost two regression catches. On this small benchmark, the original model is the better default. The six held-out cases support case-by-case inspection. A broader claim would need more functions, realistic diffs and multiple runs. [Inspect the original outputs](results/baseline.json) · [Inspect the tuned outputs](results/tuned.json)

## Replay the results

From the repository root with Node 22 and Python 3.11 or 3.12, run the saved generations through Vitest. This path also runs in GitHub Actions and needs no model download.

```sh
npm ci
python3 replay.py
```

## Reproduce the experiment on Apple Silicon

The recorded run used an Apple M1 Pro with 16 GB unified memory, macOS 26.5.1, Python 3.12.10, MLX-LM 0.32.2 and Vitest 3.2.7. Install uv and Node 22, then run from the repository root.

```sh
npm ci
uv sync --locked --python 3.12
uv run python fixtures.py
uv run python evaluate.py baseline
uv run python train.py
uv run python evaluate.py tuned
uv run python replay.py
```

Training updates 1.319 million adapter parameters across four layers of a frozen 4-bit model. The configuration uses rank 8, batch size 1, 60 steps, a 512-token sequence limit and loss on the answer tokens. Both evaluation runs use the same prompt, greedy decoding, a 192-token output limit and one attempt per diff. The evaluation commands replace the corresponding JSON results. [Training configuration](train.yaml) · [Model weights](https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit)

## Generate a test for a diff

```sh
uv run python generate.py < examples/free-shipping.diff
uv run python generate.py --adapter adapters < examples/free-shipping.diff
```

For your own PR, save a one-file diff with `git diff main...HEAD -- src/shipping.ts` and pipe it into the same command. The CLI prints TypeScript and requests a relative import for a test placed beside the changed module. Version one targets exported functions with one numeric argument and a numeric or boolean result. Review the generated assertion before adding it to your project.

## Dataset and scoring

The 30 original synthetic pairs are grouped by domain. Training uses 10 booking and 10 pagination fixes. Validation uses four timer fixes. The six pricing fixes are held out for the final comparison. Several domains intentionally share bug patterns such as rounding and inclusive boundaries. Only the diff and task instructions enter the model prompt. [Dataset source and reference assertions](fixtures.py)

The evaluator accepts one direct literal assertion in the requested Vitest template. It runs that test against the fixed function first, then checks for an assertion failure on the original if the fixed run passed. Output outside the supported grammar receives zero credit, including extra tests or commentary. Infrastructure errors stop the evaluation. This measures success under a narrow output contract and can reject otherwise useful TypeScript. [Scoring implementation](evaluate.py)
