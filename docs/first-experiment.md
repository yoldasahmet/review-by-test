# Why the first fine-tune scored 1/6

The six original pricing cases are a historical benchmark. The saved outputs are in [baseline.json](../results/baseline.json) and [tuned.json](../results/tuned.json). Replay classifies complete outputs. The table also shows what happened when we exercised recoverable numeric assertions from unsupported outputs in the controlled fixtures.

| Function | Original input → expected | Old → fixed return | Original outcome | Tuned input → expected | Old → fixed return | Tuned outcome |
|---|---|---|---|---|---|---|
| `freeShipping` | `50 → true` | `false → true` | Red/green | `51 → true` | `true → true` | Missed boundary |
| `discountCents` | `15 → 1`, `true → 1` | `2 → 1` for 15 | Unsupported, two tests and a boolean argument | `11 → 2` | `2 → 1` | Expected the old value |
| `amountDue` | `-10 → 0` | `-25 → 0` | Red/green | `3 → 0` | `-12 → 0` | Red/green |
| `feeCents` | `100 → 99` | `100 → 99` | Red/green | `99 → 99` | `99 → 99` | Missed boundary |
| `priceWithTax` | `100 → 120` | `120 → 120` | Rounding made no difference | `1 → 1.2` | `1.2 → 1` | Unsupported, misplaced parenthesis. The intended assertion also used the old value. |
| `bundlePrice` | `15 → 60`, `true → 60` | `75 → 60` for 15 | Unsupported, two tests and a boolean argument | `5 → 20` | `25 → 22` | Expected neither value |

We evaluated numeric inputs only, per the exported functions' signatures. The tuned tax output was `expect(priceWithTax(1).toBe(1.2));`. We evaluated `priceWithTax(1)` separately to recover the two actual returns. The saved output still earns zero credit.

## Shipping threshold

The old predicate was `n > 50`. The fix changed it to `n >= 50`. Both predicates are true at 51 and false below 50. Only 50 distinguishes them. The original model chose 50 and asserted the fixed value, `true`. The tuned model chose 51 and asserted a value both revisions return.

The tuned output met the full grammar on 5/6 cases versus 4/6 for the original. It lost the original model's catches on shipping and fees without adding a new catch. Two of its supported outputs chose distinguishing inputs but asserted the wrong answer. This is evidence about the saved generations. A plausible hypothesis is that the adapter learned the canonical test shape more readily than it learned to trace the two revisions. Identifying which aspects of the data or training schedule caused that behavior would require a controlled comparison.
