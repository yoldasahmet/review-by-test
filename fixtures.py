"""Original one-function bug fixes. Fields are domain, name, old/new expression, witness."""
import difflib
import json
from pathlib import Path

ROWS = {
    "bookings": [
        ("canReserve", "n < 9", "n <= 9", 9, True),
        ("availableSeats", "10 - n - 1", "10 - n", 3, 7),
        ("waitlistSize", "n - 8", "Math.max(0, n - 8)", 3, 0),
        ("courtCount", "Math.floor(n / 4)", "Math.ceil(n / 4)", 5, 2),
        ("canCancel", "n > 24", "n >= 24", 24, True),
        ("guestLimit", "Math.max(n, 4)", "Math.min(n, 4)", 6, 4),
        ("hasQuorum", "n === 6", "n >= 6", 7, True),
        ("sessionMinutes", "n * 30", "n * 45", 2, 90),
        ("isFull", "n > 12", "n >= 12", 12, True),
        ("spareRackets", "Math.abs(8 - n)", "Math.max(0, 8 - n)", 10, 0),
    ],
    "pagination": [
        ("pageOffset", "n * 20", "(n - 1) * 20", 1, 0),
        ("pageCount", "Math.floor(n / 10)", "Math.ceil(n / 10)", 11, 2),
        ("hasNext", "n >= 20", "n > 20", 20, False),
        ("previousPage", "n - 1", "Math.max(1, n - 1)", 1, 1),
        ("pageSize", "Math.max(100, n)", "Math.min(100, n)", 101, 100),
        ("firstIndex", "n * 5 + 1", "n * 5", 0, 0),
        ("isFirstPage", "n === 0", "n === 1", 1, True),
        ("lastIndex", "n * 10", "n * 10 - 1", 2, 19),
        ("isValidPage", "n >= 0", "n >= 1", 0, False),
        ("nextCursor", "n + 1", "n + 25", 0, 25),
    ],
    "timers": [
        ("wholeMinutes", "Math.ceil(n / 60)", "Math.floor(n / 60)", 61, 1),
        ("expired", "n < 0", "n <= 0", 0, True),
        ("remaining", "30 - n", "Math.max(0, 30 - n)", 31, 0),
        ("milliseconds", "n * 100", "n * 1000", 1, 1000),
    ],
    "pricing": [
        ("freeShipping", "n > 50", "n >= 50", 50, True),
        ("discountCents", "Math.ceil(n / 10)", "Math.floor(n / 10)", 19, 1),
        ("amountDue", "n - 15", "Math.max(0, n - 15)", 10, 0),
        ("feeCents", "Math.max(n, 99)", "Math.min(n, 99)", 100, 99),
        ("priceWithTax", "n * 1.2", "Math.round(n * 1.2)", 3, 4),
        ("bundlePrice", "n * 5", "Math.floor(n / 3) * 12 + (n % 3) * 5", 3, 12),
    ],
}

def source(name, expression):
    return f"export function {name}(n: number) {{\n  return {expression};\n}}\n"

def diff(old, new):
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        "a/src/subject.ts", "b/src/subject.ts"))

def messages(patch, module="./subject"):
    return [{"role": "system", "content": (
        "Write one TypeScript Vitest regression test for the fix. Output only code. "
        f"Import {{ expect, test }} from 'vitest' and the changed function from '{module}'. "
        "Use test('description', () => { expect(fn(number)).toBe(literal); }); "
        "with a numeric or boolean expected value. Choose an input that fails before the fix."
    )}, {"role": "user", "content": patch}]

def reference(name, arg, value):
    return (f"import {{ expect, test }} from 'vitest';\nimport {{ {name} }} from './subject';\n\n"
            f"test('{name} regression', () => {{\n  expect({name}({arg})).toBe({json.dumps(value)});\n}});")

def cases(split):
    domains = {"train": ["bookings", "pagination"], "valid": ["timers"], "test": ["pricing"]}
    for domain in domains[split]:
        for name, before, after, arg, value in ROWS[domain]:
            yield name, source(name, before), source(name, after), reference(name, arg, value)

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    for split in ("train", "valid", "test"):
        with Path(f"data/{split}.jsonl").open("w") as out:
            for name, old, new, answer in cases(split):
                chat = messages(diff(old, new)) + [{"role": "assistant", "content": answer}]
                print(json.dumps({"messages": chat}), file=out)
