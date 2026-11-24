"""Regression checks for the diff and generated-output grammars."""
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import Trial, literal_assertion, score_trials
from fixtures import reference
from generate import diff_module


class AssertionGrammarTest(unittest.TestCase):
    def test_preserves_numeric_source_literals(self):
        code = reference("changed", 1, 0).replace("changed(1)).toBe(0)", "changed(-0)).toBe(-0.0)")
        self.assertEqual(literal_assertion(code, "changed"), ("-0", "-0.0"))

    def test_rejects_extra_code_before_running_vitest(self):
        code = reference("changed", 1, True)
        outputs = [
            code + "\nconsole.log('extra');",
            code.replace("changed(1)", "changed(1 + 0)"),
            code.replace("changed(1)", "changed(true)"),
            code.replace("changed(1)", "changed(1٠)"),
            code.replace("from './subject'", "from '../subject'"),
            code.replace("changed regression", "changed\rregression"),
            code.replace("expect(changed(1))", "expect(changed(1).toBe(true))"),
        ]
        with patch("evaluate.subprocess.run") as run:
            scores = score_trials([Trial("changed", "", "", output) for output in outputs])
        run.assert_not_called()
        self.assertEqual(scores, [{"status": "outside_format"}] * len(outputs))


class DiffGrammarTest(unittest.TestCase):
    def test_uses_changed_basename(self):
        example = Path(__file__).with_name("examples") / "shipping.diff"
        self.assertEqual(diff_module(example.read_text()), "./shipping")

    def test_requires_an_existing_file_with_a_hunk(self):
        with self.assertRaises(ValueError):
            diff_module("--- /dev/null\n+++ b/src/shipping.ts\n@@ -0,0 +1 @@\n+return true;\n")


if __name__ == "__main__":
    unittest.main()
