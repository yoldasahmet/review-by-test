"""Checks for the review fixtures' exact refactors and the comment gate."""
import unittest

from review_experiment import calibrate, confusion, gate
from review_fixtures import ReviewCase, flip_comparison, source, swap_arguments, refactor


def fn(body, parameter="x"):
    return source("f", parameter, body)


class RefactorTest(unittest.TestCase):
    def test_flips_only_bare_comparisons(self):
        self.assertEqual(flip_comparison(fn("return x >= 0 && x <= 90;")), fn("return 0 <= x && 90 >= x;"))
        self.assertEqual(flip_comparison(fn("if (x === 0) return 0;\nreturn x;")), fn("if (0 === x) return 0;\nreturn x;"))
        # Higher-precedence operators on either side would change the parse, so they are left alone.
        for body in ("return x * 2 > 5;", "return x > 5 + 1;", "return x % 4 === 0;", "return -x > 5;"):
            self.assertIsNone(flip_comparison(fn(body)), body)

    def test_swaps_only_simple_arguments(self):
        self.assertEqual(swap_arguments(fn("return Math.max(0, x - 8);")), fn("return Math.max(x - 8, 0);"))
        self.assertIsNone(swap_arguments(fn("return Math.max(0, Math.round(x));")))

    def test_every_refactor_changes_the_text(self):
        for index, body in enumerate(("return x + 1;", "return x < 3;", "return Math.min(x, 4);")):
            label, head = refactor(fn(body), index)
            self.assertNotEqual(head, fn(body), label)


class GateTest(unittest.TestCase):
    def cases(self):
        return [ReviewCase("test", "d", name, "", "", verdict, "c") for name, verdict in
                (("a", "bug"), ("b", "bug"), ("c", "ok"), ("d", "ok"))]

    def test_comments_need_a_flag_and_a_proof(self):
        proofs = [{"proof": {"status": status}} for status in ("red_green", "passes_both", "passes_both", "passes_both")]
        counts = gate(self.cases(), [True, True, True, False], proofs)
        self.assertEqual((counts["comments_on_bug_prs"], counts["advisory_on_bug_prs"]), (1, 1))
        self.assertEqual((counts["flagged_refactor_prs"], counts["comments_on_refactor_prs"]), (1, 0))

    def test_threshold_is_chosen_on_balanced_accuracy(self):
        rows = [dict(gold=gold, margin=margin) for gold, margin in
                (("bug", 3.0), ("bug", 2.0), ("ok", 1.0), ("ok", 2.5))]
        threshold = calibrate(rows)
        self.assertEqual(threshold, 1.5)
        self.assertEqual(confusion(rows, threshold)["flagged_refactor_prs"], 1)


if __name__ == "__main__":
    unittest.main()
