"""Tests for Tukey HSD compact letter display (maximal-clique CLD)."""

import unittest

from stats_analysis import TukeyPair, _assign_cld_letters


def _pair(group1: str, group2: str, reject: bool) -> TukeyPair:
    return TukeyPair(
        group1=group1,
        group2=group2,
        mean_diff=0.0,
        p_value=0.5,
        ci_low=0.0,
        ci_high=0.0,
        reject=reject,
    )


class AssignCldLettersTest(unittest.TestCase):
    def test_star_hub_gets_concatenated_letters(self):
        """Hub non-significant vs both leaves that differ from each other -> ``ab``."""
        variants = ["A", "B", "C"]
        variant_means = {"A": 1.0, "B": 10.0, "C": 2.0}
        tukey_pairs = [
            _pair("A", "B", False),
            _pair("B", "C", False),
            _pair("A", "C", True),
        ]
        cld = _assign_cld_letters(variants, tukey_pairs, variant_means)
        self.assertEqual(cld["B"], "ab")
        self.assertEqual(cld["A"], "a")
        self.assertEqual(cld["C"], "b")

    def test_fully_non_significant_single_clique(self):
        variants = ["A", "B", "C"]
        variant_means = {"A": 1.0, "B": 2.0, "C": 3.0}
        tukey_pairs = [
            _pair("A", "B", False),
            _pair("B", "C", False),
            _pair("A", "C", False),
        ]
        cld = _assign_cld_letters(variants, tukey_pairs, variant_means)
        self.assertEqual(cld["A"], "a")
        self.assertEqual(cld["B"], "a")
        self.assertEqual(cld["C"], "a")


if __name__ == "__main__":
    unittest.main()
