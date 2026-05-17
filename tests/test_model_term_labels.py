"""Tests for ANOVA / OLS term display names (vart_/povt)."""

import unittest

from stats_analysis import display_anova_term, display_ols_term


class ModelTermLabelsTest(unittest.TestCase):
    def test_ols_terms(self):
        self.assertEqual(display_ols_term("Intercept"), "Intercept")
        self.assertEqual(display_ols_term("C(variant)[T.3V]"), "vart_3V")
        self.assertEqual(display_ols_term("C(replicate)[T.2]"), "povt2")
        self.assertEqual(
            display_ols_term("C(variant)[T.3V]:C(replicate)[T.2]"),
            "vart_3V:povt2",
        )
        self.assertEqual(
            display_ols_term("C(replicate)[T.1]:C(variant)[T.Control]"),
            "vart_Control:povt1",
        )

    def test_anova_terms(self):
        self.assertEqual(display_anova_term("C(variant)"), "vart")
        self.assertEqual(display_anova_term("C(replicate)"), "povt")
        self.assertEqual(display_anova_term("C(variant):C(replicate)"), "vart:povt")
        self.assertEqual(display_anova_term("Residual"), "Residual")


if __name__ == "__main__":
    unittest.main()
