"""Tests for Hodges–Lehmann pseudo-median confidence interval (Wilcoxon-based)."""

import unittest

import numpy as np

from stats_analysis import _hodges_lehmann_pseudo_median_ci, _wilcoxon_location_test


class HodgesLehmannCiTest(unittest.TestCase):
    def test_interval_brackets_hl_point_estimate(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.normal(12.0, 2.5, size=25)
            walsh = [(x[i] + x[j]) / 2.0 for i in range(len(x)) for j in range(i, len(x))]
            s = np.sort(walsh)
            est = float(np.median(s))
            lo, hi = _hodges_lehmann_pseudo_median_ci(s, len(x), alpha=0.05)
            self.assertLessEqual(lo, est, msg=(lo, est, hi))
            self.assertLessEqual(est, hi, msg=(lo, est, hi))

    def test_symmetric_about_zero_includes_zero(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5])
        walsh = [(x[i] + x[j]) / 2.0 for i in range(len(x)) for j in range(i, len(x))]
        s = np.sort(walsh)
        lo, hi = _hodges_lehmann_pseudo_median_ci(s, len(x), alpha=0.05)
        self.assertLessEqual(lo, 0.0)
        self.assertLessEqual(0.0, hi)

    def test_location_test_returns_rounded_hl_ci(self):
        x = np.linspace(10.0, 30.0, 15)
        out = _wilcoxon_location_test(x, location=float(np.mean(x)))
        self.assertIn("ci_low", out)
        self.assertIn("ci_high", out)
        self.assertLess(out["ci_low"], out["ci_high"])
        self.assertLessEqual(out["ci_low"], out["expected"])
        self.assertLessEqual(out["expected"], out["ci_high"])

    def test_differs_from_naive_percentile_ci(self):
        """Walsh 2.5/97.5 percentiles are not the same as the Wilcoxon-null interval."""
        x = np.arange(20, dtype=float) * 1.3 + 4.0
        n = len(x)
        walsh = [(x[i] + x[j]) / 2.0 for i in range(n) for j in range(i, n)]
        s = np.sort(walsh)
        proper_lo, proper_hi = _hodges_lehmann_pseudo_median_ci(s, n, alpha=0.05)
        naive_lo = float(np.percentile(s, 2.5))
        naive_hi = float(np.percentile(s, 97.5))
        self.assertNotAlmostEqual(proper_lo, naive_lo)
        self.assertNotAlmostEqual(proper_hi, naive_hi)


if __name__ == "__main__":
    unittest.main()
