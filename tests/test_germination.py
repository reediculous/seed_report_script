"""Tests for Всхожесть counting (alive vs dead seeds in raw files)."""

import os
import tempfile
import unittest

from data_parser import germination_stats, parse_file


class GerminationStatsTest(unittest.TestCase):
    def test_zero_equals_n_syntax(self):
        content = "\n".join(
            ["10 20 30", "15 25", "", "0=2", ""]
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            path = tmp.name
        try:
            fd = parse_file(path)
            g = germination_stats(fd)
            self.assertEqual(g.alive, 2)
            self.assertEqual(g.dead, 2)
            self.assertEqual(g.total, 4)
            self.assertAlmostEqual(g.rate, 0.5)
        finally:
            os.unlink(path)

    def test_trailing_zero_lines(self):
        lines = ["20 30 40"] * 5 + ["0"] * 3
        content = "\n".join(lines)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            path = tmp.name
        try:
            fd = parse_file(path)
            g = germination_stats(fd)
            self.assertEqual(g.alive, 5)
            self.assertEqual(g.dead, 3)
            self.assertEqual(g.total, 8)
            self.assertAlmostEqual(g.rate, 5 / 8)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
