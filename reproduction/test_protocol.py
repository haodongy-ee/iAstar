from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from summarize_32 import bootstrap_ci, summarize


class SummaryTests(unittest.TestCase):
    def test_single_value_ci_is_degenerate(self) -> None:
        self.assertEqual(bootstrap_ci(np.array([3.0]), np.random.default_rng(1), 100), (3.0, 3.0))

    def test_summary_and_paired_relative_change(self) -> None:
        rows = []
        for seed, astar_nodes, iastar_nodes in ((1, 100, 80), (2, 120, 90), (3, 110, 88)):
            for case in range(2):
                for method, nodes in (("A*", astar_nodes), ("iA*", iastar_nodes)):
                    rows.append(
                        {
                            "training_seed": seed,
                            "eval_seed": 7,
                            "case_id": case,
                            "method": method,
                            "success": 1,
                            "path_length": 10.0,
                            "expanded_nodes": nodes,
                            "runtime_ms": 2.0,
                        }
                    )
        summary, comparisons = summarize(pd.DataFrame(rows), draws=500, seed=9)
        self.assertEqual(set(summary["method"]), {"A*", "iA*"})
        expanded = comparisons[(comparisons["method"] == "iA*") & (comparisons["metric"] == "expanded_nodes")].iloc[0]
        self.assertLess(expanded["relative_change_percent"], 0)
        self.assertEqual(expanded["paired_runs"], 3)

    def test_missing_columns_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            summarize(pd.DataFrame({"method": ["A*"]}), draws=10)


if __name__ == "__main__":
    unittest.main()
