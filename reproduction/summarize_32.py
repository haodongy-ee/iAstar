"""Aggregate multi-seed 32x32 results with run-level bootstrap intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_METRICS = (
    "success_rate",
    "path_length",
    "expanded_nodes",
    "runtime_median_ms",
    "runtime_p95_ms",
)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return float(low), float(high)


def aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "training_seed",
        "eval_seed",
        "case_id",
        "method",
        "success",
        "path_length",
        "expanded_nodes",
        "runtime_ms",
        "runtime_p95_ms",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    rows = []
    for keys, group in frame.groupby(["training_seed", "eval_seed", "method"], sort=False):
        training_seed, eval_seed, method = keys
        rows.append(
            {
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "method": method,
                "cases": len(group),
                "success_rate": float(group["success"].mean()),
                "path_length": float(group["path_length"].mean()),
                "expanded_nodes": float(group["expanded_nodes"].mean()),
                "runtime_median_ms": float(group["runtime_ms"].median()),
                "runtime_p95_ms": float(np.percentile(group["runtime_p95_ms"], 95)),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    frame: pd.DataFrame, draws: int = 10_000, seed: int = 2026
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    runs = aggregate_runs(frame)

    summary_rows = []
    for method, group in runs.groupby("method", sort=False):
        row: dict[str, float | int | str] = {"method": method, "runs": len(group)}
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy()
            low, high = bootstrap_ci(values, rng, draws)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)

    baseline = runs[runs["method"] == "A*"].set_index(["training_seed", "eval_seed"])
    comparison_rows = []
    for method in (m for m in runs["method"].unique() if m != "A*"):
        candidate = runs[runs["method"] == method].set_index(["training_seed", "eval_seed"])
        paired = candidate.join(baseline, lsuffix="_candidate", rsuffix="_baseline", how="inner")
        for metric in ("path_length", "expanded_nodes", "runtime_median_ms"):
            base = paired[f"{metric}_baseline"].to_numpy()
            delta = 100.0 * (paired[f"{metric}_candidate"].to_numpy() - base) / base
            low, high = bootstrap_ci(delta, rng, draws)
            comparison_rows.append(
                {
                    "method": method,
                    "baseline": "A*",
                    "metric": metric,
                    "paired_runs": len(delta),
                    "relative_change_percent": float(delta.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return runs, pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction/results/week2"))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    combined = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    runs, summary, comparisons = summarize(combined, args.bootstrap_draws, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "metrics_all.csv", index=False)
    runs.to_csv(args.output_dir / "runs_by_method.csv", index=False)
    summary.to_csv(args.output_dir / "summary_by_method.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_vs_astar.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved aggregate results to {args.output_dir}")


if __name__ == "__main__":
    main()
