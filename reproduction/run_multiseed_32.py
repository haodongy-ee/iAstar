"""Train/evaluate several iA* seeds and produce one statistical summary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2026, 3407])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--neural-astar", type=Path, default=Path("model/nastar/neural_astar_cnn_32.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction/results/week2"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for seed in args.seeds:
        checkpoint = Path(f"model/iastar/week2/iastar_seed_{seed}.pkl")
        output = args.output_dir / f"metrics_seed_{seed}.csv"
        if not (args.skip_existing and checkpoint.exists()):
            run(
                [
                    sys.executable,
                    "reproduction/train_iastar_quick.py",
                    "--epochs", str(args.epochs),
                    "--samples", str(args.train_samples),
                    "--batch-size", str(args.batch_size),
                    "--seed", str(seed),
                    "--output", str(checkpoint),
                ]
            )
        if not (args.skip_existing and output.exists()):
            run(
                [
                    sys.executable,
                    "reproduction/evaluate_32.py",
                    "--iastar", str(checkpoint),
                    "--neural-astar", str(args.neural_astar),
                    "--samples", str(args.eval_samples),
                    "--seed", str(args.eval_seed),
                    "--training-seed", str(seed),
                    "--warmup", str(args.warmup),
                    "--timing-repeats", str(args.timing_repeats),
                    "--output", str(output),
                ]
            )
        metrics.append(output)

    run(
        [
            sys.executable,
            "reproduction/summarize_32.py",
            *map(str, metrics),
            "--output-dir", str(args.output_dir),
        ]
    )


if __name__ == "__main__":
    main()
