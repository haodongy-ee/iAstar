"""Evaluate Vanilla A*, Neural A*, and iA* under a controlled 32x32 protocol."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from neural_astar.planner import NeuralAstar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import MazeDataset
from dastar import dastar
from iastar import iastar


def geometric_length(paths: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[[[2**0.5, 1.0, 2**0.5], [1.0, 0.0, 1.0], [2**0.5, 1.0, 2**0.5]]]],
        device=paths.device,
    )
    padded = F.pad(paths.float(), (1, 1, 1, 1))
    return (F.conv2d(padded, kernel) * paths.float()).sum((1, 2, 3)) / 2.0


def valid_path(path: np.ndarray, free: np.ndarray, start: np.ndarray, goal: np.ndarray) -> bool:
    cells = path.astype(bool)
    if (
        np.any(cells & ~free.astype(bool))
        or not np.any(cells & start.astype(bool))
        or not np.any(cells & goal.astype(bool))
    ):
        return False
    source = tuple(np.argwhere(cells & start.astype(bool))[0])
    targets = {tuple(x) for x in np.argwhere(cells & goal.astype(bool))}
    queue = deque([source])
    seen = {source}
    while queue:
        r, c = queue.popleft()
        if (r, c) in targets:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (
                    (dr or dc)
                    and 0 <= nr < cells.shape[0]
                    and 0 <= nc < cells.shape[1]
                    and cells[nr, nc]
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(
    planner: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object],
    maps: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    device: torch.device,
    repeats: int,
) -> tuple[object, np.ndarray]:
    timings = []
    output = None
    for _ in range(repeats):
        synchronize(device)
        begin = time.perf_counter_ns()
        output = planner(maps, start, goal)
        synchronize(device)
        timings.append((time.perf_counter_ns() - begin) / 1e6)
    assert output is not None
    return output, np.asarray(timings, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("planning-datasets/mpd/instances/032/mazes_032_moore_c8.npz"))
    parser.add_argument("--iastar", type=Path, default=Path("model/iastar/iastar_quick_32.pkl"))
    parser.add_argument("--neural-astar", type=Path, default=Path("model/nastar/neural_astar_cnn_32.pth"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026, help="Seed used to sample fixed test starts.")
    parser.add_argument("--training-seed", type=int, default=-1, help="Training seed recorded in the output.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warm-up calls per planner.")
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reproduction/results/metrics_32.csv"))
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.timing_repeats < 1:
        parser.error("samples and timing-repeats must be positive; warmup must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MazeDataset samples a random start inside __getitem__. Materializing once
    # prevents warm-up/repeats from silently changing the evaluation task.
    dataset = MazeDataset(str(args.data), "test")
    cases = []
    for index in range(min(args.samples, len(dataset))):
        maps, start, goal, _ = dataset[index]
        cases.append(tuple(torch.from_numpy(x[None]).to(device) for x in (maps, start, goal)))

    vanilla = dastar(device=str(device), is_training=False, output_path_list=False, w=2.0).to(device).eval()
    neural = NeuralAstar(encoder_arch="CNN", encoder_depth=4, Tmax=1.0).to(device).eval()
    neural.load_state_dict(torch.load(args.neural_astar, map_location=device, weights_only=True), strict=True)
    imperative = iastar(
        encoder_input=3,
        encoder_arch="UNet",
        encoder_depth=4,
        device=str(device),
        is_training=False,
        output_path_list=False,
        w=2.0,
    ).to(device).eval()
    checkpoint = torch.load(args.iastar, map_location=device, weights_only=True)
    imperative.encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)

    planners: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]] = {
        "A*": lambda m, s, g: vanilla(m, s, g, m),
        "Neural A*": lambda m, s, g: neural(m, s, g),
        "iA*": lambda m, s, g: imperative(m, s, g),
    }

    with torch.inference_mode():
        for method_index, planner in enumerate(planners.values()):
            for warmup_index in range(args.warmup):
                planner(*cases[(method_index + warmup_index) % len(cases)])
        synchronize(device)

        rows: list[dict[str, float | int | str]] = []
        names = list(planners)
        for case_id, (maps, start, goal) in enumerate(cases):
            order = names[case_id % len(names) :] + names[: case_id % len(names)]
            for order_index, method in enumerate(order):
                output, timings = timed_call(
                    planners[method], maps, start, goal, device, args.timing_repeats
                )
                path = output.paths[0, 0].detach().cpu().numpy()
                rows.append(
                    {
                        "training_seed": args.training_seed,
                        "eval_seed": args.seed,
                        "case_id": case_id,
                        "order_index": order_index,
                        "method": method,
                        "success": int(
                            valid_path(
                                path,
                                maps[0, 0].cpu().numpy(),
                                start[0, 0].cpu().numpy(),
                                goal[0, 0].cpu().numpy(),
                            )
                        ),
                        "path_length": float(geometric_length(output.paths)[0].item()),
                        "expanded_nodes": int(output.histories[0].sum().item()),
                        "runtime_ms": float(np.median(timings)),
                        "runtime_p95_ms": float(np.percentile(timings, 95)),
                        "timing_repeats": args.timing_repeats,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"device={device}; eval_seed={args.seed}; training_seed={args.training_seed}; "
        f"cases={len(cases)}; warmup={args.warmup}; repeats={args.timing_repeats}"
    )
    print(f"{'method':<12} {'success':>9} {'path_len':>10} {'expanded':>10} {'median_ms':>11} {'p95_ms':>10}")
    for method in planners:
        group = [row for row in rows if row["method"] == method]
        print(
            f"{method:<12} {np.mean([r['success'] for r in group]):>8.1%} "
            f"{np.mean([r['path_length'] for r in group]):>10.3f} "
            f"{np.mean([r['expanded_nodes'] for r in group]):>10.2f} "
            f"{np.median([r['runtime_ms'] for r in group]):>11.3f} "
            f"{np.percentile([r['runtime_p95_ms'] for r in group], 95):>10.3f}"
        )
    print(f"Saved per-case results to {args.output}")


if __name__ == "__main__":
    main()
