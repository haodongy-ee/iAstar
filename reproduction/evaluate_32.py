"""Evaluate Vanilla A*, Neural A*, and iA* on the same fixed 32x32 test cases."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from neural_astar.planner import NeuralAstar
from torch.utils.data import DataLoader, Subset

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
    if np.any(cells & ~free.astype(bool)) or not np.any(cells & start.astype(bool)) or not np.any(cells & goal.astype(bool)):
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
                if (dr or dc) and 0 <= nr < cells.shape[0] and 0 <= nc < cells.shape[1] and cells[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("planning-datasets/mpd/instances/032/mazes_032_moore_c8.npz"))
    parser.add_argument("--iastar", type=Path, default=Path("model/iastar/iastar_quick_32.pkl"))
    parser.add_argument("--neural-astar", type=Path, default=Path("model/nastar/neural_astar_cnn_32.pth"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("reproduction/results/metrics_32.csv"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = MazeDataset(str(args.data), "test")
    loader = DataLoader(Subset(dataset, range(min(args.samples, len(dataset)))), batch_size=1, shuffle=False, num_workers=0)

    vanilla = dastar(device=str(device), is_training=False, output_path_list=False, w=2.0).to(device).eval()
    neural = NeuralAstar(encoder_arch="CNN", encoder_depth=4, Tmax=1.0).to(device).eval()
    neural.load_state_dict(torch.load(args.neural_astar, map_location=device), strict=True)
    imperative = iastar(encoder_input=3, encoder_arch="UNet", encoder_depth=4, device=str(device), is_training=False, output_path_list=False, w=2.0).to(device).eval()
    imperative.encoder.load_state_dict(torch.load(args.iastar, map_location=device)["model_state_dict"], strict=True)

    planners = {
        "A*": lambda m, s, g: vanilla(m, s, g, m),
        "Neural A*": lambda m, s, g: neural(m, s, g),
        "iA*": lambda m, s, g: imperative(m, s, g),
    }
    rows: list[dict[str, float | int | str]] = []
    with torch.inference_mode():
        for case_id, (maps, start, goal, _) in enumerate(loader):
            maps, start, goal = maps.to(device), start.to(device), goal.to(device)
            for method, planner in planners.items():
                begin = time.perf_counter_ns()
                output = planner(maps, start, goal)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                runtime_ms = (time.perf_counter_ns() - begin) / 1e6
                path = output.paths[0, 0].detach().cpu().numpy()
                rows.append(
                    {
                        "case_id": case_id,
                        "method": method,
                        "success": int(valid_path(path, maps[0, 0].cpu().numpy(), start[0, 0].cpu().numpy(), goal[0, 0].cpu().numpy())),
                        "path_length": float(geometric_length(output.paths)[0].item()),
                        "expanded_nodes": int(output.histories[0].sum().item()),
                        "runtime_ms": runtime_ms,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"device={device}; fixed seed={args.seed}; cases={len(rows) // 3}")
    print(f"{'method':<12} {'success':>9} {'path_len':>10} {'expanded':>10} {'runtime_ms':>11}")
    for method in planners:
        group = [row for row in rows if row["method"] == method]
        print(
            f"{method:<12} {np.mean([r['success'] for r in group]):>8.1%} "
            f"{np.mean([r['path_length'] for r in group]):>10.3f} "
            f"{np.mean([r['expanded_nodes'] for r in group]):>10.2f} "
            f"{np.mean([r['runtime_ms'] for r in group]):>11.3f}"
        )
    print(f"Saved per-case results to {args.output}")


if __name__ == "__main__":
    main()
