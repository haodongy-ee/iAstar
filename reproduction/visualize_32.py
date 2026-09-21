"""Render a fixed 32x32 search-area/path comparison for the three planners."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from neural_astar.planner import NeuralAstar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import MazeDataset
from dastar import dastar
from iastar import iastar


def overlay(free: np.ndarray, history: np.ndarray, path: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    # 0 obstacle, 1 free, 2 explored, 3 path, 4 start, 5 goal
    image = free.astype(np.uint8)
    image[history.astype(bool)] = 2
    image[path.astype(bool)] = 3
    image[start.astype(bool)] = 4
    image[goal.astype(bool)] = 5
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("planning-datasets/mpd/instances/032/mazes_032_moore_c8.npz"))
    parser.add_argument("--iastar", type=Path, default=Path("model/iastar/iastar_quick_32.pkl"))
    parser.add_argument("--neural-astar", type=Path, default=Path("model/nastar/neural_astar_cnn_32.pth"))
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("reproduction/results/search_comparison_32.png"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MazeDataset(str(args.data), "test")
    # Advance the seeded dataset RNG exactly as evaluate_32.py does so that the
    # selected case has the same randomly sampled start cell as the CSV row.
    for index in range(args.case + 1):
        maps, start, goal, _ = dataset[index]
    maps = torch.from_numpy(maps[None]).to(device)
    start = torch.from_numpy(start[None]).to(device)
    goal = torch.from_numpy(goal[None]).to(device)

    vanilla = dastar(device=str(device), is_training=False, output_path_list=False, w=2.0).to(device).eval()
    neural = NeuralAstar(encoder_arch="CNN", encoder_depth=4, Tmax=1.0).to(device).eval()
    neural.load_state_dict(torch.load(args.neural_astar, map_location=device), strict=True)
    imperative = iastar(encoder_input=3, encoder_arch="UNet", encoder_depth=4, device=str(device), is_training=False, output_path_list=False, w=2.0).to(device).eval()
    imperative.encoder.load_state_dict(torch.load(args.iastar, map_location=device)["model_state_dict"], strict=True)

    with torch.inference_mode():
        outputs = {
            "Vanilla A*": vanilla(maps, start, goal, maps),
            "Neural A*": neural(maps, start, goal),
            "iA* (locally trained)": imperative(maps, start, goal),
        }

    colors = ListedColormap(["#151515", "#f5f5f5", "#75c893", "#e84a5f", "#2878b5", "#f5a623"])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
    for axis, (name, output) in zip(axes, outputs.items()):
        history = output.histories[0, 0].cpu().numpy()
        path = output.paths[0, 0].cpu().numpy()
        image = overlay(maps[0, 0].cpu().numpy(), history, path, start[0, 0].cpu().numpy(), goal[0, 0].cpu().numpy())
        axis.imshow(image, cmap=colors, vmin=0, vmax=5, interpolation="nearest")
        axis.set_title(f"{name}\nexpanded={int(history.sum())}, path cells={int(path.sum())}", fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("32×32 MP maze: explored area (green) and path (red)", fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.82, wspace=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
