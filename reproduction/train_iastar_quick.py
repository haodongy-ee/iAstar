"""Small CPU-friendly iA* training run for a reproducible 32x32 smoke test.

This is deliberately not a paper-scale reproduction. It uses the current official
repository's self-supervised implementation:
L1(differentiable search history, planner's own detached path).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import MazeDataset
from iastar import iastar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("planning-datasets/mpd/instances/032/mazes_032_moore_c8.npz"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("model/iastar/iastar_quick_32.pkl"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = MazeDataset(str(args.data), "train")
    subset = Subset(dataset, range(min(args.samples, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = iastar(
        encoder_input=3,
        encoder_arch="UNet",
        encoder_depth=4,
        device=str(device),
        Tmax=0.25,
        is_training=True,
        output_path_list=False,
        w=2.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-3)

    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        batches = 0
        for maps, start, goal, _ in loader:
            maps, start, goal = maps.to(device), start.to(device), goal.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(maps, start, goal)
            loss = F.l1_loss(output.histories, output.paths.float().detach())
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        print(f"epoch {epoch + 1:02d}/{args.epochs}: loss={total / batches:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.encoder.state_dict(),
            "quick_training": True,
            "epochs": args.epochs,
            "samples": len(subset),
            "seed": args.seed,
        },
        args.output,
    )
    print(f"Saved {args.output} in {time.perf_counter() - started:.1f}s on {device}")


if __name__ == "__main__":
    main()
