"""Convert the official Neural A* Lightning checkpoint to a plain state dict."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from neural_astar.planner import NeuralAstar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("model/nastar/neural_astar_cnn_32.pth"), type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state = {
        re.split(r"planner\.", key)[-1]: value
        for key, value in checkpoint["state_dict"].items()
        if "planner." in key
    }
    model = NeuralAstar(encoder_arch="CNN", encoder_depth=4, Tmax=1.0)
    model.load_state_dict(state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved converted checkpoint to {args.output}")


if __name__ == "__main__":
    main()
