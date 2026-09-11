# iA* Week 1 Reproduction Report

## 1. Scope and reproducibility status

Paper: **iA*: Imperative Learning-based A* Search for Path Planning** (Chen, Yang, and Wang, IEEE RA-L 2025).

- iA* repository commit: `3e991fd6686cc59251e273e5d567fe62f8a66deb`
- Neural A* repository commit: `473edbbd7d20df3e2440b55989f87e720be2d06e`
- Dataset: official MP `mazes_032_moore_c8.npz`, test split, 100 maps
- Fixed test seed: `2026`
- Local verification platform: Python 3.10, PyTorch 2.14.0 CPU, NumPy 1.26.4
- Recommended Windows environment: separate Conda environment `iastar-repro`, Python 3.10

The official iA* Google Drive checkpoint was downloaded but is currently corrupted: `torch.load` raises `PytorchStreamReader failed reading zip archive: failed finding central directory`. This is independently reported in the repository's open Issue #2. Consequently, this pilot uses a quick iA* model trained for 5 epochs on 100 official 32×32 training instances. The iA* numbers below are a smoke-test result, **not** a reproduction of the paper's final numbers.

Primary sources:

- Paper: https://arxiv.org/html/2403.15870v4
- iA* official code: https://github.com/sair-lab/iAstar
- Broken-checkpoint issue: https://github.com/sair-lab/iAstar/issues/2
- Neural A* official code: https://github.com/omron-sinicx/neural-astar

## 2. Introduction: the research problem

Classical A* can return an optimal path when its assumptions and heuristic conditions hold, but its explored region and runtime grow on large maps. Learning-based planners can narrow the search, yet supervised approaches such as Neural A* require expert/optimal paths as labels and may overfit to training-map patterns. iA* addresses this tension with bilevel optimization: a neural upper level predicts per-node guidance, while a differentiable A* lower level produces the current path and supplies the training signal. The intended outcome is a smaller search area with near-optimal path quality and better generalization to unseen maps.

## 3. Methodology in five points

### 3.1 Input

For a grid of height `H` and width `W`, one planning instance is a three-layer tensor

\[
x \in \mathbb{R}^{3\times H\times W},
\]

containing:

1. the binary traversability/obstacle map;
2. a one-hot start map;
3. a one-hot goal map.

In this experiment, `H=W=32`. The three maps tell the model where motion is allowed, where planning begins, and where it must end.

### 3.2 Output

iA* returns two spatial outputs:

- a collision-free, near-optimal path matrix `mu` from start to goal;
- a search-history/closed-list matrix `C_mu` showing all expanded nodes.

The first measures solution quality; the second measures search efficiency. A good learned planner should keep the path short while shrinking the expanded region.

### 3.3 Cost map

The U-Net encoder `f_theta(x)` predicts one scalar for every grid cell, forming a map `P`. Conceptually, `P(i,j)` expresses how desirable or costly it is for the search to expand cell `(i,j)`. iA* selects a node using the score

\[
f(n)=s(n)+h(n)+p(n),
\]

where `s(n)` is accumulated path cost, `h(n)` is heuristic distance to the goal, and `p(n)` is learned guidance. Low-score cells are preferred. Unlike a fixed hand-designed heuristic, this guidance is learned from map, start, and goal structure.

### 3.4 Differentiable A*

Ordinary A* contains a discrete `argmin`, which blocks gradient propagation. Differentiable A* stores the open list, closed list, costs, start, goal, and selected node as matrices. It replaces hard node selection during backpropagation with a straight-through softmax: the forward pass still selects one hard cell, while the backward pass uses the soft probabilities to provide gradients. This lets the loss update the encoder through the search history. The final backtraced path remains discrete; the differentiable training signal primarily passes through the search process/history.

### 3.5 Self-supervised loss

The paper defines an upper-level objective combining unnecessary search and path length:

\[
U(\mu^*)=w_a L_a(\mu^*)+w_l L_l(\mu^*),
\]

where

\[
L_a(\mu)=\sum(C_\mu-\mu)
\]

counts expanded cells outside the final path, and `L_l` is geometric path length (orthogonal steps cost `1`; diagonal steps cost `sqrt(2)`). The lower-level differentiable A* generates the current solution `mu*`, so training does not require a fixed expert path label.

Important implementation note: the current official repository's `train.py --useIL` uses

```python
L1(search_history, found_path.detach())
```

as its self-supervised loss. It pulls the explored region toward the planner's own current path and also avoids ground-truth labels. This code-level loss is related to, but not algebraically identical to, the combined area-plus-length objective written in the paper. A formal reproduction report must disclose this difference.

## 4. 32×32 pilot experiment

All three methods used the same 100 test maps and the same seeded start locations. Runtime is wall-clock inference time per instance on CPU and includes the neural encoder when applicable. It excludes environment creation, model loading, and dataset loading.

| Method | Success rate | Mean path length | Mean expanded nodes | Mean runtime (ms) |
| --- | ---: | ---: | ---: | ---: |
| Vanilla A* | 100.0% | 34.084 | 70.99 | 13.975 |
| Neural A* (official checkpoint) | 100.0% | 37.267 | 60.76 | 13.999 |
| iA* (5-epoch quick training) | 100.0% | 34.360 | 55.79 | 20.334 |

Relative to Vanilla A*, the quick-trained iA* expanded about `21.4%` fewer nodes and produced paths about `0.81%` longer on average. Neural A* expanded about `14.4%` fewer nodes but produced paths about `9.34%` longer. On this CPU implementation, both learned planners were slower per instance because encoder computation and a matrix-based differentiable search are included; this should not be generalized to the paper's GPU or optimized runtime setting.

## 5. Interpretation and limitations

This run completes the engineering objective: the official dataset and implementations execute, all three planners are evaluated under one protocol, and a comparison image is generated. It does not establish that the quick-trained iA* outperforms the published method or generalizes to Penn routing.

Main limitations:

- the released iA* pretrained checkpoint is unusable, so iA* was only trained for 5 epochs on 100 maps;
- only one 32×32 MP maze family and one random seed were tested;
- CPU wall-clock results are sensitive to hardware, PyTorch version, warm-up, and Python overhead;
- the current repository implementation differs from the paper's written loss;
- the grid task is a controlled precursor to Route2Study, not yet a real Penn road-network experiment.

## 6. Windows execution (separate from Route2Study)

Create a sibling project, not a folder inside `E:\Route2Study`:

```powershell
cd E:\
git clone https://github.com/sair-lab/iAstar.git iAstar_reproduction
git clone --depth 1 https://github.com/omron-sinicx/neural-astar.git neural-astar
cd E:\iAstar_reproduction
conda env create -f reproduction\environment.yml
conda activate iastar-repro
```

Install PyTorch first. CPU-only is the most portable option:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --no-deps E:\neural-astar
```

Convert the valid Neural A* checkpoint, quick-train iA*, evaluate, and render:

```powershell
python reproduction\convert_neural_astar_checkpoint.py --input "E:\neural-astar\model\mazes_032_moore_c8\lightning_logs\version_0\checkpoints\epoch=33-step=272.ckpt"
python reproduction\train_iastar_quick.py --epochs 5 --samples 100 --batch-size 25
python reproduction\evaluate_32.py --samples 100
python reproduction\visualize_32.py --case 81
```

Expected outputs:

- `reproduction\results\metrics_32.csv`: one row per case and method;
- `reproduction\results\summary_32.csv`: aggregate table;
- `reproduction\results\search_comparison_32.png`: path/search-area comparison;
- `model\iastar\iastar_quick_32.pkl`: quick-trained iA* checkpoint;
- `model\nastar\neural_astar_cnn_32.pth`: converted official Neural A* checkpoint.

