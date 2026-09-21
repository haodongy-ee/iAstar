# iA* Week 2: controlled multi-seed protocol

Current dated status and the exam pause are recorded in
[`STATUS_2026-09-21.md`](STATUS_2026-09-21.md).

Week 1 established that the code, dataset, local training, three planners, and visualization pipeline work. Week 2 changes the experiment from a single pilot run into a controlled multi-seed study.

## Protocol

- Train iA* independently with seeds `1234`, `2026`, and `3407`.
- Use all 800 official 32x32 training maps and 20 epochs per seed as the next stability milestone. The paper-scale 200-epoch run remains the final target.
- Evaluate every checkpoint on the same 100 materialized test cases and the same sampled starts (`eval_seed=2026`).
- Warm up each planner before timing, time each planner five times per case, synchronize CUDA around every measurement, and report median and P95 latency.
- Rotate planner execution order by case to reduce systematic cache and order bias.
- Report per-training-run means, standard deviation, and a run-level 95% bootstrap confidence interval.
- Compare learned planners with A* using paired relative changes for path length, expanded nodes, and runtime.

## Run

```powershell
conda activate iastar-repro
python reproduction\run_multiseed_32.py
```

For a fast pipeline check before the longer run:

```powershell
python reproduction\run_multiseed_32.py `
  --epochs 2 --train-samples 100 --eval-samples 10 `
  --warmup 1 --timing-repeats 2 `
  --output-dir reproduction\results\week2_smoke
```

Use `--skip-existing` to resume without retraining completed seeds. Outputs are written under the selected result directory; model checkpoints remain ignored by Git.

## Interpretation guardrails

The 20-epoch result is an intermediate stability milestone, not a claim that the paper's final result has been reproduced. Runtime is hardware- and implementation-dependent. With only three training seeds, confidence intervals are descriptive and should be reported alongside the raw per-run values. The current repository loss also remains a code-level approximation of the objective written in the paper, as documented in `WEEK1_REPORT.md`.
