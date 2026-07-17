# Reward Pareto Analysis

This tool analyzes reward distributions and full-dimensional Pareto convexity from three
data sources:

1. Saved training and evaluation reward pickles under `logs/rewards/`.
2. Rollout images referenced by `logs/media.jsonl`, scored offline with configured rewards.
3. Images generated from saved checkpoints and scored on evaluation prompts.

Pairwise reward projections, faceted convex-hull plots, windowed hull animations, hull-area
curves, and the retired per-group hypervolume plot are intentionally not generated. They do
not preserve dominance relationships in the original multi-reward space.

## Usage

```bash
python -m tools.reward_pareto_analysis.analyze \
    -c tools/reward_pareto_analysis/default.yaml
```

The default configuration enables only saved reward-pickle analysis. That path discovers
reward names and applicable reward combinations from the pickle data and does not require
the `model` or `rewards` sections. Enable `images_analysis` to rescore saved rollout images,
or `evaluation` to generate and score checkpoint images; either source requires `rewards`.

A minimal reward-pickle configuration is:

```yaml
run_name: "sd3-5_lora_nft_20260716_204333"
save_dir: "saves"

rewards_analysis:
  enabled: true

compute:
  max_workers: 0

output:
  dir: "analysis_output"
```

Pareto metrics are computed independently by step. `compute.max_workers: 0` selects up to
16 worker processes automatically; set a positive integer to choose an explicit limit.
The aggregation phase groups rows by step once, so its cost grows linearly rather than
quadratically with the number of steps.

The Pareto outputs contain front size, absolute and relative convexification hypervolume
gaps, and the convex-supported Pareto ratio. Convexity-depth values are not retained,
aggregated, or exported. The supported-point classifier solves the reward-weight dual LP
with one variable per reward dimension; if HiGHS dual simplex returns an indeterminate
status, it retries with the interior-point solver before reporting an error.

During Pareto analysis, the command reports the resolved worker count, reward dimension,
step and group counts, a per-step geometry progress bar, stage timings, generated artifact
counts, and the output directory. Geometry failures include the affected step and reward
combination in the raised error.

## Modules

- `analyze.py`: configuration, data-source workflows, caching, and output dispatch.
- `reward_logs.py`: group-aware training reward and evaluation reward pickle readers.
- `plots.py`: 1-D distributions, reward percentiles, exact hypervolume calculations, and
  group-aware Pareto-convexity metrics and figures.
- `media_logs.py`: rollout image discovery from `media.jsonl`.
- `reward_scoring.py`: offline CLIP and PickScore inference.
- `checkpoint_evaluation.py`: checkpoint discovery and batched evaluation generation.

## Output Structure

```text
analysis_output/reward_pareto_analysis/<run_name>/
├── train_rewards/
│   └── reward_combinations/<reward_a__reward_b...>/
│       ├── reward_percentiles.png
│       └── pareto_convexity/
│           ├── overview.png
│           ├── per_group_metrics.csv
│           ├── metadata.json
│           └── figures/
│               ├── pdf/*.pdf
│               └── svg/*.svg
└── eval_rewards/<dataset>/
    └── reward_combinations/<reward_a__reward_b...>/
        └── reward_percentiles.png
```

Pareto-convexity outputs require prompt-group indices. Evaluation reward files without group
indices still receive reward-percentile plots but do not receive misleading group metrics.
