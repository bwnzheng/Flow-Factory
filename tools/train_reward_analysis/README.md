# Train Reward Analysis

This is an offline experiment for `notes/train-reward-analysis.md`. It reads only
the saved train reward pickles at:

```text
saves/<run_name>/logs/rewards/train_step_*.pkl
```

It does not load reward models, checkpoints, or a training configuration from
outside the saved run. It reads raw rewards from the PKLs and, when present, the
`run_context` already saved in `logs/media.jsonl`. Each prompt's rollout group
is centered independently with the uniform mean of its original raw rewards.
The analysis does not use SRC probabilities or any reweighted statistic.

## Required provenance

The saved PKL schema preserves per-prompt reward arrays and, for SRC runs,
`src_groups[].probabilities`; it does **not** itself preserve `reward.weight`.
Current locally saved media manifests begin with a `run_context` whose resolved
`configuration.reward.*.weight` provides those historical weights, and the
tool recovers them automatically. For older runs without that context, copy the
historical positive weights into `reward_weights` in the analysis YAML, with an
optional per-run fallback. The tool fails if an active reward remains unknown;
it never silently assumes equal weights.

## Run

Copy and fill the default configuration:

```yaml
save_dir: "saves"
reward_weights:
  # Needed only if this run has no logs/media.jsonl run_context:
  # pick_score: 1.0
  # clip_score: 1.0
  # ocr_reward: 1.0

runs:
  - name: "sd3-5_lora_nft_20260808_215750"
    label: "SRC-NFT"

plot:
  # Positive odd centered moving-average window; 1 disables smoothing.
  smoothing_window: 5

output:
  dir: "analysis_output/train_reward_analysis"
```

Then run:

```bash
python -m tools.train_reward_analysis.analyze \
  -c tools/train_reward_analysis/default.yaml
```

To compare the saved SRC-NFT run with its non-SRC NFT counterpart in the same
figures, run:

```bash
python -m tools.train_reward_analysis.analyze \
  -c tools/train_reward_analysis/nft_src_vs_uniform.yaml
```

## Metrics

For each prompt-local frozen reward matrix `r` and positive scalarization
weights `w`, the tool computes:

```text
reward_advantage[i, k] = r[i, k] - mean_i(r[i, k])
scalar_advantage[i] = sum_k(w[k] * r[i, k]) - mean_i(sum_k(w[k] * r[i, k]))
conflict_score[i, k] = w[k] * reward_advantage[i, k] * scalar_advantage[i]
sample_lower_bound[i] = min_k(conflict_score[i, k])
per_reward_disagreement[k] = mean_i(reward_advantage[i, k] * scalar_advantage[i] < 0)
```

Positive conflict scores mean the named reward supports the scalar training
direction; negative values mean it opposes that direction. The lower bound
keeps the weakest reward score for every sample, so strong agreement on one
reward cannot hide opposition on another.

Each step reports a macro-average over prompt groups, never a recentered pool
of samples from different prompts. The `metrics.csv` output is tidy/long-form:

- `per_reward_conflict_score` is the prompt-group mean raw conflict score for
  each active reward.
- `per_reward_disagreement` is the prompt-group fraction of samples whose
  centered reward direction opposes the weighted scalar direction for each
  active reward. Exact zero products are treated as non-conflicting.
- `reward_concordance_lower_bound` is the prompt-group mean of each sample's
  weakest raw conflict score. It is the sample-wise reward-concordance lower
  bound under the frozen uniform reference.

The output directory also contains `metadata.json`,
`per_reward_conflict_score/<reward>.png` for every active reward combination,
`per_reward_disagreement/<reward>.png` for every active reward combination,
and `reward_concordance_lower_bound.png` for the overall lower-bound curve.
When multiple runs are configured, every figure overlays their `run_label`
trajectories. All runs use exactly the same raw-reward calculation.
Every plotted curve is smoothed independently with the centered moving-average
window in `plot.smoothing_window`. At the first and last few recorded steps,
the average uses the available in-range points. The original unsmoothed curve
is retained as a same-color transparent background trace; use `1` when the
foreground should equal the raw values.
