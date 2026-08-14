# Reward Concordance Analysis

This is an offline experiment for `notes/reward-disagreement.md`. It reads only
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

output:
  dir: "analysis_output/reward_disagreement_analysis"
```

Then run:

```bash
python -m tools.reward_disagreement_analysis.analyze \
  -c tools/reward_disagreement_analysis/default.yaml
```

To compare the saved SRC-NFT run with its non-SRC NFT counterpart in the same
figures, run:

```bash
python -m tools.reward_disagreement_analysis.analyze \
  -c tools/reward_disagreement_analysis/nft_src_vs_uniform.yaml
```

## Metrics

For each prompt-local frozen reward matrix `r` and positive scalarization
weights `w`, the tool computes:

```text
reward_advantage[i, k] = r[i, k] - mean_i(r[i, k])
scalar_advantage[i] = sum_k(w[k] * r[i, k]) - mean_i(sum_k(w[k] * r[i, k]))
conflict_score[i, k] = w[k] * reward_advantage[i, k] * scalar_advantage[i]
sample_lower_bound[i] = min_k(conflict_score[i, k])
```

Positive conflict scores mean the named reward supports the scalar training
direction; negative values mean it opposes that direction. The lower bound
keeps the weakest reward score for every sample, so strong agreement on one
reward cannot hide opposition on another.

Each step reports a macro-average over prompt groups, never a recentered pool
of samples from different prompts. The `metrics.csv` output is tidy/long-form:

- `per_reward_conflict_score` is the prompt-group mean raw conflict score for
  each active reward.
- `reward_concordance_lower_bound` is the prompt-group mean of each sample's
  weakest raw conflict score. It is the sample-wise reward-concordance lower
  bound under the frozen uniform reference.

The output directory also contains `metadata.json`,
`per_reward_conflict_score/<reward>.png` for every active reward combination,
and `reward_concordance_lower_bound.png` for the overall lower-bound curve.
When multiple runs are configured, every figure overlays their `run_label`
trajectories. All runs use exactly the same raw-reward calculation.
