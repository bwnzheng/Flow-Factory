# Reward Disagreement Analysis

This is an offline experiment for `notes/reward-disagreement.md`. It reads only
the saved train reward pickles at:

```text
saves/<run_name>/logs/rewards/train_step_*.pkl
```

It does not load reward models, checkpoints, or a training configuration from
outside the saved run. It reads raw rewards from the PKLs and, when present, the
`run_context` already saved in `logs/media.jsonl`. Each prompt's rollout group
is centered independently with the uniform mean of its original raw rewards.
SRC probabilities are used only after that natural decision has been computed,
to report effective training-mass statistics.

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
single_reward_advantage[i, k] = r[i, k] - mean_i(r[i, k])
scalar_advantage[i] = sum_k(w[k] * r[i, k]) - mean_i(sum_k(w[k] * r[i, k]))
disagreement[i, k] = single_reward_advantage[i, k] * scalar_advantage[i] < 0
has_conflict[i] = any_k(disagreement[i, k])
```

The strict inequality means an exact-zero advantage is non-conflicting. There
is no validity threshold and no filtered denominator: all samples in the frozen
prompt-local rollout group carry their original uniform mass.

Each step reports a macro-average over prompt groups, never a recentered pool
of samples from different prompts. The `metrics.csv` output is tidy/long-form:

- `natural_per_reward_disagreement_rate` is the uniform fraction of samples
  for which one named reward conflicts with the scalar decision. It identifies
  the source of natural-rollout conflict and is the only per-reward metric.
- `natural_conflict_mass` is the uniform sample mass with at least one
  conflicting reward.
- `effective_conflict_mass` is the corresponding SRC probability mass and is
  emitted only for groups with saved `src_groups[].probabilities`.

The output directory also contains `metadata.json`,
`per_reward_disagreement/<reward>.png` for every active reward combination,
and `conflict_mass.png` comparing Uniform with Effective conflict mass.
When multiple runs are configured, every figure overlays their `run_label`
trajectories. A non-SRC run contributes only Uniform conflict mass; an SRC run
contributes both Uniform and Effective conflict mass.
