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

analysis:
  reward_epsilon: 1.0e-8
  scalar_epsilon: 1.0e-8

output:
  dir: "analysis_output/reward_disagreement_analysis"
```

Then run:

```bash
python -m tools.reward_disagreement_analysis.analyze \
  -c tools/reward_disagreement_analysis/default.yaml
```

## Metrics

For each prompt-local frozen reward matrix `r` and positive scalarization
weights `w`, the tool computes:

```text
single_reward_advantage[i, k] = r[i, k] - mean_i(r[i, k])
scalar_advantage[i] = sum_k(w[k] * r[i, k]) - mean_i(sum_k(w[k] * r[i, k]))
disagreement[i, k] = single_reward_advantage[i, k] * scalar_advantage[i] < 0
```

Near-zero single-reward or scalar advantages are neutral. They are excluded
from the corresponding rate denominator. Fully concordant samples must be
valid on every active dimension, so neutral samples cannot inflate FCR.

Each step reports a macro-average over prompt groups, never a recentered pool
of samples from different prompts. The `metrics.csv` output is tidy/long-form:

- `natural_disagreement_rate` and `natural_fully_concordant_ratio` describe
  samples produced by the current policy under uniform group mass.
- `effective_*` rows use saved SRC probabilities and are emitted only for
  groups with `src_groups[].probabilities`.
- `*_disagreement_count_probability` records the conflict-count distribution
  over fully valid samples.
- `scalar_advantage_identity_max_abs_error` audits the equivalence between
  centered scalar rewards and weighted centered reward dimensions.

The output directory also contains `metadata.json`, an all-reward disagreement
overview, group metrics, and `per_reward_disagreement/<reward>.png` for every
active reward combination.
