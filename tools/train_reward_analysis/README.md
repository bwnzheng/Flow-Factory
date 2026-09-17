# Train Reward Analysis

This is an offline experiment for `notes/train-reward-analysis.md`. It reads only
the saved train reward pickles at:

```text
saves/<run_name>/logs/rewards/train_step_*.pkl
```

It does not load reward models, checkpoints, or a training configuration from
outside the saved run. It reads raw rewards from the PKLs and, when present, the
`run_context` already saved in `logs/media.jsonl`. Each prompt's rollout group
is standardized independently: every reward and the weighted scalar reward are
centered with their prompt-local mean and divided by their prompt-local
population standard deviation. Zero-variance quantities are represented by
zero. The analysis does not use SRC probabilities or any reweighted statistic.

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
  # Set true only for SRC-Reweight runs. GA, SRC-Evolve, and uniform runs
  # should keep this false and are plotted without sample-weight groups.
  # - src_reweight: true
  - name: "sd3-5_lora_nft_20260808_215750"
    label: "SRC-NFT"

plot:
  # Positive odd centered moving-average window; 1 disables smoothing.
  smoothing_window: 5

output:
  dir: "analysis_output/train_reward_analysis"
  plot_format: "png"  # Matplotlib output format: png or pdf.
  cache_mode: "regenerate"  # "regenerate" writes plot_data.json; "reuse" redraws from it.
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
reward_advantage[i, k] = (r[i, k] - mean_i(r[i, k])) / std_i(r[i, k])
scalar_advantage[i] = (sum_k(w[k] * r[i, k]) - mean_i(sum_k(w[k] * r[i, k]))) / std_i(sum_k(w[k] * r[i, k]))
conflict_score[i, k] = w[k] * reward_advantage[i, k] * scalar_advantage[i]
sample_lower_bound[i] = min_k(conflict_score[i, k])
per_reward_disagreement[k] = mean_i(reward_advantage[i, k] * scalar_advantage[i] < 0)
per_reward_bottleneck_rate[k] = mean_i(argmin_j conflict_score[i, j] == k)
agreement_count[i] = sum_k(reward_advantage[i, k] * scalar_advantage[i] >= 0)
agreement_count_distribution[c] = mean_i(agreement_count[i] == c)   for c = 0..K
mean_agreement_count = mean_i(agreement_count[i])
```

Positive conflict scores mean the named reward supports the scalar training
direction; negative values mean it opposes that direction. The lower bound
keeps the weakest reward score for every sample, so strong agreement on one
reward cannot hide opposition on another.

Each step reports a macro-average over prompt groups, never a recentered pool
of samples from different prompts. The `metrics.csv` output is tidy/long-form:

- `per_reward_conflict_score` is the prompt-group mean standardized conflict score for
  each active reward.
- `per_reward_disagreement` is the prompt-group fraction of samples whose
  centered reward direction opposes the weighted scalar direction for each
  active reward. Exact zero products are treated as non-conflicting.
- `per_reward_bottleneck_rate` is the prompt-group fraction of samples for
  which each reward has the minimum weighted standardized contribution.
- `standardized_reward_covariance` is the prompt-local population covariance
  between each pair of standardized rewards, macro-averaged over prompt groups
  at each step. It is computed independently for each active reward combination;
  SRC probabilities are not used.
- `reward_concordance_lower_bound` is the prompt-group mean of each sample's
  weakest standardized conflict score. It is the sample-wise reward-concordance lower
  bound under the frozen uniform reference.
- `agreement_count_c<c>` is the prompt-group fraction of samples whose agreeing
  reward count equals `c`, macro-averaged over prompt groups. Per-reward
  disagreement rates are marginals, so they cannot tell polarized conflict
  (a few samples opposing several rewards at once) from diffuse conflict (many
  samples each opposing one reward) — the distribution can. Note that `c = 0`
  is unreachable: with the strictly positive weights this tool accepts, a sample
  below the group mean on every reward is also below the mean of the weighted
  scalar, so the lowest populated bin is `c = 1`.
- `mean_agreement_count` is the prompt-group mean agreeing reward count per
  sample. It is a derived quantity — exactly `K - sum_k(per_reward_disagreement[k])`
  — and carries no information beyond the per-reward disagreement rates; it is
  reported for readability only, and plotted as one bounded scalar per
  combination against a dashed `all K rewards agree` ceiling.
- `fully_concordant_sample_rate` is the prompt-group fraction of samples that
  agree with the weighted scalar on every active reward, i.e. the `c = K` bin of
  the agreement-count distribution. It is named separately because `K` varies
  per reward combination.

The output directory also contains `metadata.json`, followed by one directory
per dataset:

```text
<dataset>/
  per_reward_conflict_score/<reward>.png
  per_reward_disagreement/<reward>.png
  per_reward_bottleneck_rate/<reward>.png
  standardized_reward_covariance/<reward_pair>.png
  reward_concordance_lower_bound.<plot_format>
  agreement_count/<reward_combination>.<plot_format>
  agreement_count_expectation/<reward_combination>.<plot_format>
```

The dataset directory is recovered from the saved run context source (for
example, `pickscore` or `ocr`), so the fixed reward combination for each
dataset is unambiguous. Runs without saved source provenance are placed under
`unknown_dataset`. Set `output.plot_format` to `png` (default) or `pdf` to choose the
format for all generated figures.
When multiple runs are configured, every figure overlays their `run_label`
trajectories. All runs use exactly the same raw-reward calculation.
Each covariance figure contains only one reward pair and overlays all configured
run trajectories. The reward combination is intentionally omitted from the
filename and title because it is fixed by the dataset directory. The diagonal
is omitted because covariance after per-group standardization is one by
definition.
Every plotted curve is smoothed independently with the centered moving-average
window in `plot.smoothing_window`. At the first and last few recorded steps,
the average uses the available in-range points. The original unsmoothed curve
is retained as a same-color transparent background trace; use `1` when the
foreground should equal the raw values.

The final tidy rows used by all figures are stored in `plot_data.json`. Set
`output.cache_mode: reuse` to skip reward-pickle analysis and redraw every figure
directly from that cache. If the cache is absent, use `regenerate` first.
