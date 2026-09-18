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
Alongside the standardized quantities it reports each reward's raw per-step
level, which the progress figure needs and no other figure uses.

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
  cache_mode: "regenerate"  # "regenerate" writes one JSON per figure; "reuse" redraws them.
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

`--cache-mode {regenerate,reuse}` overrides `output.cache_mode` for one
invocation, so redrawing from existing per-figure JSON files needs no edit to
the config file:

```bash
python -m tools.train_reward_analysis.analyze \
  -c tools/train_reward_analysis/nft_src_vs_uniform.yaml --cache-mode reuse
```

Omitting the flag leaves the config file's value in charge; the override applies
on top of it, and applies to nothing else.

Both stages use worker processes: the per-run/step metric computation runs on
`os.cpu_count()` workers, and the figures are spread over at most one worker per
figure (`metadata.json` records both as `analysis_workers` and
`plot_workers`). Figure workers are spawned rather than forked, because
matplotlib is already imported in the parent process.

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
positive_fully_concordant_sample_rate = mean_i(reward_advantage[i, k] > 0 for all k)
negative_fully_concordant_sample_rate = mean_i(reward_advantage[i, k] < 0 for all k)
per_reward_mean_reward[k] = mean_i(r[i, k])
```

Positive conflict scores mean the named reward supports the scalar training
direction; negative values mean it opposes that direction. The lower bound
keeps the weakest reward score for every sample, so strong agreement on one
reward cannot hide opposition on another.

Each step reports a macro-average over prompt groups, never a recentered pool
of samples from different prompts. These values become the line points in the
per-figure JSON files:

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
  combination against a dashed `all K rewards concordant` ceiling.
- `fully_concordant_sample_rate` is the prompt-group fraction of samples that
  agree with the weighted scalar on every active reward, i.e. the `c = K` bin of
  the agreement-count distribution. It is named separately because `K` varies
  per reward combination.
- `positive_fully_concordant_sample_rate` and
  `negative_fully_concordant_sample_rate` split that `c = K` bin by direction.
  Because the weights are strictly positive, a sample above its group mean on
  every reward is necessarily above the mean of the weighted scalar too, so
  "every standardized reward is positive" is already the positive-aligned case,
  and symmetrically for negative. The two halves are disjoint, and a reward
  with no group variance standardizes to an exact zero, which agrees with the
  scalar without being either positive or negative — so the two halves can sum
  to less than `fully_concordant_sample_rate`, never more.
- `per_reward_mean_reward` is the raw, unstandardized reward level at that step,
  macro-averaged over prompt groups. A reward that some groups drop as missing
  (an OCR score that only applies to part of the prompt set) is averaged over
  its own active groups only, so it is not diluted by groups that never score
  it. Unlike every other metric here it is not comparable across rewards on
  different scales; it exists so the progress figure has an auditable source.

One naming split runs through this output on purpose: figure titles, axis labels,
and legends say *concordant*, while files named after the original metric
(`agreement_count`,
`agreement_count_expectation`) keep that name, while `concordance` is named for
its content.

The output directory contains `metadata.json`, followed by one directory per
dataset. Every image has a same-stem JSON file beside it:

```text
<dataset>/
  per_reward_conflict_score/<reward>.{json,<plot_format>}
  per_reward_disagreement/<reward>.{json,<plot_format>}
  per_reward_bottleneck_rate/<reward>.{json,<plot_format>}
  standardized_reward_covariance/<reward_pair>.{json,<plot_format>}
  reward_concordance_lower_bound.{json,<plot_format>}
  agreement_count.{json,<plot_format>}
  agreement_count_expectation.{json,<plot_format>}
  training_progress/
    <run_label>.{json,<plot_format>}
    concordance.{json,<plot_format>}
```

Each JSON is organized around the image rather than around internal metric
records. Its `series` list gives every drawn line a label and a compact list of
`[training_step, raw_value]` points. Axis, smoothing, legend, and non-default
style settings are included only because they are needed to redraw that image.
For example:

```json
{
  "spec_version": 6,
  "title": "Concordant sample rate [pickscore]",
  "x_label": "Training step",
  "series": [
    {
      "label": "SRC-NFT | positive concordant rate",
      "points": [
        [0.0, 18.75],
        [10.0, 20.3125]
      ],
      "color": "#2a78d6"
    }
  ],
  "left": {
    "label": "Percent of concordant samples"
  },
  "smoothing_window": 5
}
```

The points are the unsmoothed values supplied to the renderer. The foreground
moving average is derived from them, so a figure can be fully reconstructed
from its JSON without loading reward pickles or rebuilding metric rows.

## Font sizes

Add `font_sizes` at the top level of one figure JSON to control its text in
points without affecting any other figure:

```json
"font_sizes": {
  "title": 16.0,
  "x_label": 13.0,
  "left_y_label": 13.0,
  "right_y_label": 13.0,
  "x_tick": 11.0,
  "left_y_tick": 11.0,
  "right_y_tick": 11.0,
  "legend": 10.0
}
```

Every entry is optional. An omitted entry keeps the established matplotlib
size, so old specs retain their existing pixels. `left_y_*` and `right_y_*`
apply independently on a dual-y figure and apply to every corresponding panel
on a broken-axis figure. `font_sizes.legend` overrides the older
`legend.fontsize` setting when both are present. All configured sizes must be
finite and strictly positive.

## Borders and multi-column legends

Set `border_width` at the top level to change the axes frame width for one
figure. The value is in points and must be finite and strictly positive:

```json
"border_width": 1.5
```

On a dual-y figure the setting applies to both axes. On a broken-axis figure it
also applies to every panel and the diagonal break marks. Omitting the field
keeps the established matplotlib width.

Legends already support multiple columns through `legend.ncol`:

```json
"legend": {
  "entries": [
    {"label": "run A", "color": "#2a78d6"},
    {"label": "run B", "color": "#eb6834"}
  ],
  "ncol": 2,
  "loc": "upper center"
}
```

`ncol` must be a positive integer. Use the optional `anchor` field when the
multi-column legend needs to sit outside the axes.

If the title or a top-anchored legend leaves too much blank space above the
plot, set the top margin for that figure as a fraction of the figure height:

```json
"top_margin": 0.04
```

This leaves 4% above the axes. The default is unchanged when the field is
omitted. Values must be in `[0.0, 1.0)`.

For extra room below the x-axis label, set the bottom margin directly:

```json
"bottom_margin": 0.14
```

This leaves 14% below the axes. It is useful when a long x-axis label or large
x-axis tick font is close to the lower image boundary.

## Broken y-axis

To use a broken y-axis for one figure, edit that figure's adjacent JSON file and
replace the axis `limits` with ascending `segments`. For example, this keeps
`0.0-0.2` and `0.8-1.0` while omitting the interval in between:

```json
{
  "spec_version": 6,
  "title": "Concordant sample rate [pickscore]",
  "x_label": "Training step",
  "series": [
    {
      "label": "SRC-NFT | positive concordant rate",
      "points": [[0.0, 0.1], [10.0, 0.9]],
      "color": "#2a78d6"
    }
  ],
  "left": {
    "label": "Concordant sample rate",
    "segments": [[0.0, 0.2], [0.8, 1.0]],
    "segment_height_ratios": [2.0, 1.0]
  },
  "smoothing_window": 1,
  "break_gap": 0.05,
  "break_mark_size": 0.012
}
```

`segments` and `segment_height_ratios` are both ordered from the lowest range
to the highest range, even though the highest range is drawn at the top. The
height ratios are optional and default to equal panel heights. `break_gap` and
`break_mark_size` are also optional; the renderer adds the diagonal break marks
automatically. An axis cannot define both `limits` and `segments`.

The original `series[].points` stay unchanged: points in an omitted interval
are clipped from the image rather than deleted from the JSON. Redraw the edited
spec with `--cache-mode reuse`. This redraws all specs indexed by
`metadata.json`, but only the edited figure changes.

For a two-y-axis figure, both `left` and `right` must define the same number of
segments and identical `segment_height_ratios`; their numeric segment bounds
may differ. The renderer rejects mismatched layouts because the stacked panels
would otherwise imply a false correspondence between the two scales.

The dataset directory is recovered from the saved run context source (for
example, `pickscore` or `ocr`), so the fixed reward combination for each
dataset is unambiguous. The agreement-count figures are therefore written per
dataset rather than per reward combination; a dataset that unexpectedly carries
two reward combinations is rejected instead of mixing incomparable
agreeing-count scales into one figure. Runs without saved source provenance are placed under
`unknown_dataset`. Set `output.plot_format` to `png` (default) or `pdf` to choose the
format for all generated figures.
When multiple runs are configured, every figure overlays their `run_label`
trajectories. All runs use exactly the same raw-reward calculation.
Each covariance figure contains only one reward pair and overlays all configured
run trajectories. The reward combination is intentionally omitted from the
filename and title because it is fixed by the dataset directory. The diagonal
is omitted because covariance after per-group standardization is one by
definition.
`training_progress/` holds one two-axis figure per run plus one autoscaled
all-run concordance figure per dataset. Full concordance is a property of the
whole reward set (`K`) rather than of one reward.

- `<run_label>` is written once per run and carries two axes. The left axis holds
  the aggregated progress curve: each reward's `per_reward_mean_reward` is mapped
  onto 0-100% of its own observed range within that run, and those per-reward
  percentages are averaged at each step. Averaging percentages rather than raw
  levels is what makes the aggregate meaningful — a reward scored 0-1 and one
  scored 0-5 contribute equally — and it keeps the figure the same size however
  many rewards are active. Read that curve for shape, never for level: a single
  outlier step sets a reward's 100% mark, and a reward that never moves has no
  range to express progress against, so it contributes a flat 0%, following the
  zero-variance convention above. The right axis holds the two concordance rates,
  which are already shares of samples and are plotted at their own value.
- `concordance` drops the reward curve and overlays every configured run, so the
  concordance rates can be read across runs without the reward curves competing
  for the same visual channels. Telling the runs apart is what that figure is
  for, so each run owns a colour *and* a marker, and the two directions are
  separated by the dash pattern instead — positive solid, negative dashed.
  Direction has only two values while runs do not, so it is the one that can
  afford the weaker channel, and the dash pattern is also what survives greyscale
  printing, where the run colours collapse together.

Which channel carries what therefore differs between the two figures: the
per-run figure colours the direction, and the all-run figure colours the run.
That is deliberate, since a single run leaves the colour channel nothing else to
distinguish, and each figure carries its own legend.

Two y-scales in the per-run figure is a deliberate exception to keeping one scale
per figure. The progress curve spans the full 0-100% by construction, while the
concordance rates occupy roughly a fifth of that range, so on a shared axis the
rates flatten into a band and the differences the figure exists to show
disappear. The split is kept honest three ways: every axis label and legend entry
marks its side as `(left)` or `(right)`, so no curve can be read against the
wrong scale; only the left axis carries gridlines, so a crossing point is never
offered as a comparison between the two scales; and the left axis keeps its fixed
0-100% while only the right one floats, because 0% and 100% mean something for a
normalized reward range and nothing in particular for a clipped sample share.

For the same reason `concordance` autoscales instead of holding 0-100%: its
magnified differences are the point. Read its tick labels for level, since a move
that looks large against a narrowed axis may be small in absolute terms. Every
plotted value is also visible in the adjacent JSON file.

Every plotted curve is smoothed independently with the centered moving-average
window in `plot.smoothing_window`. At the first and last few recorded steps,
the average uses the available in-range points. The original unsmoothed curve
is retained as a same-color transparent background trace; use `1` when the
foreground should equal the raw values.

Set `output.cache_mode: reuse` to skip reward-pickle analysis and redraw every
figure directly from the per-figure JSON files indexed by `metadata.json`. If
the index or any listed JSON is absent, use `regenerate` first. A JSON whose
`spec_version` is unsupported is rejected rather than being drawn with changed
semantics. The current renderer writes version 6 and still reads versions 1-5.

The former aggregate `plot_data.json` and `metrics.csv` are no longer written:
they repeated dataset, run, reward, and metric identifiers on every point and
largely duplicated each other. After a successful `regenerate` writes the new
figure JSON files, those two exact legacy files are removed from the output
directory. Other images, metadata, and unrelated files are left untouched.
