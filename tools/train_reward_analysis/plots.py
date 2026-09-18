# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, dependency-light figures for reward-concordance trajectories."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# The two signed halves of full concordance, and the aggregate reward-progress
# curve that shares their axes. These are the opening three slots of the
# validated reference palette, whose slot *order* is what keeps neighbouring
# series separable under colour-vision deficiency; that opening three validates
# on every pair, not only adjacent ones, so any of the three reads against any
# other. Concordance direction keeps its colour in both training-progress
# figures, so the two stay mutually readable. The slots sit below 3:1 contrast on
# a light surface, so both figures ship a legend and every plotted value also
# appears in metrics.csv.
_CONCORDANCE_METRICS = (
    ("positive_fully_concordant_sample_rate", "positive concordant rate"),
    ("negative_fully_concordant_sample_rate", "negative concordant rate"),
)
_CONCORDANCE_METRIC_NAMES = {name for name, _ in _CONCORDANCE_METRICS}
_CONCORDANCE_COLORS = ("#2a78d6", "#eb6834")  # blue, orange
_REWARD_PROGRESS_COLOR = "#1baf7a"  # aqua


def plot_per_reward_conflict_score_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one standardized conflict-score trajectory figure for each reward.

    Args:
        rows: Tidy metric rows produced by the offline analysis.
        output_dir: Directory that receives combination-specific figures.
        smoothing_window: Positive odd number of adjacent recorded steps used
            for centered moving-average smoothing. ``1`` disables smoothing.
    """
    by_dataset_reward: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] != "per_reward_conflict_score":
            continue
        reward = str(row["reward"])
        if reward:
            by_dataset_reward[(str(row.get("dataset", "unknown_dataset")), reward)].append(row)

    for (dataset, reward), reward_rows in by_dataset_reward.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for run_index, (label, line_rows) in enumerate(sorted(_group_by_run(reward_rows).items())):
            raw_steps, raw_values = _series(line_rows)
            raw_line = axis.plot(
                raw_steps,
                raw_values,
                alpha=0.22,
                linewidth=1.0,
                label="_nolegend_",
                zorder=1,
            )[0]
            steps, values = _smoothed_series(line_rows, smoothing_window)
            axis.plot(
                steps,
                values,
                color=raw_line.get_color(),
                marker="o",
                markersize=3,
                label=label,
                zorder=2,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"{reward} conflict score [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Mean standardized conflict score")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.90))
        figure.tight_layout()
        path = (
            Path(output_dir)
            / _filename_component(dataset)
            / "per_reward_conflict_score"
            / f"{_filename_component(reward)}.{plot_format}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_reward_concordance_lower_bound_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one sample-wise reward-concordance lower-bound figure per reward set.

    Args:
        rows: Tidy metric rows produced by the offline analysis.
        output_dir: Directory that receives combination-specific figures.
        smoothing_window: Positive odd number of adjacent recorded steps used
            for centered moving-average smoothing. ``1`` disables smoothing.
    """
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "reward_concordance_lower_bound":
            by_dataset[str(row.get("dataset", "unknown_dataset"))].append(row)

    for dataset, combination_rows in by_dataset.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for label, line_rows in sorted(_group_by_run(combination_rows).items()):
            raw_steps, raw_values = _series(line_rows)
            raw_line = axis.plot(
                raw_steps,
                raw_values,
                alpha=0.22,
                linewidth=1.0,
                label="_nolegend_",
                zorder=1,
            )[0]
            steps, values = _smoothed_series(line_rows, smoothing_window)
            axis.plot(
                steps,
                values,
                color=raw_line.get_color(),
                marker="o",
                markersize=3,
                label=label,
                zorder=2,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"Reward-concordance lower bound [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Mean weakest standardized conflict score")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = (
            Path(output_dir)
            / _filename_component(dataset)
            / f"reward_concordance_lower_bound.{plot_format}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_standardized_reward_covariance_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one standardized reward-pair covariance trajectory per reward pair.

    Each curve is a prompt-group macro-average at one recorded training step.
    Every off-diagonal reward pair receives its own figure; all configured runs
    are overlaid in that figure for direct step-by-step comparison.
    """
    by_dataset_combination_pair: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        if row["metric"] != "standardized_reward_covariance":
            continue
        pair = str(row.get("reward_pair", ""))
        if pair:
            by_dataset_combination_pair[
                (
                    str(row.get("dataset", "unknown_dataset")),
                    str(row["reward_combination"]),
                    pair,
                )
            ].append(row)

    run_styles = ["-", "--", ":", "-."]
    for (dataset, combination, pair), pair_rows in sorted(by_dataset_combination_pair.items()):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for index, (label, line_rows) in enumerate(sorted(_group_by_run(pair_rows).items())):
            raw_steps, raw_values = _series(line_rows)
            linestyle = run_styles[index % len(run_styles)]
            axis.plot(
                raw_steps,
                raw_values,
                color=f"C{index % 10}",
                alpha=0.18,
                linewidth=0.9,
                linestyle=linestyle,
                label="_nolegend_",
                zorder=1,
            )
            steps, values = _smoothed_series(line_rows, smoothing_window)
            axis.plot(
                steps,
                values,
                color=f"C{index % 10}",
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=label,
                zorder=2,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"Standardized reward covariance: {pair.replace('__', ' + ')}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Prompt-local standardized covariance")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        filename = f"{_filename_component(pair)}.{plot_format}"
        path = (
            Path(output_dir)
            / _filename_component(dataset)
            / "standardized_reward_covariance"
            / filename
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_per_reward_disagreement_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one per-reward disagreement-rate trajectory figure for each reward."""
    by_dataset_reward: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] != "per_reward_disagreement":
            continue
        reward = str(row["reward"])
        if reward:
            by_dataset_reward[(str(row.get("dataset", "unknown_dataset")), reward)].append(row)

    for (dataset, reward), reward_rows in by_dataset_reward.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for run_index, (label, line_rows) in enumerate(sorted(_group_by_run(reward_rows).items())):
            raw_steps, raw_values = _series(line_rows)
            raw_line = axis.plot(
                raw_steps,
                raw_values,
                alpha=0.22,
                linewidth=1.0,
                label="_nolegend_",
                zorder=1,
            )[0]
            steps, values = _smoothed_series(line_rows, smoothing_window)
            axis.plot(
                steps,
                values,
                color=raw_line.get_color(),
                marker="o",
                markersize=3,
                label=label,
                zorder=2,
            )
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{reward} disagreement [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Per-reward disagreement rate")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.90))
        figure.tight_layout()
        path = (
            Path(output_dir)
            / _filename_component(dataset)
            / "per_reward_disagreement"
            / f"{_filename_component(reward)}.{plot_format}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_agreement_count_distribution_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one sample agreement-count distribution figure per reward combination.

    Per-reward disagreement rates are marginals: they cannot separate polarized
    conflict (few samples opposing many rewards at once) from diffuse conflict
    (many samples opposing a single reward). The agreement-count distribution
    keeps that joint structure, so a rising ``c = n_rewards`` curve beside a
    rising ``c = 1`` curve means conflict is spreading rather than deepening.

    Bins that no run ever populates are left out: with the strictly positive
    weights this tool accepts, ``c = 0`` is unreachable, because a sample below
    the group mean on every reward is also below the mean of the weighted scalar.

    One figure is written per dataset, because a dataset fixes the reward set
    (and therefore ``K``) that its runs were trained against. A dataset carrying
    more than one reward combination is rejected instead of mixing incomparable
    counts into one figure.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["metric"]).startswith("agreement_count_c"):
            grouped[str(row.get("dataset", "unknown_dataset"))].append(row)

    for dataset, dataset_rows in grouped.items():
        _reject_mixed_reward_combinations(dataset, dataset_rows)
        counts = sorted(
            {int(str(row["reward_pair"]).removeprefix("count_")) for row in dataset_rows}
        )
        by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for label, line_rows in sorted(_group_by_run(dataset_rows).items()):
            bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in line_rows:
                bins[str(row["reward_pair"])].append(row)
            by_run[label] = bins
        active_counts = [
            agreeing_count
            for agreeing_count in counts
            if any(
                float(row["value"]) != 0.0
                for bins in by_run.values()
                for row in bins.get(f"count_{agreeing_count}", ())
            )
        ]
        figure, axis = plt.subplots(figsize=(9, 5))
        legend_handles = []
        for run_index, (label, bins) in enumerate(by_run.items()):
            style = ("-", "--", "-.", ":")[run_index % 4]
            for position, agreeing_count in enumerate(active_counts):
                count_rows = bins.get(f"count_{agreeing_count}")
                if not count_rows:
                    continue
                color = f"C{position % 10}"
                legend_label = f"{label} | c={agreeing_count}"
                steps, values = _smoothed_series(count_rows, smoothing_window)
                axis.plot(
                    steps,
                    values,
                    linestyle=style,
                    color=color,
                    marker="o",
                    markersize=2.5,
                    # Series share a color within one agreeing count, so run
                    # index is carried by the dash pattern alone. Marking every
                    # step would fill the dash gaps and erase that cue.
                    markevery=max(1, len(steps) // 12),
                    label="_nolegend_",
                )
                legend_handles.append(
                    Line2D([], [], linestyle=style, color=color, linewidth=1.5, label=legend_label)
                )
        axis.set_title(f"Sample concordance-count distribution [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Sample fraction")
        axis.grid(alpha=0.25)
        # Marker-free handles with a longer line: a handle copied from the series
        # carries its centre marker, which covers the single dash gap that fits
        # in a short handle, so both runs would look identical in the legend.
        axis.legend(handles=legend_handles, fontsize=7, loc="best", handlelength=2.8)
        figure.tight_layout()
        path = Path(output_dir) / _filename_component(dataset) / f"agreement_count.{plot_format}"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_agreement_count_expectation_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one mean agreement-count figure per reward combination.

    The expected count is exactly ``n_rewards`` minus the summed per-reward
    disagreement rates, so this figure restates the marginal disagreement rows.
    It earns its place as a single bounded scalar: read against the dashed
    ceiling, it answers "how far from full concordance is this group" without
    the reader comparing several marginal curves. The ceiling is drawn only when
    the matching agreement-count bins are present in ``rows``.
    """
    rows = list(rows)
    ceilings: dict[str, int] = {}
    for row in rows:
        if str(row["metric"]).startswith("agreement_count_c"):
            dataset = str(row.get("dataset", "unknown_dataset"))
            agreeing_count = int(str(row["reward_pair"]).removeprefix("count_"))
            ceilings[dataset] = max(ceilings.get(dataset, 0), agreeing_count)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "mean_agreement_count":
            grouped[str(row.get("dataset", "unknown_dataset"))].append(row)

    for dataset, dataset_rows in grouped.items():
        _reject_mixed_reward_combinations(dataset, dataset_rows)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for label, line_rows in sorted(_group_by_run(dataset_rows).items()):
            raw_steps, raw_values = _series(line_rows)
            raw_line = axis.plot(
                raw_steps,
                raw_values,
                alpha=0.22,
                linewidth=1.0,
                label="_nolegend_",
                zorder=1,
            )[0]
            steps, values = _smoothed_series(line_rows, smoothing_window)
            axis.plot(
                steps,
                values,
                color=raw_line.get_color(),
                marker="o",
                markersize=3,
                label=label,
                zorder=2,
            )
        ceiling = ceilings.get(dataset)
        if ceiling is not None:
            axis.axhline(
                ceiling,
                color="black",
                linewidth=0.8,
                alpha=0.4,
                linestyle="--",
                label=f"all {ceiling} rewards concordant",
            )
        axis.set_title(f"Mean concordance count [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Mean concordant reward count per sample")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = (
            Path(output_dir)
            / _filename_component(dataset)
            / f"agreement_count_expectation.{plot_format}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_per_reward_weighted_advantage_sign_trajectories(rows, output_dir, smoothing_window=5, plot_format="png"):
    """Plot standardized advantage means split by sample weight and sign."""
    names = {"weight_ge_1_adv_positive":"weight≥1, adv>0", "weight_ge_1_adv_negative":"weight≥1, adv<0", "weight_lt_1_adv_positive":"weight<1, adv>0", "weight_lt_1_adv_negative":"weight<1, adv<0", "adv_positive":"adv>0", "adv_negative":"adv<0"}
    grouped = defaultdict(list)
    for row in rows:
        if row["metric"] in names and row.get("reward"):
            grouped[(str(row.get("dataset", "unknown_dataset")), str(row["reward"]))].append(row)
    for (dataset, reward), reward_rows in grouped.items():
        figure, axis = plt.subplots(figsize=(9, 5))
        for run_index, (label, line_rows) in enumerate(sorted(_group_by_run(reward_rows).items())):
            by_metric = defaultdict(list)
            for row in line_rows: by_metric[row["metric"]].append(row)
            for metric, legend in names.items():
                if metric in by_metric:
                    metric_rows = sorted(by_metric[metric], key=lambda row: int(row["step"]))
                    steps, values = _smoothed_series(metric_rows, smoothing_window)
                    counts = np.asarray([float(row.get("sample_count", np.nan)) for row in metric_rows], dtype=float)
                    size = np.clip(1.5 + 0.8 * np.sqrt(np.maximum(counts, 0.0)), 2.0, 6.0)
                    style = ("-", "--", "-.", ":")[run_index % 4]
                    width = 2.5 if metric.startswith("weight_ge_1") else 1.3
                    line = axis.plot(steps, values, linestyle=style, linewidth=width, label=f"{label} | {legend}")[0]
                    axis.scatter(steps, values, s=np.square(size), color=line.get_color(), alpha=0.85, zorder=3)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"{reward} weighted advantage sign [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Mean standardized advantage")
        axis.grid(alpha=0.25); axis.legend(fontsize=7, loc="best")
        figure.tight_layout()
        path = Path(output_dir) / _filename_component(dataset) / "per_reward_weighted_advantage_sign" / f"{_filename_component(reward)}.{plot_format}"
        path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def plot_per_reward_weighted_advantage_count_trajectories(rows, output_dir, smoothing_window=5, plot_format="png"):
    """Plot average sample counts split by sample-weight and advantage sign."""
    names = {"sample_count_weight_ge_1_adv_positive":"weight≥1, adv>0", "sample_count_weight_ge_1_adv_negative":"weight≥1, adv<0", "sample_count_weight_ge_1_adv_zero":"weight≥1, adv=0", "sample_count_weight_lt_1_adv_positive":"weight<1, adv>0", "sample_count_weight_lt_1_adv_negative":"weight<1, adv<0", "sample_count_weight_lt_1_adv_zero":"weight<1, adv=0", "sample_count_adv_positive":"adv>0", "sample_count_adv_negative":"adv<0", "sample_count_adv_zero":"adv=0"}
    grouped = defaultdict(list)
    for row in rows:
        if row["metric"] in names and row.get("reward"):
            grouped[(str(row.get("dataset", "unknown_dataset")), str(row["reward"]))].append(row)
    for (dataset, reward), reward_rows in grouped.items():
        figure, axis = plt.subplots(figsize=(9, 5))
        for run_index, (label, line_rows) in enumerate(sorted(_group_by_run(reward_rows).items())):
            by_metric = defaultdict(list)
            for row in line_rows: by_metric[row["metric"]].append(row)
            for metric, legend in names.items():
                if metric in by_metric:
                    steps, values = _smoothed_series(by_metric[metric], smoothing_window)
                    style = ("-", "--", "-.", ":")[run_index % 4]
                    width = 2.5 if "weight_ge_1" in metric else 1.3
                    axis.plot(steps, values, linestyle=style, linewidth=width, marker="o", markersize=2.5, label=f"{label} | {legend}")
        axis.set_title(f"{reward} weighted advantage sample count [{dataset}]")
        axis.set_xlabel("Training step"); axis.set_ylabel("Mean sample count per group"); axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best"); figure.tight_layout()
        path = Path(output_dir) / _filename_component(dataset) / "per_reward_weighted_advantage_count" / f"{_filename_component(reward)}.{plot_format}"
        path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def plot_per_reward_bottleneck_rate_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one per-reward bottleneck-rate trajectory figure for each reward."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "per_reward_bottleneck_rate" and row.get("reward"):
            grouped[(str(row.get("dataset", "unknown_dataset")), str(row["reward"]))].append(row)
    for (dataset, reward), reward_rows in grouped.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for label, line_rows in sorted(_group_by_run(reward_rows).items()):
            raw_steps, raw_values = _series(line_rows)
            raw_line = axis.plot(raw_steps, raw_values, alpha=0.22, linewidth=1.0, label="_nolegend_")[0]
            steps, values = _smoothed_series(line_rows, smoothing_window)
            axis.plot(steps, values, color=raw_line.get_color(), marker="o", markersize=3, label=label)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{reward} bottleneck rate [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Per-reward bottleneck rate")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.90))
        figure.tight_layout()
        path = Path(output_dir) / _filename_component(dataset) / "per_reward_bottleneck_rate" / f"{_filename_component(reward)}.{plot_format}"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)
        plt.close(figure)


def plot_run_training_progress_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one reward-progress and full-concordance figure per run and dataset.

    Rewards are aggregated into a single progress curve rather than drawn
    separately: each reward is mapped onto 0-100% of its own observed range
    within the run, and those per-reward percentages are averaged at each step.
    A reward set of any size therefore reads as one curve, which is the point —
    a per-reward version grows unreadable as rewards are added.

    Averaging percentages rather than raw levels is what makes the aggregate
    meaningful: a reward scored 0-1 and one scored 0-5 contribute equally, so
    the curve tracks how far each reward has moved through its own range. Read
    it for shape, not for level; a single outlier step sets a reward's 100% mark,
    and a reward that never moves contributes a flat 0%.

    The two concordance curves are already shares of samples, so they are plotted
    at their own value and are never rescaled. All three curves are bounded
    0-100% by construction with no free scale parameter, so one axis carries
    them and no second y-scale is needed.
    """
    for dataset, rewards, by_run in _group_progress_rows(rows):
        for label, keyed in by_run.items():
            steps, progress = _mean_reward_progress(keyed, rewards)
            if steps.size == 0:
                continue
            figure, progress_axis = plt.subplots(figsize=(8, 4.5))
            rate_axis = progress_axis.twinx()
            _plot_percent_curve(
                progress_axis,
                steps,
                progress,
                _REWARD_PROGRESS_COLOR,
                "-",
                "o",
                smoothing_window,
            )
            handles = [
                Line2D(
                    [],
                    [],
                    color=_REWARD_PROGRESS_COLOR,
                    linewidth=1.8,
                    label="mean reward progress (left)",
                )
            ]
            for offset, (metric, legend_name) in enumerate(_CONCORDANCE_METRICS):
                fraction_rows = keyed.get((metric, ""))
                if not fraction_rows:
                    continue
                _plot_percent_rows(
                    rate_axis,
                    fraction_rows,
                    _percent_of_sample_share,
                    _CONCORDANCE_COLORS[offset],
                    "-",
                    "o",
                    smoothing_window,
                )
                handles.append(
                    Line2D(
                        [],
                        [],
                        color=_CONCORDANCE_COLORS[offset],
                        linewidth=1.8,
                        label=f"{legend_name} (right)",
                    )
                )
            _lock_percent_axis(progress_axis, "Reward progress % (left, own min-max)")
            _autoscale_rate_axis(rate_axis, "Concordant samples % (right)")
            progress_axis.set_title(f"Reward progress and concordance rate [{dataset}] | {label}")
            progress_axis.set_xlabel("Training step")
            progress_axis.legend(handles=handles, fontsize=8, loc="best")
            figure.tight_layout()
            path = (
                Path(output_dir)
                / _filename_component(dataset)
                / "training_progress"
                / f"{_filename_component(label)}.{plot_format}"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=180)
            plt.close(figure)


def plot_concordance_rate_trajectories(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    smoothing_window: int = 5,
    plot_format: str = "png",
) -> None:
    """Write one all-run full-concordance figure per dataset.

    This figure drops the reward-progress curve so the runs can carry the axes
    alone: with no other family competing for the same channels, run identity is
    read from colour, dash pattern, and marker together rather than from the dash
    pattern by itself. Concordance direction keeps the colours it has in the
    per-run figure, so the two figures in this folder stay mutually readable.
    """
    for dataset, _rewards, by_run in _group_progress_rows(rows):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        handles = []
        for run_index, (label, keyed) in enumerate(by_run.items()):
            linestyle = ("-", "--", "-.", ":")[run_index % 4]
            marker = ("o", "s", "^", "D")[run_index % 4]
            for offset, (metric, legend_name) in enumerate(_CONCORDANCE_METRICS):
                fraction_rows = keyed.get((metric, ""))
                if not fraction_rows:
                    continue
                _plot_percent_rows(
                    axis,
                    fraction_rows,
                    _percent_of_sample_share,
                    _CONCORDANCE_COLORS[offset],
                    linestyle,
                    marker,
                    smoothing_window,
                )
                handles.append(
                    Line2D(
                        [],
                        [],
                        color=_CONCORDANCE_COLORS[offset],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.8,
                        label=f"{label} | {legend_name}",
                    )
                )
        _autoscale_rate_axis(axis, "Percent of concordant samples")
        axis.grid(alpha=0.25)
        axis.set_title(f"Concordant sample rate [{dataset}]")
        axis.set_xlabel("Training step")
        axis.legend(handles=handles, fontsize=8, loc="best")
        figure.tight_layout()
        path = (
            Path(output_dir)
            / _filename_component(dataset)
            / "training_progress"
            / f"concordance.{plot_format}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def _group_progress_rows(
    rows: Iterable[dict[str, Any]],
) -> list[tuple[str, list[str], dict[str, dict[tuple[str, str], list[dict[str, Any]]]]]]:
    """Group progress rows by dataset and then by run, keyed by metric and reward.

    A dataset fixes the reward set its runs were trained against, so a dataset
    carrying two combinations is rejected rather than averaged into one curve.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "per_reward_mean_reward" or row["metric"] in _CONCORDANCE_METRIC_NAMES:
            grouped[str(row.get("dataset", "unknown_dataset"))].append(row)

    result = []
    for dataset, dataset_rows in grouped.items():
        _reject_mixed_reward_combinations(dataset, dataset_rows)
        rewards = sorted(
            {
                str(row["reward"])
                for row in dataset_rows
                if row["metric"] == "per_reward_mean_reward" and row["reward"]
            }
        )
        by_run: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
        for label, line_rows in sorted(_group_by_run(dataset_rows).items()):
            keyed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in line_rows:
                keyed[(str(row["metric"]), str(row["reward"]))].append(row)
            by_run[label] = keyed
        result.append((dataset, rewards, by_run))
    return result


def _mean_reward_progress(
    keyed: dict[tuple[str, str], list[dict[str, Any]]],
    rewards: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Average every reward's own 0-100% progress into one curve per step.

    Each reward is normalized to its own range before averaging, so a reward
    scored 0-1 and one scored 0-5 contribute equally to the aggregate. A step
    where a reward is absent is averaged over the rewards that are present.
    """
    collected: dict[int, list[float]] = defaultdict(list)
    for reward in rewards:
        reward_rows = keyed.get(("per_reward_mean_reward", reward))
        if not reward_rows:
            continue
        steps, values = _series(reward_rows)
        for step, percent in zip(steps, _percent_of_own_range(values)):
            collected[int(step)].append(float(percent))
    ordered = sorted(collected)
    return (
        np.asarray(ordered, dtype=np.int64),
        np.asarray([float(np.mean(collected[step])) for step in ordered]),
    )


def _plot_percent_curve(
    axis: Any,
    steps: np.ndarray,
    percent: np.ndarray,
    color: str,
    linestyle: str,
    marker: str,
    smoothing_window: int,
) -> None:
    """Draw one faint raw trace with its smoothed foreground on a percent axis."""
    axis.plot(
        steps,
        percent,
        color=color,
        linestyle=linestyle,
        alpha=0.18,
        linewidth=0.9,
        label="_nolegend_",
        zorder=1,
    )
    axis.plot(
        steps,
        _moving_average(percent, smoothing_window),
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
        marker=marker,
        markersize=3,
        # Marking every step would fill the dash gaps and hide the cue that
        # carries run identity.
        markevery=max(1, len(steps) // 12),
        label="_nolegend_",
        zorder=2,
    )


def _plot_percent_rows(
    axis: Any,
    rows: Iterable[dict[str, Any]],
    transform: Any,
    color: str,
    linestyle: str,
    marker: str,
    smoothing_window: int,
) -> None:
    """Draw one percent series read from its own metric rows."""
    steps, values = _series(rows)
    _plot_percent_curve(
        axis,
        steps,
        transform(values),
        color,
        linestyle,
        marker,
        smoothing_window,
    )


def _lock_percent_axis(axis: Any, label: str) -> None:
    """Fix an axis to 0-100% so a later draw cannot widen it.

    Reward progress is anchored at both ends by its own normalization — 0% is a
    reward's lowest observed value and 100% its highest — so that scale carries
    meaning and is worth fixing. Without disabling autoscale, the next draw
    unstales the pending autoscale and reapplies matplotlib's default margins,
    which reads as data poking past 0-100%.
    """
    axis.set_autoscaley_on(False)
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel(label)
    axis.grid(alpha=0.25)


def _autoscale_rate_axis(axis: Any, label: str) -> None:
    """Let a sample-rate axis follow its own band instead of a fixed 0-100%.

    Concordance rates occupy a narrow band — roughly a fifth of the axis in these
    runs — so pinning them to 0-100% flattens exactly the differences the figure
    exists to show. Their bounds are arbitrary rather than anchored, which is why
    the rate axis is the one that gets to float while progress stays fixed.

    A twin rate axis is left without a grid: gridlines that appear to serve both
    sides invite reading a crossing point as a comparison of the two scales,
    when only the fixed left axis supports that.
    """
    axis.set_autoscaley_on(True)
    axis.set_ylabel(label)


def _percent_of_own_range(values: np.ndarray) -> np.ndarray:
    """Map one series onto 0-100% of its own observed range.

    The mapping is defined by the raw series, before smoothing, so the smoothed
    trace stays inside 0-100%. A constant series has no range to express
    progress against and becomes flat 0%, following the zero-variance
    convention used elsewhere in this tool.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values)
    return 100.0 * (values - low) / (high - low)


def _percent_of_sample_share(values: np.ndarray) -> np.ndarray:
    """Show an already-normalized sample share as a percentage."""
    return 100.0 * np.asarray(values, dtype=np.float64)


def _reject_mixed_reward_combinations(dataset: str, rows: Iterable[dict[str, Any]]) -> None:
    """Fail fast when one dataset carries more than one reward combination.

    A dataset fixes the reward set its runs were trained against, so its samples
    are only comparable while that set — and therefore the agreeing-count scale —
    stays constant.
    """
    combinations = sorted({str(row["reward_combination"]) for row in rows})
    if len(combinations) > 1:
        raise ValueError(
            f"Dataset {dataset!r} carries more than one reward combination {combinations}; "
            "agreement-count statistics are only comparable within a single reward set."
        )


def _group_by_run(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row["value"])
        if np.isfinite(value):
            grouped[str(row["run_label"])].append(row)
    return grouped


def _series(rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    return (
        np.asarray([int(row["step"]) for row in ordered]),
        np.asarray([float(row["value"]) for row in ordered]),
    )


def _smoothed_series(
    rows: Iterable[dict[str, Any]], smoothing_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return one sorted trajectory with centered moving-average smoothing."""
    steps, values = _series(rows)
    return steps, _moving_average(values, smoothing_window)


def _moving_average(values: np.ndarray, smoothing_window: int) -> np.ndarray:
    """Smooth one already-ordered series with a centered moving average."""
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer.")
    values = np.asarray(values, dtype=np.float64)
    if smoothing_window == 1 or values.size < 2:
        return values
    radius = smoothing_window // 2
    smoothed = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        smoothed[index] = values[start:end].mean()
    return smoothed


def _filename_component(value: str) -> str:
    """Return a portable filename component while preserving readable reward names."""
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
