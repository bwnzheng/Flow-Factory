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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        for run_index, (label, bins) in enumerate(by_run.items()):
            style = ("-", "--", "-.", ":")[run_index % 4]
            for position, agreeing_count in enumerate(active_counts):
                count_rows = bins.get(f"count_{agreeing_count}")
                if not count_rows:
                    continue
                steps, values = _smoothed_series(count_rows, smoothing_window)
                axis.plot(
                    steps,
                    values,
                    linestyle=style,
                    color=f"C{position % 10}",
                    marker="o",
                    markersize=2.5,
                    label=f"{label} | c={agreeing_count}",
                )
        axis.set_title(f"Sample agreement-count distribution [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Sample fraction")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")
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
                label=f"all {ceiling} rewards agree",
            )
        axis.set_title(f"Mean agreement count [{dataset}]")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Mean agreeing reward count per sample")
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
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer.")
    steps, values = _series(rows)
    if smoothing_window == 1 or values.size < 2:
        return steps, values
    radius = smoothing_window // 2
    smoothed = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        smoothed[index] = values[start:end].mean()
    return steps, smoothed


def _filename_component(value: str) -> str:
    """Return a portable filename component while preserving readable reward names."""
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
