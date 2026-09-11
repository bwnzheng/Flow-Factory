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
        for label, line_rows in sorted(_group_by_run(reward_rows).items()):
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
        for label, line_rows in sorted(_group_by_run(reward_rows).items()):
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
