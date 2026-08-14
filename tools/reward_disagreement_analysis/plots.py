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
    rows: Iterable[dict[str, Any]], output_dir: str | Path, smoothing_window: int = 5
) -> None:
    """Write one raw conflict-score trajectory figure for each reward.

    Args:
        rows: Tidy metric rows produced by the offline analysis.
        output_dir: Directory that receives combination-specific figures.
        smoothing_window: Positive odd number of adjacent recorded steps used
            for centered moving-average smoothing. ``1`` disables smoothing.
    """
    by_combination_reward: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] != "per_reward_conflict_score":
            continue
        reward = str(row["reward"])
        if reward:
            by_combination_reward[(str(row["reward_combination"]), reward)].append(row)

    for (combination, reward), reward_rows in by_combination_reward.items():
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
        axis.set_title(f"{reward} conflict score: {combination.replace('__', ' + ')}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Mean raw conflict score")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.90))
        figure.tight_layout()
        path = (
            Path(output_dir)
            / combination
            / "per_reward_conflict_score"
            / f"{_filename_component(reward)}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_reward_concordance_lower_bound_trajectories(
    rows: Iterable[dict[str, Any]], output_dir: str | Path, smoothing_window: int = 5
) -> None:
    """Write one sample-wise reward-concordance lower-bound figure per reward set.

    Args:
        rows: Tidy metric rows produced by the offline analysis.
        output_dir: Directory that receives combination-specific figures.
        smoothing_window: Positive odd number of adjacent recorded steps used
            for centered moving-average smoothing. ``1`` disables smoothing.
    """
    by_combination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "reward_concordance_lower_bound":
            by_combination[str(row["reward_combination"])].append(row)

    for combination, combination_rows in by_combination.items():
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
        axis.set_title(f"Reward-concordance lower bound: {combination.replace('__', ' + ')}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Mean weakest raw conflict score")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = Path(output_dir) / combination / "reward_concordance_lower_bound.png"
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
