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

"""Small, dependency-light figures for reward-disagreement trajectories."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_per_reward_disagreement_trajectories(
    rows: Iterable[dict[str, Any]], output_dir: str | Path
) -> None:
    """Write one natural disagreement trajectory figure for each reward.

    Args:
        rows: Tidy metric rows produced by the offline analysis.
        output_dir: Directory that receives combination-specific figures.
    """
    by_combination_reward: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] != "natural_per_reward_disagreement_rate":
            continue
        reward = str(row["reward"])
        if reward:
            by_combination_reward[(str(row["reward_combination"]), reward)].append(row)

    for (combination, reward), reward_rows in by_combination_reward.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for label, line_rows in sorted(_group_by_run(reward_rows).items()):
            steps, values = _series(line_rows)
            axis.plot(steps, values, marker="o", markersize=3, label=label)
        axis.set_title(f"{reward} disagreement: {combination.replace('__', ' + ')}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Natural disagreement rate")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = (
            Path(output_dir)
            / combination
            / "per_reward_disagreement"
            / f"{_filename_component(reward)}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_conflict_mass_trajectories(rows: Iterable[dict[str, Any]], output_dir: str | Path) -> None:
    """Write one Uniform-versus-Effective conflict-mass figure per reward set.

    Args:
        rows: Tidy metric rows produced by the offline analysis.
        output_dir: Directory that receives combination-specific figures.
    """
    metrics = {"natural_conflict_mass", "effective_conflict_mass"}
    by_combination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in metrics:
            by_combination[str(row["reward_combination"])].append(row)

    labels = {
        "natural_conflict_mass": "Uniform conflict mass",
        "effective_conflict_mass": "Effective conflict mass",
    }
    for combination, combination_rows in by_combination.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for (run_label, metric), line_rows in sorted(
            _group_by_run_metric(combination_rows).items()
        ):
            steps, values = _series(line_rows)
            axis.plot(
                steps,
                values,
                linestyle="-" if metric == "natural_conflict_mass" else "--",
                marker="o",
                markersize=3,
                label=f"{run_label} · {labels[metric]}",
            )
        axis.set_title(f"Conflict mass: {combination.replace('__', ' + ')}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Conflict mass")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = Path(output_dir) / combination / "conflict_mass.png"
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


def _group_by_run_metric(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row["value"])
        if np.isfinite(value):
            grouped[(str(row["run_label"]), str(row["metric"]))].append(row)
    return grouped


def _series(rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    return (
        np.asarray([int(row["step"]) for row in ordered]),
        np.asarray([float(row["value"]) for row in ordered]),
    )


def _filename_component(value: str) -> str:
    """Return a portable filename component while preserving readable reward names."""
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
