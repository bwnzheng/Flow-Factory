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


def plot_disagreement_trajectories(rows: Iterable[dict[str, Any]], output_dir: str | Path) -> None:
    """Write one natural/effective disagreement trajectory figure per reward set."""
    by_combination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in {"natural_disagreement_rate", "effective_disagreement_rate"}:
            by_combination[str(row["reward_combination"])].append(row)

    for combination, combination_rows in by_combination.items():
        if not combination_rows:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        grouped = _group_lines(combination_rows)
        for (label, reward, metric), line_rows in sorted(grouped.items()):
            steps, values = _series(line_rows)
            linestyle = "-" if metric == "natural_disagreement_rate" else "--"
            metric_label = "uniform" if metric == "natural_disagreement_rate" else "effective"
            axis.plot(
                steps,
                values,
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=(f"{label} · {reward} · {metric_label}"),
            )
        axis.set_title(f"Reward disagreement: {combination.replace('__', ' + ')}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Disagreement rate")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        path = Path(output_dir) / combination / "disagreement_rates.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def plot_per_reward_disagreement_trajectories(
    rows: Iterable[dict[str, Any]], output_dir: str | Path
) -> None:
    """Write one focused disagreement-rate trajectory figure for each reward."""
    by_combination_reward: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] not in {"natural_disagreement_rate", "effective_disagreement_rate"}:
            continue
        reward = str(row["reward"])
        if reward:
            by_combination_reward[(str(row["reward_combination"]), reward)].append(row)

    for (combination, reward), reward_rows in by_combination_reward.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for (label, _reward, metric), line_rows in sorted(_group_lines(reward_rows).items()):
            steps, values = _series(line_rows)
            linestyle = "-" if metric == "natural_disagreement_rate" else "--"
            mass_label = "Uniform" if metric == "natural_disagreement_rate" else "Effective"
            axis.plot(
                steps,
                values,
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=f"{label} · {mass_label}",
            )
        axis.set_title(f"{reward} disagreement: {combination.replace('__', ' + ')}")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Disagreement rate")
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


def plot_group_metric_trajectories(rows: Iterable[dict[str, Any]], output_dir: str | Path) -> None:
    """Write FCR, valid mass, and disagreement-count trajectory figures."""
    metrics = (
        "natural_fully_concordant_ratio",
        "natural_fully_concordant_positive_ratio",
        "natural_fully_concordant_negative_ratio",
        "natural_fully_valid_ratio",
        "natural_mean_disagreement_count",
        "effective_fully_concordant_ratio",
        "effective_fully_concordant_positive_ratio",
        "effective_fully_concordant_negative_ratio",
        "effective_fully_valid_mass",
        "effective_mean_disagreement_count",
    )
    by_combination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in metrics:
            by_combination[str(row["reward_combination"])].append(row)

    labels = {
        "natural_fully_concordant_ratio": "Uniform FCR",
        "natural_fully_concordant_positive_ratio": "Uniform positive FCR",
        "natural_fully_concordant_negative_ratio": "Uniform negative FCR",
        "natural_fully_valid_ratio": "Uniform fully-valid ratio",
        "natural_mean_disagreement_count": "Uniform mean disagreement count",
        "effective_fully_concordant_ratio": "Effective FCR",
        "effective_fully_concordant_positive_ratio": "Effective positive FCR",
        "effective_fully_concordant_negative_ratio": "Effective negative FCR",
        "effective_fully_valid_mass": "Effective fully-valid mass",
        "effective_mean_disagreement_count": "Effective mean disagreement count",
    }
    for combination, combination_rows in by_combination.items():
        figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        metric_groups = (
            ("natural_fully_concordant_ratio", "effective_fully_concordant_ratio"),
            (
                "natural_fully_concordant_positive_ratio",
                "effective_fully_concordant_positive_ratio",
            ),
            (
                "natural_fully_concordant_negative_ratio",
                "effective_fully_concordant_negative_ratio",
            ),
            ("natural_fully_valid_ratio", "effective_fully_valid_mass"),
            ("natural_mean_disagreement_count", "effective_mean_disagreement_count"),
        )
        for axis, metric_pair in zip(axes.flat, metric_groups):
            metric_rows = [row for row in combination_rows if row["metric"] in metric_pair]
            for (label, _reward, metric), line_rows in sorted(_group_lines(metric_rows).items()):
                steps, values = _series(line_rows)
                linestyle = "-" if metric.startswith("natural_") else "--"
                axis.plot(
                    steps,
                    values,
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                    label=(f"{label} · {labels[metric]}"),
                )
            axis.set_title(" / ".join(labels[metric] for metric in metric_pair))
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
            if "count" not in " ".join(metric_pair):
                axis.set_ylim(-0.02, 1.02)
        axes.flat[-1].axis("off")
        figure.suptitle(f"Prompt-local group metrics: {combination.replace('__', ' + ')}")
        figure.tight_layout()
        path = Path(output_dir) / combination / "group_metrics.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def _group_lines(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row["value"])
        if np.isfinite(value):
            grouped[(str(row["run_label"]), str(row["reward"]), str(row["metric"]))].append(row)
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
