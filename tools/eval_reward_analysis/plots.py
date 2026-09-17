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

"""Matplotlib visualizations for aggregate reward covariance metrics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Union

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_covariance_matrix(
    covariance: np.ndarray,
    reward_names: Sequence[str],
    output_path: Union[str, Path],
    title: str,
) -> None:
    """Write an annotated heatmap for one checkpoint's aggregate covariance.

    Args:
        covariance: Square covariance matrix averaged across prompt groups.
        reward_names: Labels for the matrix rows and columns.
        output_path: PNG or PDF path to write.
        title: Figure title identifying the checkpoint and source.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    labels = [str(name) for name in reward_names]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance must be square, got {matrix.shape}.")
    if matrix.shape[0] != len(labels):
        raise ValueError(
            f"covariance dimension ({matrix.shape[0]}) does not match reward_names ({len(labels)})."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must contain only finite values.")

    size = max(4.5, 1.1 * len(labels) + 2.0)
    figure, axis = plt.subplots(figsize=(size, size))
    limit = float(np.max(np.abs(matrix))) or 1.0
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Covariance")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(title)
    axis.set_xlabel("Reward")
    axis.set_ylabel("Reward")
    threshold = limit * 0.55
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3g}",
                ha="center",
                va="center",
                color="white" if abs(matrix[row, column]) > threshold else "black",
            )
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def plot_agreement_count_distribution(
    distribution: Sequence[float],
    output_path: Union[str, Path],
    title: str,
) -> None:
    """Write one run's agreeing-count distribution as a bar chart.

    Args:
        distribution: Sample fraction per agreeing count, indexed by count.
        output_path: PNG or PDF path to write.
        title: Figure title identifying the checkpoint and source.
    """
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError(f"distribution must list counts 0..n_rewards, got shape {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("distribution must be finite and non-negative.")
    # A sample below the prompt mean on every reward is also below the mean of the
    # weighted scalar, so c = 0 is unreachable and its bin is always empty.
    counts = [count for count in range(1, values.size) if values[count] > 0.0]
    figure, axis = plt.subplots(figsize=(max(4.5, 1.0 * len(counts) + 3.0), 4.5))
    axis.bar(
        np.arange(len(counts), dtype=np.float64),
        [values[count] for count in counts],
        width=0.7,
    )
    axis.set_xticks(np.arange(len(counts), dtype=np.float64), [f"c={count}" for count in counts])
    axis.set_xlabel("Agreeing rewards per sample")
    axis.set_ylabel("Sample fraction")
    axis.set_title(title)
    axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def plot_jsr_curves(
    curves: dict[str, Sequence[float]],
    q_grid: Sequence[float],
    output_path: Union[str, Path],
    title: str = "Joint Success Rate",
) -> None:
    """Write JSR curves against the reference-model percentile axis."""
    q = np.asarray(q_grid, dtype=float)
    if q.ndim != 1 or not len(q) or np.any((q < 0) | (q > 1)):
        raise ValueError("q_grid must contain values in [0, 1].")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for name, values in curves.items():
        y = np.asarray(values, dtype=float)
        if y.shape != q.shape or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError(f"Invalid JSR curve for {name!r}.")
        axis.plot(q, y, label=name)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Base-model reference percentile q",
        ylabel="Joint Success Rate",
        title=title,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
