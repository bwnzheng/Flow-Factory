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
