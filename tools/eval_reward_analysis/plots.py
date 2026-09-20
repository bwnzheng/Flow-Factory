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

"""Figures and the figure-shaped data behind them for the eval analysis.

Each builder turns finished metrics into a spec, and the shared renderer in
:mod:`tools.figures` draws that spec. Nothing draws from the metrics directly,
so the spec beside each image is a complete description of it: the image can be
recovered from the data alone, which is what ``figure_mode: reuse`` does.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tools.figures import (
    BAR_SERIES_KIND,
    CATEGORICAL_COLORS,
    Figure,
    FigureAxis,
    FigureLegend,
    FigureMatrix,
    FigureSeries,
    FigureSpec,
    LegendEntry,
    MatrixFigureSpec,
)

# Stems inside one experiment's plots directory.
COVARIANCE_STEM = "covariance_matrix"
AGREEMENT_COUNT_STEM = "agreement_count"
JSR_STEM = "jsr_curves"

# The heatmap annotates every cell, so a cell reads dark-on-light only when its
# magnitude is far from the middle of the colour scale.
_ANNOTATION_COLOR_THRESHOLD = 0.55


def build_covariance_figure(
    covariance: np.ndarray,
    reward_names: Sequence[str],
    title: str,
) -> tuple[str, MatrixFigureSpec]:
    """Build the annotated covariance heatmap for one checkpoint's aggregate.

    Args:
        covariance: Square covariance matrix averaged across prompt groups.
        reward_names: Labels for the matrix rows and columns.
        title: Figure title identifying the checkpoint and source.

    Returns:
        The figure's path stem and its spec.
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
    spec = MatrixFigureSpec(
        title=title,
        matrix=FigureMatrix(
            values=matrix.tolist(),
            row_labels=labels,
            column_labels=labels,
            colorbar_label="Covariance",
            colormap="RdBu_r",
            symmetric_limits=True,
            annotation_format=".3g",
            annotation_color_threshold=_ANNOTATION_COLOR_THRESHOLD,
            x_label="Reward",
            y_label="Reward",
        ),
        figsize=[size, size],
    )
    return COVARIANCE_STEM, spec


def build_agreement_count_figure(
    distribution: Sequence[float],
    title: str,
) -> tuple[str, FigureSpec]:
    """Build one run's agreeing-count distribution as a labelled bar chart.

    Args:
        distribution: Sample fraction per agreeing count, indexed by count.
        title: Figure title identifying the checkpoint and source.

    Returns:
        The figure's path stem and its spec.
    """
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError(f"distribution must list counts 0..n_rewards, got shape {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("distribution must be finite and non-negative.")
    # A sample below the prompt mean on every reward is also below the mean of the
    # weighted scalar, so c = 0 is unreachable and its bin is always empty.
    counts = [count for count in range(1, values.size) if values[count] > 0.0]
    heights = [float(values[count]) for count in counts]
    positions = [float(index) for index in range(len(counts))]
    top = max(heights) if heights else 0.0
    spec = FigureSpec(
        title=title,
        x_label="Agreeing rewards per sample",
        left=FigureAxis(
            label="Sample fraction",
            limits=[0.0, top * 1.15 if top > 0.0 else 1.0],
            grid=True,
            grid_axis="y",
        ),
        series=[
            FigureSeries(
                label="sample fraction",
                points=[[position, height] for position, height in zip(positions, heights)],
                color=CATEGORICAL_COLORS[0],
                kind=BAR_SERIES_KIND,
                value_labels=True,
                value_label_format=".3f",
                value_label_fontsize=8.0,
            )
        ],
        figsize=[max(4.5, 1.0 * len(counts) + 3.0), 4.5],
        x_ticks=positions,
        x_tick_labels=[f"c={count}" for count in counts],
    )
    return AGREEMENT_COUNT_STEM, spec


def build_jsr_figure(
    curves: dict[str, Sequence[float]],
    q_grid: Sequence[float],
    title: str = "Joint Success Rate",
) -> tuple[str, FigureSpec]:
    """Build the JSR curves against the reference-model percentile axis.

    Args:
        curves: One JSR curve per compared run, labelled by run.
        q_grid: Reference percentile axis shared by every curve.
        title: Figure title.

    Returns:
        The figure's path stem and its spec.
    """
    q = np.asarray(q_grid, dtype=float)
    if q.ndim != 1 or not len(q) or np.any((q < 0) | (q > 1)):
        raise ValueError("q_grid must contain values in [0, 1].")
    series: list[FigureSeries] = []
    entries: list[LegendEntry] = []
    for index, (name, values) in enumerate(curves.items()):
        y = np.asarray(values, dtype=float)
        if y.shape != q.shape or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError(f"Invalid JSR curve for {name!r}.")
        color = CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)]
        points = [[float(x), float(value)] for x, value in zip(q, y)]
        series.append(
            FigureSeries(
                label=name,
                points=points,
                color=color,
                linewidth=1.8,
                markersize=4.0,
                # The default q grid has 101 points, so a marker per point would
                # read as a solid band rather than a line.
                markevery=max(1, len(points) // 12),
                faint_raw_trace=False,
            )
        )
        entries.append(LegendEntry(label=name, color=color, linewidth=1.8, marker=""))
    spec = FigureSpec(
        title=title,
        x_label="Base-model reference percentile q",
        left=FigureAxis(label="Joint Success Rate", limits=[0.0, 1.0], grid=True),
        series=series,
        legend=FigureLegend(entries=entries),
        smoothing_window=1,
        figsize=[7.0, 4.5],
        x_limits=[0.0, 1.0],
    )
    return JSR_STEM, spec


def figures_by_stem(outputs: Sequence[tuple[str, Figure]]) -> dict[str, Figure]:
    """Index built figures by stem, rejecting duplicate names."""
    stems = [stem for stem, _ in outputs]
    if len(stems) != len(set(stems)):
        duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
        raise ValueError(f"Figure builders produced duplicate output stems: {duplicates}")
    return dict(outputs)
