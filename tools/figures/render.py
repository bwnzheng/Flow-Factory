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

"""Draw a figure spec, and nothing else.

Nothing here reads analysis results: every input is a spec, so an image can
always be recovered from the data file beside it. Figure builders live with the
tools that own their metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from tools.figures.spec import (
    BAR_SERIES_KIND,
    Figure,
    FigureAxis,
    FigureFontSizes,
    FigureLegend,
    FigureSeries,
    FigureSpec,
    MatrixFigureSpec,
    validate_spec,
)


def render_figure(
    spec: Figure, output_dir: str | Path, stem: str, plot_format: str = "png"
) -> Path:
    """Draw one spec and save it as ``<stem>.<plot_format>``."""
    validate_spec(spec)
    if isinstance(spec, MatrixFigureSpec):
        return _render_matrix_figure(spec, output_dir, stem, plot_format)
    if spec.left.segments is not None:
        return _render_broken_figure(spec, output_dir, stem, plot_format)
    return _render_continuous_figure(spec, output_dir, stem, plot_format)


def _render_continuous_figure(
    spec: FigureSpec, output_dir: str | Path, stem: str, plot_format: str
) -> Path:
    """Draw the ordinary one-panel layout without changing its established pixels."""
    figure, base_axis = plt.subplots(figsize=tuple(spec.figsize))
    axes = {"left": base_axis}
    if spec.right is not None:
        axes["right"] = base_axis.twinx()
    _set_border_width(axes.values(), spec.border_width)

    for series in spec.series:
        _draw_series(axes[series.axis], series, spec.smoothing_window)
    for hline in spec.hlines:
        base_axis.axhline(
            hline.y,
            color=hline.color,
            linewidth=0.8,
            alpha=hline.alpha,
            linestyle=hline.linestyle,
            label="_nolegend_",
        )

    _configure_axis(
        base_axis,
        spec.left,
        spec.font_sizes.left_y_label,
        spec.font_sizes.left_y_tick,
    )
    if spec.right is not None:
        _configure_axis(
            axes["right"],
            spec.right,
            spec.font_sizes.right_y_label,
            spec.font_sizes.right_y_tick,
        )
    title = base_axis.set_title(spec.title)
    x_label = base_axis.set_xlabel(spec.x_label)
    if spec.font_sizes.title is not None:
        title.set_fontsize(spec.font_sizes.title)
    if spec.font_sizes.x_label is not None:
        x_label.set_fontsize(spec.font_sizes.x_label)
    if spec.font_sizes.x_tick is not None:
        base_axis.tick_params(axis="x", labelsize=spec.font_sizes.x_tick)
    _configure_x_axis(base_axis, spec)
    if spec.legend is not None and spec.legend.entries:
        base_axis.legend(**_legend_kwargs(spec.legend, spec.font_sizes.legend))

    figure.tight_layout()
    if spec.top_margin is not None or spec.bottom_margin is not None:
        adjustments = {}
        if spec.top_margin is not None:
            adjustments["top"] = 1.0 - spec.top_margin
        if spec.bottom_margin is not None:
            adjustments["bottom"] = spec.bottom_margin
        figure.subplots_adjust(**adjustments)
    path = Path(output_dir) / f"{stem}.{plot_format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_broken_figure(
    spec: FigureSpec, output_dir: str | Path, stem: str, plot_format: str
) -> Path:
    """Draw ascending y-axis segments as vertically stacked shared-x panels."""
    left_segments = spec.left.segments
    if left_segments is None:
        raise ValueError("A broken figure requires left.segments.")
    ratios = spec.left.segment_height_ratios or [1.0] * len(left_segments)
    figure, panel_axes = plt.subplots(
        len(left_segments),
        1,
        sharex=True,
        figsize=tuple(spec.figsize),
        gridspec_kw={
            "height_ratios": list(reversed(ratios)),
            "hspace": spec.break_gap,
        },
    )
    left_axes = list(np.asarray(panel_axes, dtype=object).reshape(-1))
    axes: dict[str, list[Axes]] = {"left": left_axes}
    if spec.right is not None:
        axes["right"] = [axis.twinx() for axis in left_axes]
    _set_border_width(
        [axis for axis_group in axes.values() for axis in axis_group],
        spec.border_width,
    )

    for series in spec.series:
        for axis in axes[series.axis]:
            _draw_series(axis, series, spec.smoothing_window)
    for hline in spec.hlines:
        for axis in left_axes:
            axis.axhline(
                hline.y,
                color=hline.color,
                linewidth=0.8,
                alpha=hline.alpha,
                linestyle=hline.linestyle,
                label="_nolegend_",
            )

    _configure_broken_panels(
        left_axes,
        spec.left,
        left_segments,
        spec.font_sizes.left_y_tick,
    )
    if spec.right is not None:
        right_segments = spec.right.segments
        if right_segments is None:
            raise ValueError("A dual-y broken figure requires right.segments.")
        _configure_broken_panels(
            axes["right"],
            spec.right,
            right_segments,
            spec.font_sizes.right_y_tick,
        )

    title = left_axes[0].set_title(spec.title)
    x_label = left_axes[-1].set_xlabel(spec.x_label)
    if spec.font_sizes.title is not None:
        title.set_fontsize(spec.font_sizes.title)
    if spec.font_sizes.x_label is not None:
        x_label.set_fontsize(spec.font_sizes.x_label)
    if spec.font_sizes.x_tick is not None:
        for axis in left_axes:
            axis.tick_params(axis="x", labelsize=spec.font_sizes.x_tick)
    _configure_x_axis(left_axes[-1], spec)
    if spec.legend is not None and spec.legend.entries:
        left_axes[0].legend(**_legend_kwargs(spec.legend, spec.font_sizes.legend))
    left_y_label = figure.supylabel(spec.left.label)
    if spec.font_sizes.left_y_label is not None:
        left_y_label.set_fontsize(spec.font_sizes.left_y_label)
    if spec.right is not None:
        right_y_label = figure.text(
            0.99,
            0.5,
            spec.right.label,
            rotation=270,
            va="center",
            ha="right",
        )
        if spec.font_sizes.right_y_label is not None:
            right_y_label.set_fontsize(spec.font_sizes.right_y_label)

    figure.subplots_adjust(
        left=0.12,
        right=0.88 if spec.right is not None else 0.96,
        bottom=(0.12 if spec.bottom_margin is None else spec.bottom_margin),
        top=(0.9 if spec.top_margin is None else 1.0 - spec.top_margin),
        hspace=spec.break_gap,
    )
    _draw_break_marks(left_axes, spec.break_mark_size, spec.border_width)
    path = Path(output_dir) / f"{stem}.{plot_format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_matrix_figure(
    spec: MatrixFigureSpec, output_dir: str | Path, stem: str, plot_format: str
) -> Path:
    """Draw one annotated heatmap with its colour bar."""
    matrix = np.asarray(spec.matrix.values, dtype=np.float64)
    row_labels = list(spec.matrix.row_labels)
    column_labels = list(spec.matrix.column_labels)
    figsize = (
        tuple(spec.figsize)
        if spec.figsize is not None
        else (max(4.5, 1.1 * len(column_labels) + 2.0),) * 2
    )
    figure, axis = plt.subplots(figsize=figsize)
    _set_border_width([axis], spec.border_width)
    limit = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if limit <= 0.0:
        limit = 1.0
    if spec.matrix.symmetric_limits:
        color_limits = {"vmin": -limit, "vmax": limit}
    else:
        color_limits = {"vmin": float(np.min(matrix)), "vmax": limit}
    image = axis.imshow(matrix, cmap=spec.matrix.colormap, **color_limits)
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label=spec.matrix.colorbar_label,
    )
    axis.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    title = axis.set_title(spec.title)
    if spec.matrix.x_label:
        axis.set_xlabel(spec.matrix.x_label)
    if spec.matrix.y_label:
        axis.set_ylabel(spec.matrix.y_label)
    if spec.font_sizes.title is not None:
        title.set_fontsize(spec.font_sizes.title)
    if spec.font_sizes.x_tick is not None:
        axis.tick_params(axis="x", labelsize=spec.font_sizes.x_tick)
    if spec.font_sizes.left_y_tick is not None:
        axis.tick_params(axis="y", labelsize=spec.font_sizes.left_y_tick)
    if spec.matrix.x_label and spec.font_sizes.x_label is not None:
        axis.xaxis.label.set_fontsize(spec.font_sizes.x_label)
    if spec.matrix.y_label and spec.font_sizes.left_y_label is not None:
        axis.yaxis.label.set_fontsize(spec.font_sizes.left_y_label)
    threshold = limit * spec.matrix.annotation_color_threshold
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:{spec.matrix.annotation_format}}",
                ha="center",
                va="center",
                fontsize=_annotation_fontsize(spec.font_sizes),
                color="white" if abs(value) > threshold else "black",
            )
    figure.tight_layout()
    path = Path(output_dir) / f"{stem}.{plot_format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _annotation_fontsize(font_sizes: FigureFontSizes) -> float | None:
    """Size cell annotations from the tick font size, which is what they sit among."""
    return font_sizes.x_tick if font_sizes.x_tick is not None else font_sizes.left_y_tick


def _configure_x_axis(axis: Axes, spec: FigureSpec) -> None:
    """Apply the optional explicit x extent and tick labelling."""
    if spec.x_limits is not None:
        axis.set_xlim(spec.x_limits[0], spec.x_limits[1])
    if spec.x_ticks is not None:
        labels = spec.x_tick_labels
        axis.set_xticks(spec.x_ticks, labels if labels is not None else None)


def _configure_broken_panels(
    axes: Sequence[Axes],
    spec: FigureAxis,
    segments: Sequence[Sequence[float]],
    tick_fontsize: float | None,
) -> None:
    """Apply segment limits and hide the adjoining panel spines."""
    for index, (axis, limits) in enumerate(zip(axes, reversed(segments))):
        axis.set_ylim(float(limits[0]), float(limits[1]))
        if spec.grid:
            axis.grid(True, alpha=0.25, axis=spec.grid_axis)
        else:
            axis.grid(False)
        if tick_fontsize is not None:
            axis.tick_params(axis="y", labelsize=tick_fontsize)
        if index > 0:
            axis.spines["top"].set_visible(False)
        if index < len(axes) - 1:
            axis.spines["bottom"].set_visible(False)
            axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def _set_border_width(axes: Iterable[Axes], width: float | None) -> None:
    """Apply one border width to every spine without changing the default when omitted."""
    if width is None:
        return
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_linewidth(width)


def _draw_break_marks(axes: Sequence[Axes], size: float, border_width: float | None = None) -> None:
    """Mark every omitted y interval on both sides of the plot."""
    line = {
        "color": "black",
        "clip_on": False,
        "linewidth": 0.8 if border_width is None else border_width,
    }
    for upper_axis, lower_axis in zip(axes[:-1], axes[1:]):
        for x_position in (0.0, 1.0):
            upper_axis.plot(
                (x_position - size, x_position + size),
                (-size, size),
                transform=upper_axis.transAxes,
                **line,
            )
            lower_axis.plot(
                (x_position - size, x_position + size),
                (1.0 - size, 1.0 + size),
                transform=lower_axis.transAxes,
                **line,
            )


def _legend_kwargs(legend: FigureLegend, fontsize: float | None = None) -> dict[str, Any]:
    handles = [
        Line2D(
            [],
            [],
            color=entry.color,
            linestyle=entry.linestyle,
            marker=entry.marker,
            linewidth=entry.linewidth,
            markersize=entry.markersize,
            alpha=entry.alpha,
            label=entry.label,
        )
        for entry in legend.entries
    ]
    kwargs: dict[str, Any] = {
        "handles": handles,
        "fontsize": legend.fontsize if fontsize is None else fontsize,
        "loc": legend.loc,
        "ncol": legend.ncol,
    }
    if legend.anchor is not None:
        kwargs["bbox_to_anchor"] = tuple(legend.anchor)
    if legend.handlelength is not None:
        kwargs["handlelength"] = legend.handlelength
    return kwargs


def _configure_axis(
    axis: Axes,
    spec: FigureAxis,
    label_fontsize: float | None = None,
    tick_fontsize: float | None = None,
) -> None:
    """Apply one axis's limits, label, and grid."""
    if spec.limits is not None:
        axis.set_ylim(spec.limits[0], spec.limits[1])
    label = axis.set_ylabel(spec.label)
    if label_fontsize is not None:
        label.set_fontsize(label_fontsize)
    if tick_fontsize is not None:
        axis.tick_params(axis="y", labelsize=tick_fontsize)
    # Passing line properties alongside False turns the grid *on*, so the two
    # cases stay separate.
    if spec.grid:
        axis.grid(True, alpha=0.25, axis=spec.grid_axis)
    else:
        axis.grid(False)


def _draw_series(axis: Axes, series: FigureSeries, smoothing_window: int) -> None:
    """Draw one bar series, or one line series as a faint raw trace under its smoothed foreground."""
    positions = np.asarray([point[0] for point in series.points], dtype=np.float64)
    values = np.asarray([point[1] for point in series.points], dtype=np.float64)
    if series.kind == BAR_SERIES_KIND:
        axis.bar(
            positions,
            values,
            width=series.bar_width,
            color=series.color,
            label="_nolegend_",
            zorder=series.zorder,
        )
        if series.value_labels:
            for position, value in zip(positions, values):
                axis.text(
                    position,
                    value,
                    f"{value:{series.value_label_format}}",
                    ha="center",
                    va="bottom",
                    fontsize=series.value_label_fontsize,
                )
        return
    if series.faint_raw_trace:
        axis.plot(
            positions,
            values,
            color=series.color,
            linestyle=series.linestyle,
            alpha=series.faint_alpha,
            linewidth=series.faint_linewidth,
            label="_nolegend_",
            zorder=series.faint_zorder,
        )
    smoothed = moving_average(values, smoothing_window)
    line = axis.plot(
        positions,
        smoothed,
        color=series.color,
        linestyle=series.linestyle,
        linewidth=series.linewidth,
        marker=series.marker,
        markersize=series.markersize,
        markevery=series.markevery,
        label="_nolegend_",
        zorder=series.zorder,
    )[0]
    if series.scatter_sizes is not None:
        axis.scatter(
            positions,
            smoothed,
            s=np.square(series.scatter_sizes),
            color=line.get_color(),
            alpha=0.85,
            zorder=3,
        )


def moving_average(values: np.ndarray, smoothing_window: int) -> np.ndarray:
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
