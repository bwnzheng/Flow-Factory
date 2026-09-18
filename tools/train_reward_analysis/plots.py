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

"""Figures and the figure-shaped data behind them.

Each figure is built in two steps. A builder turns the tidy metric rows into a
:class:`FigureSpec` — the lines, their points, and the styling — and
:func:`render_figure` draws that spec. Nothing draws from the rows directly, so
the spec beside each image is a complete description of it: the image can be
recovered from the data alone, which is what ``cache_mode: reuse`` does.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from tools.train_reward_analysis.figure_spec import (
    FigureAxis,
    FigureHLine,
    FigureLegend,
    FigureSeries,
    FigureSpec,
    LegendEntry,
    validate_spec,
    write_spec,
)

# Fixed-order categorical slots for the training-progress figures. The slot
# *order* is what keeps neighbouring series separable under colour-vision
# deficiency, so slots are taken in sequence and never cycled or reordered; the
# opening three validate on every pair, not only adjacent ones, so any of the
# three reads against any other. The slots sit below 3:1 contrast on a light
# surface, so both figures ship a legend and every plotted value also sits in
# the figure's own data file.
_CATEGORICAL_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

# Which channel carries what differs between the two training-progress figures,
# because they answer different questions. The per-run figure has one run and
# two directions, so direction takes the colour. The all-run figure exists to
# tell runs apart, so the runs take the colour and the marker, and direction
# falls back to the dash pattern — which is also the channel that survives
# greyscale printing.
_CONCORDANCE_METRICS = (
    ("positive_fully_concordant_sample_rate", "positive concordant rate"),
    ("negative_fully_concordant_sample_rate", "negative concordant rate"),
)
_CONCORDANCE_METRIC_NAMES = {name for name, _ in _CONCORDANCE_METRICS}
_CONCORDANCE_DIRECTION_COLORS = (_CATEGORICAL_COLORS[0], _CATEGORICAL_COLORS[1])
_CONCORDANCE_DIRECTION_LINESTYLES = ("-", "--")
_REWARD_PROGRESS_COLOR = _CATEGORICAL_COLORS[2]
_RUN_MARKERS = ("o", "s", "^", "D", "v", "X", "P", "*")

# Style shared by the reward-progress and concordance figures.
_PERCENT_LINEWIDTH = 1.8
_PERCENT_MARKERSIZE = 5.0
_PERCENT_FAINT_ALPHA = 0.18
_PERCENT_FAINT_LINEWIDTH = 0.9

# One spec per figure, addressed by its path stem inside the output directory.
FigureOutput = tuple[str, FigureSpec]


def render_figure(
    spec: FigureSpec, output_dir: str | Path, stem: str, plot_format: str = "png"
) -> Path:
    """Draw one spec and save it as ``<stem>.<plot_format>``."""
    validate_spec(spec)
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
    if spec.legend is not None and spec.legend.entries:
        base_axis.legend(**_legend_kwargs(spec.legend, spec.font_sizes.legend))

    figure.tight_layout()
    if spec.top_margin is not None:
        figure.subplots_adjust(top=1.0 - spec.top_margin)
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
        bottom=0.12,
        top=(0.9 if spec.top_margin is None else 1.0 - spec.top_margin),
        hspace=spec.break_gap,
    )
    _draw_break_marks(left_axes, spec.break_mark_size, spec.border_width)
    path = Path(output_dir) / f"{stem}.{plot_format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


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
            axis.grid(True, alpha=0.25)
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
        axis.grid(True, alpha=0.25)
    else:
        axis.grid(False)


def _draw_series(axis: Axes, series: FigureSeries, smoothing_window: int) -> None:
    """Draw one series as a faint raw trace under its smoothed foreground."""
    steps = np.asarray([point[0] for point in series.points], dtype=np.float64)
    values = np.asarray([point[1] for point in series.points], dtype=np.float64)
    if series.faint_raw_trace:
        axis.plot(
            steps,
            values,
            color=series.color,
            linestyle=series.linestyle,
            alpha=series.faint_alpha,
            linewidth=series.faint_linewidth,
            label="_nolegend_",
            zorder=series.faint_zorder,
        )
    smoothed = _moving_average(values, smoothing_window)
    line = axis.plot(
        steps,
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
            steps,
            smoothed,
            s=np.square(series.scatter_sizes),
            color=line.get_color(),
            alpha=0.85,
            zorder=3,
        )


def build_per_reward_conflict_score_figures(rows: Iterable[dict[str, Any]]) -> list[FigureOutput]:
    """One standardized conflict-score trajectory figure per reward."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] != "per_reward_conflict_score":
            continue
        reward = str(row["reward"])
        if reward:
            grouped[(str(row.get("dataset", "unknown_dataset")), reward)].append(row)

    outputs: list[FigureOutput] = []
    for (dataset, reward), reward_rows in grouped.items():
        series, entries = _run_series(_group_by_run(reward_rows))
        outputs.append(
            (
                f"{_filename_component(dataset)}/per_reward_conflict_score/"
                f"{_filename_component(reward)}",
                FigureSpec(
                    title=f"{reward} conflict score [{dataset}]",
                    x_label="Training step",
                    left=FigureAxis("Mean standardized conflict score"),
                    series=series,
                    hlines=[FigureHLine(0.0)],
                    legend=FigureLegend(entries=entries, loc="upper right", anchor=[1.0, 0.90]),
                ),
            )
        )
    return outputs


def build_reward_concordance_lower_bound_figures(
    rows: Iterable[dict[str, Any]],
) -> list[FigureOutput]:
    """One sample-wise reward-concordance lower-bound figure per reward set."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == "reward_concordance_lower_bound":
            grouped[str(row.get("dataset", "unknown_dataset"))].append(row)

    outputs: list[FigureOutput] = []
    for dataset, combination_rows in grouped.items():
        series, entries = _run_series(_group_by_run(combination_rows))
        outputs.append(
            (
                f"{_filename_component(dataset)}/reward_concordance_lower_bound",
                FigureSpec(
                    title=f"Reward-concordance lower bound [{dataset}]",
                    x_label="Training step",
                    left=FigureAxis("Mean weakest standardized conflict score"),
                    series=series,
                    hlines=[FigureHLine(0.0)],
                    legend=FigureLegend(entries=entries),
                ),
            )
        )
    return outputs


def build_standardized_reward_covariance_figures(
    rows: Iterable[dict[str, Any]],
) -> list[FigureOutput]:
    """One standardized reward-pair covariance trajectory per reward pair."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] != "standardized_reward_covariance":
            continue
        pair = str(row.get("reward_pair", ""))
        if pair:
            grouped[
                (
                    str(row.get("dataset", "unknown_dataset")),
                    str(row["reward_combination"]),
                    pair,
                )
            ].append(row)

    run_styles = ["-", "--", ":", "-."]
    outputs: list[FigureOutput] = []
    for (dataset, _combination, pair), pair_rows in sorted(grouped.items()):
        series: list[FigureSeries] = []
        entries: list[LegendEntry] = []
        for index, (label, line_rows) in enumerate(sorted(_group_by_run(pair_rows).items())):
            color = f"C{index % 10}"
            style = run_styles[index % len(run_styles)]
            item = FigureSeries(
                label=label,
                points=_points(line_rows),
                color=color,
                linestyle=style,
                faint_alpha=0.18,
                faint_linewidth=0.9,
            )
            series.append(item)
            entries.append(_entry_for(item))
        outputs.append(
            (
                f"{_filename_component(dataset)}/standardized_reward_covariance/"
                f"{_filename_component(pair)}",
                FigureSpec(
                    title=f"Standardized reward covariance: {pair.replace('__', ' + ')}",
                    x_label="Training step",
                    left=FigureAxis("Prompt-local standardized covariance"),
                    series=series,
                    hlines=[FigureHLine(0.0)],
                    legend=FigureLegend(entries=entries),
                ),
            )
        )
    return outputs


def build_per_reward_disagreement_figures(rows: Iterable[dict[str, Any]]) -> list[FigureOutput]:
    """One per-reward disagreement-rate trajectory figure per reward."""
    return _per_reward_unit_rate_figures(
        rows,
        metric="per_reward_disagreement",
        folder="per_reward_disagreement",
        title=f"{{reward}} disagreement [{{dataset}}]",
        y_label="Per-reward disagreement rate",
    )


def build_per_reward_bottleneck_rate_figures(rows: Iterable[dict[str, Any]]) -> list[FigureOutput]:
    """One per-reward bottleneck-rate trajectory figure per reward."""
    return _per_reward_unit_rate_figures(
        rows,
        metric="per_reward_bottleneck_rate",
        folder="per_reward_bottleneck_rate",
        title=f"{{reward}} bottleneck rate [{{dataset}}]",
        y_label="Per-reward bottleneck rate",
    )


def _per_reward_unit_rate_figures(
    rows: Iterable[dict[str, Any]],
    metric: str,
    folder: str,
    title: str,
    y_label: str,
) -> list[FigureOutput]:
    """Build the per-reward figures whose rates live on a fixed 0-1 scale."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == metric and row.get("reward"):
            grouped[(str(row.get("dataset", "unknown_dataset")), str(row["reward"]))].append(row)

    outputs: list[FigureOutput] = []
    for (dataset, reward), reward_rows in grouped.items():
        series, entries = _run_series(_group_by_run(reward_rows))
        if metric == "per_reward_bottleneck_rate":
            series = [replace(item, faint_zorder=2.0) for item in series]
        outputs.append(
            (
                f"{_filename_component(dataset)}/{folder}/{_filename_component(reward)}",
                FigureSpec(
                    title=title.format(reward=reward, dataset=dataset),
                    x_label="Training step",
                    left=FigureAxis(y_label, limits=[0.0, 1.0]),
                    series=series,
                    legend=FigureLegend(entries=entries, loc="upper right", anchor=[1.0, 0.90]),
                ),
            )
        )
    return outputs


def build_agreement_count_distribution_figures(
    rows: Iterable[dict[str, Any]],
) -> list[FigureOutput]:
    """One sample concordance-count distribution figure per reward combination.

    Per-reward disagreement rates are marginals: they cannot separate polarized
    conflict (few samples opposing many rewards at once) from diffuse conflict
    (many samples opposing a single reward). The count distribution keeps that
    joint structure, so a rising ``c = n_rewards`` curve beside a rising ``c = 1``
    curve means conflict is spreading rather than deepening.

    Bins that no run ever populates are left out: with the strictly positive
    weights this tool accepts, ``c = 0`` is unreachable, because a sample below
    the group mean on every reward is also below the mean of the weighted scalar.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["metric"]).startswith("agreement_count_c"):
            grouped[str(row.get("dataset", "unknown_dataset"))].append(row)

    outputs: list[FigureOutput] = []
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
        series: list[FigureSeries] = []
        entries: list[LegendEntry] = []
        for run_index, (label, bins) in enumerate(by_run.items()):
            style = ("-", "--", "-.", ":")[run_index % 4]
            for position, agreeing_count in enumerate(active_counts):
                count_rows = bins.get(f"count_{agreeing_count}")
                if not count_rows:
                    continue
                points = _points(count_rows)
                color = f"C{position % 10}"
                series.append(
                    FigureSeries(
                        label=f"{label} | c={agreeing_count}",
                        points=points,
                        color=color,
                        linestyle=style,
                        markersize=2.5,
                        # Series share a colour within one agreeing count, so run
                        # index is carried by the dash pattern alone. Marking
                        # every step would fill the dash gaps and erase that cue.
                        markevery=max(1, len(points) // 12),
                        faint_raw_trace=False,
                    )
                )
                # Marker-free handles with a longer line: a handle copied from
                # the series carries its centre marker, which covers the single
                # dash gap that fits in a short handle, so both runs would look
                # identical in the legend.
                entries.append(
                    LegendEntry(
                        label=f"{label} | c={agreeing_count}",
                        color=color,
                        linestyle=style,
                        linewidth=1.5,
                    )
                )
        outputs.append(
            (
                f"{_filename_component(dataset)}/agreement_count",
                FigureSpec(
                    title=f"Sample concordance-count distribution [{dataset}]",
                    x_label="Training step",
                    left=FigureAxis("Sample fraction"),
                    series=series,
                    legend=FigureLegend(entries=entries, fontsize=7.0, handlelength=2.8),
                    figsize=[9.0, 5.0],
                ),
            )
        )
    return outputs


def build_agreement_count_expectation_figures(
    rows: Iterable[dict[str, Any]],
) -> list[FigureOutput]:
    """One mean concordance-count figure per reward combination.

    The expected count is exactly ``n_rewards`` minus the summed per-reward
    disagreement rates, so this figure restates the marginal disagreement rows.
    It earns its place as a single bounded scalar: read against the dashed
    ceiling, it answers "how far from full concordance is this group" without the
    reader comparing several marginal curves.
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

    outputs: list[FigureOutput] = []
    for dataset, dataset_rows in grouped.items():
        _reject_mixed_reward_combinations(dataset, dataset_rows)
        series, entries = _run_series(_group_by_run(dataset_rows))
        hlines: list[FigureHLine] = []
        ceiling = ceilings.get(dataset)
        if ceiling is not None:
            label = f"all {ceiling} rewards concordant"
            hlines.append(FigureHLine(y=float(ceiling), label=label, linestyle="--", alpha=0.4))
            entries.append(
                LegendEntry(
                    label=label,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.4,
                )
            )
        outputs.append(
            (
                f"{_filename_component(dataset)}/agreement_count_expectation",
                FigureSpec(
                    title=f"Mean concordance count [{dataset}]",
                    x_label="Training step",
                    left=FigureAxis("Mean concordant reward count per sample"),
                    series=series,
                    hlines=hlines,
                    legend=FigureLegend(entries=entries),
                ),
            )
        )
    return outputs


def build_per_reward_weighted_advantage_sign_figures(
    rows: Iterable[dict[str, Any]],
) -> list[FigureOutput]:
    """One standardized-advantage-by-sign figure per reward, split by sample weight."""
    names = {
        "weight_ge_1_adv_positive": "weight≥1, adv>0",
        "weight_ge_1_adv_negative": "weight≥1, adv<0",
        "weight_lt_1_adv_positive": "weight<1, adv>0",
        "weight_lt_1_adv_negative": "weight<1, adv<0",
        "adv_positive": "adv>0",
        "adv_negative": "adv<0",
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in names and row.get("reward"):
            grouped[(str(row.get("dataset", "unknown_dataset")), str(row["reward"]))].append(row)

    outputs: list[FigureOutput] = []
    for (dataset, reward), reward_rows in grouped.items():
        series: list[FigureSeries] = []
        entries: list[LegendEntry] = []
        by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for label, line_rows in sorted(_group_by_run(reward_rows).items()):
            by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in line_rows:
                by_metric[str(row["metric"])].append(row)
            by_run[label] = by_metric
        # These two figures draw every series without a colour, so matplotlib's
        # cycle paints each *series* in draw order rather than each run. That is
        # the opposite of the per-run figures above, where the faint raw trace
        # consumes one cycle slot per run.
        series_index = 0
        for run_index, (label, by_metric) in enumerate(by_run.items()):
            style = ("-", "--", "-.", ":")[run_index % 4]
            for metric, legend_name in names.items():
                metric_rows = by_metric.get(metric)
                if not metric_rows:
                    continue
                ordered = sorted(metric_rows, key=lambda row: int(row["step"]))
                counts = np.asarray(
                    [float(row.get("sample_count", np.nan)) for row in ordered], dtype=float
                )
                color = f"C{series_index % 10}"
                series_index += 1
                item = FigureSeries(
                    label=f"{label} | {legend_name}",
                    points=_points(ordered),
                    color=color,
                    linestyle=style,
                    marker="",
                    linewidth=2.5 if metric.startswith("weight_ge_1") else 1.3,
                    faint_raw_trace=False,
                    scatter_sizes=np.clip(
                        1.5 + 0.8 * np.sqrt(np.maximum(counts, 0.0)), 2.0, 6.0
                    ).tolist(),
                )
                series.append(item)
                entries.append(_entry_for(item))
        outputs.append(
            (
                f"{_filename_component(dataset)}/per_reward_weighted_advantage_sign/"
                f"{_filename_component(reward)}",
                FigureSpec(
                    title=f"{reward} weighted advantage sign [{dataset}]",
                    x_label="Training step",
                    left=FigureAxis("Mean standardized advantage"),
                    series=series,
                    hlines=[FigureHLine(0.0)],
                    legend=FigureLegend(entries=entries, fontsize=7.0),
                    figsize=[9.0, 5.0],
                ),
            )
        )
    return outputs


def build_per_reward_weighted_advantage_count_figures(
    rows: Iterable[dict[str, Any]],
) -> list[FigureOutput]:
    """One average-sample-count-by-sign figure per reward, split by sample weight."""
    names = {
        "sample_count_weight_ge_1_adv_positive": "weight≥1, adv>0",
        "sample_count_weight_ge_1_adv_negative": "weight≥1, adv<0",
        "sample_count_weight_ge_1_adv_zero": "weight≥1, adv=0",
        "sample_count_weight_lt_1_adv_positive": "weight<1, adv>0",
        "sample_count_weight_lt_1_adv_negative": "weight<1, adv<0",
        "sample_count_weight_lt_1_adv_zero": "weight<1, adv=0",
        "sample_count_adv_positive": "adv>0",
        "sample_count_adv_negative": "adv<0",
        "sample_count_adv_zero": "adv=0",
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in names and row.get("reward"):
            grouped[(str(row.get("dataset", "unknown_dataset")), str(row["reward"]))].append(row)

    outputs: list[FigureOutput] = []
    for (dataset, reward), reward_rows in grouped.items():
        series: list[FigureSeries] = []
        entries: list[LegendEntry] = []
        by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for label, line_rows in sorted(_group_by_run(reward_rows).items()):
            by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in line_rows:
                by_metric[str(row["metric"])].append(row)
            by_run[label] = by_metric
        series_index = 0
        for run_index, (label, by_metric) in enumerate(by_run.items()):
            style = ("-", "--", "-.", ":")[run_index % 4]
            for metric, legend_name in names.items():
                metric_rows = by_metric.get(metric)
                if not metric_rows:
                    continue
                color = f"C{series_index % 10}"
                series_index += 1
                item = FigureSeries(
                    label=f"{label} | {legend_name}",
                    points=_points(metric_rows),
                    color=color,
                    linestyle=style,
                    linewidth=2.5 if "weight_ge_1" in metric else 1.3,
                    markersize=2.5,
                    faint_raw_trace=False,
                )
                series.append(item)
                entries.append(_entry_for(item))
        outputs.append(
            (
                f"{_filename_component(dataset)}/per_reward_weighted_advantage_count/"
                f"{_filename_component(reward)}",
                FigureSpec(
                    title=f"{reward} weighted advantage sample count [{dataset}]",
                    x_label="Training step",
                    left=FigureAxis("Mean sample count per group"),
                    series=series,
                    legend=FigureLegend(entries=entries, fontsize=7.0),
                    figsize=[9.0, 5.0],
                ),
            )
        )
    return outputs


def build_run_training_progress_figures(rows: Iterable[dict[str, Any]]) -> list[FigureOutput]:
    """One reward-progress and concordance-rate figure per run and dataset.

    Rewards are aggregated into a single progress curve rather than drawn
    separately: each reward is mapped onto 0-100% of its own observed range
    within the run, and those per-reward percentages are averaged at each step.
    A reward set of any size therefore reads as one curve, which is the point —
    a per-reward version grows unreadable as rewards are added.

    Averaging percentages rather than raw levels is what makes the aggregate
    meaningful: a reward scored 0-1 and one scored 0-5 contribute equally, so the
    curve tracks how far each reward has moved through its own range. Read it for
    shape, not for level; a single outlier step sets a reward's 100% mark, and a
    reward that never moves contributes a flat 0%.

    The figure carries two axes. Progress keeps the fixed 0-100% because that
    range is anchored by the normalization itself; the concordance rates float to
    their own band, which is what makes their shape legible. The split is marked
    three ways so no curve can be read against the wrong scale: every axis label
    and legend entry names its side, only the left axis carries gridlines, and
    the fixed scale stays on the side whose bounds are anchored rather than
    chosen.
    """
    outputs: list[FigureOutput] = []
    for dataset, rewards, by_run in _group_progress_rows(rows):
        for label, keyed in by_run.items():
            steps, progress = _mean_reward_progress(keyed, rewards)
            if steps.size == 0:
                continue
            series = [
                FigureSeries(
                    label="mean reward progress (left)",
                    points=_as_points(steps, progress),
                    color=_REWARD_PROGRESS_COLOR,
                    linewidth=_PERCENT_LINEWIDTH,
                    markersize=_PERCENT_MARKERSIZE,
                    markevery=_mark_every(len(steps)),
                    faint_alpha=_PERCENT_FAINT_ALPHA,
                    faint_linewidth=_PERCENT_FAINT_LINEWIDTH,
                )
            ]
            entries = [
                LegendEntry(
                    label="mean reward progress (left)",
                    color=_REWARD_PROGRESS_COLOR,
                    linewidth=_PERCENT_LINEWIDTH,
                )
            ]
            for offset, (metric, legend_name) in enumerate(_CONCORDANCE_METRICS):
                concordance_points = _percent_points(keyed.get((metric, "")))
                if not concordance_points:
                    continue
                series.append(
                    FigureSeries(
                        label=f"{legend_name} (right)",
                        points=concordance_points,
                        color=_CONCORDANCE_DIRECTION_COLORS[offset],
                        axis="right",
                        linewidth=_PERCENT_LINEWIDTH,
                        markersize=_PERCENT_MARKERSIZE,
                        markevery=_mark_every(len(concordance_points)),
                        faint_alpha=_PERCENT_FAINT_ALPHA,
                        faint_linewidth=_PERCENT_FAINT_LINEWIDTH,
                    )
                )
                entries.append(
                    LegendEntry(
                        label=f"{legend_name} (right)",
                        color=_CONCORDANCE_DIRECTION_COLORS[offset],
                        linewidth=_PERCENT_LINEWIDTH,
                        markersize=_PERCENT_MARKERSIZE,
                    )
                )
            outputs.append(
                (
                    f"{_filename_component(dataset)}/training_progress/"
                    f"{_filename_component(label)}",
                    FigureSpec(
                        title=f"Reward progress and concordance rate [{dataset}] | {label}",
                        x_label="Training step",
                        left=FigureAxis(
                            "Reward progress % (left, own min-max)", limits=[0.0, 100.0]
                        ),
                        right=FigureAxis("Concordant samples % (right)", grid=False),
                        series=series,
                        legend=FigureLegend(entries=entries),
                    ),
                )
            )
    return outputs


def build_concordance_rate_figures(rows: Iterable[dict[str, Any]]) -> list[FigureOutput]:
    """One all-run concordance-rate figure per dataset.

    This figure drops the reward-progress curve so the runs can carry the axes
    alone, because telling the runs apart is what it is for. Each run therefore
    owns a colour and a marker, and the two concordance directions are separated
    by the dash pattern instead: positive solid, negative dashed. Direction is
    the one axis with only two values, so it is the one that can afford the
    weakest channel, and the dash pattern is also what survives greyscale
    printing, where the run colours collapse together.
    """
    outputs: list[FigureOutput] = []
    for dataset, _rewards, by_run in _group_progress_rows(rows):
        if len(by_run) > len(_CATEGORICAL_COLORS):
            raise ValueError(
                f"Dataset {dataset!r} overlays {len(by_run)} runs but the validated palette "
                f"holds {len(_CATEGORICAL_COLORS)}. Split the runs across figures instead of "
                "cycling colours, which would give two runs the same hue."
            )
        series: list[FigureSeries] = []
        entries: list[LegendEntry] = []
        for run_index, (label, keyed) in enumerate(by_run.items()):
            color = _CATEGORICAL_COLORS[run_index]
            marker = _RUN_MARKERS[run_index]
            for offset, (metric, legend_name) in enumerate(_CONCORDANCE_METRICS):
                points = _percent_points(keyed.get((metric, "")))
                if not points:
                    continue
                linestyle = _CONCORDANCE_DIRECTION_LINESTYLES[offset]
                series.append(
                    FigureSeries(
                        label=f"{label} | {legend_name}",
                        points=points,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=_PERCENT_LINEWIDTH,
                        markersize=_PERCENT_MARKERSIZE,
                        markevery=_mark_every(len(points)),
                        faint_alpha=_PERCENT_FAINT_ALPHA,
                        faint_linewidth=_PERCENT_FAINT_LINEWIDTH,
                    )
                )
                entries.append(
                    LegendEntry(
                        label=f"{label} | {legend_name}",
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=_PERCENT_LINEWIDTH,
                    )
                )
        outputs.append(
            (
                f"{_filename_component(dataset)}/training_progress/concordance",
                FigureSpec(
                    title=f"Concordant sample rate [{dataset}]",
                    x_label="Training step",
                    left=FigureAxis("Percent of concordant samples"),
                    series=series,
                    legend=FigureLegend(entries=entries),
                ),
            )
        )
    return outputs


_BUILDERS: tuple[Callable[[Iterable[dict[str, Any]]], list[FigureOutput]], ...] = (
    build_per_reward_conflict_score_figures,
    build_per_reward_disagreement_figures,
    build_per_reward_bottleneck_rate_figures,
    build_agreement_count_distribution_figures,
    build_agreement_count_expectation_figures,
    build_per_reward_weighted_advantage_sign_figures,
    build_per_reward_weighted_advantage_count_figures,
    build_standardized_reward_covariance_figures,
    build_reward_concordance_lower_bound_figures,
    build_run_training_progress_figures,
    build_concordance_rate_figures,
)


def build_figures(rows: Iterable[dict[str, Any]], smoothing_window: int = 5) -> list[FigureOutput]:
    """Build every figure's data from the tidy metric rows.

    The configured smoothing window is applied here rather than in each builder,
    so a spec can never carry a figure's own default and silently ignore the
    config.
    """
    rows = list(rows)
    outputs: list[FigureOutput] = []
    for builder in _BUILDERS:
        outputs.extend(builder(rows))
    return [(stem, replace(spec, smoothing_window=smoothing_window)) for stem, spec in outputs]


def write_figure_data(outputs: Sequence[FigureOutput], output_dir: str | Path) -> list[str]:
    """Write one spec beside each image, returning the stems that were written."""
    stems = [stem for stem, _ in outputs]
    if len(stems) != len(set(stems)):
        duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
        raise ValueError(f"Figure builders produced duplicate output stems: {duplicates}")

    written: list[str] = []
    for stem, spec in outputs:
        write_spec(spec, Path(output_dir) / Path(stem).parent, Path(stem).name)
        written.append(stem)
    return written


def _entry_for(series: FigureSeries, label: str | None = None) -> LegendEntry:
    """Build the legend entry for a series, mirroring how it is drawn."""
    return LegendEntry(
        label=series.label if label is None else label,
        color=series.color,
        linestyle=series.linestyle,
        marker=series.marker,
        linewidth=series.linewidth,
        markersize=series.markersize,
    )


def _run_series(
    by_run: dict[str, list[dict[str, Any]]],
) -> tuple[list[FigureSeries], list[LegendEntry]]:
    """Build the common per-run trajectory series and its legend entries."""
    series: list[FigureSeries] = []
    entries: list[LegendEntry] = []
    for index, (label, line_rows) in enumerate(by_run.items()):
        color = f"C{index % 10}"
        item = FigureSeries(label=label, points=_points(line_rows), color=color)
        series.append(item)
        entries.append(_entry_for(item))
    return series, entries


def _points(rows: Iterable[dict[str, Any]]) -> list[list[float]]:
    """Return one ``[step, value]`` pair per recorded step, in step order."""
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    return [
        [float(row["step"]), float(row["value"])]
        for row in ordered
        if np.isfinite(float(row["value"]))
    ]


def _as_points(steps: np.ndarray, values: np.ndarray) -> list[list[float]]:
    return [[float(step), float(value)] for step, value in zip(steps, values)]


def _percent_points(rows: Iterable[dict[str, Any]] | None) -> list[list[float]]:
    """Return a sample-share series already expressed in percent."""
    if not rows:
        return []
    return [[step, 100.0 * value] for step, value in _points(rows)]


def _mark_every(point_count: int) -> int:
    return max(1, point_count // 12)


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
    per_step: dict[int, list[float]] = defaultdict(list)
    for reward in rewards:
        reward_rows = keyed.get(("per_reward_mean_reward", reward))
        if not reward_rows:
            continue
        points = _points(reward_rows)
        values = np.asarray([value for _, value in points], dtype=np.float64)
        for (step, _), percent in zip(points, _percent_of_own_range(values)):
            per_step[int(step)].append(float(percent))
    ordered = sorted(per_step)
    return (
        np.asarray(ordered, dtype=np.int64),
        np.asarray([float(np.mean(per_step[step])) for step in ordered]),
    )


def _percent_of_own_range(values: np.ndarray) -> np.ndarray:
    """Map one series onto 0-100% of its own observed range.

    The mapping is defined by the raw series, before smoothing, so the smoothed
    trace stays inside 0-100%. A constant series has no range to express progress
    against and becomes flat 0%, following the zero-variance convention used
    elsewhere in this tool.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values)
    return 100.0 * (values - low) / (high - low)


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
            "concordance-count statistics are only comparable within a single reward set."
        )


def _group_by_run(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by run label, in label order.

    The order decides which colour each run gets, so it is fixed by the label
    rather than by the order runs appear in the config: reordering the config
    must not repaint the figures.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row["value"])
        if np.isfinite(value):
            grouped[str(row["run_label"])].append(row)
    return {label: grouped[label] for label in sorted(grouped)}


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
