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

"""Figure-shaped data: one spec per figure, holding every line and point.

A spec is what the renderer needs to draw a figure and nothing else. It is
written beside its own image as ``<figure>.json``, so the data for a figure and
the figure travel together, and ``cache_mode: reuse`` redraws from the specs
without touching the reward pickles again.

The spec is deliberately figure-shaped rather than normalized: a reader can
open one file and see the lines that figure draws, in order, each with its own
points. Default style values are omitted to keep the data compact; everything
that differs from those versioned defaults is stored with the points.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bumped whenever the spec's shape or meaning changes. Readers list compatible
# historical versions explicitly so incompatible data is never rendered with
# changed semantics.
SPEC_VERSION = 6
SUPPORTED_SPEC_VERSIONS = (1, 2, 3, 4, 5, SPEC_VERSION)


@dataclass(frozen=True)
class FigureSeries:
    """One drawn line: its legend label, its points, and how it is styled.

    ``points`` are the values this figure plots at each recorded step, before
    any smoothing. The renderer derives the smoothed foreground and the faint
    unsmoothed background from them.
    """

    label: str
    points: list[list[float]]
    color: str
    linestyle: str = "-"
    marker: str = "o"
    linewidth: float = 1.5
    markersize: float = 3.0
    markevery: int | None = None
    axis: str = "left"
    faint_raw_trace: bool = True
    faint_alpha: float = 0.22
    faint_linewidth: float = 1.0
    faint_zorder: float = 1.0
    zorder: float = 2.0
    scatter_sizes: list[float] | None = None


@dataclass(frozen=True)
class FigureAxis:
    """One y-axis, optionally split into ascending non-overlapping segments."""

    label: str
    limits: list[float] | None = None
    grid: bool = True
    segments: list[list[float]] | None = None
    segment_height_ratios: list[float] | None = None


@dataclass(frozen=True)
class FigureHLine:
    """A horizontal reference line spanning the axis."""

    y: float
    label: str = ""
    color: str = "black"
    linestyle: str = "-"
    alpha: float = 0.5


@dataclass(frozen=True)
class LegendEntry:
    """One legend row, which need not match any drawn series' style."""

    label: str
    color: str
    linestyle: str = "-"
    marker: str = ""
    linewidth: float = 1.5
    markersize: float = 6.0
    alpha: float = 1.0


@dataclass(frozen=True)
class FigureLegend:
    """The legend box, listed explicitly rather than read off the axes."""

    entries: list[LegendEntry]
    loc: str = "best"
    fontsize: float = 8.0
    ncol: int = 1
    anchor: list[float] | None = None
    handlelength: float | None = None


@dataclass(frozen=True)
class FigureFontSizes:
    """Optional per-figure font sizes in points."""

    title: float | None = None
    x_label: float | None = None
    left_y_label: float | None = None
    right_y_label: float | None = None
    x_tick: float | None = None
    left_y_tick: float | None = None
    right_y_tick: float | None = None
    legend: float | None = None


@dataclass(frozen=True)
class FigureSpec:
    """Everything needed to redraw one figure, and nothing more."""

    title: str
    x_label: str
    left: FigureAxis
    series: list[FigureSeries] = field(default_factory=list)
    right: FigureAxis | None = None
    hlines: list[FigureHLine] = field(default_factory=list)
    legend: FigureLegend | None = None
    smoothing_window: int = 5
    figsize: list[float] = field(default_factory=lambda: [8.0, 4.5])
    break_gap: float = 0.05
    break_mark_size: float = 0.012
    border_width: float | None = None
    top_margin: float | None = None
    bottom_margin: float | None = None
    font_sizes: FigureFontSizes = field(default_factory=FigureFontSizes)


def to_json(spec: FigureSpec) -> str:
    """Serialize one spec, keeping each point on its own line.

    ``json.dumps(indent=...)`` would explode every point pair across three
    lines, which makes a 61-step series unreadable. Numeric lists are therefore
    kept inline and a list of points gets one row per point, so the file reads
    like the line it describes.
    """
    validate_spec(spec)
    return _dumps({"spec_version": SPEC_VERSION, **_as_mapping(spec)}, 0) + "\n"


def write_spec(spec: FigureSpec, directory: str | Path, stem: str) -> Path:
    """Write one spec next to the image that shares its name."""
    path = Path(directory) / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(spec), encoding="utf-8")
    return path


def read_spec(path: str | Path) -> FigureSpec:
    """Load one spec written by :func:`to_json`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Figure spec must be a JSON object: {path}")
    version = raw.get("spec_version")
    if version not in SUPPORTED_SPEC_VERSIONS:
        raise ValueError(
            f"Figure data {path} was written by unsupported spec_version {version!r}; supported "
            f"versions are {list(SUPPORTED_SPEC_VERSIONS)}. Re-run with "
            "output.cache_mode: regenerate."
        )
    spec = _spec_from_mapping(raw, str(path))
    validate_spec(spec, str(path))
    return spec


def validate_spec(spec: FigureSpec, context: str = "figure spec") -> None:
    """Reject layouts that cannot be represented without ambiguous axes."""
    if spec.smoothing_window < 1 or spec.smoothing_window % 2 == 0:
        raise ValueError(f"{context}: smoothing_window must be a positive odd integer.")
    if len(spec.figsize) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in spec.figsize
    ):
        raise ValueError(f"{context}: figsize must contain two finite positive values.")
    if not math.isfinite(spec.break_gap) or not 0.0 <= spec.break_gap < 0.5:
        raise ValueError(f"{context}: break_gap must be finite and in [0, 0.5).")
    if (
        not math.isfinite(spec.break_mark_size)
        or spec.break_mark_size <= 0.0
        or spec.break_mark_size >= 0.1
    ):
        raise ValueError(
            f"{context}: break_mark_size must be finite, positive, and smaller than 0.1."
        )
    if spec.border_width is not None and (
        not math.isfinite(spec.border_width) or spec.border_width <= 0.0
    ):
        raise ValueError(f"{context}: border_width must be finite and strictly positive.")
    if spec.top_margin is not None and (
        not math.isfinite(spec.top_margin) or not 0.0 <= spec.top_margin < 1.0
    ):
        raise ValueError(f"{context}: top_margin must be finite and in [0.0, 1.0).")
    if spec.bottom_margin is not None and (
        not math.isfinite(spec.bottom_margin) or not 0.0 <= spec.bottom_margin < 1.0
    ):
        raise ValueError(f"{context}: bottom_margin must be finite and in [0.0, 1.0).")
    if spec.legend is not None and (
        isinstance(spec.legend.ncol, bool)
        or not isinstance(spec.legend.ncol, int)
        or spec.legend.ncol < 1
    ):
        raise ValueError(f"{context}: legend.ncol must be a positive integer.")
    for name in _FONT_SIZE_FIELDS:
        value = getattr(spec.font_sizes, name)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{context}: font_sizes.{name} must be finite and strictly positive.")

    _validate_axis(spec.left, "left", context)
    if spec.right is not None:
        _validate_axis(spec.right, "right", context)

    for series in spec.series:
        if series.axis not in {"left", "right"}:
            raise ValueError(
                f"{context}: series {series.label!r} uses unknown axis {series.axis!r}."
            )
        if series.axis == "right" and spec.right is None:
            raise ValueError(
                f"{context}: series {series.label!r} uses the right axis, but no right axis exists."
            )

    left_count = len(spec.left.segments or [])
    right_count = len(spec.right.segments or []) if spec.right is not None else 0
    if spec.right is not None and (left_count or right_count):
        if left_count != right_count:
            raise ValueError(
                f"{context}: a dual-y broken figure requires left and right axes to define the "
                f"same number of segments, got left({left_count}) and right({right_count})."
            )
        left_ratios = spec.left.segment_height_ratios or [1.0] * left_count
        right_ratios = spec.right.segment_height_ratios or [1.0] * right_count
        if left_ratios != right_ratios:
            raise ValueError(
                f"{context}: dual-y broken axes must use identical segment_height_ratios, got "
                f"left({left_ratios}) and right({right_ratios})."
            )


def _validate_axis(axis: FigureAxis, name: str, context: str) -> None:
    """Validate one continuous or segmented y-axis."""
    if axis.limits is not None:
        if len(axis.limits) != 2:
            raise ValueError(f"{context}: {name}.limits must contain [lower, upper].")
        lower, upper = axis.limits
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"{context}: {name}.limits must be finite and strictly increasing, got "
                f"{axis.limits}."
            )
    if axis.segments is None:
        if axis.segment_height_ratios is not None:
            raise ValueError(f"{context}: {name}.segment_height_ratios requires {name}.segments.")
        return
    if axis.limits is not None:
        raise ValueError(f"{context}: {name} cannot define both limits and segments.")
    if len(axis.segments) < 2:
        raise ValueError(f"{context}: {name}.segments must contain at least two ranges.")

    previous_upper: float | None = None
    for index, segment in enumerate(axis.segments):
        if len(segment) != 2:
            raise ValueError(f"{context}: {name}.segments[{index}] must contain [lower, upper].")
        lower, upper = segment
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"{context}: {name}.segments[{index}] must be finite and strictly increasing, "
                f"got {segment}."
            )
        if previous_upper is not None and lower <= previous_upper:
            raise ValueError(
                f"{context}: {name}.segments must be ascending and separated; segment {index} "
                f"starts at {lower} after the previous segment ended at {previous_upper}."
            )
        previous_upper = upper

    if axis.segment_height_ratios is None:
        return
    if len(axis.segment_height_ratios) != len(axis.segments):
        raise ValueError(
            f"{context}: {name}.segment_height_ratios must have one value per segment."
        )
    if any(not math.isfinite(ratio) or ratio <= 0.0 for ratio in axis.segment_height_ratios):
        raise ValueError(
            f"{context}: {name}.segment_height_ratios must be finite and strictly positive."
        )


def _as_mapping(spec: FigureSpec) -> dict[str, Any]:
    """Lay one spec out in the order its keys should read."""
    result = {
        "title": spec.title,
        "x_label": spec.x_label,
        "series": [_series_to_mapping(series) for series in spec.series],
        "left": _axis_to_mapping(spec.left),
        "smoothing_window": spec.smoothing_window,
    }
    if spec.right is not None:
        result["right"] = _axis_to_mapping(spec.right)
    if spec.hlines:
        result["hlines"] = [_hline_to_mapping(hline) for hline in spec.hlines]
    if spec.legend is not None:
        result["legend"] = _legend_to_mapping(spec.legend)
    if spec.figsize != [8.0, 4.5]:
        result["figsize"] = list(spec.figsize)
    if spec.break_gap != 0.05:
        result["break_gap"] = spec.break_gap
    if spec.break_mark_size != 0.012:
        result["break_mark_size"] = spec.break_mark_size
    if spec.border_width is not None:
        result["border_width"] = spec.border_width
    if spec.top_margin is not None:
        result["top_margin"] = spec.top_margin
    if spec.bottom_margin is not None:
        result["bottom_margin"] = spec.bottom_margin
    font_sizes = _font_sizes_to_mapping(spec.font_sizes)
    if font_sizes:
        result["font_sizes"] = font_sizes
    return result


def _axis_to_mapping(axis: FigureAxis) -> dict[str, Any]:
    result: dict[str, Any] = {"label": axis.label}
    if axis.limits is not None:
        result["limits"] = axis.limits
    if not axis.grid:
        result["grid"] = False
    if axis.segments is not None:
        result["segments"] = axis.segments
    if axis.segment_height_ratios is not None:
        result["segment_height_ratios"] = axis.segment_height_ratios
    return result


def _series_to_mapping(series: FigureSeries) -> dict[str, Any]:
    """Keep the visible data first and omit style values equal to defaults."""
    result: dict[str, Any] = {
        "label": series.label,
        "points": series.points,
        "color": series.color,
    }
    defaults = FigureSeries(label="", points=[], color="")
    for name in (
        "linestyle",
        "marker",
        "linewidth",
        "markersize",
        "markevery",
        "axis",
        "faint_raw_trace",
        "faint_alpha",
        "faint_linewidth",
        "faint_zorder",
        "zorder",
        "scatter_sizes",
    ):
        value = getattr(series, name)
        if value != getattr(defaults, name):
            result[name] = value
    return result


def _hline_to_mapping(hline: FigureHLine) -> dict[str, Any]:
    result: dict[str, Any] = {"y": hline.y}
    defaults = FigureHLine(y=0.0)
    for name in ("label", "color", "linestyle", "alpha"):
        value = getattr(hline, name)
        if value != getattr(defaults, name):
            result[name] = value
    return result


def _legend_to_mapping(legend: FigureLegend) -> dict[str, Any]:
    result: dict[str, Any] = {
        "entries": [_legend_entry_to_mapping(entry) for entry in legend.entries]
    }
    defaults = FigureLegend(entries=[])
    for name in ("loc", "fontsize", "ncol", "anchor", "handlelength"):
        value = getattr(legend, name)
        if value != getattr(defaults, name):
            result[name] = value
    return result


_FONT_SIZE_FIELDS = (
    "title",
    "x_label",
    "left_y_label",
    "right_y_label",
    "x_tick",
    "left_y_tick",
    "right_y_tick",
    "legend",
)


def _font_sizes_to_mapping(font_sizes: FigureFontSizes) -> dict[str, float]:
    """Serialize only explicitly configured font sizes."""
    return {
        name: value
        for name in _FONT_SIZE_FIELDS
        if (value := getattr(font_sizes, name)) is not None
    }


def _legend_entry_to_mapping(entry: LegendEntry) -> dict[str, Any]:
    result: dict[str, Any] = {"label": entry.label, "color": entry.color}
    defaults = LegendEntry(label="", color="")
    for name in ("linestyle", "marker", "linewidth", "markersize", "alpha"):
        value = getattr(entry, name)
        if value != getattr(defaults, name):
            result[name] = value
    return result


def _spec_from_mapping(raw: dict[str, Any], path: str) -> FigureSpec:
    """Rebuild a spec from its serialized form, rejecting unknown shapes."""
    legend = raw.get("legend")
    right = raw.get("right")
    font_sizes = raw.get("font_sizes")
    return FigureSpec(
        title=str(raw["title"]),
        x_label=str(raw["x_label"]),
        left=_axis_from_mapping(raw["left"], path),
        series=[_series_from_mapping(item, path) for item in raw.get("series", [])],
        right=None if right is None else _axis_from_mapping(right, path),
        hlines=[_hline_from_mapping(item, path) for item in raw.get("hlines", [])],
        legend=None if legend is None else _legend_from_mapping(legend, path),
        smoothing_window=int(raw["smoothing_window"]),
        figsize=[float(value) for value in raw.get("figsize", [8.0, 4.5])],
        break_gap=float(raw.get("break_gap", 0.05)),
        break_mark_size=float(raw.get("break_mark_size", 0.012)),
        border_width=(None if raw.get("border_width") is None else float(raw["border_width"])),
        top_margin=None if raw.get("top_margin") is None else float(raw["top_margin"]),
        bottom_margin=(None if raw.get("bottom_margin") is None else float(raw["bottom_margin"])),
        font_sizes=(
            FigureFontSizes() if font_sizes is None else _font_sizes_from_mapping(font_sizes, path)
        ),
    )


def _font_sizes_from_mapping(raw: Any, path: str) -> FigureFontSizes:
    """Load the optional font-size overrides from JSON."""
    if not isinstance(raw, dict):
        raise ValueError(f"Figure font_sizes must be a JSON object: {path}")
    unknown = sorted(set(raw) - set(_FONT_SIZE_FIELDS))
    if unknown:
        raise ValueError(f"Figure font_sizes contains unknown fields {unknown}: {path}")
    return FigureFontSizes(
        **{name: None if raw.get(name) is None else float(raw[name]) for name in _FONT_SIZE_FIELDS}
    )


def _axis_from_mapping(raw: dict[str, Any], path: str) -> FigureAxis:
    limits = raw.get("limits")
    segments = raw.get("segments")
    ratios = raw.get("segment_height_ratios")
    return FigureAxis(
        label=str(raw["label"]),
        limits=None if limits is None else [float(value) for value in limits],
        grid=bool(raw.get("grid", True)),
        segments=(
            None
            if segments is None
            else [[float(lower), float(upper)] for lower, upper in segments]
        ),
        segment_height_ratios=(None if ratios is None else [float(value) for value in ratios]),
    )


def _hline_from_mapping(raw: dict[str, Any], path: str) -> FigureHLine:
    return FigureHLine(
        y=float(raw["y"]),
        label=str(raw.get("label", "")),
        color=str(raw.get("color", "black")),
        linestyle=str(raw.get("linestyle", "-")),
        alpha=float(raw.get("alpha", 0.5)),
    )


def _legend_from_mapping(raw: dict[str, Any], path: str) -> FigureLegend:
    anchor = raw.get("anchor")
    handlelength = raw.get("handlelength")
    ncol = raw.get("ncol", 1)
    if isinstance(ncol, bool) or not isinstance(ncol, int) or ncol < 1:
        raise ValueError(f"Figure legend.ncol must be a positive integer: {path}")
    return FigureLegend(
        entries=[
            LegendEntry(
                label=str(item["label"]),
                color=str(item["color"]),
                linestyle=str(item.get("linestyle", "-")),
                marker=str(item.get("marker", "")),
                linewidth=float(item.get("linewidth", 1.5)),
                markersize=float(item.get("markersize", 6.0)),
                alpha=float(item.get("alpha", 1.0)),
            )
            for item in raw.get("entries", [])
        ],
        loc=str(raw.get("loc", "best")),
        fontsize=float(raw.get("fontsize", 8.0)),
        ncol=ncol,
        anchor=None if anchor is None else [float(value) for value in anchor],
        handlelength=None if handlelength is None else float(handlelength),
    )


def _series_from_mapping(raw: dict[str, Any], path: str) -> FigureSeries:
    points = raw.get("points")
    if not isinstance(points, list):
        raise ValueError(f"Figure series needs a list of points: {path}")
    sizes = raw.get("scatter_sizes")
    markevery = raw.get("markevery")
    return FigureSeries(
        label=str(raw["label"]),
        points=[[float(step), float(value)] for step, value in points],
        color=str(raw["color"]),
        linestyle=str(raw.get("linestyle", "-")),
        marker=str(raw.get("marker", "o")),
        linewidth=float(raw.get("linewidth", 1.5)),
        markersize=float(raw.get("markersize", 3.0)),
        markevery=None if markevery is None else int(markevery),
        axis=str(raw.get("axis", "left")),
        faint_raw_trace=bool(raw.get("faint_raw_trace", True)),
        faint_alpha=float(raw.get("faint_alpha", 0.22)),
        faint_linewidth=float(raw.get("faint_linewidth", 1.0)),
        faint_zorder=float(raw.get("faint_zorder", 1.0)),
        zorder=float(raw.get("zorder", 2.0)),
        scatter_sizes=None if sizes is None else [float(value) for value in sizes],
    )


def _dumps(value: Any, indent: int) -> str:
    """Render one spec as readable JSON with numeric rows kept inline."""
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = [
            f"{pad}  {_json_atom(key)}: {_dumps(item, indent + 1)}" for key, item in value.items()
        ]
        return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if _is_number_row(value):
            return "[" + ", ".join(_json_atom(item) for item in value) + "]"
        rows = [f"{pad}  {_dumps(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(rows) + f"\n{pad}]"
    return _json_atom(value)


def _is_number_row(value: list[Any]) -> bool:
    """Whether a list is a flat run of numbers that belongs on one line."""
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _json_atom(value: Any) -> str:
    """Serialize one scalar as strict, Unicode-preserving JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Figure data contains a non-finite value: {value!r}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
