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

"""Reward-distribution and full-dimensional Pareto-convexity plots."""

from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Set backend before importing pyplot to avoid GUI dependency and circular
# import issues with non-standard matplotlib installations.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_distribution_1d(
    all_steps: Dict[int, Dict[str, Any]],
    reward_name: str,
    output_path: str,
    title: str = "Reward Distribution",
    label_name: str = "Step",
) -> None:
    """Plot 1D reward distributions (overlaid KDE + strip) across steps/epochs.

    Args:
        all_steps: ``{step: {"points": np.ndarray (N, 1), ...}}``
        reward_name: Display name for the single reward dimension.
        output_path: Where to save the PNG.
        title: Figure title.
        label_name: Colorbar label.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = sorted(all_steps.keys())
    if not steps:
        return

    # Strip plot
    fig, (ax_strip, ax_dist) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [1, 2]},
        layout="constrained",
    )

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(steps), vmax=max(steps))

    # Upper panel: strip + range bar per step
    for step in steps:
        data = all_steps[step]
        pts = data.get("points")
        if pts is None or len(pts) == 0:
            continue
        vals = pts[:, 0] if pts.ndim == 2 else pts
        color = cmap(norm(step))
        y_jitter = np.full(len(vals), step, dtype=float)
        ax_strip.scatter(vals, y_jitter, color=color, alpha=0.5, s=8, edgecolors="none")

    ax_strip.set_xlabel(reward_name)
    ax_strip.set_ylabel(label_name)

    # Lower panel: overlaid KDE histograms
    for step in steps:
        data = all_steps[step]
        pts = data.get("points")
        if pts is None or len(pts) == 0:
            continue
        vals = pts[:, 0] if pts.ndim == 2 else pts
        color = cmap(norm(step))
        ax_dist.hist(
            vals,
            bins="auto",
            density=True,
            alpha=0.3,
            color=color,
            histtype="stepfilled",
            linewidth=0.5,
            edgecolor=color,
            label=f"{label_name} {step}",
        )

    ax_dist.set_xlabel(reward_name)
    ax_dist.set_ylabel("Density")
    if len(steps) <= 15:
        ax_dist.legend(fontsize=7, loc="best")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_strip, ax_dist], shrink=0.92, pad=0.02)
    cbar.set_label(label_name)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_percentiles(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    window_size: int = 20,
    title: str = "Reward Percentile Trends",
    label_name: str = "Step",
    force_per_step: bool = False,
) -> None:
    """Plot percentiles of each reward dimension.

    When *force_per_step* is False and ``len(steps) >= window_size``, pools
    reward points across consecutive windows to suppress per-step noise.
    When *force_per_step* is True (e.g. eval datasets with sparse data),
    always uses one data point per step.  Shows median, mean, IQR band,
    and min/max.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = sorted(all_steps.keys())
    if not steps:
        return

    # Build windows — or fall back to per-step when forced or too few steps
    windows: List[Tuple[float, List[int]]] = []
    if force_per_step or len(steps) < window_size:
        for s in steps:
            windows.append((float(s), [s]))
    else:
        lo = 0
        while lo + window_size <= len(steps):
            w_steps = steps[lo : lo + window_size]
            mid = sum(w_steps) / len(w_steps)
            windows.append((mid, w_steps))
            lo += window_size

    n_rewards = len(reward_names)
    fig, axes = plt.subplots(n_rewards, 1, figsize=(10, 3.5 * n_rewards), squeeze=False)

    for ri, rname in enumerate(reward_names):
        ax = axes[ri][0]

        # Pre-pool data per window
        pooled: List[Tuple[int, np.ndarray]] = []
        for wi, (mid, w_steps) in enumerate(windows):
            chunks = []
            for s in w_steps:
                pts = all_steps.get(s, {}).get("points")
                if pts is not None and pts.shape[1] > ri and len(pts) > 0:
                    chunks.append(pts[:, ri])
            if chunks:
                pooled.append((wi, np.concatenate(chunks)))

        statistics = {
            wi: (
                float(np.percentile(values, 25)),
                float(np.percentile(values, 50)),
                float(np.percentile(values, 75)),
                float(values.min()),
                float(values.max()),
                float(values.mean()),
            )
            for wi, values in pooled
        }

        # Assemble ordered lists
        xs, q25s, medians, q75s, mins, maxs, means = [], [], [], [], [], [], []
        for wi, (mid, _) in enumerate(windows):
            result = statistics.get(wi)
            if result is None:
                continue
            q25, q50, q75, vmin, vmax, vmean = result
            xs.append(mid)
            q25s.append(q25)
            medians.append(q50)
            q75s.append(q75)
            mins.append(vmin)
            maxs.append(vmax)
            means.append(vmean)

        ax.fill_between(xs, q25s, q75s, alpha=0.25, color="#2196F3", label="25%-75%")
        ax.plot(xs, medians, "o-", color="#1565C0", linewidth=1.5, markersize=4, label="Median")
        ax.plot(xs, means, "s-", color="#0D47A1", linewidth=1.2, markersize=4, label="Mean")
        ax.plot(xs, mins, ":", color="#90CAF9", linewidth=0.7, label="Min")
        ax.plot(xs, maxs, "--", color="#FFAB91", linewidth=0.7, label="Max")

        ax.set_ylabel(rname)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    axes[-1][0].set_xlabel(label_name)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the Pareto-optimal subset of *points* (maximization in all dims)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2-D point array, got {points.shape}")
    if len(points) == 0:
        return np.empty((0, points.shape[1]))
    if not np.isfinite(points).all():
        raise ValueError("Pareto points must all be finite")

    unique = np.unique(points, axis=0)
    is_pareto = np.ones(len(unique), dtype=bool)
    for i, point in enumerate(unique):
        dominates = np.all(unique >= point, axis=1) & np.any(unique > point, axis=1)
        is_pareto[i] = not np.any(dominates)
    pareto = unique[is_pareto]
    return pareto[np.lexsort(tuple(-pareto[:, i] for i in reversed(range(pareto.shape[1]))))]


def _hypervolume(pareto: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume indicator (Zitzler & Thiele, 1999) via HSO recursion.

    Computes the Lebesgue measure of the union of axis-aligned orthants from
    each Pareto point to the reference point:

        HV = Λ( ⋃_{a∈A} [r₁,a₁] × [r₂,a₂] × … × [r_d,a_d] )

    Implementation: sort by first dimension ascending, sweep left to right.
    At each point i, the vertical slice ``(x_i − x_{i-1})`` is multiplied by
    the (d−1)-dimensional hypervolume of the points at or to its right
    (``pts[i:]``), per the Hypervolume by Slicing Objectives (HSO) scheme.
    """
    if len(pareto) == 0:
        return 0.0
    dim = pareto.shape[1]

    if dim == 1:
        return max(pareto[:, 0].max() - ref[0], 0.0)

    # Sort by first dimension ascending (sweep left to right)
    pts = pareto[np.argsort(pareto[:, 0])]
    hv = 0.0
    prev_x = ref[0]

    for i, pt in enumerate(pts):
        x = pt[0]
        if x <= prev_x:
            continue
        # Take points at or to the right of the current one (pts[i:]).
        # As we sweep left→right, the "height" of each vertical slice is
        # determined by the best remaining (rightward) points in all other
        # dimensions, not the already-processed leftward ones.
        remaining = pts[i:, 1:]
        # Filter: only points that dominate ref in remaining dims
        mask = np.all(remaining >= ref[1:], axis=1)
        if mask.sum() > 0:
            hv_slice = _hypervolume(remaining[mask], ref[1:])
            hv += (x - prev_x) * hv_slice
        prev_x = x

    return hv


def _convexified_hypervolume(pareto: np.ndarray, ref: np.ndarray) -> float:
    """Return the exact dominated volume after convexifying Pareto points.

    The dominated region of ``conv(pareto)`` equals the convex hull of the
    union of boxes ``[ref, p]``.  Every D-dimensional box contributes all
    ``2**D`` vertices; omitting multi-coordinate reference projections gives
    an incorrect volume in three or more dimensions.
    """
    points = np.asarray(pareto, dtype=float)
    reference = np.asarray(ref, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2-D point array, got {points.shape}")
    if reference.shape != (points.shape[1],):
        raise ValueError(
            f"Reference shape {reference.shape} does not match dimension {points.shape[1]}"
        )
    if len(points) == 0:
        return 0.0
    if not np.isfinite(points).all() or not np.isfinite(reference).all():
        raise ValueError("Hypervolume inputs must all be finite")
    if np.any(points < reference - 1e-12):
        raise ValueError("Every Pareto point must weakly dominate the reference point")

    points = _compute_pareto_front(points)
    dim = points.shape[1]
    if dim == 1:
        return float(max(points[:, 0].max() - reference[0], 0.0))

    masks = np.asarray(list(itertools.product((0.0, 1.0), repeat=dim)))
    vertices = np.vstack([reference + masks * (point - reference) for point in points])
    vertices = np.unique(vertices, axis=0)
    if len(vertices) <= dim:
        return 0.0
    if np.linalg.matrix_rank(vertices[1:] - vertices[0], tol=1e-12) < dim:
        return 0.0

    try:
        from scipy.spatial import ConvexHull  # type: ignore[import-untyped]
        from scipy.spatial import QhullError  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("scipy is required for exact convexified hypervolume") from exc

    try:
        return float(ConvexHull(vertices).volume)
    except QhullError as exc:
        raise RuntimeError(
            "Qhull could not compute the exact convexified dominated volume"
        ) from exc


def _is_convex_supported(
    point: np.ndarray,
    pareto: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    """Return whether a Pareto point maximizes a nonnegative reward weighting."""
    try:
        from scipy.optimize import linprog  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("scipy is required for convex-support LPs") from exc

    point = np.asarray(point, dtype=float)
    pareto = np.asarray(pareto, dtype=float)
    # A point is convex-supported exactly when some nonnegative reward weights
    # make its weighted sum at least as large as every other Pareto point. This
    # dual formulation has one variable per reward dimension, rather than one
    # per Pareto point, and avoids the degenerate one-hot feasible solution in
    # the equivalent convex-combination LP.
    gains = pareto - point
    gain_scale = float(np.max(np.abs(gains)))
    if gain_scale == 0.0:
        return True
    scaled_gains = gains / gain_scale
    solver_tolerance = min(1e-7, max(1e-10, tolerance / gain_scale))
    objective = np.zeros(pareto.shape[1])
    solver_messages: List[str] = []
    for method in ("highs-ds", "highs-ipm"):
        result = linprog(
            objective,
            A_ub=scaled_gains,
            b_ub=np.zeros(len(pareto)),
            A_eq=np.ones((1, pareto.shape[1])),
            b_eq=np.ones(1),
            bounds=[(0.0, None)] * pareto.shape[1],
            method=method,
            options={
                "primal_feasibility_tolerance": solver_tolerance,
                "dual_feasibility_tolerance": solver_tolerance,
            },
        )
        if result.success:
            weights = np.asarray(result.x, dtype=float)
            validation_tolerance = 10.0 * solver_tolerance
            if (
                np.all(np.isfinite(weights))
                and float(np.min(weights)) >= -validation_tolerance
                and abs(float(np.sum(weights)) - 1.0) <= validation_tolerance
                and float(np.max(scaled_gains @ weights)) <= validation_tolerance
            ):
                return True
            solver_messages.append(f"{method}: returned an invalid feasible solution")
            continue
        if result.status == 2:
            return False
        solver_messages.append(f"{method}: {result.message}")

    raise RuntimeError(
        "Convex-support LP failed with both HiGHS solvers: " + "; ".join(solver_messages)
    )


def _compute_convexity_metrics(
    points: np.ndarray,
    ref: np.ndarray,
    tolerance: float = 1e-9,
) -> Dict[str, float]:
    """Compute exact convexification-HV and reference-free convexity metrics."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(ref, dtype=float)
    pareto = _compute_pareto_front(points)
    if len(pareto) == 0:
        return {
            "pareto_size": 0,
            "hv_discrete": 0.0,
            "hv_convexified": 0.0,
            "hv_gap_abs": 0.0,
            "hv_gap_ratio": float("nan"),
            "supported_ratio": float("nan"),
        }

    hv_discrete = float(_hypervolume(pareto, reference))
    hv_convexified = _convexified_hypervolume(pareto, reference)
    raw_gap = hv_convexified - hv_discrete
    volume_tolerance = tolerance * max(1.0, abs(hv_convexified))
    if raw_gap < -volume_tolerance:
        raise RuntimeError(
            "Convexified hypervolume is smaller than discrete hypervolume: "
            f"{hv_convexified} < {hv_discrete}"
        )
    gap = max(raw_gap, 0.0)
    ratio = gap / hv_convexified if hv_convexified > volume_tolerance else float("nan")

    supported = np.asarray(
        [_is_convex_supported(point, pareto, tolerance=tolerance) for point in pareto],
        dtype=bool,
    )
    return {
        "pareto_size": int(len(pareto)),
        "hv_discrete": hv_discrete,
        "hv_convexified": hv_convexified,
        "hv_gap_abs": gap,
        "hv_gap_ratio": float(ratio),
        "supported_ratio": float(np.mean(supported)),
    }


MetricSummary = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _metric_summaries(rows: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, MetricSummary]:
    """Aggregate metrics after grouping the group-level rows by step once."""
    rows_by_step: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_step.setdefault(int(row["step"]), []).append(row)
    steps = np.asarray(sorted(rows_by_step), dtype=int)

    summaries: Dict[str, MetricSummary] = {}
    for metric in metrics:
        means: List[float] = []
        medians: List[float] = []
        q25s: List[float] = []
        q75s: List[float] = []
        for step in steps:
            values = np.asarray(
                [float(row[metric]) for row in rows_by_step[int(step)]], dtype=float
            )
            values = values[np.isfinite(values)]
            if len(values) == 0:
                means.append(float("nan"))
                medians.append(float("nan"))
                q25s.append(float("nan"))
                q75s.append(float("nan"))
                continue
            means.append(float(values.mean()))
            medians.append(float(np.median(values)))
            q25s.append(float(np.quantile(values, 0.25)))
            q75s.append(float(np.quantile(values, 0.75)))
        summaries[metric] = (
            steps,
            np.asarray(means),
            np.asarray(medians),
            np.asarray(q25s),
            np.asarray(q75s),
        )
    return summaries


def _metric_summary(rows: List[Dict[str, Any]], metric: str) -> MetricSummary:
    """Aggregate one group-level metric into per-step mean/median/IQR arrays."""
    return _metric_summaries(rows, [metric])[metric]


def _draw_metric(
    ax: Any,
    summary: MetricSummary,
    ylabel: str,
    label_name: str,
    color: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    steps, means, medians, q25s, q75s = summary
    ax.fill_between(steps, q25s, q75s, color=color, alpha=0.18, label="IQR")
    ax.plot(steps, means, color=color, linewidth=1.8, label="Mean")
    ax.plot(steps, medians, color=color, linewidth=1.2, linestyle="--", label="Median")
    ax.set_xlabel(label_name)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _adaptive_metric_ylim(
    summary: MetricSummary,
    metric: str,
    bounds: Tuple[float, float] = (0.0, 1.0),
    padding_fraction: float = 0.08,
    minimum_span: float = 0.02,
    upper_boundary_headroom: float = 0.02,
) -> Tuple[float, float]:
    """Choose data-adaptive y-limits while validating the metric domain."""
    _, means, medians, q25s, q75s = summary
    values = np.concatenate([means, medians, q25s, q75s])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return bounds

    lower_bound, upper_bound = bounds
    observed_lower = float(values.min())
    observed_upper = float(values.max())
    if observed_lower < lower_bound or observed_upper > upper_bound:
        raise ValueError(
            f"Plotted {metric} statistics fall outside [{lower_bound}, {upper_bound}]: "
            f"[{observed_lower}, {observed_upper}]"
        )
    observed_span = max(observed_upper - observed_lower, 0.0)
    target_span = max(observed_span * (1.0 + 2.0 * padding_fraction), minimum_span)
    target_span = min(target_span, upper_bound - lower_bound)
    center = 0.5 * (observed_lower + observed_upper)
    lower = center - 0.5 * target_span
    upper = center + 0.5 * target_span
    if lower < lower_bound:
        upper += lower_bound - lower
        lower = lower_bound
    if upper > upper_bound:
        lower -= upper - upper_bound
        upper = upper_bound
    if np.isclose(observed_upper, upper_bound):
        upper += upper_boundary_headroom * (upper_bound - lower_bound)
    return max(lower, lower_bound), upper


def _compute_step_convexity_rows(
    step: int,
    points: np.ndarray,
    prompt_idx: np.ndarray,
    lows: np.ndarray,
    scales: np.ndarray,
    reference: np.ndarray,
    reward_names: List[str],
    tolerance: float,
) -> List[Dict[str, Any]]:
    """Compute all group metrics for one step; safe for process workers."""
    normalized = (points - lows) / scales
    if np.any(normalized < -tolerance) or np.any(normalized > 1.0 + tolerance):
        raise ValueError(f"Step {step} falls outside the fixed normalization bounds")
    normalized = np.clip(normalized, 0.0, 1.0)
    rows: List[Dict[str, Any]] = []
    for gid in sorted(set(prompt_idx.tolist())):
        group_points = normalized[prompt_idx == gid]
        try:
            metrics = _compute_convexity_metrics(group_points, reference, tolerance=tolerance)
        except Exception as exc:
            print(
                f"[ERROR] Pareto metric computation failed for step={step}, group_id={gid}, "
                f"n_points={len(group_points)}: {type(exc).__name__}: {exc}"
            )
            print(f"  group_points (normalized):\n{group_points}")
            print(f"  reference: {reference}")
            raise
        rows.append(
            {
                "step": int(step),
                "reward_combination": "__".join(reward_names),
                "group_id": int(gid),
                "n_samples": int(len(group_points)),
                **metrics,
            }
        )
    return rows


def _resolve_worker_count(n_items: int, max_workers: int) -> int:
    """Resolve an explicit or auto-sized process count for step computations."""
    if n_items < 1:
        raise ValueError(f"n_items must be positive, got {n_items}")
    if max_workers < 0:
        raise ValueError(f"max_workers must be non-negative, got {max_workers}")
    worker_limit = min(os.cpu_count() or 4, 16) if max_workers == 0 else max_workers
    return min(worker_limit, n_items)


def plot_pareto_convexity_metrics(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_dir: str,
    title: str = "Pareto Convexity Metrics",
    label_name: str = "Step",
    normalization_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    tolerance: float = 1e-9,
    max_workers: int = 0,
) -> None:
    """Compute and plot group-aware Pareto convexification metrics.

    A fixed run-wide affine normalization is applied before all metrics.  The
    normalized reference point is the origin.  The function writes one PNG
    overview, four publication-ready vector PDFs, four editable SVGs,
    group-level CSV data, and a JSON metadata record into ``output_dir``.

    Args:
        all_steps: Mapping from step to aligned reward points and prompt-group
            indices.
        reward_names: Reward dimensions in the same order as point columns.
        output_dir: Directory for the overview, vector figures, CSV, and metadata.
        title: Overview title.
        label_name: X-axis label.
        normalization_bounds: Optional fixed min/max pair for each reward.
        tolerance: Numerical tolerance in normalized reward coordinates.
        max_workers: Process workers for independent step computations. Zero
            selects up to 16 workers based on available CPUs.
    """
    if len(reward_names) < 2:
        return
    total_start = time.perf_counter()
    steps_without_groups = [
        int(step)
        for step, data in all_steps.items()
        if data.get("points") is not None
        and len(data.get("points")) > 0
        and data.get("prompt_idx") is None
    ]
    if steps_without_groups:
        for filename in ("overview.png", "per_group_metrics.csv", "metadata.json"):
            path = os.path.join(output_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
        figures_dir = os.path.join(output_dir, "figures")
        if os.path.isdir(figures_dir):
            shutil.rmtree(figures_dir)
        print(
            "  [Plot] Pareto convexity skipped because prompt-group indices "
            f"are missing at steps {steps_without_groups}"
        )
        return
    dim = len(reward_names)
    point_blocks: List[np.ndarray] = []
    for step, data in sorted(all_steps.items()):
        points = np.asarray(data.get("points", []), dtype=float)
        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1] != dim:
            raise ValueError(f"Step {step} has point shape {points.shape}, expected (*, {dim})")
        if not np.isfinite(points).all():
            raise ValueError(f"Step {step} contains non-finite reward points")
        point_blocks.append(points)
    if not point_blocks:
        print(f"  [Plot] No valid points for {reward_names}; convexity plots skipped")
        return

    all_points = np.vstack(point_blocks)
    if normalization_bounds is None:
        lows = all_points.min(axis=0)
        highs = all_points.max(axis=0)
        bounds = {name: (float(lows[i]), float(highs[i])) for i, name in enumerate(reward_names)}
    else:
        missing = [name for name in reward_names if name not in normalization_bounds]
        if missing:
            raise ValueError(f"Missing normalization bounds for rewards: {missing}")
        bounds = {name: normalization_bounds[name] for name in reward_names}
        lows = np.asarray([bounds[name][0] for name in reward_names], dtype=float)
        highs = np.asarray([bounds[name][1] for name in reward_names], dtype=float)

    scales = highs - lows
    degenerate = [reward_names[i] for i in range(dim) if scales[i] <= tolerance]
    if degenerate:
        raise ValueError(f"Degenerate normalization ranges for rewards: {degenerate}")
    reference = np.zeros(dim, dtype=float)

    step_items: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for step, data in sorted(all_steps.items()):
        points = np.asarray(data.get("points", []), dtype=float)
        if points.size == 0:
            continue
        prompt_idx = data.get("prompt_idx")
        if prompt_idx is None:
            prompt_idx = np.zeros(len(points), dtype=int)
        prompt_idx = np.asarray(prompt_idx, dtype=int)
        if len(prompt_idx) != len(points):
            raise ValueError(
                f"Step {step} has {len(points)} points but " f"{len(prompt_idx)} prompt indices"
            )
        step_items.append((int(step), points, prompt_idx))

    resolved_workers = _resolve_worker_count(len(step_items), max_workers)
    execution_workers = 1 if len(step_items) < 5 else resolved_workers
    total_groups = 0
    group_sizes: List[int] = []
    for _, _, prompt_idx in step_items:
        _, counts = np.unique(prompt_idx, return_counts=True)
        total_groups += len(counts)
        group_sizes.extend(int(count) for count in counts)
    group_size_range = (
        str(group_sizes[0])
        if len(set(group_sizes)) == 1
        else f"{min(group_sizes)}-{max(group_sizes)}"
    )
    worker_setting = "auto" if max_workers == 0 else "explicit"
    normalization_label = (
        "derived from this reward combination"
        if normalization_bounds is None
        else "fixed run-wide min/max"
    )
    combination_label = " + ".join(reward_names)
    print(f"\n[Pareto] {combination_label}")
    print(f"  Dimension: {dim}")
    print(f"  Steps: {len(step_items)}")
    print(f"  Groups: {total_groups}")
    print(f"  Samples/group: {group_size_range}")
    print(
        f"  Workers: {execution_workers} " f"(compute.max_workers({max_workers}, {worker_setting}))"
    )
    print(f"  Normalization: {normalization_label}")

    geometry_start = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    if len(step_items) < 5:
        for step, points, prompt_idx in tqdm(
            step_items,
            desc=f"  Pareto geometry ({dim}D)",
            unit="step",
        ):
            try:
                rows.extend(
                    _compute_step_convexity_rows(
                        step,
                        points,
                        prompt_idx,
                        lows,
                        scales,
                        reference,
                        reward_names,
                        tolerance,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Pareto geometry failed at step({step}) for "
                    f"reward_combination({combination_label})"
                ) from exc
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = {
                executor.submit(
                    _compute_step_convexity_rows,
                    step,
                    points,
                    prompt_idx,
                    lows,
                    scales,
                    reference,
                    reward_names,
                    tolerance,
                ): step
                for step, points, prompt_idx in step_items
            }
            rows_by_step: Dict[int, List[Dict[str, Any]]] = {}
            with tqdm(
                total=len(futures),
                desc=f"  Pareto geometry ({dim}D)",
                unit="step",
            ) as progress:
                for future in as_completed(futures):
                    step = futures[future]
                    try:
                        rows_by_step[step] = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Pareto geometry failed at step({step}) for "
                            f"reward_combination({combination_label})"
                        ) from exc
                    progress.update(1)
        for step in sorted(rows_by_step):
            rows.extend(rows_by_step[step])
    geometry_elapsed = time.perf_counter() - geometry_start
    print(f"  Geometry computation: {geometry_elapsed:.1f}s")

    if not rows:
        print(f"  [Plot] No group metrics for {reward_names}; output skipped")
        return

    aggregation_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    pdf_dir = os.path.join(figures_dir, "pdf")
    svg_dir = os.path.join(figures_dir, "svg")
    if os.path.isdir(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    fieldnames = [
        "step",
        "reward_combination",
        "group_id",
        "n_samples",
        "pareto_size",
        "hv_discrete",
        "hv_convexified",
        "hv_gap_abs",
        "hv_gap_ratio",
        "supported_ratio",
    ]
    with open(os.path.join(output_dir, "per_group_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "pareto_size",
        "hv_gap_abs",
        "hv_gap_ratio",
        "supported_ratio",
    ]
    summaries = _metric_summaries(rows, metric_names)
    groups_per_step: Dict[str, int] = {}
    for row in rows:
        step_key = str(int(row["step"]))
        groups_per_step[step_key] = groups_per_step.get(step_key, 0) + 1

    metadata = {
        "metric_version": 3,
        "reward_names": list(reward_names),
        "dimension": dim,
        "n_steps": len({int(row["step"]) for row in rows}),
        "groups_per_step": dict(sorted(groups_per_step.items(), key=lambda item: int(item[0]))),
        "normalization": {
            name: {"min": float(bounds[name][0]), "max": float(bounds[name][1])}
            for name in reward_names
        },
        "reference_point": reference.tolist(),
        "tolerance": tolerance,
        "compute_workers": execution_workers,
        "definitions": {
            "hv_gap_abs": "HV(conv(P); r) - HV(P; r)",
            "hv_gap_ratio": "hv_gap_abs / HV(conv(P); r)",
            "supported_ratio": "fraction of Pareto points not convexly dominated",
        },
        "interpretation": {
            "pareto_size": "Higher means more observed non-dominated trade-offs; it is not a direct quality or convexity score.",
            "hv_gap_abs": "Values closer to zero are more convex; higher values mean stronger volume-level non-convexity in normalized coordinates.",
            "hv_gap_ratio": "Values closer to zero are more convex; higher values mean a larger fraction of convexified dominated volume is missing.",
            "supported_ratio": "Values closer to one are more convex; lower values mean more Pareto points lie below the convex envelope.",
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    aggregation_elapsed = time.perf_counter() - aggregation_start
    print(f"  Metric aggregation/export: {aggregation_elapsed:.1f}s")

    figure_start = time.perf_counter()
    ratio_ylim = _adaptive_metric_ylim(summaries["hv_gap_ratio"], "hv_gap_ratio")
    supported_ylim = _adaptive_metric_ylim(summaries["supported_ratio"], "supported_ratio")
    plot_specs = [
        (
            "pareto_size",
            "Pareto Front Size",
            "#2E7D32",
            None,
            "pareto_front_size.pdf",
        ),
        (
            "hv_gap_abs",
            "Absolute Convexification HV Gap",
            "#C62828",
            (0.0, None),
            "convexification_hv_gap_absolute.pdf",
        ),
        (
            "hv_gap_ratio",
            "Relative Convexification HV Gap",
            "#1565C0",
            ratio_ylim,
            "convexification_hv_gap_relative.pdf",
        ),
        (
            "supported_ratio",
            "Convex-Supported Pareto Ratio",
            "#00838F",
            supported_ylim,
            "convex_supported_ratio.pdf",
        ),
    ]

    with plt.rc_context(
        {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 10}
    ):
        fig = plt.figure(figsize=(13.5, 11.5))
        grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.48))
        metric_axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
        ]
        for ax, (metric, ylabel, color, ylim, _) in zip(metric_axes, plot_specs):
            _draw_metric(ax, summaries[metric], ylabel, label_name, color, ylim)
        interpretation_ax = fig.add_subplot(grid[2, :])
        interpretation_ax.axis("off")
        interpretation_ax.text(
            0.0,
            1.0,
            "How to interpret (qualitative; no universal cutoff)\n\n"
            "Pareto front size — non-dominated samples per prompt group.\n"
            "  ↑ more observed trade-offs; ↓ fewer. Not a quality/convexity score.\n\n"
            "Absolute HV gap — HV(conv(P)) − HV(P), in normalized space.\n"
            "  0 / ↓ closer to convex; ↑ stronger volume-level non-convexity.",
            va="top",
            fontsize=8.6,
            linespacing=1.18,
        )
        interpretation_ax.text(
            0.52,
            1.0,
            "Relative HV gap — absolute gap / HV(conv(P)).\n"
            "  0 / ↓ closer to convex; ↑ larger missing convexified-volume fraction.\n\n"
            "Convex-supported ratio — supported / all Pareto points.\n"
            "  1 / ↑ closer to convex; ↓ more points below the convex envelope.\n\n"
            f"Rewards: {', '.join(reward_names)} | Dimension: {dim} | "
            f"Steps: {metadata['n_steps']}\n"
            "Computed within each prompt group, then summarized across groups.\n"
            "Fixed run-wide normalization; HV reference = origin; shading = group IQR.",
            va="top",
            fontsize=8.6,
            linespacing=1.18,
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(output_dir, "overview.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

        for metric, ylabel, color, ylim, filename in plot_specs:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            _draw_metric(ax, summaries[metric], ylabel, label_name, color, ylim)
            fig.tight_layout()
            fig.savefig(os.path.join(pdf_dir, filename), bbox_inches="tight")
            svg_filename = f"{os.path.splitext(filename)[0]}.svg"
            fig.savefig(os.path.join(svg_dir, svg_filename), bbox_inches="tight")
            plt.close(fig)

    figure_elapsed = time.perf_counter() - figure_start
    print(f"  Figure generation: {figure_elapsed:.1f}s")
    print("  Wrote: 1 PNG, 4 PDFs, 4 SVGs, 1 CSV, 1 JSON")
    print(f"  Output: {output_dir}")
    print(f"  Completed in {time.perf_counter() - total_start:.1f}s")
