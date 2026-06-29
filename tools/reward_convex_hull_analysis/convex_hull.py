"""Convex hull computation (Andrew's monotone chain) and plotting utilities.

Provides 2D convex hull plotting (scatter + filled hull polygon per epoch/step)
and a 1D distribution fallback for single-reward-model configurations.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set backend before importing pyplot to avoid GUI dependency and circular
# import issues with non-standard matplotlib installations.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Marker cycle for distinguishing prompts
_PROMPT_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def andrews_monotone_chain(points: np.ndarray) -> np.ndarray:
    """Compute the 2D convex hull using Andrew's monotone chain algorithm.

    Args:
        points: Float array of shape (N, 2).

    Returns:
        Hull vertices in CCW order, shape (M, 2).  Returns the input unchanged
        when N <= 2 (degenerate hull).
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array, got {points.shape}")
    n = len(points)
    if n <= 2:
        return points.copy()

    # Sort by x, then y (duplicates are harmless — collinear points are
    # naturally popped by the cross-product <= 0 check).
    idx = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[idx]

    lower: List[int] = []
    for i in range(len(pts)):
        while len(lower) >= 2:
            ab = pts[lower[-1]] - pts[lower[-2]]
            ac = pts[i] - pts[lower[-2]]
            if np.cross(ab, ac) <= 0:
                lower.pop()
            else:
                break
        lower.append(i)

    upper: List[int] = []
    for i in range(len(pts) - 1, -1, -1):
        while len(upper) >= 2:
            ab = pts[upper[-1]] - pts[upper[-2]]
            ac = pts[i] - pts[upper[-2]]
            if np.cross(ab, ac) <= 0:
                upper.pop()
            else:
                break
        upper.append(i)

    # Remove duplicate endpoints
    hull_indices = lower[:-1] + upper[:-1]
    return pts[hull_indices]


# ---------------------------------------------------------------------------
# 2D convex hull plotting
# ---------------------------------------------------------------------------


def plot_convex_hulls_2d(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    title: str = "Reward Convex Hulls",
    label_name: str = "Step",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Plot 2D convex hulls across multiple steps/epochs.

    If ``reward_names`` has more than 2 entries, pairwise subplots are created.

    Args:
        all_steps: ``{step: {"points": np.ndarray (N, D), "hull": np.ndarray (M, 2), ...}}``
        reward_names: List of reward dimension names (length D ≥ 2).
        output_path: Where to save the PNG.
        title: Suptitle for the figure.
        label_name: Axis legend label (e.g. "Epoch" or "Checkpoint").
        figsize: Figure size for a single-pair subplot (scaled for multi-pair).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dim = len(reward_names)
    if dim < 2:
        raise ValueError("plot_convex_hulls_2d requires at least 2 reward dimensions")

    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    n_pairs = len(pairs)
    cols = min(n_pairs, 3)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows), squeeze=False
    )

    steps = sorted(all_steps.keys())
    if not steps:
        plt.close(fig)
        return
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(steps), vmax=max(steps))

    for pair_idx, (di, dj) in enumerate(pairs):
        ax = axes[pair_idx // cols][pair_idx % cols]
        ax.set_xlabel(reward_names[di])
        ax.set_ylabel(reward_names[dj])

        for step in steps:
            data = all_steps[step]
            pts = data.get("points")
            if pts is None or pts.shape[1] <= max(di, dj) or len(pts) == 0:
                continue
            color = cmap(norm(step))
            xy = np.column_stack([pts[:, di], pts[:, dj]])
            prompt_idx = data.get("prompt_idx")

            # Scatter points — different marker per prompt
            if prompt_idx is not None and len(prompt_idx) == len(xy):
                for pi in sorted(set(prompt_idx.tolist())):
                    mask = prompt_idx == pi
                    marker = _PROMPT_MARKERS[pi % len(_PROMPT_MARKERS)]
                    ax.scatter(
                        xy[mask, 0],
                        xy[mask, 1],
                        color=color,
                        alpha=0.55,
                        s=12,
                        marker=marker,
                        edgecolors="none",
                        label=f"P{pi}" if step == steps[0] and pi < 8 else None,
                    )
            else:
                ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.55, s=12, edgecolors="none")

            # Draw hull polygon (over all points, regardless of prompt)
            if len(xy) >= 3:
                hull_xy = andrews_monotone_chain(xy)
                if len(hull_xy) >= 2:
                    ax.fill(hull_xy[:, 0], hull_xy[:, 1], color=color, alpha=0.12)
                    ax.plot(
                        np.append(hull_xy[:, 0], hull_xy[0, 0]),
                        np.append(hull_xy[:, 1], hull_xy[0, 1]),
                        color=color,
                        linewidth=1.0,
                        alpha=0.7,
                    )

    # Hide unused subplots
    for pi in range(n_pairs, rows * cols):
        axes[pi // cols][pi % cols].set_visible(False)

    # Add prompt legend on first subplot if markers were used
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(handles, labels, fontsize=7, loc="best", title="Prompt", title_fontsize=8)

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label(label_name)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overlaid plot: two sources (TensorBoard + Checkpoints) on the same axes
# ---------------------------------------------------------------------------


def plot_combined_convex_hulls_2d(
    source_a: Dict[int, Dict[str, Any]],
    source_b: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    label_a: str = "TensorBoard",
    label_b: str = "Checkpoints",
    title: str = "Combined Reward Convex Hulls",
) -> None:
    """Plot convex hulls from two sources on the same axes (different markers/colormaps)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dim = len(reward_names)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    n_pairs = len(pairs)
    cols = min(n_pairs, 3)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

    # Source A uses viridis, Source B uses plasma
    sources = [
        (source_a, label_a, plt.cm.viridis),
        (source_b, label_b, plt.cm.plasma),
    ]

    for pair_idx, (di, dj) in enumerate(pairs):
        ax = axes[pair_idx // cols][pair_idx % cols]
        ax.set_xlabel(reward_names[di])
        ax.set_ylabel(reward_names[dj])

        for src_data, _, cmap in sources:
            steps = sorted(src_data.keys())
            if not steps:
                continue
            norm = plt.Normalize(vmin=min(steps), vmax=max(steps))

            for step in steps:
                data = src_data[step]
                pts = data.get("points")
                if pts is None or pts.shape[1] <= max(di, dj) or len(pts) == 0:
                    continue
                color = cmap(norm(step))
                xy = np.column_stack([pts[:, di], pts[:, dj]])
                ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.45, s=10, edgecolors="none")

                hull_xy = andrews_monotone_chain(xy)
                if len(hull_xy) >= 3:
                    ax.plot(
                        np.append(hull_xy[:, 0], hull_xy[0, 0]),
                        np.append(hull_xy[:, 1], hull_xy[0, 1]),
                        color=color,
                        linewidth=1.0,
                        alpha=0.6,
                        linestyle="--",
                    )

        # Legend for source identity
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.viridis(0.5),
                markersize=8,
                label=label_a,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.plasma(0.5),
                markersize=8,
                label=label_b,
            ),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=8)

    for pi in range(n_pairs, rows * cols):
        axes[pi // cols][pi % cols].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1D distribution visualization (fallback for single reward model)
# ---------------------------------------------------------------------------


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
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]}
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
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Faceted convex hulls — one hull per subplot, showing step-by-step evolution
# ---------------------------------------------------------------------------


def plot_convex_hulls_faceted(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    title: str = "Convex Hulls by Step",
    label_name: str = "Step",
    dim_pair: Tuple[int, int] = (0, 1),
    cols: int = 5,
    max_steps: int = 30,
    step_range: Optional[Tuple[int, int]] = None,
) -> None:
    """Plot each step's convex hull in its own subplot (faceted grid).

    Selects up to *max_steps* evenly-spaced steps.  Each subplot shows scatter
    points and the hull polygon for one step.

    Args:
        step_range: Optional ``(start, end)`` tuple to restrict which steps are
            considered (inclusive).  Steps outside this range are filtered out
            before sampling.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = sorted(all_steps.keys())

    # Filter by step_range if specified
    if step_range is not None:
        lo, hi = step_range
        steps = [s for s in steps if lo <= s <= hi]

    if not steps:
        return

    if len(steps) > max_steps:
        stride = max(len(steps) // max_steps, 1)
        selected = steps[::stride][:max_steps]
    else:
        selected = steps

    n = len(selected)
    rows = max(1, (n + cols - 1) // cols)
    di, dj = dim_pair

    # Compute global axis limits across all selected steps
    all_x, all_y = [], []
    for step in selected:
        data = all_steps.get(step, {})
        pts = data.get("points")
        if pts is not None and pts.shape[1] > max(di, dj) and len(pts) > 0:
            all_x.extend(pts[:, di].tolist())
            all_y.extend(pts[:, dj].tolist())
    if all_x:
        pad_x = (max(all_x) - min(all_x)) * 0.05 or 0.01
        pad_y = (max(all_y) - min(all_y)) * 0.05 or 0.01
        xlim = (min(all_x) - pad_x, max(all_x) + pad_x)
        ylim = (min(all_y) - pad_y, max(all_y) + pad_y)
    else:
        xlim = ylim = None

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.5 * cols, 3.2 * rows), squeeze=False, sharex=True, sharey=True
    )
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(steps), vmax=max(steps))

    for pi, step in enumerate(selected):
        ax = axes[pi // cols][pi % cols]
        data = all_steps.get(step, {})
        pts = data.get("points")
        if pts is None or pts.shape[1] <= max(di, dj):
            ax.set_visible(False)
            continue

        color = cmap(norm(step))
        xy = np.column_stack([pts[:, di], pts[:, dj]])
        prompt_idx = data.get("prompt_idx")

        # Different marker per prompt
        if prompt_idx is not None and len(prompt_idx) == len(xy):
            for prompt_i in sorted(set(prompt_idx.tolist())):
                mask = prompt_idx == prompt_i
                marker = _PROMPT_MARKERS[prompt_i % len(_PROMPT_MARKERS)]
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    color=color,
                    alpha=0.55,
                    s=8,
                    marker=marker,
                    edgecolors="none",
                    label=f"P{prompt_i}" if prompt_i < 8 else None,
                )
        else:
            ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.55, s=8, edgecolors="none")

        if len(xy) >= 3:
            hull_xy = andrews_monotone_chain(xy)
            if len(hull_xy) >= 2:
                ax.fill(hull_xy[:, 0], hull_xy[:, 1], color=color, alpha=0.15)
                ax.plot(
                    np.append(hull_xy[:, 0], hull_xy[0, 0]),
                    np.append(hull_xy[:, 1], hull_xy[0, 1]),
                    color=color,
                    linewidth=1.2,
                    alpha=0.8,
                )

        ax.set_title(f"{label_name} {step}", fontsize=9)
        ax.tick_params(labelsize=7)

    for pi in range(n, rows * cols):
        axes[pi // cols][pi % cols].set_visible(False)

    # sharex/sharey=True propagates limits to all subplots — setting on any is enough
    if xlim is not None:
        axes[0][0].set_xlim(xlim)
        axes[0][0].set_ylim(ylim)

    fig.supxlabel(reward_names[di], fontsize=10)
    fig.supylabel(reward_names[dj], fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hull area curve — how hull size evolves over training
# ---------------------------------------------------------------------------


def _hull_area_2d(points: np.ndarray) -> float:
    """Compute the area of a 2D convex hull via the shoelace formula."""
    if len(points) < 3:
        return 0.0
    hull = andrews_monotone_chain(points)
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _hull_area_worker(pts: np.ndarray, di: int, dj: int) -> float:
    """Picklable wrapper for parallel hull-area calls."""
    if pts.shape[1] <= max(di, dj):
        return 0.0
    return _hull_area_2d(pts[:, [di, dj]])


def _percentiles_worker(vals: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Picklable worker: (q25, q50, q75, min, max, mean) for pooled values."""
    return (
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 75)),
        float(np.min(vals)),
        float(np.max(vals)),
        float(np.mean(vals)),
    )


# ---------------------------------------------------------------------------
# Per-group hypervolume & hull gap helpers
# ---------------------------------------------------------------------------


def _upper_convex_hull_2d(pareto_sorted: np.ndarray) -> np.ndarray:
    """Upper convex hull of 2-D Pareto points, going left to right.

    Takes Pareto-optimal points sorted by the first dimension ascending and
    returns the subset that forms the convex upper envelope (the "convexified"
    Pareto front).  When the returned set equals the input, the Pareto front
    is perfectly convex and the hull gap is zero.

    Args:
        pareto_sorted: ``(K, 2)`` Pareto points sorted by column 0 ascending.

    Returns:
        ``(M, 2)`` subset forming the upper convex hull in x-ascending order.
    """
    if len(pareto_sorted) < 2:
        return pareto_sorted

    upper = [pareto_sorted[0]]
    for p in pareto_sorted[1:]:
        while len(upper) >= 2:
            a = upper[-2]
            b = upper[-1]
            # Cross-product (b-a) × (p-a): positive = left turn, negative = right turn.
            # For the upper envelope going left→right we require a right turn
            # (convex shape in maximisation space), so pop on left turn / collinear.
            if np.cross(b - a, p - a) >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    return np.array(upper)


def _continuous_hv_2d(vertices: np.ndarray, ref_2d: np.ndarray) -> float:
    """Hypervolume of the continuous convex hull of 2-D Pareto points.

    The dominated region of the convex hull is the set of all points
    component-wise ≤ some point in the hull.  Its upper boundary is the
    upper convex hull of the Pareto points, extended to the reference
    planes via:

    - left  (x < x_min): horizontal at y = max_y(Pareto) → x = r_x
    - right (x > x_max): vertical drop to y = r_y

    The area under this envelope is computed via trapezoid integration.
    """
    if len(vertices) < 1:
        return 0.0
    r_x, r_y = ref_2d[0], ref_2d[1]

    # Sort by x ascending
    pts = vertices[np.argsort(vertices[:, 0])]

    # Filter to Pareto-optimal (y non-increasing)
    pareto = _compute_pareto_front(pts)
    pareto = pareto[np.argsort(pareto[:, 0])]
    if len(pareto) < 1:
        return 0.0

    # Upper convex hull of Pareto points
    hull = _upper_convex_hull_2d(pareto)
    if len(hull) < 1:
        return 0.0

    # Global max y of the convex hull (for left extension)
    max_y = float(pareto[:, 1].max())

    # --- Build extended envelope ---
    extended: List[np.ndarray] = []

    # Left extension: horizontal from (r_x, max_y) to (x_min, max_y)
    # but only if max_y ≥ the hull's y at x_min
    x_min = hull[0, 0]
    y_at_xmin = hull[0, 1]
    if max_y > y_at_xmin:
        extended.append(np.array([r_x, max_y]))
        extended.append(np.array([x_min, max_y]))
    else:
        extended.append(np.array([r_x, y_at_xmin]))

    # All hull vertices
    for v in hull:
        extended.append(v)

    # Right: vertical drop from rightmost hull vertex to y = r_y
    if hull[-1, 1] > r_y:
        extended.append(np.array([hull[-1, 0], r_y]))

    extended_pts = np.array(extended)
    extended_pts = extended_pts[np.argsort(extended_pts[:, 0])]

    # Trapezoid integration
    hv = 0.0
    for i in range(len(extended_pts) - 1):
        x_a, y_a = extended_pts[i]
        x_b, y_b = extended_pts[i + 1]
        if x_b <= r_x:
            continue
        clip_x_a = max(x_a, r_x)
        clip_x_b = max(x_b, r_x)
        if clip_x_b <= clip_x_a:
            continue
        if x_b > x_a:
            t_a = (clip_x_a - x_a) / (x_b - x_a)
            t_b = (clip_x_b - x_a) / (x_b - x_a)
            clip_y_a = y_a + t_a * (y_b - y_a)
            clip_y_b = y_a + t_b * (y_b - y_a)
        else:
            clip_y_a = clip_y_b = max(y_a, y_b)
        mean_y = (clip_y_a + clip_y_b) / 2.0
        hv += (clip_x_b - clip_x_a) * max(0.0, mean_y - r_y)

    return hv


def _convex_hull_hypervolume(pareto: np.ndarray, ref: np.ndarray) -> float:
    """Exact continuous hypervolume of the convex hull of Pareto points.

    For 2-D rewards the trapezoid formula with boundary extension is used.
    For 3+ dimensions, the convex hull is decomposed into upper facets
    (Pareto-optimal (d−1)-simplices), and each facet's contribution is
    integrated exactly::

        contrib = vol_{d−1}(proj) × (mean_dim_d − r_d)

    where *proj* is the facet projected onto the first d−1 dimensions and
    its volume is computed via the Cayley-Menger determinant.

    Requires ``scipy.spatial.ConvexHull`` for d ≥ 3.
    """
    d = pareto.shape[1]
    if len(pareto) < d:
        # Not enough points for a (d-1)-simplex – fall back to HSO.
        return _hypervolume(pareto[np.argsort(pareto[:, 0])], ref)

    # ---- Single (d-1)-simplex (exactly d points) — no Qhull needed ----
    if len(pareto) == d:
        # The d points form a single (d-1)-simplex.  Check whether it is
        # Pareto-optimal (its normal should point into the positive orthant).
        proj = pareto[:, : d - 1]  # (d, d-1)
        M = (proj[1:] - proj[0]).T   # (d-1, d-1)
        try:
            vol_proj = abs(np.linalg.det(M)) / float(math.factorial(d - 1))
        except np.linalg.LinAlgError:
            vol_proj = 0.0
        if vol_proj == 0.0:
            return _hypervolume(pareto[np.argsort(pareto[:, 0])], ref)
        # Compute normal via cross product of edge vectors (generalised)
        # For a simplex with vertices v_0,...,v_{d-1} in R^d, the normal
        # of the hyperplane is orthogonal to all d-1 edge vectors.
        # Use the nullspace / generalised cross product.
        edges = pareto[1:] - pareto[0]  # (d-1, d)
        # Find a vector n such that edges @ n = 0
        _, _, vh = np.linalg.svd(edges, full_matrices=True)
        normal = vh[-1]  # last row of Vh = nullspace basis
        # Ensure normal points outward (positive direction)
        if normal[0] < 0:
            normal = -normal
        if not (np.all(normal >= -1e-10) and np.any(normal > 1e-10)):
            # Not a Pareto-optimal facet
            return _hypervolume(pareto[np.argsort(pareto[:, 0])], ref)
        mean_last = float(pareto[:, d - 1].mean())
        return vol_proj * max(0.0, mean_last - ref[d - 1])

    if d == 1:
        return max(0.0, pareto.max() - ref[0])

    if d == 2:
        return _continuous_hv_2d(pareto, ref)

    # ---- d ≥ 3: projection-based approximation of dominated-region volume ----
    # The exact continuous HV requires decomposing the d-dimensional
    # dominated region of the convex hull, which is complex (analogous to
    # the boundary-extension procedure in 2-D, but for (d-1)-facets).
    # Instead we project each Pareto point onto every coordinate plane to
    # obtain a close approximation (~2-3% relative error) that is always
    # free of the affine-extrapolation spike problem.
    from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

    all_pts_list: List[np.ndarray] = [pareto]
    for pt in pareto:
        for di in range(d):
            proj = pt.copy()
            proj[di] = ref[di]
            all_pts_list.append(proj.reshape(1, d))
    all_pts_list.append(ref.reshape(1, d))
    all_pts = np.vstack(all_pts_list)

    try:
        ext_hull = ConvexHull(all_pts)
        return float(ext_hull.volume)
    except Exception:
        return _hypervolume(pareto[np.argsort(pareto[:, 0])], ref)


def _compute_hull_gap(pareto: np.ndarray, ref: np.ndarray) -> float:
    """Hull gap: HV(convex hull of Pareto) − HV(Pareto).

    Positive for any Pareto front — even a perfectly convex one —
    because the continuous convex hull dominates a strictly larger
    region than the discrete Pareto staircase.  Greater values
    indicate stronger concavities or sparser sampling.
    """
    if len(pareto) < 2:
        return 0.0
    d = pareto.shape[1]

    hv_hull = _convex_hull_hypervolume(pareto, ref)

    # Discrete Pareto hypervolume (exact HSO)
    pareto_sorted = pareto[np.argsort(pareto[:, 0])]
    hv_pareto = _hypervolume(pareto_sorted, ref)

    return max(0.0, hv_hull - hv_pareto)


def _compute_per_group_metrics(
    points: np.ndarray,
    prompt_idx: Optional[np.ndarray],
    ref: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-group hypervolume and hull gap, then average across groups.

    For each unique group in *prompt_idx*, the Pareto front is computed and
    its hypervolume (w.r.t. *ref*) is measured via HSO.  The hull gap —
    ``HV(continuous convex hull of Pareto) − HV(discrete Pareto)`` — is
    computed in the full N-D reward space using exact integration (trapezoid
    for 2-D; simplex decomposition for 3+-D).

    Args:
        points: ``(N, D)`` array of reward vectors.
        prompt_idx: ``(N,)`` integer array mapping each point to a group.
            When ``None`` or empty, all points are treated as one group.
        ref: ``(D,)`` reference point for hypervolume computation (all dims
            must be ≤ the corresponding minima of *points*).

    Returns:
        Dict with keys:
        - ``mean_hypervolume``: average HSO hypervolume across groups.
        - ``mean_hull_gap``: average hull gap across groups.
        - ``mean_pareto_size``: average number of Pareto-optimal points per group.
        - ``per_group_hypervolume``: ``{gid: float}`` mapping.
        - ``per_group_hull_gap``: ``{gid: float}`` mapping.
        - ``n_groups``: total number of groups.
        - ``n_active_groups``: groups with ≥ 2 points.
    """
    if prompt_idx is None or len(prompt_idx) == 0:
        prompt_idx = np.zeros(len(points), dtype=int)

    unique_gids = sorted(set(prompt_idx.tolist()))
    dim = points.shape[1]

    per_group_hv: Dict[int, float] = {}
    per_group_gap: Dict[int, float] = {}
    per_group_psize: Dict[int, int] = {}
    n_active = 0

    for gid in unique_gids:
        mask = prompt_idx == gid
        group_pts = points[mask]

        if len(group_pts) < 2:
            per_group_hv[gid] = 0.0
            per_group_gap[gid] = 0.0
            per_group_psize[gid] = 0
            continue

        n_active += 1

        # Hypervolume (all reward dimensions, exact HSO)
        pareto = _compute_pareto_front(group_pts)
        hv = _hypervolume(pareto, ref) if len(pareto) > 0 else 0.0
        per_group_hv[gid] = hv
        per_group_psize[gid] = len(pareto)

        # Hull gap (N-D): HV(continuous convex hull) − HV(discrete Pareto)
        if dim >= 2 and len(group_pts) >= dim + 1:
            gap = _compute_hull_gap(pareto, ref)
        else:
            gap = 0.0
        per_group_gap[gid] = gap

    mean_hv = float(np.mean(list(per_group_hv.values()))) if per_group_hv else 0.0
    mean_gap = float(np.mean(list(per_group_gap.values()))) if per_group_gap else 0.0
    mean_psize = float(np.mean(list(per_group_psize.values()))) if per_group_psize else 0.0

    return {
        "mean_hypervolume": mean_hv,
        "mean_hull_gap": mean_gap,
        "mean_pareto_size": mean_psize,
        "per_group_hypervolume": per_group_hv,
        "per_group_hull_gap": per_group_gap,
        "n_groups": len(unique_gids),
        "n_active_groups": n_active,
    }


def plot_hull_area_curve(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    title: str = "Convex Hull Area by Step",
    label_name: str = "Step",
) -> None:
    """Plot convex hull area vs step for each reward dimension pair."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dim = len(reward_names)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    n_pairs = len(pairs)
    cols = min(n_pairs, 2)
    rows = (n_pairs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    steps = sorted(all_steps.keys())
    if not steps:
        plt.close(fig)
        return

    for pi, (di, dj) in enumerate(pairs):
        ax = axes[pi // cols][pi % cols]

        # Pre-compute hull area per step in parallel
        from tools.reward_convex_hull_analysis.parallel import compute_map

        items = [
            (step, data["points"], di, dj)
            for step in steps
            for data in [all_steps.get(step, {})]
            if data.get("points") is not None
            and data["points"].shape[1] > max(di, dj)
            and len(data["points"]) >= 3
        ]
        raw = compute_map(_hull_area_worker, items)
        xs = sorted(raw.keys())
        areas = [raw[s] for s in xs]

        ax.plot(xs, areas, "o-", markersize=4, linewidth=1.2, color="#2196F3")
        ax.set_xlabel(label_name)
        ax.set_ylabel(f"Area ({reward_names[di]} vs {reward_names[dj]})")
        ax.grid(True, alpha=0.3)

    for pi in range(n_pairs, rows * cols):
        axes[pi // cols][pi % cols].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Window-averaged convex hulls — smooth trend by pooling steps into groups
# ---------------------------------------------------------------------------


def plot_convex_hulls_windows(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    window_size: int = 20,
    title: str = "Convex Hull Trend (Window-Averaged)",
    label_name: str = "Step Range",
) -> None:
    """Pool reward points across fixed-size windows of steps to smooth the trend.

    Groups steps into windows of *window_size* steps each (last window may be
    smaller).  Pools all reward points within each window and plots one convex
    hull per window.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = sorted(all_steps.keys())

    # Build fixed-size windows
    windows: List[Tuple[float, List[int]]] = []
    lo = 0
    while lo + window_size <= len(steps):
        hi = lo + window_size
        w_steps = steps[lo:hi]
        mid = sum(w_steps) / len(w_steps) if w_steps else 0
        windows.append((mid, w_steps))
        lo = hi
    nw = len(windows)
    if nw == 0:
        return

    dim = len(reward_names)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    n_pairs = len(pairs)

    # Fixed axis limits — by convention, PickScore and CLIP have known ranges.
    # These make cross-run comparisons meaningful.
    _DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
        "clip_score": (0.05, 0.5),
        "pick_score": (0.4, 1.2),
        "CLIP": (0.05, 0.5),
        "PickScore": (0.4, 1.2),
    }
    limits: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
    for di, dj in pairs:
        lx = _DEFAULT_LIMITS.get(reward_names[di], (None, None))
        ly = _DEFAULT_LIMITS.get(reward_names[dj], (None, None))
        if lx[0] is not None and ly[0] is not None:
            limits[(di, dj)] = (lx[0], lx[1], ly[0], ly[1])

    cols = min(n_pairs, 3)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

    # Gray ramp: early windows light, late windows dark (exponential)
    if nw > 1:
        t = np.linspace(0, 3.0, nw)
        t_norm = (np.exp(t) - 1) / (np.exp(3.0) - 1)  # 0→1, exponential
    else:
        t_norm = [0.5]
    # 0→1 maps to light gray (0.85) → dark gray (0.15)
    colors = [str(0.85 - 0.70 * v) for v in t_norm]

    for pair_idx, (di, dj) in enumerate(pairs):
        ax = axes[pair_idx // cols][pair_idx % cols]
        ax.set_xlabel(reward_names[di])
        ax.set_ylabel(reward_names[dj])

        # Apply global limits
        if (di, dj) in limits:
            x0, x1, y0, y1 = limits[(di, dj)]
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

        # --- Pre-pool points per window (reused for hull + centroid) ---
        window_xy: List[Optional[np.ndarray]] = []
        for _wi, (_mid, w_steps) in enumerate(windows):
            pool = []
            for s in w_steps:
                pts = all_steps.get(s, {}).get("points")
                if pts is not None and pts.shape[1] > max(di, dj) and len(pts) > 0:
                    pool.append(pts)
            if pool:
                combined = np.vstack(pool)
                window_xy.append(np.column_stack([combined[:, di], combined[:, dj]]))
            else:
                window_xy.append(None)

        # --- Draw hulls + centroids in one pass ---
        centroids: List[Tuple[Optional[float], Optional[float]]] = []
        for wi, (_mid, w_steps) in enumerate(windows):
            xy = window_xy[wi]
            if xy is None:
                centroids.append((None, None))
                continue

            c = colors[wi]
            label = f"{w_steps[0]}-{w_steps[-1]}"

            ax.scatter(xy[:, 0], xy[:, 1], color=c, alpha=0.6, s=6, edgecolors="none", label=label)

            if len(xy) >= 3:
                hull_xy = andrews_monotone_chain(xy)
                if len(hull_xy) >= 2:
                    ax.fill(hull_xy[:, 0], hull_xy[:, 1], color=c, alpha=0.08)
                    ax.plot(
                        np.append(hull_xy[:, 0], hull_xy[0, 0]),
                        np.append(hull_xy[:, 1], hull_xy[0, 1]),
                        color=c,
                        linewidth=2.0,
                        alpha=0.85,
                    )

            cx = np.mean(xy[:, 0])
            cy = np.mean(xy[:, 1])
            centroids.append((cx, cy))
            ax.plot(
                cx, cy, "o", color=c, markersize=6, markeredgecolor="white", markeredgewidth=0.5
            )

        fontsize = 6 if nw <= 20 else 5
        ax.legend(fontsize=fontsize, loc="best", title="Window", title_fontsize=7)

        # Connect consecutive valid centroids with dashed lines
        valid = [(cx, cy) for cx, cy in centroids if cx is not None]
        if len(valid) >= 2:
            cxs, cys = zip(*valid)
            ax.plot(cxs, cys, "k--", linewidth=1.0, alpha=0.5)

    for pi in range(n_pairs, rows * cols):
        axes[pi // cols][pi % cols].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convex_hulls_windows_cumulative(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_dir: str,
    window_size: int = 20,
    title: str = "Convex Hull — Cumulative Windows",
    label_name: str = "Step",
) -> None:
    """Generate cumulative window plots — one PNG per window, each pooling all
    data from windows 0..*i*, saved under *output_dir*/.

    This creates a frame-by-frame progression showing how the reward convex hull
    grows as more training steps are included.
    """
    os.makedirs(output_dir, exist_ok=True)

    steps = sorted(all_steps.keys())

    # Build windows (same logic as plot_convex_hulls_windows)
    windows: List[Tuple[float, List[int]]] = []
    lo = 0
    while lo + window_size <= len(steps):
        hi = lo + window_size
        w_steps = steps[lo:hi]
        mid = sum(w_steps) / len(w_steps) if w_steps else 0
        windows.append((mid, w_steps))
        lo = hi

    if not windows:
        return

    dim = len(reward_names)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    n_pairs = len(pairs)

    cols = min(n_pairs, 2)
    rows = (n_pairs + cols - 1) // cols

    # Fixed axis limits
    _DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
        "clip_score": (0.05, 0.5),
        "pick_score": (0.4, 1.2),
        "CLIP": (0.05, 0.5),
        "PickScore": (0.4, 1.2),
    }

    # --- Pre-compute per-window point arrays (pool once, reuse across frames) ---
    window_arrays: List[np.ndarray] = []
    all_pool: List[np.ndarray] = []
    for _mid, w_steps in windows:
        pool = []
        for s in w_steps:
            pts = all_steps.get(s, {}).get("points")
            if pts is not None and pts.shape[1] >= dim and len(pts) > 0:
                pool.append(pts[:, :dim])
        arr = np.vstack(pool) if pool else np.empty((0, dim))
        window_arrays.append(arr)
        if len(arr) > 0:
            all_pool.append(arr)

    # --- Global axis limits (from all points, kept fixed across frames) ---
    global_limits: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
    if all_pool:
        all_pts = np.vstack(all_pool)
        for di, dj in pairs:
            dx = all_pts[:, di]
            dy = all_pts[:, dj]
            margin = 0.05
            x0, x1 = dx.min() - margin, dx.max() + margin
            y0, y1 = dy.min() - margin, dy.max() + margin
            lx = _DEFAULT_LIMITS.get(reward_names[di])
            ly = _DEFAULT_LIMITS.get(reward_names[dj])
            if lx is not None:
                x0, x1 = lx[0], lx[1]
            if ly is not None:
                y0, y1 = ly[0], ly[1]
            global_limits[(di, dj)] = (x0, x1, y0, y1)

    # --- Color ramp (same as plot_convex_hulls_windows) ---
    nw = len(windows)
    if nw > 1:
        t = np.linspace(0, 3.0, nw)
        t_norm = (np.exp(t) - 1) / (np.exp(3.0) - 1)
    else:
        t_norm = [0.5]
    colors = [str(0.85 - 0.70 * v) for v in t_norm]  # light → dark gray

    # --- Parallel frame generation ---

    def _draw_one(wi: int) -> None:
        mid, w_steps = windows[wi]

        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows), squeeze=False)

        for pair_idx, (di, dj) in enumerate(pairs):
            ax = axes[pair_idx // cols][pair_idx % cols]
            ax.set_xlabel(reward_names[di])
            ax.set_ylabel(reward_names[dj])

            if (di, dj) in global_limits:
                ax.set_xlim(global_limits[(di, dj)][0], global_limits[(di, dj)][1])
                ax.set_ylim(global_limits[(di, dj)][2], global_limits[(di, dj)][3])

            # Draw each window's hull in its own color, overlaid
            total_pts = 0
            total_wins = 0
            for wj in range(wi + 1):
                arr = window_arrays[wj]
                if len(arr) == 0:
                    continue
                total_pts += len(arr)
                total_wins += 1
                c = colors[wj]
                xy = np.column_stack([arr[:, di], arr[:, dj]])

                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    color=c,
                    alpha=0.6,
                    s=6,
                    edgecolors="none",
                )

                if len(xy) >= 3:
                    hull_xy = andrews_monotone_chain(xy)
                    if len(hull_xy) >= 2:
                        ax.fill(hull_xy[:, 0], hull_xy[:, 1], color=c, alpha=0.08)
                        ax.plot(
                            np.append(hull_xy[:, 0], hull_xy[0, 0]),
                            np.append(hull_xy[:, 1], hull_xy[0, 1]),
                            color=c,
                            linewidth=2.0,
                            alpha=0.85,
                        )

            ax.text(
                0.02,
                0.98,
                f"Window {wi + 1}/{nw} | {label_name}s {w_steps[0]}–{w_steps[-1]} | {total_pts} pts | {total_wins} windows",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )

        for pi in range(n_pairs, rows * cols):
            axes[pi // cols][pi % cols].set_visible(False)

        fig.suptitle(
            f"{title} — Cumulative up to Window {wi + 1}/{nw} ({label_name} ≈{mid:.0f})",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(
            os.path.join(output_dir, f"frame_{wi:04d}.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)

    max_workers = min(len(windows), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_draw_one, range(len(windows))))

    print(f"  [Plot] Saved {len(windows)} cumulative frames → {output_dir}/")


# ---------------------------------------------------------------------------
# Reward percentile trends — how reward distribution evolves per step
# ---------------------------------------------------------------------------


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

        # Compute percentiles in parallel
        from tools.reward_convex_hull_analysis.parallel import compute_map

        items = [(wi, vals) for wi, vals in pooled]
        raw = compute_map(_percentiles_worker, items)

        # Assemble ordered lists
        xs, q25s, medians, q75s, mins, maxs, means = [], [], [], [], [], [], []
        for wi, (mid, _) in enumerate(windows):
            result = raw.get(wi)
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


# ---------------------------------------------------------------------------
# Pareto front evolution — cumulative Pareto-optimal set over training
# ---------------------------------------------------------------------------


def _compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the Pareto-optimal subset of *points* (maximization in all dims)."""
    if len(points) == 0:
        return np.empty((0, points.shape[1]))
    # Sort by first dim descending
    idx = np.argsort(-points[:, 0])
    sorted_pts = points[idx]
    is_pareto = np.ones(len(sorted_pts), dtype=bool)
    for i in range(len(sorted_pts)):
        if not is_pareto[i]:
            continue
        # This point dominates any later point that is <= in all dims
        for j in range(i + 1, len(sorted_pts)):
            if not is_pareto[j]:
                continue
            if all(sorted_pts[i] >= sorted_pts[j]):
                is_pareto[j] = False
    return sorted_pts[is_pareto]


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


def plot_per_group_hypervolume_and_gap(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    title: str = "Per-Group Pareto Size & Hull Gap",
    label_name: str = "Step",
) -> None:
    """Plot per-group-averaged Pareto-front size and hull gap over steps.

    At each step, points are split by *prompt_idx* into prompt groups.
    Within each group the number of Pareto-optimal points and the hull gap
    (continuous convex-hull HV minus discrete HSO HV) are computed.
    Values are averaged across groups per step.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = sorted(all_steps.keys())
    dim = len(reward_names)
    if len(steps) == 0:
        return

    # --- Global reference point (consistent across all steps/groups) ---
    global_pts_list = []
    for step in steps:
        pts = all_steps.get(step, {}).get("points")
        if pts is not None and pts.shape[1] >= dim and len(pts) > 0:
            global_pts_list.append(pts[:, :dim])
    if global_pts_list:
        global_ref = np.vstack(global_pts_list).min(axis=0) - 0.01
    else:
        global_ref = np.zeros(dim)

    # --- Per-step computation (parallel across steps) ---
    from tools.reward_convex_hull_analysis.parallel import compute_map

    step_items = []
    empty_steps: List[int] = []
    for step in steps:
        data = all_steps.get(step, {})
        pts = data.get("points")
        if pts is None or pts.shape[1] < dim or len(pts) == 0:
            empty_steps.append(step)
            continue
        step_items.append((step, pts[:, :dim].copy(), data.get("prompt_idx"), global_ref))

    raw = compute_map(_compute_per_group_metrics, step_items)

    mean_psizes: List[float] = []
    mean_gaps: List[Optional[float]] = []
    plot_steps: List[int] = []
    for step in steps:
        if step in empty_steps:
            mean_psizes.append(0.0)
            mean_gaps.append(None)
        else:
            m = raw.get(step)
            if m is not None:
                mean_psizes.append(m["mean_pareto_size"])
                mean_gaps.append(m["mean_hull_gap"])
            else:
                mean_psizes.append(0.0)
                mean_gaps.append(None)
        plot_steps.append(step)

    # --- Plot ---
    # Pareto size on left axis; hull gap on right (2+ rewards only).
    show_gap = dim >= 2

    fig, ax_left = plt.subplots(figsize=(10, 5))
    color_left = "#2E7D32"
    ax_left.plot(plot_steps, mean_psizes, "o-", color=color_left, linewidth=1.5, markersize=4,
                 label="Mean Pareto Size (per-group)")
    ax_left.set_xlabel(label_name)
    ax_left.set_ylabel("Pareto Front Size", color=color_left)
    ax_left.tick_params(axis="y", labelcolor=color_left)
    ax_left.grid(True, alpha=0.3)

    if show_gap:
        ax_gap = ax_left.twinx()
        color_gap = "#C62828"
        gap_vals = [g if g is not None else float("nan") for g in mean_gaps]
        ax_gap.plot(plot_steps, gap_vals, "s--", color=color_gap, linewidth=1.2, markersize=4,
                    label="Mean Hull Gap (per-group)")
        ax_gap.set_ylabel("Hull Gap (HV_hull − HV_HSO)", color=color_gap)
        ax_gap.tick_params(axis="y", labelcolor=color_gap)

        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_gap.get_legend_handles_labels()
        ax_left.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")
    else:
        ax_left.legend(fontsize=9, loc="best")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto_front_evolution_cumulative(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    title: str = "Pareto Front Evolution",
    label_name: str = "Step",
) -> None:
    """Track cumulative Pareto front size and hypervolume over training steps.

    Works with any number of reward dimensions (≥1).  At each step, all
    reward vectors seen so far are pooled, and the N-dimensional Pareto front
    is computed.  Plots the number of Pareto-optimal points and an estimated
    hypervolume over steps.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = sorted(all_steps.keys())
    dim = len(reward_names)
    if len(steps) == 0:
        return

    # --- Compute global reference point once from all steps ---
    global_pts_list = []
    for step in steps:
        pts = all_steps.get(step, {}).get("points")
        if pts is not None and pts.shape[1] >= dim and len(pts) > 0:
            global_pts_list.append(pts[:, :dim])
    if global_pts_list:
        global_ref = np.vstack(global_pts_list).min(axis=0) - 0.01
    else:
        global_ref = np.zeros(dim)

    cumulative = []  # list of (N, D) arrays
    pareto_counts = []
    hypervolumes = []

    for step in steps:
        pts = all_steps.get(step, {}).get("points")
        if pts is not None and pts.shape[1] >= dim and len(pts) > 0:
            cumulative.append(pts[:, :dim])

        if cumulative:
            all_pts = np.vstack(cumulative)
        else:
            all_pts = np.empty((0, dim))

        pareto = _compute_pareto_front(all_pts)
        pareto_counts.append(len(pareto))

        if len(pareto) > 0:
            hv = _hypervolume(pareto, global_ref)
        else:
            hv = 0.0
        hypervolumes.append(hv)

    fig, ax_count = plt.subplots(figsize=(10, 5))
    ax_hv = ax_count.twinx()

    ax_count.plot(steps, pareto_counts, "-", color="#2E7D32", linewidth=1.5, label="Pareto Size")
    ax_hv.plot(steps, hypervolumes, "-", color="#C62828", linewidth=1.5, label="Hypervolume")

    ax_count.set_ylabel("Pareto Front Size", color="#2E7D32")
    ax_hv.set_ylabel("Hypervolume", color="#C62828")
    ax_count.set_xlabel(label_name)
    ax_count.tick_params(axis="y", labelcolor="#2E7D32")
    ax_hv.tick_params(axis="y", labelcolor="#C62828")
    ax_count.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax_count.get_legend_handles_labels()
    lines2, labels2 = ax_hv.get_legend_handles_labels()
    ax_count.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
