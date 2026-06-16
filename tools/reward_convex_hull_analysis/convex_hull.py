"""Convex hull computation (Andrew's monotone chain) and plotting utilities.

Provides 2D convex hull plotting (scatter + filled hull polygon per epoch/step)
and a 1D distribution fallback for single-reward-model configurations.
"""

from __future__ import annotations

import os
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

    # Remove duplicate points
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts

    # Sort by x, then y
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]

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
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(steps), vmax=max(steps)) if steps else plt.Normalize(0, 1)

    for pair_idx, (di, dj) in enumerate(pairs):
        ax = axes[pair_idx // cols][pair_idx % cols]
        ax.set_xlabel(reward_names[di])
        ax.set_ylabel(reward_names[dj])

        for step in steps:
            data = all_steps[step]
            pts = data.get("points")
            if pts is None or len(pts) < di or len(pts) < dj:
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
                if pts is None:
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
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(steps), vmax=max(steps)) if steps else plt.Normalize(0, 1)

    # Strip plot
    fig, (ax_strip, ax_dist) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]}
    )

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
    norm = plt.Normalize(vmin=min(steps), vmax=max(steps)) if steps else plt.Normalize(0, 1)

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
            for pi in sorted(set(prompt_idx.tolist())):
                mask = prompt_idx == pi
                marker = _PROMPT_MARKERS[pi % len(_PROMPT_MARKERS)]
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    color=color,
                    alpha=0.55,
                    s=8,
                    marker=marker,
                    edgecolors="none",
                    label=f"P{pi}" if pi < 8 else None,
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

    # Apply unified axis limits (sharex/sharey propagates from first visible subplot)
    if xlim is not None:
        for row in range(rows):
            for col in range(cols):
                ax = axes[row][col]
                if ax.get_visible():
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    break
            else:
                continue
            break

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

    for pi, (di, dj) in enumerate(pairs):
        ax = axes[pi // cols][pi % cols]
        areas: List[float] = []
        xs: List[int] = []
        for step in steps:
            data = all_steps[step]
            pts = data.get("points")
            if pts is None or pts.shape[1] <= max(di, dj) or len(pts) < 3:
                continue
            area = _hull_area_2d(pts[:, [di, dj]])
            areas.append(area)
            xs.append(step)

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

        for wi, (mid, w_steps) in enumerate(windows):
            pool = []
            for s in w_steps:
                pts = all_steps.get(s, {}).get("points")
                if pts is not None and pts.shape[1] > max(di, dj) and len(pts) > 0:
                    pool.append(pts)
            if not pool:
                continue
            combined = np.vstack(pool)
            xy = np.column_stack([combined[:, di], combined[:, dj]])

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

        fontsize = 6 if nw <= 20 else 5
        ax.legend(fontsize=fontsize, loc="best", title="Window", title_fontsize=7)

        # --- Draw centroid drift lines ---
        centroids = []
        for wi, (mid, w_steps) in enumerate(windows):
            pool = []
            for s in w_steps:
                pts = all_steps.get(s, {}).get("points")
                if pts is not None and pts.shape[1] > max(di, dj) and len(pts) > 0:
                    pool.append(pts)
            if not pool:
                centroids.append((None, None))
                continue
            combined = np.vstack(pool)
            cx = np.mean(combined[:, di])
            cy = np.mean(combined[:, dj])
            centroids.append((cx, cy))
            # Mark centroid with a small dot
            c = colors[wi]
            ax.plot(
                cx, cy, "o", color=c, markersize=6, markeredgecolor="white", markeredgewidth=0.5
            )

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


# ---------------------------------------------------------------------------
# Reward quantile trends — how reward distribution evolves per step
# ---------------------------------------------------------------------------


def plot_reward_quantiles(
    all_steps: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_path: str,
    window_size: int = 20,
    title: str = "Reward Quantile Trends",
    label_name: str = "Step",
) -> None:
    """Plot quantiles of each reward dimension, pooled in windows.

    Pools reward points across *window_size* consecutive steps (same as the
    window-averaged hull plot) to suppress per-step noise.  Shows median,
    mean, IQR band, and min/max per window.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = sorted(all_steps.keys())

    # Build windows (same logic as plot_convex_hulls_windows)
    windows: List[Tuple[float, List[int]]] = []
    lo = 0
    while lo + window_size <= len(steps):
        hi = lo + window_size
        w_steps = steps[lo:hi]
        mid = sum(w_steps) / len(w_steps)
        windows.append((mid, w_steps))
        lo = hi

    n_rewards = len(reward_names)
    fig, axes = plt.subplots(n_rewards, 1, figsize=(10, 3.5 * n_rewards), squeeze=False)

    for ri, rname in enumerate(reward_names):
        ax = axes[ri][0]
        xs, medians, q25s, q75s, mins, maxs, means = [], [], [], [], [], [], []

        for mid, w_steps in windows:
            pool = []
            for s in w_steps:
                pts = all_steps.get(s, {}).get("points")
                if pts is not None and pts.shape[1] > ri and len(pts) > 0:
                    pool.append(pts[:, ri])
            if not pool:
                continue
            vals = np.concatenate(pool)
            xs.append(mid)
            q25s.append(np.percentile(vals, 25))
            medians.append(np.percentile(vals, 50))
            q75s.append(np.percentile(vals, 75))
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
            means.append(np.mean(vals))

        ax.fill_between(xs, q25s, q75s, alpha=0.25, color="#2196F3", label="25%-75%")
        ax.plot(xs, medians, "o-", color="#1565C0", linewidth=1.5, markersize=4, label="Median")
        ax.plot(xs, means, "s-", color="#0D47A1", linewidth=1.2, markersize=4, label="Mean")
        ax.plot(xs, mins, "--", color="#90CAF9", linewidth=0.7, label="Min")
        ax.plot(xs, maxs, "--", color="#90CAF9", linewidth=0.7, label="Max")

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
    """Exact hypervolume via recursive dimension reduction.

    For 2D: sort by x desc, sum rectangular slices.
    For N>2: sort by first dim desc, for each point recurse on remaining
    dims with the set of points above it.
    """
    if len(pareto) == 0:
        return 0.0
    dim = pareto.shape[1]

    if dim == 1:
        return max(pareto[:, 0].max() - ref[0], 0.0)

    # Sort by first dimension descending
    pts = pareto[np.argsort(-pareto[:, 0])]
    hv = 0.0
    prev_x = ref[0]

    for i, pt in enumerate(pts):
        x = pt[0]
        if x <= prev_x:
            continue
        # Slice: points above this one in remaining dims
        remaining = pts[: i + 1, 1:]  # include current point
        # Filter: only points that dominate ref in remaining dims
        mask = np.all(remaining >= ref[1:], axis=1)
        if mask.sum() > 0:
            hv_slice = _hypervolume(remaining[mask], ref[1:])
            hv += (x - prev_x) * hv_slice
        prev_x = x

    return hv


def plot_pareto_front_evolution(
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
            ref = all_pts.min(axis=0) - 0.01
            hv = _hypervolume(pareto, ref)
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
