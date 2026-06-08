"""Convex hull computation (Andrew's monotone chain) and plotting utilities.

Provides 2D convex hull plotting (scatter + filled hull polygon per epoch/step)
and a 1D distribution fallback for single-reward-model configurations.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows),
                             squeeze=False)

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

            # Scatter points
            ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.55, s=12, edgecolors="none")

            # Draw hull polygon (computed in 2D for accurate per-pair display)
            if len(xy) >= 3:
                hull_xy = andrews_monotone_chain(xy)
                if len(hull_xy) >= 2:
                    ax.fill(hull_xy[:, 0], hull_xy[:, 1], color=color, alpha=0.12)
                    ax.plot(np.append(hull_xy[:, 0], hull_xy[0, 0]),
                            np.append(hull_xy[:, 1], hull_xy[0, 1]),
                            color=color, linewidth=1.0, alpha=0.7)

    # Hide unused subplots
    for pi in range(n_pairs, rows * cols):
        axes[pi // cols][pi % cols].set_visible(False)

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
                ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.45, s=10,
                           edgecolors="none")

                hull_xy = andrews_monotone_chain(xy)
                if len(hull_xy) >= 3:
                    ax.plot(np.append(hull_xy[:, 0], hull_xy[0, 0]),
                            np.append(hull_xy[:, 1], hull_xy[0, 1]),
                            color=color, linewidth=1.0, alpha=0.6, linestyle="--")

        # Legend for source identity
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.viridis(0.5),
                   markersize=8, label=label_a),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.plasma(0.5),
                   markersize=8, label=label_b),
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
    fig, (ax_strip, ax_dist) = plt.subplots(2, 1, figsize=(10, 8),
                                            gridspec_kw={"height_ratios": [1, 2]})

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
        ax_dist.hist(vals, bins="auto", density=True, alpha=0.3, color=color,
                     histtype="stepfilled", linewidth=0.5, edgecolor=color,
                     label=f"{label_name} {step}")

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
