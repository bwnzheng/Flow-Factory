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

# src/flow_factory/trainers/crossover/pareto.py
"""
Pareto-front utilities for crossover sample selection and filtering.

Used both for selecting which parents to crossover (non-dominated only)
and for final filtering of dominated samples before advantage computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ============================================================================
# Core Pareto computation
# ============================================================================


def compute_pareto_mask(points: np.ndarray) -> np.ndarray:
    """Compute the Pareto (non-dominated) boolean mask for a set of points.

    All dimensions are treated as **maximisation** objectives.  A point *A*
    dominates *B* when *A >= B* in every dimension and *A > B* in at least one.

    Args:
        points: ``(N, d)`` array of *N* points in *d*-dim reward space.
            NaN values are treated as missing: points with any NaN are
            automatically kept (mask = True).

    Returns:
        ``(N,)`` boolean array — ``True`` for non-dominated points.
    """
    N = points.shape[0]
    if N <= 1:
        return np.ones(N, dtype=bool)

    # Handle NaN: keep those samples automatically
    nan_mask = np.any(np.isnan(points), axis=1)

    is_pareto = np.ones(N, dtype=bool)

    # Sort by first dimension descending for early termination
    order = np.argsort(-points[:, 0])
    sorted_pts = points[order]

    for i in range(N):
        if not is_pareto[order[i]]:
            continue
        for j in range(i + 1, N):
            if not is_pareto[order[j]]:
                continue
            if np.all(sorted_pts[i] >= sorted_pts[j]):
                is_pareto[order[j]] = False

    # Restore NaN-marked samples
    is_pareto[nan_mask] = True

    return is_pareto


# ============================================================================
# Per-group filtering
# ============================================================================


def filter_by_group(
    gathered_rewards: Dict[str, np.ndarray],
    group_indices: np.ndarray,
    parent_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply Pareto filtering per group and return a global keep mask.

    For each group (all K parents + M children assigned to the same prompt),
    compute the non-dominated subset via :func:`compute_pareto_mask`.  The
    per-group masks are merged into a single ``(S,)`` boolean array.

    Args:
        gathered_rewards: ``{reward_name: array(S,)}`` — per-reward scores
            gathered across all ranks.
        group_indices: ``(S,)`` integer array mapping each sample to its group.
        parent_mask: Optional ``(S,)`` boolean — ``True`` for parent samples,
            used only for stats.

    Returns:
        Tuple of:
        - **pareto_mask**: ``(S,)`` bool — ``True`` = non-dominated (keep).
        - **stats**: Dict with keys ``total``, ``kept``, ``discarded``,
          ``parents_kept``, ``parents_discarded``, ``children_kept``,
          ``children_discarded``.
    """
    reward_keys = list(gathered_rewards.keys())
    if not reward_keys:
        return np.ones(len(group_indices), dtype=bool), _empty_stats()

    # Build (S, R) rewards matrix
    stack = np.stack([gathered_rewards[k].astype(np.float64) for k in reward_keys], axis=1)

    num_groups = int(group_indices.max()) + 1
    pareto_mask = np.ones(len(group_indices), dtype=bool)

    for g in range(num_groups):
        idx = np.where(group_indices == g)[0]
        if len(idx) <= 1:
            continue
        group_mask = compute_pareto_mask(stack[idx])
        pareto_mask[idx] = group_mask

    # Build stats
    total = len(pareto_mask)
    kept = int(pareto_mask.sum())
    discarded = total - kept

    stats: Dict[str, int] = {"total": total, "kept": kept, "discarded": discarded}
    if parent_mask is not None:
        p_mask = parent_mask.astype(bool)
        parents_total = int(p_mask.sum())
        parents_kept = int((pareto_mask & p_mask).sum())
        stats["parents_kept"] = parents_kept
        stats["parents_discarded"] = parents_total - parents_kept
        stats["children_kept"] = kept - parents_kept
        stats["children_discarded"] = discarded - (parents_total - parents_kept)

    return pareto_mask, stats


def _empty_stats() -> Dict[str, int]:
    return {
        "total": 0,
        "kept": 0,
        "discarded": 0,
        "parents_kept": 0,
        "parents_discarded": 0,
        "children_kept": 0,
        "children_discarded": 0,
    }


# ============================================================================
# Parent selection (for selective crossover)
# ============================================================================


def select_non_dominated_parents(
    parent_rewards: Dict[str, torch.Tensor],
    group_ids: List[int],
) -> np.ndarray:
    """Identify non-dominated parents within each group.

    Used during sampling to decide which parents should produce children.

    Args:
        parent_rewards: ``{reward_name: tensor(K,)}`` — per-reward scores
            for the K parents on this rank.
        group_ids: List of *unique_id* integers aligned with parent_rewards.

    Returns:
        ``(K,)`` boolean numpy array — ``True`` for non-dominated parents.
    """
    reward_keys = list(parent_rewards.keys())
    if not reward_keys:
        return np.ones(len(group_ids), dtype=bool)

    stack = np.stack([parent_rewards[k].cpu().float().numpy() for k in reward_keys], axis=1)
    unique_groups = np.unique(group_ids)
    mask = np.ones(len(group_ids), dtype=bool)

    for g in unique_groups:
        idx = np.where(np.array(group_ids) == g)[0]
        if len(idx) <= 1:
            continue
        group_mask = compute_pareto_mask(stack[idx])
        mask[idx] = group_mask

    return mask
