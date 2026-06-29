"""Rewards pickle reader — loads pre-computed scores from ``logs/rewards/*.pkl``.

Converts pickle data into the standard ``{step: {"points": ndarray(N, D), ...}}``
format used by all plotting functions in ``convex_hull.py``.

Reward keys are discovered dynamically — the reader is not hardcoded to a specific
set of reward names.  Any key that is not ``step`` or ``prompts`` is treated as a
reward dimension.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_reward_keys(rewards_dir: str) -> List[str]:
    """Scan all ``*.pkl`` files in *rewards_dir* and return sorted reward key names.

    Reward keys are every top-level key except ``step`` and ``prompts``.
    """
    keys: set = set()
    if not os.path.isdir(rewards_dir):
        return []
    for fname in sorted(os.listdir(rewards_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(rewards_dir, fname), "rb") as f:
            data = pickle.load(f)
        for k in data:
            if k not in ("step", "prompts"):
                keys.add(k)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Train pickles (per-prompt list-of-arrays)
# ---------------------------------------------------------------------------


def load_train_rewards(
    rewards_dir: str,
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    """Load all ``train_step_*.pkl`` files.

    Each train pickle has the format::

        {
            "step": <int>,
            "prompts": [str, ...],             # 48 prompts
            "reward_a": [ndarray(16,), ...],   # 48 arrays of 16 scores each
            "reward_b": [ndarray(16,), ...],
            ...
        }

    Per-prompt arrays are concatenated into a single flat 1-D array per reward
    dimension, then stacked into ``(N, D)`` points.

    Returns:
        ``(step_data, reward_names)`` where *step_data* maps ``step`` →
        ``{"points": ndarray(N, D), "step": int, "n_total": int, "n_valid": int}``
        and *reward_names* is the sorted list of discovered reward keys.
    """
    reward_names = discover_reward_keys(rewards_dir)
    if not reward_names:
        return {}, []

    result: Dict[int, Dict[str, Any]] = {}

    for fname in sorted(os.listdir(rewards_dir)):
        if not fname.startswith("train_step_") or not fname.endswith(".pkl"):
            continue
        filepath = os.path.join(rewards_dir, fname)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        step = int(data["step"])

        # Collect scores per reward dimension; also build per-prompt group
        # indices from the first reward's array-of-arrays structure.
        dim_scores = []
        prompt_idx_built: Optional[np.ndarray] = None
        for rname in reward_names:
            arr_list = data.get(rname)
            if arr_list is None:
                # Missing reward in this file — skip the whole file
                break
            flat = np.concatenate([np.asarray(a) for a in arr_list])
            dim_scores.append(flat)
            if prompt_idx_built is None:
                # First reward determines the prompt→sample mapping
                prompt_idx_built = np.concatenate([
                    np.full(len(np.asarray(a)), i) for i, a in enumerate(arr_list)
                ])
        else:
            # All reward keys present — stack into (N, D) points
            pts = np.column_stack(dim_scores)
            mask = np.isfinite(pts).all(axis=1)
            pts = pts[mask]
            result[step] = {
                "points": pts,
                "step": step,
                "n_total": len(dim_scores[0]),
                "n_valid": len(pts),
            }
            if prompt_idx_built is not None:
                result[step]["prompt_idx"] = prompt_idx_built[mask]
                prompts_raw = data.get("prompts")
                if prompts_raw is not None:
                    result[step]["prompt_labels"] = list(prompts_raw)

    # Sort by step
    result = dict(sorted(result.items()))
    return result, reward_names


# ---------------------------------------------------------------------------
# Eval pickles (flat arrays)
# ---------------------------------------------------------------------------


def load_eval_rewards(
    rewards_dir: str,
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    """Load all ``eval_*_step_*.pkl`` files.

    Each eval pickle has the format::

        {
            "step": <int>,
            "reward_a": ndarray(N,),
            "reward_b": ndarray(N,),
            ...
        }

    Returns:
        ``(step_data, reward_names)`` — same structure as :func:`load_train_rewards`.
    """
    reward_names = discover_reward_keys(rewards_dir)
    if not reward_names:
        return {}, []

    result: Dict[int, Dict[str, Any]] = {}

    for fname in sorted(os.listdir(rewards_dir)):
        if not fname.startswith("eval_") or not fname.endswith(".pkl"):
            continue
        filepath = os.path.join(rewards_dir, fname)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        step = int(data["step"])

        dim_scores = []
        for rname in reward_names:
            arr = data.get(rname)
            if arr is None:
                break
            dim_scores.append(np.asarray(arr).flatten())
        else:
            pts = np.column_stack(dim_scores)
            mask = np.isfinite(pts).all(axis=1)
            pts = pts[mask]
            result[step] = {
                "points": pts,
                "step": step,
                "n_total": len(dim_scores[0]),
                "n_valid": len(pts),
            }

    result = dict(sorted(result.items()))
    return result, reward_names
