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


def _file_reward_keys(data: Dict) -> List[str]:
    """Extract sorted reward key names from a single pkl file's data dict."""
    return sorted(k for k in data if k not in ("step", "prompts"))


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

    Each file is processed with its **own** reward keys so that files with
    different reward subsets (e.g. train vs eval) are not silently skipped.

    Returns:
        ``(step_data, reward_names)`` where *step_data* maps ``step`` →
        ``{"points": ndarray(N, D), "step": int, "n_total": int, "n_valid": int}``
        and *reward_names* is the **intersection** of reward keys across all
        loaded files (guaranteeing uniform dimension for downstream plotting).
    """
    if not os.path.isdir(rewards_dir):
        return {}, []

    result: Dict[int, Dict[str, Any]] = {}
    all_key_sets: List[set] = []

    for fname in sorted(os.listdir(rewards_dir)):
        if not fname.startswith("train_step_") or not fname.endswith(".pkl"):
            continue
        filepath = os.path.join(rewards_dir, fname)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        step = int(data["step"])

        # Use this file's own reward keys, NOT a global union
        file_keys = _file_reward_keys(data)
        if not file_keys:
            continue

        dim_scores = []
        prompt_idx_built: Optional[np.ndarray] = None
        for rname in file_keys:
            arr_list = data.get(rname)
            if arr_list is None:
                continue
            flat = np.concatenate([np.asarray(a) for a in arr_list])
            dim_scores.append(flat)
            if prompt_idx_built is None:
                prompt_idx_built = np.concatenate([
                    np.full(len(np.asarray(a)), i) for i, a in enumerate(arr_list)
                ])

        if not dim_scores:
            continue

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

        all_key_sets.append(set(file_keys))

    # Sort by step
    result = dict(sorted(result.items()))

    # Intersection of keys across all loaded files for uniform dimensions
    if all_key_sets:
        common_keys = sorted(set.intersection(*all_key_sets))
    else:
        common_keys = []
    return result, common_keys


# ---------------------------------------------------------------------------
# Eval pickles (flat arrays)
# ---------------------------------------------------------------------------


def _parse_eval_dataset(fname: str) -> str:
    """Extract dataset name from an eval pkl filename.

    Filenames are ``{eval_dataset_slug}_step_XXXXXX.pkl`` where *dataset_slug*
    is ``eval_<name>`` (e.g. ``eval_default``, ``eval_pickscore``).

    Returns the original dataset name (e.g. ``"default"``, ``"pickscore"``).
    """
    # Strip leading "eval_" and trailing "_step_XXXXXX.pkl"
    base = fname[:-4] if fname.endswith(".pkl") else fname  # strip .pkl
    if base.startswith("eval_"):
        inner = base[5:]  # remove "eval_" prefix
        # Remove "_step_XXXXXX" suffix
        parts = inner.rsplit("_step_", 1)
        if len(parts) == 2:
            return parts[0]
    return "unknown"


def load_eval_rewards(
    rewards_dir: str,
) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], Dict[str, List[str]]]:
    """Load all ``eval_*_step_*.pkl`` files, grouped by dataset.

    Each eval pickle has the format::

        {
            "step": <int>,
            "reward_a": ndarray(N,),
            "reward_b": ndarray(N,),
            ...
        }

    Each file is processed with its **own** reward keys so that files with
    different reward subsets (e.g. ``eval_pickscore`` vs ``eval_ocr``) are
    not silently skipped.

    Returns:
        ``(dataset_data, dataset_reward_names)`` where:

        * *dataset_data* maps dataset name → ``{step: {...}}``
        * *dataset_reward_names* maps dataset name → sorted list of reward
          keys (intersection across all steps within that dataset)
    """
    if not os.path.isdir(rewards_dir):
        return {}, {}

    # Group by dataset name
    by_dataset: Dict[str, Dict[int, Dict[str, Any]]] = {}
    by_dataset_key_sets: Dict[str, List[set]] = {}

    for fname in sorted(os.listdir(rewards_dir)):
        if not fname.startswith("eval_") or not fname.endswith(".pkl"):
            continue
        filepath = os.path.join(rewards_dir, fname)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        step = int(data["step"])
        ds_name = _parse_eval_dataset(fname)

        # Use this file's own reward keys, NOT a global union
        file_keys = _file_reward_keys(data)
        if not file_keys:
            continue

        dim_scores = []
        for rname in file_keys:
            arr = data.get(rname)
            if arr is None:
                continue
            dim_scores.append(np.asarray(arr).flatten())

        if not dim_scores:
            continue

        pts = np.column_stack(dim_scores)
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]

        if ds_name not in by_dataset:
            by_dataset[ds_name] = {}
            by_dataset_key_sets[ds_name] = []
        by_dataset[ds_name][step] = {
            "points": pts,
            "step": step,
            "n_total": len(dim_scores[0]),
            "n_valid": len(pts),
        }
        by_dataset_key_sets[ds_name].append(set(file_keys))

    # Sort each dataset by step
    for ds_name in by_dataset:
        by_dataset[ds_name] = dict(sorted(by_dataset[ds_name].items()))

    # Compute per-dataset reward names (intersection across steps)
    dataset_reward_names: Dict[str, List[str]] = {}
    for ds_name, key_sets in by_dataset_key_sets.items():
        if key_sets:
            dataset_reward_names[ds_name] = sorted(set.intersection(*key_sets))
        else:
            dataset_reward_names[ds_name] = []

    return by_dataset, dataset_reward_names
