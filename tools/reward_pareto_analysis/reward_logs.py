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

"""Rewards pickle reader — loads pre-computed scores from ``logs/rewards/*.pkl``.

Converts pickle data into the standard ``{step: {"points": ndarray(N, D), ...}}``
format used by the reward Pareto plotting functions in ``plots.py``.

Reward keys are discovered dynamically — the reader is not hardcoded to a specific
set of reward names.  Any key that is not ``step`` or ``prompts`` is treated as a
reward dimension.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Train pickles (per-prompt list-of-arrays)
# ---------------------------------------------------------------------------


def _file_reward_keys(data: Dict) -> List[str]:
    """Extract sorted reward key names from a single pkl file's data dict."""
    return sorted(k for k in data if k not in ("step", "prompts"))


def load_train_rewards(
    rewards_dir: str,
) -> Tuple[Dict[Tuple[str, ...], Dict[int, Dict[str, Any]]], List[str]]:
    """Load all ``train_step_*.pkl`` files.

    Each train pickle has the format::

        {
            "step": <int>,
            "prompts": [str, ...],             # 48 prompts
            "reward_a": [ndarray(16,), ...],   # 48 arrays of 16 scores each
            "reward_b": [ndarray(16,), ...],
            ...
        }

    Groups are partitioned by their exact set of fully available rewards. A
    reward is available only when every score in that prompt group is finite;
    an all-NaN group means the reward is not applicable. Partial missingness
    and infinities are rejected because silently filtering individual samples
    would break sample alignment across reward dimensions.

    Args:
        rewards_dir: Directory containing ``train_step_*.pkl`` files.

    Returns:
        ``(combination_data, reward_names)`` where *combination_data* maps an
        exact reward-name tuple to its step data, and *reward_names* is the
        sorted union of reward keys found in the train files.
    """
    if not os.path.isdir(rewards_dir):
        return {}, []

    loaded: List[Tuple[str, Dict[str, Any]]] = []
    reward_name_set: set[str] = set()
    for fname in sorted(os.listdir(rewards_dir)):
        if not fname.startswith("train_step_") or not fname.endswith(".pkl"):
            continue
        filepath = os.path.join(rewards_dir, fname)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        loaded.append((filepath, data))
        reward_name_set.update(_file_reward_keys(data))

    reward_names = sorted(reward_name_set)
    if not loaded or not reward_names:
        return {}, reward_names

    result: Dict[Tuple[str, ...], Dict[int, Dict[str, Any]]] = {}
    expected_combo_by_prompt: Dict[str, Tuple[str, ...]] = {}
    expected_samples_by_prompt: Dict[str, int] = {}

    for filepath, data in loaded:
        step = int(data["step"])
        file_keys = _file_reward_keys(data)
        prompts_raw = data.get("prompts")
        if prompts_raw is not None:
            n_groups = len(prompts_raw)
        elif file_keys:
            n_groups = len(data[file_keys[0]])
        else:
            continue

        for reward_name in file_keys:
            if len(data[reward_name]) != n_groups:
                raise ValueError(
                    f"Reward {reward_name!r} has {len(data[reward_name])} groups "
                    f"but step {step} has {n_groups} prompts ({filepath})"
                )

        per_combo_points: Dict[Tuple[str, ...], List[np.ndarray]] = {}
        per_combo_prompt_idx: Dict[Tuple[str, ...], List[np.ndarray]] = {}
        per_combo_group_ids: Dict[Tuple[str, ...], List[int]] = {}

        for gid in range(n_groups):
            arrays: Dict[str, np.ndarray] = {}
            sample_counts: set[int] = set()
            available: List[str] = []

            for reward_name in reward_names:
                if reward_name not in data:
                    continue
                values = np.asarray(data[reward_name][gid], dtype=float).reshape(-1)
                arrays[reward_name] = values
                sample_counts.add(len(values))

                if np.isinf(values).any():
                    raise ValueError(
                        f"Infinite reward at step {step}, group {gid}, "
                        f"reward {reward_name!r} ({filepath})"
                    )
                finite = np.isfinite(values)
                if finite.all():
                    available.append(reward_name)
                elif np.isnan(values).all():
                    continue
                else:
                    raise ValueError(
                        f"Partially missing reward at step {step}, group {gid}, "
                        f"reward {reward_name!r}: {int(finite.sum())}/{len(values)} "
                        f"finite values ({filepath})"
                    )

            if len(sample_counts) != 1:
                raise ValueError(
                    f"Sample counts differ across rewards at step {step}, "
                    f"group {gid}: {sorted(sample_counts)} ({filepath})"
                )
            n_samples = next(iter(sample_counts))
            combination = tuple(available)
            if not combination:
                raise ValueError(f"No finite rewards at step {step}, group {gid} ({filepath})")

            if prompts_raw is not None:
                prompt = str(prompts_raw[gid])
                expected_combo = expected_combo_by_prompt.setdefault(prompt, combination)
                if combination != expected_combo:
                    raise ValueError(
                        f"Reward combination changed for prompt {prompt!r} at "
                        f"step {step}, group {gid}: expected {expected_combo}, "
                        f"got {combination} ({filepath})"
                    )
                expected_samples = expected_samples_by_prompt.setdefault(prompt, n_samples)
                if n_samples != expected_samples:
                    raise ValueError(
                        f"Sample count changed for prompt {prompt!r} at step {step}, "
                        f"group {gid}: expected {expected_samples}, got {n_samples} "
                        f"({filepath})"
                    )

            points = np.column_stack([arrays[name] for name in combination])
            per_combo_points.setdefault(combination, []).append(points)
            per_combo_prompt_idx.setdefault(combination, []).append(
                np.full(n_samples, gid, dtype=int)
            )
            per_combo_group_ids.setdefault(combination, []).append(gid)

        for combination, point_blocks in per_combo_points.items():
            if step in result.setdefault(combination, {}):
                raise ValueError(f"Duplicate train step {step} for {combination}")
            points = np.vstack(point_blocks)
            prompt_idx = np.concatenate(per_combo_prompt_idx[combination])
            record: Dict[str, Any] = {
                "points": points,
                "prompt_idx": prompt_idx,
                "group_ids": per_combo_group_ids[combination],
                "step": step,
                "n_total": len(points),
                "n_valid": len(points),
                "n_groups": len(per_combo_group_ids[combination]),
            }
            if prompts_raw is not None:
                record["prompt_labels"] = list(prompts_raw)
            result[combination][step] = record

    for combination in result:
        result[combination] = dict(sorted(result[combination].items()))
    return dict(sorted(result.items())), reward_names


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
