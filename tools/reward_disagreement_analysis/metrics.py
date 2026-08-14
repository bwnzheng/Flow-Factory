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

"""Prompt-local conflict-mass metrics for multi-reward scalarization.

Every call operates on one frozen rollout group. Reward and scalar advantages
are centered with the original group's uniform reward means. SRC probabilities,
when saved for that group, only reweight the resulting sample-level conflict
indicator; they never redefine the advantages or per-reward conflict source.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_reward_disagreement_metrics(
    rewards: np.ndarray,
    reward_weights: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute natural and optional effective conflict mass for one prompt group.

    Args:
        rewards: Finite raw rewards shaped ``(group_size, n_rewards)``.
        reward_weights: Positive scalarization weights shaped ``(n_rewards,)``.
        sample_weights: Optional nonnegative SRC probabilities shaped
            ``(group_size,)``. They are normalized to a probability vector and
            aggregate only the already-computed conflict indicator.

    Returns:
        Group-level natural per-reward disagreement and overall conflict mass.
        When ``sample_weights`` is supplied, also returns effective conflict
        mass. Exact-zero advantages are non-conflicting because the definition
        uses a strict negative product.
    """
    matrix = _validate_rewards(rewards)
    group_size, n_rewards = matrix.shape
    weights = _validate_reward_weights(reward_weights, n_rewards)

    reward_advantages = matrix - matrix.mean(axis=0, keepdims=True)
    scalar_advantages = (matrix @ weights) - (matrix @ weights).mean()
    disagreement = reward_advantages * scalar_advantages[:, None] < 0.0
    has_conflict = disagreement.any(axis=1)

    result: dict[str, Any] = {
        "group_size": group_size,
        "natural_per_reward_disagreement_rate": disagreement.mean(axis=0),
        "natural_conflict_mass": float(has_conflict.mean()),
    }
    if sample_weights is not None:
        probabilities = _validate_probabilities(sample_weights, group_size)
        result["effective_conflict_mass"] = float(probabilities @ has_conflict)
    return result


def aggregate_group_metrics(group_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average prompt-local conflict metrics across groups.

    Args:
        group_metrics: Metrics from prompt-local rollout groups at one step and
            reward combination.

    Returns:
        One equally prompt-weighted summary. Effective conflict mass is
        averaged only across groups that saved SRC probabilities.
    """
    if not group_metrics:
        raise ValueError("Cannot aggregate an empty collection of prompt-group metrics.")

    n_rewards = len(group_metrics[0]["natural_per_reward_disagreement_rate"])
    for metrics in group_metrics:
        if len(metrics["natural_per_reward_disagreement_rate"]) != n_rewards:
            raise ValueError("All groups must have the same number of active rewards.")

    result: dict[str, Any] = {
        "n_groups": len(group_metrics),
        "mean_group_size": float(np.mean([metrics["group_size"] for metrics in group_metrics])),
        "natural_per_reward_disagreement_rate": np.mean(
            [metrics["natural_per_reward_disagreement_rate"] for metrics in group_metrics], axis=0
        ),
        "natural_conflict_mass": float(
            np.mean([metrics["natural_conflict_mass"] for metrics in group_metrics])
        ),
    }
    effective = [metrics for metrics in group_metrics if "effective_conflict_mass" in metrics]
    if effective:
        result.update(
            {
                "n_effective_groups": len(effective),
                "effective_conflict_mass": float(
                    np.mean([metrics["effective_conflict_mass"] for metrics in effective])
                ),
            }
        )
    return result


def _validate_rewards(rewards: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rewards, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(
            "rewards must have shape (group_size >= 2, n_rewards >= 2), " f"got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("rewards must contain only finite values.")
    return matrix


def _validate_reward_weights(reward_weights: np.ndarray, n_rewards: int) -> np.ndarray:
    weights = np.asarray(reward_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (n_rewards,):
        raise ValueError(f"reward_weights must have shape ({n_rewards},), got {weights.shape}.")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError(
            "reward_weights must be finite and strictly positive for every active reward."
        )
    return weights


def _validate_probabilities(sample_weights: np.ndarray, group_size: int) -> np.ndarray:
    probabilities = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if probabilities.shape != (group_size,):
        raise ValueError(
            f"sample_weights must have shape ({group_size},), got {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("sample_weights must be finite and nonnegative.")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("sample_weights must have positive total mass.")
    return probabilities / total
