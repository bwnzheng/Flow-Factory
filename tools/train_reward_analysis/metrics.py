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

"""Prompt-local standardized sample-wise reward-concordance metrics.

Every call operates on one frozen rollout group. It uses only raw saved rewards
and the historical scalarization weights, with the original uniform group mean
and population standard deviation as the reference. No SRC probability or
reweighted statistic is used.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_reward_concordance_metrics(
    rewards: np.ndarray,
    reward_weights: np.ndarray,
) -> dict[str, Any]:
    """Compute standardized conflict scores, disagreement rates, and lower bound.

    Args:
        rewards: Finite raw rewards shaped ``(group_size, n_rewards)``.
        reward_weights: Positive scalarization weights shaped ``(n_rewards,)``.

    Returns:
        The mean standardized conflict score and disagreement rate for every
        reward, plus the mean of each sample's weakest reward score.
    """
    matrix = _validate_rewards(rewards)
    group_size, n_rewards = matrix.shape
    weights = _validate_reward_weights(reward_weights, n_rewards)

    centered_rewards = _standardize_centered(matrix, axis=0)
    scalar_rewards = matrix @ weights
    scalar_advantages = _standardize_centered(scalar_rewards, axis=0)
    conflict_scores = weights[None, :] * centered_rewards * scalar_advantages[:, None]
    disagreement = (centered_rewards * scalar_advantages[:, None] < 0.0).mean(axis=0)

    return {
        "group_size": group_size,
        "per_reward_conflict_score": conflict_scores.mean(axis=0),
        "per_reward_disagreement": disagreement,
        "reward_concordance_lower_bound": float(conflict_scores.min(axis=1).mean()),
    }


def aggregate_group_metrics(group_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average prompt-local reward-concordance metrics across groups.

    Args:
        group_metrics: Metrics from prompt-local rollout groups at one step and
            reward combination.

    Returns:
        An equally prompt-weighted summary without probability reweighting.
    """
    if not group_metrics:
        raise ValueError("Cannot aggregate an empty collection of prompt-group metrics.")

    n_rewards = len(group_metrics[0]["per_reward_conflict_score"])
    for metrics in group_metrics:
        if len(metrics["per_reward_conflict_score"]) != n_rewards:
            raise ValueError("All groups must have the same number of active rewards.")

    return {
        "n_groups": len(group_metrics),
        "mean_group_size": float(np.mean([metrics["group_size"] for metrics in group_metrics])),
        "per_reward_conflict_score": np.mean(
            [metrics["per_reward_conflict_score"] for metrics in group_metrics], axis=0
        ),
        "per_reward_disagreement": np.mean(
            [metrics["per_reward_disagreement"] for metrics in group_metrics], axis=0
        ),
        "reward_concordance_lower_bound": float(
            np.mean([metrics["reward_concordance_lower_bound"] for metrics in group_metrics])
        ),
    }


def _validate_rewards(rewards: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rewards, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(
            "rewards must have shape (group_size >= 2, n_rewards >= 2), " f"got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("rewards must contain only finite values.")
    return matrix


def _standardize_centered(values: np.ndarray, axis: int) -> np.ndarray:
    """Center values and divide by their population standard deviation."""
    centered = values - np.mean(values, axis=axis, keepdims=True)
    scale = np.std(values, axis=axis, keepdims=True, ddof=0)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0.0)


def _validate_reward_weights(reward_weights: np.ndarray, n_rewards: int) -> np.ndarray:
    weights = np.asarray(reward_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (n_rewards,):
        raise ValueError(f"reward_weights must have shape ({n_rewards},), got {weights.shape}.")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError(
            "reward_weights must be finite and strictly positive for every active reward."
        )
    return weights
