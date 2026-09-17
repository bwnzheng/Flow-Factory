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

"""Pure prompt-local reward geometry metrics."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from .jsr import (
    build_reference_thresholds,
    compute_jsr,
    paired_prompt_bootstrap,
    reference_quantile,
)


def compute_group_metrics(
    rewards: np.ndarray, reward_weights: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """Compute covariance geometry and agreement counts for one prompt's rollouts.

    Args:
        rewards: Finite matrix shaped ``(samples, rewards)``.
        reward_weights: Positive scalarization weights shaped ``(rewards,)``. The
            agreement count compares every reward against the weighted scalar, so
            the weights must be the ones the compared checkpoints were trained
            with, otherwise the counts are not the same quantity.

    Returns:
        Reward means, covariance, correlation, negative-correlation metrics, and
        the prompt-local agreement-count distribution over ``0..n_rewards``.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[0] < 2 or rewards.shape[1] < 2:
        raise ValueError(f"rewards must be (samples >= 2, rewards >= 2), got {rewards.shape}.")
    if not np.isfinite(rewards).all():
        raise ValueError("Rewards must be finite.")
    weights = _validate_reward_weights(reward_weights, rewards.shape[1])
    covariance = np.cov(rewards, rowvar=False, ddof=1)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = np.divide(covariance, scale, out=np.eye(rewards.shape[1]), where=scale > 0)
    # With nonzero variance this is exactly covariance after per-reward z-scoring.
    # Keep the existing finite convention (diag=1, zero-variance cross terms=0).
    standardized_covariance = correlation.copy()
    upper = correlation[np.triu_indices(rewards.shape[1], k=1)]
    # Agreement counts follow the training-side reward-concordance tool exactly:
    # center and population-standardize each reward inside the prompt, and
    # standardize the weighted scalar the same way. A sample's agreeing count is
    # how many rewards point the same way as the aggregate objective on it, so an
    # exact zero product counts as agreeing, matching the training-side rule.
    centered = rewards - rewards.mean(axis=0, keepdims=True)
    reward_scale = rewards.std(axis=0, ddof=0, keepdims=True)
    reward_z = np.divide(
        centered, reward_scale, out=np.zeros_like(centered), where=reward_scale > 0.0
    )
    scalar = rewards @ weights
    scalar_centered = scalar - scalar.mean()
    scalar_scale = scalar.std(ddof=0)
    scalar_z = np.divide(
        scalar_centered,
        scalar_scale,
        out=np.zeros_like(scalar_centered),
        where=scalar_scale > 0.0,
    )
    agreeing_counts = (reward_z * scalar_z[:, None] >= 0.0).sum(axis=1)
    agreement_count_distribution = (
        np.bincount(agreeing_counts, minlength=rewards.shape[1] + 1) / rewards.shape[0]
    )
    return {
        "mean": rewards.mean(0),
        "covariance": covariance,
        "standardized_covariance": standardized_covariance,
        "correlation": correlation,
        "negative_pairwise_correlation_ratio": float((upper < 0).mean()),
        "mean_negative_pairwise_correlation": (
            float(upper[upper < 0].mean()) if (upper < 0).any() else 0.0
        ),
        "agreement_count_distribution": agreement_count_distribution,
        "mean_agreement_count": float(agreeing_counts.mean()),
        "fully_concordant_sample_rate": float(agreement_count_distribution[rewards.shape[1]]),
    }


def _validate_reward_weights(reward_weights: np.ndarray, n_rewards: int) -> np.ndarray:
    weights = np.asarray(reward_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (n_rewards,):
        raise ValueError(f"reward_weights must have shape ({n_rewards},), got {weights.shape}.")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("reward_weights must be finite and strictly positive.")
    return weights


def aggregate_group_metrics(
    groups: List[Dict[str, Union[np.ndarray, float]]],
) -> Dict[str, Union[np.ndarray, float]]:
    """Macro-average prompt-local metrics without pooling prompts.

    Args:
        groups: Prompt-local metrics with a common reward dimension.

    Returns:
        Equally prompt-weighted aggregate metrics.
    """
    if not groups:
        raise ValueError("Cannot aggregate no prompt groups.")
    keys = (
        "mean",
        "covariance",
        "standardized_covariance",
        "correlation",
        "negative_pairwise_correlation_ratio",
        "mean_negative_pairwise_correlation",
        "agreement_count_distribution",
        "mean_agreement_count",
        "fully_concordant_sample_rate",
    )
    return {key: np.mean([group[key] for group in groups], axis=0) for key in keys}
