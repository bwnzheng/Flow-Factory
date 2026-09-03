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


def compute_group_metrics(rewards: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """Compute covariance geometry for one prompt's repeated rollouts.

    Args:
        rewards: Finite matrix shaped ``(samples, rewards)``.

    Returns:
        Reward means, covariance, correlation, and negative-correlation metrics.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[0] < 2 or rewards.shape[1] < 2:
        raise ValueError(f"rewards must be (samples >= 2, rewards >= 2), got {rewards.shape}.")
    if not np.isfinite(rewards).all():
        raise ValueError("Rewards must be finite.")
    covariance = np.cov(rewards, rowvar=False, ddof=1)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = np.divide(covariance, scale, out=np.eye(rewards.shape[1]), where=scale > 0)
    # With nonzero variance this is exactly covariance after per-reward z-scoring.
    # Keep the existing finite convention (diag=1, zero-variance cross terms=0).
    standardized_covariance = correlation.copy()
    upper = correlation[np.triu_indices(rewards.shape[1], k=1)]
    return {
        "mean": rewards.mean(0),
        "covariance": covariance,
        "standardized_covariance": standardized_covariance,
        "correlation": correlation,
        "negative_pairwise_correlation_ratio": float((upper < 0).mean()),
        "mean_negative_pairwise_correlation": (
            float(upper[upper < 0].mean()) if (upper < 0).any() else 0.0
        ),
    }


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
    )
    return {key: np.mean([group[key] for group in groups], axis=0) for key in keys}
