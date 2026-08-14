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

"""Prompt-local metrics for multi-reward scalarization disagreement.

Every call operates on one frozen rollout group.  The function intentionally
centers each reward dimension with the uniform group mean before it optionally
uses SRC probabilities for effective-training-mass aggregation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_reward_disagreement_metrics(
    rewards: np.ndarray,
    reward_weights: np.ndarray,
    sample_weights: np.ndarray | None = None,
    reward_eps: float | np.ndarray = 1e-8,
    scalar_eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute natural and optional effective disagreement for one prompt group.

    Args:
        rewards: Finite raw rewards shaped ``(group_size, n_rewards)``.
        reward_weights: Positive scalarization weights shaped ``(n_rewards,)``.
        sample_weights: Optional nonnegative SRC probabilities shaped
            ``(group_size,)``. They are normalized to a probability vector and
            are used only to aggregate already-computed natural decisions.
        reward_eps: Scalar or per-reward threshold for neutral single-reward
            advantages.
        scalar_eps: Threshold for neutral scalar advantages.

    Returns:
        A dictionary containing group-level natural metrics. When
        ``sample_weights`` is supplied, it also contains effective metrics.
        Undefined ratios are represented by ``NaN``; callers must aggregate
        them over eligible prompt groups rather than treating them as zero.
    """
    matrix = _validate_rewards(rewards)
    group_size, n_rewards = matrix.shape
    weights = _validate_reward_weights(reward_weights, n_rewards)
    reward_thresholds = _as_reward_thresholds(reward_eps, n_rewards)
    scalar_threshold = _validate_nonnegative_scalar(scalar_eps, "scalar_eps")

    reward_advantages = matrix - matrix.mean(axis=0, keepdims=True)
    scalar_rewards = matrix @ weights
    scalar_advantages = scalar_rewards - scalar_rewards.mean()
    scalar_advantages_from_dimensions = reward_advantages @ weights
    identity_error = float(np.max(np.abs(scalar_advantages - scalar_advantages_from_dimensions)))

    valid = (np.abs(reward_advantages) > reward_thresholds[None, :]) & (
        np.abs(scalar_advantages[:, None]) > scalar_threshold
    )
    disagreement = (reward_advantages * scalar_advantages[:, None] < 0.0) & valid
    fully_valid = valid.all(axis=1)
    fully_concordant = fully_valid & (~disagreement).all(axis=1)
    disagreement_count = disagreement.sum(axis=1)

    result: dict[str, Any] = {
        "group_size": group_size,
        "disagreement_rate_per_reward": _rates(disagreement, valid),
        "valid_ratio_per_reward": valid.mean(axis=0),
        "fully_valid_ratio": float(fully_valid.mean()),
        "fully_concordant_ratio": _masked_ratio(fully_concordant, fully_valid),
        "fully_concordant_positive_ratio": _masked_ratio(
            fully_concordant & (scalar_advantages > scalar_threshold),
            fully_valid & (scalar_advantages > scalar_threshold),
        ),
        "fully_concordant_negative_ratio": _masked_ratio(
            fully_concordant & (scalar_advantages < -scalar_threshold),
            fully_valid & (scalar_advantages < -scalar_threshold),
        ),
        "mean_disagreement_count": _masked_mean(disagreement_count, fully_valid),
        "disagreement_count_histogram": _count_histogram(
            disagreement_count,
            fully_valid,
            n_rewards,
        ),
        "scalar_advantage_identity_max_abs_error": identity_error,
    }

    if sample_weights is not None:
        probabilities = _validate_probabilities(sample_weights, group_size)
        valid_mass = (probabilities[:, None] * valid).sum(axis=0)
        fully_valid_mass = float(probabilities @ fully_valid)
        result.update(
            {
                "effective_disagreement_rate_per_reward": _weighted_rates(
                    probabilities,
                    disagreement,
                    valid_mass,
                ),
                "effective_valid_mass_per_reward": valid_mass,
                "effective_fully_valid_mass": fully_valid_mass,
                "effective_fully_concordant_ratio": _weighted_masked_ratio(
                    probabilities,
                    fully_concordant,
                    fully_valid,
                ),
                "effective_fully_concordant_positive_ratio": _weighted_masked_ratio(
                    probabilities,
                    fully_concordant & (scalar_advantages > scalar_threshold),
                    fully_valid & (scalar_advantages > scalar_threshold),
                ),
                "effective_fully_concordant_negative_ratio": _weighted_masked_ratio(
                    probabilities,
                    fully_concordant & (scalar_advantages < -scalar_threshold),
                    fully_valid & (scalar_advantages < -scalar_threshold),
                ),
                "effective_mean_disagreement_count": _weighted_masked_mean(
                    probabilities,
                    disagreement_count,
                    fully_valid,
                ),
                "effective_disagreement_count_histogram": _weighted_count_histogram(
                    probabilities,
                    disagreement_count,
                    fully_valid,
                    n_rewards,
                ),
            }
        )
    return result


def aggregate_group_metrics(group_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average metrics across prompt-local groups.

    A step is an equally weighted collection of prompts, not one large rollout
    group. Undefined ratios from groups with no eligible samples are ignored.
    Effective metrics are averaged only across groups that carry SRC
    probabilities.
    """
    if not group_metrics:
        raise ValueError("Cannot aggregate an empty collection of prompt-group metrics.")

    n_rewards = len(group_metrics[0]["disagreement_rate_per_reward"])
    for metrics in group_metrics:
        if len(metrics["disagreement_rate_per_reward"]) != n_rewards:
            raise ValueError("All groups must have the same number of active rewards.")

    result: dict[str, Any] = {
        "n_groups": len(group_metrics),
        "mean_group_size": _nanmean_scalar([metrics["group_size"] for metrics in group_metrics]),
        "disagreement_rate_per_reward": _nanmean_array(
            [metrics["disagreement_rate_per_reward"] for metrics in group_metrics]
        ),
        "valid_ratio_per_reward": _nanmean_array(
            [metrics["valid_ratio_per_reward"] for metrics in group_metrics]
        ),
        "fully_valid_ratio": _nanmean_scalar(
            [metrics["fully_valid_ratio"] for metrics in group_metrics]
        ),
        "fully_concordant_ratio": _nanmean_scalar(
            [metrics["fully_concordant_ratio"] for metrics in group_metrics]
        ),
        "fully_concordant_positive_ratio": _nanmean_scalar(
            [metrics["fully_concordant_positive_ratio"] for metrics in group_metrics]
        ),
        "fully_concordant_negative_ratio": _nanmean_scalar(
            [metrics["fully_concordant_negative_ratio"] for metrics in group_metrics]
        ),
        "mean_disagreement_count": _nanmean_scalar(
            [metrics["mean_disagreement_count"] for metrics in group_metrics]
        ),
        "disagreement_count_histogram": _nanmean_array(
            [metrics["disagreement_count_histogram"] for metrics in group_metrics]
        ),
        "scalar_advantage_identity_max_abs_error": max(
            metrics["scalar_advantage_identity_max_abs_error"] for metrics in group_metrics
        ),
    }

    effective = [
        metrics for metrics in group_metrics if "effective_disagreement_rate_per_reward" in metrics
    ]
    if effective:
        result.update(
            {
                "n_effective_groups": len(effective),
                "effective_disagreement_rate_per_reward": _nanmean_array(
                    [metrics["effective_disagreement_rate_per_reward"] for metrics in effective]
                ),
                "effective_valid_mass_per_reward": _nanmean_array(
                    [metrics["effective_valid_mass_per_reward"] for metrics in effective]
                ),
                "effective_fully_valid_mass": _nanmean_scalar(
                    [metrics["effective_fully_valid_mass"] for metrics in effective]
                ),
                "effective_fully_concordant_ratio": _nanmean_scalar(
                    [metrics["effective_fully_concordant_ratio"] for metrics in effective]
                ),
                "effective_fully_concordant_positive_ratio": _nanmean_scalar(
                    [metrics["effective_fully_concordant_positive_ratio"] for metrics in effective]
                ),
                "effective_fully_concordant_negative_ratio": _nanmean_scalar(
                    [metrics["effective_fully_concordant_negative_ratio"] for metrics in effective]
                ),
                "effective_mean_disagreement_count": _nanmean_scalar(
                    [metrics["effective_mean_disagreement_count"] for metrics in effective]
                ),
                "effective_disagreement_count_histogram": _nanmean_array(
                    [metrics["effective_disagreement_count_histogram"] for metrics in effective]
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


def _as_reward_thresholds(reward_eps: float | np.ndarray, n_rewards: int) -> np.ndarray:
    thresholds = np.asarray(reward_eps, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.full(n_rewards, float(thresholds))
    if thresholds.shape != (n_rewards,):
        raise ValueError(
            f"reward_eps must be scalar or shape ({n_rewards},), got {thresholds.shape}."
        )
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0.0):
        raise ValueError("reward_eps must contain finite, nonnegative values.")
    return thresholds


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}.")
    return scalar


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


def _rates(disagreement: np.ndarray, valid: np.ndarray) -> np.ndarray:
    numerators = disagreement.sum(axis=0, dtype=np.float64)
    denominators = valid.sum(axis=0, dtype=np.float64)
    return _divide_or_nan(numerators, denominators)


def _weighted_rates(
    probabilities: np.ndarray,
    disagreement: np.ndarray,
    valid_mass: np.ndarray,
) -> np.ndarray:
    numerators = (probabilities[:, None] * disagreement).sum(axis=0)
    return _divide_or_nan(numerators, valid_mass)


def _masked_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denominator_count = int(denominator.sum())
    if denominator_count == 0:
        return float("nan")
    return float(numerator.sum() / denominator_count)


def _weighted_masked_ratio(
    probabilities: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> float:
    denominator_mass = float(probabilities @ denominator)
    if denominator_mass == 0.0:
        return float("nan")
    return float((probabilities @ numerator) / denominator_mass)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(values[mask].mean())


def _weighted_masked_mean(
    probabilities: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> float:
    mask_mass = float(probabilities @ mask)
    if mask_mass == 0.0:
        return float("nan")
    return float(probabilities @ (values * mask) / mask_mass)


def _count_histogram(values: np.ndarray, mask: np.ndarray, n_rewards: int) -> np.ndarray:
    if not mask.any():
        return np.full(n_rewards + 1, np.nan)
    counts = np.bincount(values[mask], minlength=n_rewards + 1)
    return counts[: n_rewards + 1] / counts.sum()


def _weighted_count_histogram(
    probabilities: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    n_rewards: int,
) -> np.ndarray:
    mass = float(probabilities @ mask)
    if mass == 0.0:
        return np.full(n_rewards + 1, np.nan)
    result = np.zeros(n_rewards + 1, dtype=np.float64)
    for count in range(n_rewards + 1):
        result[count] = float(probabilities[(values == count) & mask].sum() / mass)
    return result


def _divide_or_nan(numerators: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    result = np.full(np.asarray(numerators).shape, np.nan, dtype=np.float64)
    valid = denominators > 0.0
    result[valid] = np.asarray(numerators, dtype=np.float64)[valid] / denominators[valid]
    return result


def _nanmean_scalar(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _nanmean_array(values: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.asarray(values, dtype=np.float64)
    result = np.full(stacked.shape[1:], np.nan, dtype=np.float64)
    for index in np.ndindex(result.shape):
        column = stacked[(slice(None),) + index]
        finite = column[np.isfinite(column)]
        if finite.size:
            result[index] = finite.mean()
    return result
