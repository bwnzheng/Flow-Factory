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

"""Prompt-local sample weighting for group-relative policy updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SRCReweightResult:
    """SRC-Reweight outputs aligned with the input sample axis."""

    contributions: np.ndarray
    weighted_contributions: np.ndarray
    scores: np.ndarray
    raw_scores: np.ndarray
    saturated_scores: np.ndarray
    normalized_scores: np.ndarray
    probabilities: np.ndarray
    loss_multipliers: np.ndarray
    uniform_advantages: np.ndarray
    weighted_advantages: np.ndarray
    effective_advantages: np.ndarray
    scalar_variances: np.ndarray
    uniform_means: np.ndarray
    weighted_means: np.ndarray
    weighted_variances: np.ndarray
    effective_sample_sizes: np.ndarray
    lower_bound_uniform: np.ndarray
    lower_bound_reweighted: np.ndarray
    degenerate_scalar_contrast: np.ndarray


def compute_src_reweight(
    reward_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    applicable: np.ndarray,
    group_indices: np.ndarray,
    interpolation: float,
    temperature: float,
    epsilon: float,
    degeneracy_threshold: float,
    score_type: Literal["raw", "saturated"] = "saturated",
) -> SRCReweightResult:
    """Compute frozen-group SRC probabilities and effective advantages.

    All arithmetic is local NumPy work. This function performs no rank gather,
    aggregation, or reduction. ``reward_matrix`` is shaped ``(R, S)`` to match
    :class:`AdvantageProcessor`'s reward-major representation.

    Args:
        reward_matrix: Reward values with shape ``(num_rewards, num_samples)``.
        weight_matrix: Fixed source-aware weights aligned with ``reward_matrix``.
        applicable: Boolean reward-applicability mask aligned with ``reward_matrix``.
        group_indices: Prompt-group identifier for every sample.
        interpolation: Mixture coefficient between uniform and concordance weights.
        temperature: Positive softmax temperature for the selected SRC scores.
        epsilon: Positive numerical safeguard for raw calibration and saturation.
        degeneracy_threshold: Scalar-variance threshold for uniform fallback.
        score_type: Frozen-group SRC score formula used for softmax weighting.

    Returns:
        SRC diagnostics, probabilities, and effective advantages aligned with samples.

    Note:
        SRC probabilities are used as an outer sample-mass multiplier. The
        optimization advantage remains centered and scaled with the original
        uniform prompt-group statistics; the weighted-centered quantities are
        retained as diagnostics for the SRC distribution itself.
    """
    rewards = np.asarray(reward_matrix, dtype=np.float64)
    weights = np.asarray(weight_matrix, dtype=np.float64)
    applicable_mask = np.asarray(applicable, dtype=bool)
    groups = np.asarray(group_indices, dtype=np.int64)

    if rewards.ndim != 2:
        raise ValueError(f"reward_matrix must have shape (R, S), got {rewards.shape}.")
    if weights.shape != rewards.shape or applicable_mask.shape != rewards.shape:
        raise ValueError(
            "weight_matrix and applicable must match reward_matrix shape; "
            f"got rewards={rewards.shape}, weights={weights.shape}, "
            f"applicable={applicable_mask.shape}."
        )
    if groups.shape != (rewards.shape[1],):
        raise ValueError(
            f"group_indices must have shape ({rewards.shape[1]},), got {groups.shape}."
        )
    if not 0.0 <= interpolation < 1.0:
        raise ValueError(f"interpolation must be in [0, 1), got {interpolation}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if degeneracy_threshold < 0.0:
        raise ValueError(f"degeneracy_threshold must be >= 0, got {degeneracy_threshold}.")
    if score_type not in {"raw", "saturated"}:
        raise ValueError(f"score_type must be 'raw' or 'saturated', got {score_type!r}.")
    if np.any(weights[applicable_mask] < 0.0):
        raise ValueError("SRC-Reweight requires nonnegative active reward weights.")

    num_rewards, num_samples = rewards.shape
    contributions = np.full((num_rewards, num_samples), np.nan, dtype=np.float64)
    weighted_contributions = np.full((num_rewards, num_samples), np.nan, dtype=np.float64)
    scores = np.zeros(num_samples, dtype=np.float64)
    raw_scores = np.zeros(num_samples, dtype=np.float64)
    saturated_scores = np.zeros(num_samples, dtype=np.float64)
    normalized_scores = np.zeros(num_samples, dtype=np.float64)
    probabilities = np.zeros(num_samples, dtype=np.float64)
    loss_multipliers = np.ones(num_samples, dtype=np.float64)
    uniform_advantages = np.zeros(num_samples, dtype=np.float64)
    weighted_advantages = np.zeros(num_samples, dtype=np.float64)
    effective_advantages = np.zeros(num_samples, dtype=np.float64)
    scalar_variances = np.zeros(num_samples, dtype=np.float64)
    uniform_means = np.zeros(num_samples, dtype=np.float64)
    weighted_means = np.zeros(num_samples, dtype=np.float64)
    weighted_variances = np.zeros(num_samples, dtype=np.float64)
    effective_sample_sizes = np.zeros(num_samples, dtype=np.float64)
    lower_bound_uniform = np.zeros(num_samples, dtype=np.float64)
    lower_bound_reweighted = np.zeros(num_samples, dtype=np.float64)
    degenerate = np.zeros(num_samples, dtype=bool)

    for group_id in np.unique(groups):
        sample_indices = np.flatnonzero(groups == group_id)
        group_size = len(sample_indices)
        if group_size < 2:
            raise ValueError(
                "SRC-Reweight requires at least two samples per prompt group; "
                f"group {int(group_id)} has {group_size}."
            )

        group_applicable = applicable_mask[:, sample_indices]
        group_weights = weights[:, sample_indices]
        if not np.all(group_applicable == group_applicable[:, :1]):
            raise ValueError(
                "SRC-Reweight requires homogeneous reward applicability within each prompt group; "
                f"group {int(group_id)} is mixed."
            )
        if not np.allclose(group_weights, group_weights[:, :1], rtol=0.0, atol=0.0):
            raise ValueError(
                "SRC-Reweight requires fixed reward weights within each prompt group; "
                f"group {int(group_id)} is mixed."
            )

        reward_weights = group_weights[:, 0]
        active = group_applicable[:, 0] & (reward_weights > 0.0)
        if int(active.sum()) < 2:
            raise ValueError(
                "SRC-Reweight requires at least two active rewards with positive weights in every "
                f"prompt group; group {int(group_id)} has {int(active.sum())}."
            )

        group_rewards = rewards[:, sample_indices]
        active_rewards = group_rewards[active]
        active_weights = reward_weights[active]
        centered_rewards = _standardize_centered(active_rewards, axis=1)
        scalar_rewards = active_weights @ active_rewards
        scalar_centered_raw = scalar_rewards - scalar_rewards.mean()
        scalar_variance = float(np.mean(np.square(scalar_centered_raw)))
        scalar_centered = _standardize_centered(scalar_rewards, axis=0)
        group_raw_contributions = (
            active_weights[:, None] * centered_rewards * scalar_centered[None, :]
        )
        saturation = scalar_centered / (np.abs(scalar_centered) + 1.0 + epsilon)
        group_saturated_contributions = (
            active_weights[:, None] * centered_rewards * saturation[None, :]
        )
        group_raw_scores = group_raw_contributions.min(axis=0)
        group_saturated_scores = group_saturated_contributions.min(axis=0)
        if score_type == "raw":
            group_contributions = group_raw_contributions
            group_scores = group_raw_scores
        else:
            group_contributions = group_saturated_contributions
            group_scores = group_saturated_scores

        contributions[np.flatnonzero(active)[:, None], sample_indices] = group_contributions
        scores[sample_indices] = group_scores
        raw_scores[sample_indices] = group_raw_scores
        saturated_scores[sample_indices] = group_saturated_scores
        scalar_variances[sample_indices] = scalar_variance
        uniform_mean = float(scalar_rewards.mean())
        uniform_means[sample_indices] = uniform_mean

        is_degenerate = scalar_variance <= degeneracy_threshold
        if is_degenerate:
            group_probabilities = np.full(group_size, 1.0 / group_size, dtype=np.float64)
            group_normalized_scores = np.zeros(group_size, dtype=np.float64)
        else:
            group_normalized_scores = group_scores
            logits = group_normalized_scores / temperature
            logits = logits - logits.max()
            concordant_probabilities = np.exp(logits)
            concordant_probabilities /= concordant_probabilities.sum()
            group_probabilities = (
                1.0 - interpolation
            ) / group_size + interpolation * concordant_probabilities

        weighted_mean = float(group_probabilities @ scalar_rewards)
        centered_weighted = scalar_rewards - weighted_mean
        weighted_reward_means = active_rewards @ group_probabilities
        weighted_centered_rewards = _standardize_centered(
            active_rewards,
            axis=1,
            means=weighted_reward_means,
            scales=np.sqrt(
                group_probabilities @ np.square(active_rewards - weighted_reward_means[:, None]).T
            ),
        )
        weighted_scalar_directions = np.sign(centered_weighted)
        group_weighted_contributions = (
            active_weights[:, None]
            * weighted_centered_rewards
            * weighted_scalar_directions[None, :]
        )
        weighted_variance = float(group_probabilities @ np.square(centered_weighted))
        uniform_centered = scalar_rewards - uniform_mean
        group_uniform_advantages = uniform_centered / np.sqrt(scalar_variance + epsilon)
        group_weighted_advantages = centered_weighted / np.sqrt(weighted_variance + epsilon)
        group_multipliers = group_size * group_probabilities

        normalized_scores[sample_indices] = group_normalized_scores
        weighted_contributions[np.flatnonzero(active)[:, None], sample_indices] = (
            group_weighted_contributions
        )
        probabilities[sample_indices] = group_probabilities
        loss_multipliers[sample_indices] = group_multipliers
        uniform_advantages[sample_indices] = group_uniform_advantages
        weighted_advantages[sample_indices] = group_weighted_advantages
        effective_advantages[sample_indices] = group_multipliers * group_uniform_advantages
        weighted_means[sample_indices] = weighted_mean
        weighted_variances[sample_indices] = weighted_variance
        effective_sample_sizes[sample_indices] = 1.0 / float(
            group_probabilities @ group_probabilities
        )
        lower_bound_uniform[sample_indices] = float(group_scores.mean())
        lower_bound_reweighted[sample_indices] = float(group_probabilities @ group_scores)
        degenerate[sample_indices] = is_degenerate

    return SRCReweightResult(
        contributions=contributions,
        weighted_contributions=weighted_contributions,
        scores=scores,
        raw_scores=raw_scores,
        saturated_scores=saturated_scores,
        normalized_scores=normalized_scores,
        probabilities=probabilities,
        loss_multipliers=loss_multipliers,
        uniform_advantages=uniform_advantages,
        weighted_advantages=weighted_advantages,
        effective_advantages=effective_advantages,
        scalar_variances=scalar_variances,
        uniform_means=uniform_means,
        weighted_means=weighted_means,
        weighted_variances=weighted_variances,
        effective_sample_sizes=effective_sample_sizes,
        lower_bound_uniform=lower_bound_uniform,
        lower_bound_reweighted=lower_bound_reweighted,
        degenerate_scalar_contrast=degenerate,
    )


def _standardize_centered(
    values: np.ndarray,
    axis: int,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Center values and divide by their population standard deviation."""
    array = np.asarray(values, dtype=np.float64)
    if means is None:
        means = np.mean(array, axis=axis, keepdims=True)
    else:
        means = np.asarray(means, dtype=np.float64)
        if axis == 1:
            means = means[:, None]
    centered = array - means
    if scales is None:
        scales = np.sqrt(np.mean(np.square(centered), axis=axis, keepdims=True))
    else:
        scales = np.asarray(scales, dtype=np.float64)
        if axis == 1:
            scales = scales[:, None]
    return np.divide(centered, scales, out=np.zeros_like(centered), where=scales > 0.0)
