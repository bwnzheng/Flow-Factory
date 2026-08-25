# Copyright 2026 Bowen-Zheng
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

"""Sample-wise Reward Concordance selection and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CovarianceGroupScore:
    """Store the covariance diagnostics for one candidate subset."""

    covariance: np.ndarray
    contribution_vector: np.ndarray
    score: float
    scalar_variance: float
    mean_scalar_reward: float
    degenerate: bool


@dataclass(frozen=True)
class SRCContributionScore:
    """Store frozen-pool Sample-wise Reward Concordance diagnostics."""

    pool_mean: np.ndarray
    scalar_rewards: np.ndarray
    centered_rewards: np.ndarray
    scalar_advantages: np.ndarray
    contribution_matrix: np.ndarray
    fitness: np.ndarray
    degenerate_scalar_contrast: bool


@dataclass(frozen=True)
class SRCSelectionResult:
    """Store scalar-elitist SRC selection diagnostics."""

    selected_indices: np.ndarray
    rejected_indices: np.ndarray
    elite_index: int
    sample_scores: SRCContributionScore
    group_score: CovarianceGroupScore
    frozen_contribution_vector: np.ndarray
    frozen_score: float
    lower_bound: float


def population_covariance(reward_matrix: np.ndarray) -> np.ndarray:
    """Compute population covariance using the ``1 / n`` convention.

    Args:
        reward_matrix: Reward matrix shaped ``(n_candidates, n_rewards)``.

    Returns:
        Symmetric population covariance matrix shaped ``(n_rewards, n_rewards)``.
    """
    matrix = _validate_reward_matrix(reward_matrix)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / matrix.shape[0]
    return (covariance + covariance.T) * 0.5


def covariance_group_score(
    reward_matrix: np.ndarray,
    weights: np.ndarray,
    objective: str,
) -> CovarianceGroupScore:
    """Score a complete candidate group by its weakest active reward contribution.

    Args:
        reward_matrix: Raw reward values shaped ``(n_candidates, n_rewards)``.
        weights: Nonnegative scalarization weights shaped ``(n_rewards,)``.
        objective: ``"standardized_grpo"`` or ``"locally_linear_nft"``. Both
            use the group-dependent scalar-reward population variance as the
            squared normalizer under the required group-local normalization.

    Returns:
        Group-level covariance score and diagnostics.
    """
    matrix = _validate_reward_matrix(reward_matrix)
    weight_vector = _validate_weights(weights, matrix.shape[1])
    covariance = population_covariance(matrix)
    covariance_weight = covariance @ weight_vector
    scalar_variance = float(weight_vector @ covariance_weight)
    scalar_rewards = matrix @ weight_vector
    active = weight_vector > 0
    variance_tolerance, _ = _scalar_tolerances(matrix, weight_vector)

    if objective in {"standardized_grpo", "locally_linear_nft"}:
        degenerate = bool(scalar_variance <= variance_tolerance)
        if degenerate:
            contribution = np.full(int(active.sum()), np.nan, dtype=np.float64)
            score = float("-inf")
        else:
            contribution = weight_vector[active] * covariance_weight[active] / scalar_variance
            score = float(np.min(contribution))
    else:
        raise ValueError(
            "Covariance objective must be 'standardized_grpo' or 'locally_linear_nft'; "
            f"got {objective!r}."
        )

    return CovarianceGroupScore(
        covariance=covariance,
        contribution_vector=contribution,
        score=score,
        scalar_variance=scalar_variance,
        mean_scalar_reward=float(np.mean(scalar_rewards)),
        degenerate=degenerate,
    )


def compute_src_contributions(
    reward_matrix: np.ndarray,
    weights: np.ndarray,
) -> SRCContributionScore:
    """Compute sign-only Sample-wise Reward Concordance against one frozen pool.

    Args:
        reward_matrix: Candidate rewards shaped ``(n_candidates, n_rewards)``.
        weights: Nonnegative scalarization weights shaped ``(n_rewards,)``.

    Returns:
        Frozen pool reference values and one weakest-coordinate fitness per sample.

    Note:
        Scalar contrast contributes only through its sign. NumPy defines
        ``sign(0) == 0``, so zero contrast yields zero contribution without
        division by zero.
    """
    matrix = _validate_reward_matrix(reward_matrix)
    weight_vector = _validate_weights(weights, matrix.shape[1])
    pool_mean = matrix.mean(axis=0)
    centered = matrix - pool_mean
    scalar_rewards = matrix @ weight_vector
    scalar_advantages = centered @ weight_vector
    scalar_directions = np.sign(scalar_advantages)
    contributions = centered * scalar_directions[:, None] * weight_vector[None, :]
    fitness = np.min(contributions[:, weight_vector > 0], axis=1)
    variance_tolerance, _ = _scalar_tolerances(matrix, weight_vector)
    scalar_variance = float(np.mean(np.square(scalar_advantages)))

    return SRCContributionScore(
        pool_mean=pool_mean,
        scalar_rewards=scalar_rewards,
        centered_rewards=centered,
        scalar_advantages=scalar_advantages,
        contribution_matrix=contributions,
        fitness=fitness,
        degenerate_scalar_contrast=bool(scalar_variance <= variance_tolerance),
    )


def select_src_group(
    reward_matrix: np.ndarray,
    weights: np.ndarray,
    target_size: int,
    objective: str,
    candidate_ids: Sequence[int] | None = None,
) -> SRCSelectionResult:
    """Select one scalar elite plus the top SRC candidates.

    Compute every contribution once against the complete candidate pool. The
    elite is ranked by scalar reward, contribution fitness, and stable ID; all
    other survivors are ranked by contribution fitness, scalar reward, and
    stable ID. Values within a scale-aware floating-point tolerance advance to
    the next tie-break field.

    Args:
        reward_matrix: Candidate rewards shaped ``(N, d)``.
        weights: Nonnegative scalarization weights shaped ``(d,)``.
        target_size: Required survivor count.
        objective: Objective used only for true selected-group diagnostics.
        candidate_ids: Stable distinct integer identifiers for deterministic ties.

    Returns:
        Selected indices and frozen-pool plus true selected-group diagnostics.
    """
    matrix = _validate_reward_matrix(reward_matrix)
    weight_vector = _validate_weights(weights, matrix.shape[1])
    num_candidates = matrix.shape[0]
    if target_size < 2 or target_size > num_candidates:
        raise ValueError(f"target_size must be in [2, {num_candidates}], got {target_size}.")

    stable_ids = np.arange(num_candidates) if candidate_ids is None else np.asarray(candidate_ids)
    if stable_ids.shape != (num_candidates,) or len(np.unique(stable_ids)) != num_candidates:
        raise ValueError("candidate_ids must contain one distinct identifier per candidate.")

    sample_scores = compute_src_contributions(matrix, weight_vector)
    _, scalar_tolerance = _scalar_tolerances(matrix, weight_vector)
    fitness_scale = float(np.max(np.abs(sample_scores.fitness)))
    fitness_tolerance = max(
        np.finfo(np.float64).tiny,
        np.finfo(np.float64).eps * 64 * max(1, num_candidates, matrix.shape[1]) * fitness_scale,
    )

    def compare_indices(left: int, right: int, elite_order: bool) -> int:
        """Compare two candidates with deterministic tolerance-aware ties."""
        fields = (
            (sample_scores.scalar_rewards, scalar_tolerance),
            (sample_scores.fitness, fitness_tolerance),
        )
        if not elite_order:
            fields = fields[::-1]
        for values, tolerance in fields:
            difference = float(values[left] - values[right])
            if difference > tolerance:
                return -1
            if difference < -tolerance:
                return 1
        if stable_ids[left] < stable_ids[right]:
            return -1
        if stable_ids[left] > stable_ids[right]:
            return 1
        return 0

    candidates = sorted(range(num_candidates), key=lambda index: stable_ids[index])
    elite_order = sorted(
        candidates,
        key=cmp_to_key(lambda left, right: compare_indices(left, right, elite_order=True)),
    )
    elite = elite_order[0]
    remaining = [index for index in candidates if index != elite]
    remaining.sort(
        key=cmp_to_key(lambda left, right: compare_indices(left, right, elite_order=False))
    )
    selected = [elite] + remaining[: target_size - 1]
    selected_set = set(selected)
    rejected = [index for index in candidates if index not in selected_set]

    selected_contributions = sample_scores.contribution_matrix[selected]
    frozen_contribution = selected_contributions.mean(axis=0)
    frozen_score = float(np.min(frozen_contribution[weight_vector > 0]))
    lower_bound = float(np.mean(sample_scores.fitness[selected]))
    tolerance = fitness_tolerance * max(1, target_size)
    if frozen_score + tolerance < lower_bound:
        raise RuntimeError(
            "SRC lower-bound invariant failed: "
            f"frozen_score({frozen_score}) < lower_bound({lower_bound})."
        )

    return SRCSelectionResult(
        selected_indices=np.asarray(selected, dtype=np.int64),
        rejected_indices=np.asarray(rejected, dtype=np.int64),
        elite_index=int(elite),
        sample_scores=sample_scores,
        group_score=covariance_group_score(matrix[selected], weight_vector, objective),
        frozen_contribution_vector=frozen_contribution,
        frozen_score=frozen_score,
        lower_bound=lower_bound,
    )


def _validate_reward_matrix(reward_matrix: np.ndarray) -> np.ndarray:
    """Validate and return a floating-point reward matrix."""
    matrix = np.asarray(reward_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"reward_matrix must be a nonempty 2D array, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("reward_matrix must contain only finite values.")
    return matrix


def _validate_weights(weights: np.ndarray, num_rewards: int) -> np.ndarray:
    """Validate SRC scalarization weights."""
    weight_vector = np.asarray(weights, dtype=np.float64)
    if weight_vector.shape != (num_rewards,):
        raise ValueError(f"weights must have shape ({num_rewards},), got {weight_vector.shape}.")
    if not np.all(np.isfinite(weight_vector)) or np.any(weight_vector < 0):
        raise ValueError("weights must be finite and nonnegative.")
    if int(np.count_nonzero(weight_vector > 0)) < 2:
        raise ValueError("SRC selection requires at least two positive reward weights.")
    return weight_vector


def _scalar_tolerances(reward_matrix: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Derive variance and mean tolerances from centered scalar-reward scale."""
    scalar_rewards = reward_matrix @ weights
    centered = scalar_rewards - np.mean(scalar_rewards)
    scale = float(np.max(np.abs(centered)))
    factor = np.finfo(np.float64).eps * 64 * max(1, reward_matrix.shape[0])
    variance_tolerance = max(np.finfo(np.float64).tiny, factor * scale**2)
    mean_tolerance = max(np.finfo(np.float64).tiny, factor * scale)
    return variance_tolerance, mean_tolerance
