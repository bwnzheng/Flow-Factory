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

"""Covariance-guided environmental selection for crossover populations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Sequence

import numpy as np

from .pareto import compute_pareto_mask


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
class CovarianceSelectionResult:
    """Store selected indices and final environmental-selection diagnostics."""

    selected_indices: np.ndarray
    rejected_indices: np.ndarray
    pareto_mask: np.ndarray
    branch: str
    group_score: CovarianceGroupScore
    degenerate_fallback: bool


@dataclass(frozen=True)
class SampleWiseContributionScore:
    """Store frozen-pool sample-wise covariance-contribution diagnostics."""

    pool_mean: np.ndarray
    scalar_rewards: np.ndarray
    centered_rewards: np.ndarray
    scalar_advantages: np.ndarray
    contribution_matrix: np.ndarray
    fitness: np.ndarray
    degenerate_scalar_contrast: bool


@dataclass(frozen=True)
class SampleWiseCovarianceSelectionResult:
    """Store scalar-elitist sample-wise covariance selection diagnostics."""

    selected_indices: np.ndarray
    rejected_indices: np.ndarray
    elite_index: int
    sample_scores: SampleWiseContributionScore
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


def sample_wise_covariance_contributions(
    reward_matrix: np.ndarray,
    weights: np.ndarray,
) -> SampleWiseContributionScore:
    """Compute per-sample covariance contributions against one frozen pool.

    Args:
        reward_matrix: Candidate rewards shaped ``(n_candidates, n_rewards)``.
        weights: Nonnegative scalarization weights shaped ``(n_rewards,)``.

    Returns:
        Frozen pool reference values and one weakest-coordinate fitness per sample.
    """
    matrix = _validate_reward_matrix(reward_matrix)
    weight_vector = _validate_weights(weights, matrix.shape[1])
    pool_mean = matrix.mean(axis=0)
    centered = matrix - pool_mean
    scalar_rewards = matrix @ weight_vector
    scalar_advantages = centered @ weight_vector
    contributions = centered * scalar_advantages[:, None] * weight_vector[None, :]
    fitness = np.min(contributions[:, weight_vector > 0], axis=1)
    variance_tolerance, _ = _scalar_tolerances(matrix, weight_vector)
    scalar_variance = float(np.mean(np.square(scalar_advantages)))

    return SampleWiseContributionScore(
        pool_mean=pool_mean,
        scalar_rewards=scalar_rewards,
        centered_rewards=centered,
        scalar_advantages=scalar_advantages,
        contribution_matrix=contributions,
        fitness=fitness,
        degenerate_scalar_contrast=bool(scalar_variance <= variance_tolerance),
    )


def select_sample_wise_covariance_group(
    reward_matrix: np.ndarray,
    weights: np.ndarray,
    target_size: int,
    objective: str,
    candidate_ids: Sequence[int] | None = None,
) -> SampleWiseCovarianceSelectionResult:
    """Select one scalar elite plus top sample-wise covariance contributors.

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

    sample_scores = sample_wise_covariance_contributions(matrix, weight_vector)
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
            "Sample-wise covariance lower-bound invariant failed: "
            f"frozen_score({frozen_score}) < lower_bound({lower_bound})."
        )

    return SampleWiseCovarianceSelectionResult(
        selected_indices=np.asarray(selected, dtype=np.int64),
        rejected_indices=np.asarray(rejected, dtype=np.int64),
        elite_index=int(elite),
        sample_scores=sample_scores,
        group_score=covariance_group_score(matrix[selected], weight_vector, objective),
        frozen_contribution_vector=frozen_contribution,
        frozen_score=frozen_score,
        lower_bound=lower_bound,
    )


def select_covariance_guided_group(
    reward_matrix: np.ndarray,
    weights: np.ndarray,
    target_size: int,
    objective: str,
    fallback_scores: np.ndarray,
    candidate_ids: Sequence[int] | None = None,
) -> CovarianceSelectionResult:
    """Select a fixed-size group with Pareto-first covariance-guided greediness.

    Args:
        reward_matrix: Raw candidate rewards shaped ``(N, d)``.
        weights: Nonnegative scalarization weights shaped ``(d,)``.
        target_size: Required survivor count.
        objective: ``"standardized_grpo"`` or ``"locally_linear_nft"``.
        fallback_scores: Absolute scalar advantages used only when every compared
            subset has a degenerate scalar-reward normalizer.
        candidate_ids: Stable integer identifiers used for deterministic ties.

    Returns:
        Selected indices plus Pareto and covariance diagnostics.
    """
    matrix = _validate_reward_matrix(reward_matrix)
    weight_vector = _validate_weights(weights, matrix.shape[1])
    num_candidates = matrix.shape[0]
    if target_size < 2 or target_size > num_candidates:
        raise ValueError(f"target_size must be in [2, {num_candidates}], got {target_size}.")

    fallback = np.asarray(fallback_scores, dtype=np.float64)
    if fallback.shape != (num_candidates,):
        raise ValueError(
            f"fallback_scores must have shape ({num_candidates},), got {fallback.shape}."
        )
    stable_ids = np.arange(num_candidates) if candidate_ids is None else np.asarray(candidate_ids)
    if stable_ids.shape != (num_candidates,) or len(np.unique(stable_ids)) != num_candidates:
        raise ValueError("candidate_ids must contain one distinct identifier per candidate.")

    pareto_mask = compute_pareto_mask(matrix)
    pareto_indices = np.flatnonzero(pareto_mask).tolist()
    degenerate_fallback = False

    if len(pareto_indices) > target_size:
        branch = "prune"
        selected = list(pareto_indices)
        while len(selected) > target_size:
            alternatives = [
                (candidate, [index for index in selected if index != candidate])
                for candidate in selected
            ]
            remove, used_fallback = _choose_greedy_action(
                alternatives,
                matrix,
                weight_vector,
                objective,
                fallback,
                stable_ids,
                action="remove",
            )
            degenerate_fallback |= used_fallback
            selected.remove(remove)
    elif len(pareto_indices) < target_size:
        branch = "fill"
        selected = list(pareto_indices)
        remaining = [index for index in range(num_candidates) if index not in selected]
        while len(selected) < target_size:
            alternatives = [(candidate, selected + [candidate]) for candidate in remaining]
            add, used_fallback = _choose_greedy_action(
                alternatives,
                matrix,
                weight_vector,
                objective,
                fallback,
                stable_ids,
                action="add",
            )
            degenerate_fallback |= used_fallback
            selected.append(add)
            remaining.remove(add)
    else:
        branch = "exact"
        selected = list(pareto_indices)

    selected = sorted(selected, key=lambda index: stable_ids[index])
    rejected = sorted(
        set(range(num_candidates)) - set(selected), key=lambda index: stable_ids[index]
    )
    final_score = covariance_group_score(matrix[selected], weight_vector, objective)
    return CovarianceSelectionResult(
        selected_indices=np.asarray(selected, dtype=np.int64),
        rejected_indices=np.asarray(rejected, dtype=np.int64),
        pareto_mask=pareto_mask,
        branch=branch,
        group_score=final_score,
        degenerate_fallback=degenerate_fallback,
    )


def _choose_greedy_action(
    alternatives: list[tuple[int, list[int]]],
    reward_matrix: np.ndarray,
    weights: np.ndarray,
    objective: str,
    fallback_scores: np.ndarray,
    candidate_ids: np.ndarray,
    action: str,
) -> tuple[int, bool]:
    """Choose one greedy add/remove action with deterministic tie-breaking."""
    scored = [
        (candidate, covariance_group_score(reward_matrix[subset], weights, objective))
        for candidate, subset in alternatives
    ]
    if all(score.degenerate for _, score in scored):
        if action == "remove":
            chosen = min(
                (candidate for candidate, _ in scored),
                key=lambda index: (fallback_scores[index], -int(candidate_ids[index])),
            )
        else:
            chosen = max(
                (candidate for candidate, _ in scored),
                key=lambda index: (fallback_scores[index], -int(candidate_ids[index])),
            )
        return chosen, True

    score_tolerance = (
        np.finfo(np.float64).eps * 64 * max(1, reward_matrix.shape[0], reward_matrix.shape[1])
    )
    variance_tolerance, mean_tolerance = _scalar_tolerances(reward_matrix, weights)

    def is_better(
        candidate: int,
        score: CovarianceGroupScore,
        best_candidate: int,
        best_score: CovarianceGroupScore,
    ) -> bool:
        if score.degenerate != best_score.degenerate:
            return not score.degenerate
        if score.score > best_score.score + score_tolerance:
            return True
        if abs(score.score - best_score.score) > score_tolerance:
            return False
        if score.scalar_variance > best_score.scalar_variance + variance_tolerance:
            return True
        if abs(score.scalar_variance - best_score.scalar_variance) > variance_tolerance:
            return False
        if score.mean_scalar_reward > best_score.mean_scalar_reward + mean_tolerance:
            return True
        if abs(score.mean_scalar_reward - best_score.mean_scalar_reward) > mean_tolerance:
            return False
        return candidate_ids[candidate] < candidate_ids[best_candidate]

    best_candidate, best_score = scored[0]
    for candidate, score in scored[1:]:
        if is_better(candidate, score, best_candidate, best_score):
            best_candidate, best_score = candidate, score
    return best_candidate, False


def _validate_reward_matrix(reward_matrix: np.ndarray) -> np.ndarray:
    """Validate and return a floating-point reward matrix."""
    matrix = np.asarray(reward_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"reward_matrix must be a nonempty 2D array, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("reward_matrix must contain only finite values.")
    return matrix


def _validate_weights(weights: np.ndarray, num_rewards: int) -> np.ndarray:
    """Validate covariance scalarization weights."""
    weight_vector = np.asarray(weights, dtype=np.float64)
    if weight_vector.shape != (num_rewards,):
        raise ValueError(f"weights must have shape ({num_rewards},), got {weight_vector.shape}.")
    if not np.all(np.isfinite(weight_vector)) or np.any(weight_vector < 0):
        raise ValueError("weights must be finite and nonnegative.")
    if int(np.count_nonzero(weight_vector > 0)) < 2:
        raise ValueError("Covariance selection requires at least two positive reward weights.")
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
