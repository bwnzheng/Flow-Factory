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
        degenerate = scalar_variance <= variance_tolerance
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
