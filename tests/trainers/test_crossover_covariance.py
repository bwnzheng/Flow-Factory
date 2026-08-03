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

from types import SimpleNamespace

import numpy as np
import pytest

from flow_factory.hparams import CrossoverArguments
from flow_factory.samples import BaseSample
from flow_factory.trainers.crossover.covariance import (
    covariance_group_score,
    population_covariance,
    select_covariance_guided_group,
)
from flow_factory.trainers.crossover.genetic_algorithm import GeneticAlgorithm
from flow_factory.trainers.crossover.pareto import compute_pareto_mask


def test_concordant_pair_has_equal_grpo_contributions():
    score = covariance_group_score(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([1.0, 1.0]),
        "standardized_grpo",
    )

    np.testing.assert_allclose(score.contribution_vector, [0.5, 0.5])
    assert score.score == pytest.approx(0.5)
    assert not score.degenerate


def test_equal_scalar_reward_specialists_are_degenerate():
    score = covariance_group_score(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, 1.0]),
        "standardized_grpo",
    )

    assert score.degenerate
    assert score.score == float("-inf")
    assert np.isnan(score.contribution_vector).all()


def test_conflicting_pair_has_negative_weakest_contribution():
    score = covariance_group_score(
        np.array([[1.0, 0.0], [0.0, 0.8]]),
        np.array([1.0, 1.0]),
        "standardized_grpo",
    )

    assert score.score < 0
    assert np.any(score.contribution_vector < 0)


def test_locally_linear_nft_includes_group_dependent_normalizer():
    rewards = np.array([[0.0, 0.0], [1.0, 1.0]])
    weights = np.array([1.0, 1.0])

    score = covariance_group_score(rewards, weights, "locally_linear_nft")

    np.testing.assert_allclose(score.contribution_vector, [0.5, 0.5])
    assert score.score == pytest.approx(0.5)


def test_covariance_score_is_permutation_and_translation_invariant():
    rewards = np.array([[0.1, 0.8], [0.7, 0.4], [0.9, 0.9]])
    weights = np.array([0.4, 0.6])
    original = covariance_group_score(rewards, weights, "standardized_grpo")
    transformed = covariance_group_score(
        rewards[[2, 0, 1]] + np.array([10.0, -3.0]),
        weights,
        "standardized_grpo",
    )

    np.testing.assert_allclose(original.covariance, transformed.covariance)
    np.testing.assert_allclose(original.contribution_vector, transformed.contribution_vector)
    assert original.score == pytest.approx(transformed.score)


@pytest.mark.parametrize("scale", [7.0, 1e-12])
def test_common_positive_reward_scaling_preserves_grpo_score(scale):
    rewards = np.array([[0.1, 0.8], [0.7, 0.4], [0.9, 0.9]])
    weights = np.array([0.4, 0.6])
    original = covariance_group_score(rewards, weights, "standardized_grpo")
    scaled = covariance_group_score(rewards * scale, weights, "standardized_grpo")

    np.testing.assert_allclose(original.contribution_vector, scaled.contribution_vector)
    assert original.score == pytest.approx(scaled.score)


def test_affine_reward_rescaling_with_inverse_weights_preserves_score_and_selection():
    rewards = np.array([[0.1, 0.8], [0.7, 0.4], [0.9, 0.9], [0.3, 0.2]], dtype=np.float64)
    weights = np.array([0.4, 0.6])
    scales = np.array([10.0, 2.0])
    offsets = np.array([-3.0, 5.0])
    transformed_rewards = rewards * scales + offsets
    transformed_weights = weights / scales

    original_score = covariance_group_score(rewards, weights, "standardized_grpo")
    transformed_score = covariance_group_score(
        transformed_rewards, transformed_weights, "standardized_grpo"
    )
    original_selection = select_covariance_guided_group(
        rewards,
        weights,
        2,
        "standardized_grpo",
        fallback_scores=np.arange(len(rewards), dtype=np.float64),
    )
    transformed_selection = select_covariance_guided_group(
        transformed_rewards,
        transformed_weights,
        2,
        "standardized_grpo",
        fallback_scores=np.arange(len(rewards), dtype=np.float64),
    )

    np.testing.assert_allclose(
        original_score.contribution_vector, transformed_score.contribution_vector
    )
    assert original_score.score == pytest.approx(transformed_score.score)
    np.testing.assert_array_equal(
        original_selection.selected_indices, transformed_selection.selected_indices
    )


@pytest.mark.parametrize(
    ("rewards", "target_size", "expected_branch"),
    [
        (np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]), 2, "prune"),
        (np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), 2, "exact"),
        (np.array([[1.0, 1.0], [0.8, 0.8], [0.2, 0.2]]), 2, "fill"),
    ],
)
def test_covariance_selector_preserves_pareto_branch_invariants(
    rewards,
    target_size,
    expected_branch,
):
    result = select_covariance_guided_group(
        rewards,
        weights=np.array([1.0, 1.0]),
        target_size=target_size,
        objective="standardized_grpo",
        fallback_scores=np.arange(len(rewards), dtype=np.float64),
    )
    pareto_indices = set(np.flatnonzero(result.pareto_mask))
    selected = set(result.selected_indices)

    assert result.branch == expected_branch
    assert len(selected) == target_size
    if len(pareto_indices) >= target_size:
        assert selected <= pareto_indices
    else:
        assert pareto_indices <= selected


def test_covariance_selector_is_deterministic_under_row_permutation():
    rewards = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.8], [0.3, 0.3]], dtype=np.float64)
    candidate_ids = np.array([10, 11, 12, 13])
    original = select_covariance_guided_group(
        rewards,
        np.array([1.0, 1.0]),
        2,
        "standardized_grpo",
        fallback_scores=np.ones(4),
        candidate_ids=candidate_ids,
    )
    permutation = np.array([2, 0, 3, 1])
    permuted = select_covariance_guided_group(
        rewards[permutation],
        np.array([1.0, 1.0]),
        2,
        "standardized_grpo",
        fallback_scores=np.ones(4),
        candidate_ids=candidate_ids[permutation],
    )

    original_ids = candidate_ids[original.selected_indices]
    permuted_ids = candidate_ids[permutation][permuted.selected_indices]
    np.testing.assert_array_equal(original_ids, permuted_ids)


def test_all_degenerate_subsets_use_absolute_advantage_fallback():
    rewards = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    result = select_covariance_guided_group(
        rewards,
        np.array([1.0, 1.0]),
        2,
        "standardized_grpo",
        fallback_scores=np.array([3.0, 1.0, 2.0]),
    )

    assert result.degenerate_fallback
    assert result.selected_indices.tolist() == [0, 2]


def test_population_covariance_uses_population_denominator():
    rewards = np.array([[0.0, 0.0], [2.0, 2.0]])

    np.testing.assert_allclose(population_covariance(rewards), [[1.0, 1.0], [1.0, 1.0]])


def test_equal_reward_vectors_are_both_pareto_nondominated():
    rewards = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]])

    np.testing.assert_array_equal(compute_pareto_mask(rewards), [True, True, False])


def test_later_equal_first_coordinate_can_dominate_earlier_candidate():
    rewards = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])

    np.testing.assert_array_equal(compute_pareto_mask(rewards), [False, True, True])


def test_degenerate_fallback_tie_preserves_lower_stable_ids():
    rewards = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    candidate_ids = np.array([10, 11, 12])

    result = select_covariance_guided_group(
        rewards,
        np.array([1.0, 1.0]),
        2,
        "standardized_grpo",
        fallback_scores=np.ones(3),
        candidate_ids=candidate_ids,
    )

    np.testing.assert_array_equal(candidate_ids[result.selected_indices], [10, 11])


def test_crossover_arguments_parse_covariance_configuration():
    args = CrossoverArguments.from_dict({"survivor_score": "covariance"})

    assert args.survivor_score == "covariance"
    assert not hasattr(args, "covariance_reward_bounds")


def test_ga_covariance_inputs_use_raw_rewards_and_source_weights():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._reward_weights = {
        "quality": {"dataset": 2.0},
        "safety": {"dataset": 0.5},
    }
    rewards = {
        "quality": np.array([0.0, 10.0]),
        "safety": np.array([-1.0, 1.0]),
    }

    reward_matrix, weights = ga._prepare_covariance_inputs(
        rewards, ["quality", "safety"], "dataset"
    )

    np.testing.assert_allclose(reward_matrix, [[0.0, -1.0], [10.0, 1.0]])
    np.testing.assert_allclose(weights, [2.0, 0.5])


def test_ga_survivor_selection_exposes_covariance_diagnostics():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._group_size = 2
    ga._advantage_aggregation = "sum"
    ga._survivor_score = "covariance"
    ga._survivor_selection_aggregation = "weighted_sum"
    ga._covariance_objective = "standardized_grpo"
    ga._reward_weights = {
        "quality": {"default": 1.0},
        "safety": {"default": 1.0},
    }
    population = [BaseSample(), BaseSample()]
    children = [BaseSample(), BaseSample()]
    pop_rewards = {
        "quality": np.array([1.0, 0.0]),
        "safety": np.array([0.0, 1.0]),
    }
    child_rewards = {
        "quality": np.array([0.8, 0.3]),
        "safety": np.array([0.8, 0.3]),
    }

    survivors, rewards, stats = ga._select_survivors(
        population,
        children,
        pop_rewards,
        child_rewards,
        ["quality", "safety"],
        ["quality", "safety"],
        None,
    )

    diagnostics = stats["covariance_selection"]
    assert len(survivors) == 2
    assert len(rewards["quality"]) == 2
    assert diagnostics["branch"] == "prune"
    assert len(diagnostics["selected_ids"]) == 2
    assert np.asarray(diagnostics["covariance_after"]).shape == (2, 2)
    assert diagnostics["score"] is not None
    assert diagnostics["selection_aggregation"] == "weighted_sum"
    assert diagnostics["policy_advantage_aggregation"] == "sum"


def test_ga_covariance_degenerate_fallback_uses_weighted_sum_with_gdpo_policy():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._group_size = 2
    ga._advantage_aggregation = "gdpo"
    ga._survivor_score = "covariance"
    ga._survivor_selection_aggregation = "weighted_sum"
    ga._covariance_objective = "standardized_grpo"
    ga._reward_weights = {
        "quality": {"default": 1.0},
        "safety": {"default": 2.0},
    }
    population = [BaseSample(), BaseSample()]
    children = [BaseSample()]
    pop_rewards = {
        "quality": np.array([0.0, 1.0]),
        "safety": np.array([1.0, 0.5]),
    }
    child_rewards = {
        "quality": np.array([2.0]),
        "safety": np.array([0.0]),
    }

    _, _, stats = ga._select_survivors(
        population,
        children,
        pop_rewards,
        child_rewards,
        ["quality", "safety"],
        ["quality", "safety"],
        None,
    )

    diagnostics = stats["covariance_selection"]
    assert diagnostics["degenerate_fallback"]
    assert diagnostics["selected_ids"] == [0, 1]
    assert diagnostics["selection_aggregation"] == "weighted_sum"
    assert diagnostics["policy_advantage_aggregation"] == "gdpo"


def test_ga_covariance_inputs_reject_nonfinite_raw_rewards():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._reward_weights = {"quality": {"default": 1.0}, "safety": {"default": 1.0}}
    rewards = {"quality": np.array([1.0, np.nan]), "safety": np.ones(2)}

    with pytest.raises(ValueError, match="non-finite reward 'quality'"):
        ga._prepare_covariance_inputs(rewards, ["quality", "safety"], None)


def test_ga_selects_covariance_objective_from_trainer_type():
    reward_weights = {
        "quality": {"default": 1.0},
        "safety": {"default": 1.0},
    }
    common = {
        "crossover": SimpleNamespace(survivor_score="covariance"),
        "advantage_aggregation": "sum",
        "global_std": False,
        "num_inference_steps": 4,
        "group_size": 2,
    }

    grpo = GeneticAlgorithm(
        crossover_strategy=None,
        adapter=None,
        accelerator=None,
        autocast=None,
        training_args=SimpleNamespace(**common, trainer_type="crossover-grpo-guard"),
        reward_buffer=None,
        reward_weights=reward_weights,
    )
    nft = GeneticAlgorithm(
        crossover_strategy=None,
        adapter=None,
        accelerator=None,
        autocast=None,
        training_args=SimpleNamespace(**common, trainer_type="crossover-nft"),
        reward_buffer=None,
        reward_weights=reward_weights,
    )

    assert grpo._covariance_objective == "standardized_grpo"
    assert nft._covariance_objective == "locally_linear_nft"


@pytest.mark.parametrize("aggregation", ["sum", "gdpo"])
@pytest.mark.parametrize("stddev_reweighting", [False, True])
@pytest.mark.parametrize("global_std", [False, True])
def test_ga_decouples_covariance_selection_from_policy_advantage_options(
    aggregation, stddev_reweighting, global_std
):
    training_args = SimpleNamespace(
        crossover=SimpleNamespace(survivor_score="covariance"),
        advantage_aggregation=aggregation,
        stddev_reweighting=stddev_reweighting,
        global_std=global_std,
        trainer_type="crossover-nft",
        num_inference_steps=4,
        group_size=2,
    )

    ga = GeneticAlgorithm(
        crossover_strategy=None,
        adapter=None,
        accelerator=None,
        autocast=None,
        training_args=training_args,
        reward_buffer=None,
        reward_weights={
            "quality": {"default": 1.0},
            "safety": {"default": 1.0},
        },
    )

    assert ga._advantage_aggregation == aggregation
    assert ga._survivor_selection_aggregation == "weighted_sum"


def test_ga_rejects_missing_reward_weights_during_initialization():
    training_args = SimpleNamespace(
        crossover=SimpleNamespace(survivor_score="covariance"),
        advantage_aggregation="sum",
        stddev_reweighting=False,
        global_std=False,
        trainer_type="crossover-grpo-guard",
        num_inference_steps=4,
        group_size=2,
    )

    with pytest.raises(ValueError, match="requires configured reward weights"):
        GeneticAlgorithm(
            crossover_strategy=None,
            adapter=None,
            accelerator=None,
            autocast=None,
            training_args=training_args,
            reward_buffer=None,
        )
