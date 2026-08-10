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

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow_factory.hparams import GAArguments
from flow_factory.samples import BaseSample, T2ISample
from flow_factory.trainers.crossover.genetic_algorithm import (
    GeneticAlgorithm,
    _prepare_ga_child_media,
)
from flow_factory.trainers.crossover.pareto import compute_pareto_mask
from flow_factory.trainers.crossover.src import (
    compute_src_contributions,
    covariance_group_score,
    population_covariance,
    select_src_group,
)


def test_concordant_pair_has_equal_grpo_contributions():
    score = covariance_group_score(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([1.0, 1.0]),
        "standardized_grpo",
    )

    np.testing.assert_allclose(score.contribution_vector, [0.5, 0.5])
    assert score.score == pytest.approx(0.5)
    assert score.degenerate is False


def test_equal_scalar_reward_specialists_are_degenerate():
    score = covariance_group_score(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, 1.0]),
        "standardized_grpo",
    )

    assert score.degenerate is True
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


def test_affine_reward_rescaling_with_inverse_weights_preserves_score():
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
    np.testing.assert_allclose(
        original_score.contribution_vector, transformed_score.contribution_vector
    )
    assert original_score.score == pytest.approx(transformed_score.score)


def test_population_covariance_uses_population_denominator():
    rewards = np.array([[0.0, 0.0], [2.0, 2.0]])

    np.testing.assert_allclose(population_covariance(rewards), [[1.0, 1.0], [1.0, 1.0]])


def test_equal_reward_vectors_are_both_pareto_nondominated():
    rewards = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]])

    np.testing.assert_array_equal(compute_pareto_mask(rewards), [True, True, False])


def test_later_equal_first_coordinate_can_dominate_earlier_candidate():
    rewards = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])

    np.testing.assert_array_equal(compute_pareto_mask(rewards), [False, True, True])


def test_ga_src_inputs_use_raw_rewards_and_source_weights():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._reward_weights = {
        "quality": {"dataset": 2.0},
        "safety": {"dataset": 0.5},
    }
    rewards = {
        "quality": np.array([0.0, 10.0]),
        "safety": np.array([-1.0, 1.0]),
    }

    reward_matrix, weights = ga._prepare_src_inputs(rewards, ["quality", "safety"], "dataset")

    np.testing.assert_allclose(reward_matrix, [[0.0, -1.0], [10.0, 1.0]])
    np.testing.assert_allclose(weights, [2.0, 0.5])


def test_ga_src_inputs_reject_nonfinite_raw_rewards():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._reward_weights = {"quality": {"default": 1.0}, "safety": {"default": 1.0}}
    rewards = {"quality": np.array([1.0, np.nan]), "safety": np.ones(2)}

    with pytest.raises(ValueError, match="non-finite reward 'quality'"):
        ga._prepare_src_inputs(rewards, ["quality", "safety"], None)


def test_ga_selects_src_diagnostic_objective_from_trainer_type():
    reward_weights = {
        "quality": {"default": 1.0},
        "safety": {"default": 1.0},
    }
    common = {
        "ga": SimpleNamespace(survivor_score="src"),
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
        training_args=SimpleNamespace(**common, trainer_type="ga_grpo_guard"),
        reward_buffer=None,
        reward_weights=reward_weights,
    )
    nft = GeneticAlgorithm(
        crossover_strategy=None,
        adapter=None,
        accelerator=None,
        autocast=None,
        training_args=SimpleNamespace(**common, trainer_type="ga_nft"),
        reward_buffer=None,
        reward_weights=reward_weights,
    )

    assert grpo._src_diagnostic_objective == "standardized_grpo"
    assert nft._src_diagnostic_objective == "locally_linear_nft"


def test_ga_rejects_missing_reward_weights_during_initialization():
    training_args = SimpleNamespace(
        ga=SimpleNamespace(survivor_score="src"),
        advantage_aggregation="sum",
        global_std=False,
        trainer_type="ga_grpo_guard",
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


def test_src_concordant_pair_has_equal_positive_fitness():
    scores = compute_src_contributions(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([1.0, 1.0]),
    )

    np.testing.assert_allclose(scores.contribution_matrix, [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(scores.fitness, [0.5, 0.5])
    assert scores.degenerate_scalar_contrast is False


def test_src_uses_only_scalar_advantage_sign_in_contributions():
    scores = compute_src_contributions(
        np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 2.0]]),
        np.array([2.0, 0.5]),
    )

    expected = (
        scores.centered_rewards
        * np.sign(scores.scalar_advantages)[:, None]
        * np.array([2.0, 0.5])[None, :]
    )
    np.testing.assert_allclose(scores.contribution_matrix, expected)
    nonzero = np.abs(scores.scalar_advantages) > 0
    magnitude_weighted = (
        scores.centered_rewards[nonzero]
        * scores.scalar_advantages[nonzero, None]
        * np.array([2.0, 0.5])[None, :]
    )
    assert not np.allclose(scores.contribution_matrix[nonzero], magnitude_weighted)


def test_src_equal_scalar_specialists_are_degenerate():
    scores = compute_src_contributions(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, 1.0]),
    )

    np.testing.assert_allclose(scores.scalar_advantages, 0.0)
    np.testing.assert_allclose(scores.contribution_matrix, 0.0)
    np.testing.assert_allclose(scores.fitness, 0.0)
    assert scores.degenerate_scalar_contrast is True


def test_src_candidate_at_pool_mean_has_zero_contribution():
    scores = compute_src_contributions(
        np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        np.array([1.0, 1.0]),
    )

    np.testing.assert_allclose(scores.centered_rewards[1], 0.0)
    assert scores.scalar_advantages[1] == pytest.approx(0.0)
    np.testing.assert_allclose(scores.contribution_matrix[1], 0.0)
    assert scores.fitness[1] == pytest.approx(0.0)


def test_src_selected_frozen_score_respects_lower_bound():
    rng = np.random.default_rng(7)
    rewards = rng.normal(size=(20, 4))
    result = select_src_group(
        rewards,
        np.array([0.1, 0.2, 0.3, 0.4]),
        target_size=7,
        objective="standardized_grpo",
    )

    assert result.frozen_score >= result.lower_bound - 1e-12


def test_src_top_k_maximizes_conditional_lower_bound():
    rewards = np.array([[10.0, 0.0], [4.0, 4.0], [0.0, 0.0], [3.0, 3.0], [2.0, 5.0]])
    target_size = 3
    result = select_src_group(
        rewards,
        np.ones(2),
        target_size=target_size,
        objective="standardized_grpo",
    )
    elite = result.elite_index
    fitness = result.sample_scores.fitness
    selected_lower_bound = float(np.mean(fitness[result.selected_indices]))
    alternatives = [index for index in range(len(rewards)) if index != elite]
    oracle = max(
        float(np.mean(fitness[[elite, *subset]]))
        for subset in combinations(alternatives, target_size - 1)
    )

    assert selected_lower_bound == pytest.approx(oracle)


def test_src_selector_is_translation_scaling_and_permutation_invariant():
    rewards = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.8], [0.3, 0.3]])
    weights = np.array([0.4, 0.6])
    candidate_ids = np.array([10, 11, 12, 13])
    original = select_src_group(
        rewards,
        weights,
        2,
        "standardized_grpo",
        candidate_ids,
    )
    permutation = np.array([2, 0, 3, 1])
    transformed = select_src_group(
        (rewards[permutation] + np.array([8.0, -3.0])) * 7.0,
        weights,
        2,
        "standardized_grpo",
        candidate_ids[permutation],
    )

    np.testing.assert_array_equal(
        candidate_ids[original.selected_indices],
        candidate_ids[permutation][transformed.selected_indices],
    )


def test_src_selector_always_retains_scalar_elite():
    rewards = np.array([[10.0, 0.0], [4.0, 4.0], [0.0, 0.0], [3.0, 3.0]])
    result = select_src_group(
        rewards,
        np.ones(2),
        2,
        "standardized_grpo",
    )
    contribution_only = np.argsort(result.sample_scores.fitness)[::-1][:2]

    assert result.elite_index == 0
    assert result.elite_index in result.selected_indices
    assert result.elite_index not in contribution_only


def test_ga_src_selection_exposes_frozen_and_true_diagnostics():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._group_size = 2
    ga._advantage_aggregation = "sum"
    ga._survivor_score = "src"
    ga._survivor_selection_aggregation = "weighted_sum"
    ga._src_diagnostic_objective = "standardized_grpo"
    ga._reward_weights = {
        "quality": {"default": 1.0},
        "safety": {"default": 1.0},
    }
    population = [BaseSample(), BaseSample()]
    children = [BaseSample(), BaseSample()]
    pop_rewards = {
        "quality": np.array([10.0, 4.0]),
        "safety": np.array([0.0, 4.0]),
    }
    child_rewards = {
        "quality": np.array([0.0, 3.0]),
        "safety": np.array([0.0, 3.0]),
    }

    survivors, _, stats = ga._select_survivors(
        population,
        children,
        pop_rewards,
        child_rewards,
        ["quality", "safety"],
        ["quality", "safety"],
        None,
    )

    diagnostics = stats["src_selection"]
    assert len(survivors) == 2
    assert diagnostics["elite_id"] == 0
    assert diagnostics["elite_id"] in diagnostics["selected_ids"]
    assert len(diagnostics["sample_fitness"]) == 4
    assert diagnostics["frozen_score"] >= diagnostics["lower_bound"] - 1e-12
    assert np.asarray(diagnostics["covariance_after"]).shape == (2, 2)
    event = stats["selection_event"]
    assert event["survivor_score"] == "src"
    assert event["reward_keys"] == ["quality", "safety"]
    assert event["valid_reward_keys"] == ["quality", "safety"]
    assert event["candidate_origin"] == ["population", "population", "child", "child"]
    np.testing.assert_array_equal(event["candidate_ids"], np.arange(4))
    np.testing.assert_array_equal(event["selected_ids"], diagnostics["selected_ids"])
    np.testing.assert_allclose(event["selection_scores"], diagnostics["sample_fitness"])
    np.testing.assert_allclose(event["candidate_rewards"]["quality"], [10.0, 4.0, 0.0, 3.0])


def test_ga_child_media_keeps_selected_and_rejected_candidates_with_full_event():
    children = [
        T2ISample(image=torch.zeros(3, 4, 4), prompt="prompt", _unique_id=9),
        T2ISample(image=torch.ones(3, 4, 4), prompt="prompt", _unique_id=9),
    ]
    event = {
        "selected_ids": np.array([0, 2], dtype=np.int64),
        "candidate_rewards": {"quality": np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float32)},
        "selection_advantages": np.array([-1.0, -0.5, 1.0, 0.5], dtype=np.float32),
        "selection_scores": np.array([-1.0, -0.5, 1.0, 0.5], dtype=np.float32),
        "pareto_mask": np.array([False, False, True, True]),
        "offspring": {
            "candidate_ids": np.array([2, 3], dtype=np.int64),
            "primary_parent_ids": np.array([0, 1], dtype=np.int64),
            "secondary_parent_ids": np.array([1, 0], dtype=np.int64),
        },
        "rejected_ids": np.array([1, 3], dtype=np.int64),
    }

    media = _prepare_ga_child_media(children, event, generation=1)

    assert len(media) == 2
    selected = media[0].extra_kwargs["_media_metadata"]["context"]["ga"]
    rejected = media[1].extra_kwargs["_media_metadata"]["context"]["ga"]
    assert selected["candidate_id"] == 2
    assert selected["selected"] is True
    assert selected["selected_order"] == 1
    assert selected["rewards"]["quality"] == pytest.approx(0.9)
    assert rejected["candidate_id"] == 3
    assert rejected["selected"] is False
    assert rejected["selected_order"] is None
    np.testing.assert_array_equal(rejected["group_selection"]["rejected_ids"], [1, 3])

    backend_media = _prepare_ga_child_media(
        children,
        event,
        generation=1,
        include_metadata=False,
    )
    assert "_media_metadata" not in backend_media[0].extra_kwargs
    assert [sample.extra_kwargs["ga_candidate_id"] for sample in backend_media] == [2, 3]


def test_ga_arguments_parse_src_configuration():
    args = GAArguments.from_dict({"survivor_score": "src"})

    assert args.survivor_score == "src"
