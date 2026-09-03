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

import numpy as np
import pytest
import torch

from flow_factory.advantage import AdvantageProcessor, compute_src_reweight
from flow_factory.hparams import Arguments
from flow_factory.samples import BaseSample


def _compute(
    rewards: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    interpolation: float = 0.6,
    score_type: str = "saturated",
    epsilon: float = 1e-8,
):
    num_rewards, num_samples = rewards.shape
    if groups is None:
        groups = np.zeros(num_samples, dtype=np.int64)
    return compute_src_reweight(
        reward_matrix=rewards,
        weight_matrix=np.ones((num_rewards, num_samples)),
        applicable=np.ones((num_rewards, num_samples), dtype=bool),
        group_indices=groups,
        interpolation=interpolation,
        temperature=0.7,
        epsilon=epsilon,
        degeneracy_threshold=1e-12,
        score_type=score_type,
    )


def test_src_probabilities_and_weighted_advantages_satisfy_group_invariants():
    result = _compute(
        np.array(
            [
                [0.0, 0.2, 0.9, 1.0],
                [0.0, 0.8, 0.7, 1.0],
            ]
        )
    )

    assert result.probabilities.sum() == pytest.approx(1.0)
    assert result.loss_multipliers.mean() == pytest.approx(1.0)
    assert result.probabilities.min() >= (1.0 - 0.6) / 4.0 - 1e-12
    assert result.effective_sample_sizes[0] <= 4.0
    assert result.effective_sample_sizes[0] > 1.0
    assert result.uniform_advantages.mean() == pytest.approx(0.0, abs=1e-8)
    assert result.uniform_advantages @ result.uniform_advantages / 4.0 == pytest.approx(
        1.0, rel=1e-6
    )
    assert result.probabilities @ result.weighted_advantages == pytest.approx(0.0, abs=1e-8)
    assert result.probabilities @ np.square(result.weighted_advantages) == pytest.approx(
        1.0, rel=1e-6
    )
    np.testing.assert_allclose(
        result.effective_advantages,
        result.loss_multipliers * result.uniform_advantages,
    )
    assert result.lower_bound_reweighted[0] >= result.lower_bound_uniform[0]


def test_src_raw_score_matches_magnitude_weighted_definition():
    rewards = np.array(
        [
            [0.0, 1.0, 4.0],
            [0.0, 2.0, 2.0],
        ]
    )
    result = _compute(rewards, score_type="raw")
    centered_rewards = rewards - rewards.mean(axis=1, keepdims=True)
    centered_rewards /= np.sqrt(np.mean(np.square(centered_rewards), axis=1, keepdims=True))
    scalar_rewards = rewards.sum(axis=0)
    scalar_centered = scalar_rewards - scalar_rewards.mean()
    scalar_centered /= np.sqrt(np.mean(np.square(scalar_centered)))
    expected_contributions = centered_rewards * scalar_centered[None, :]
    expected_scores = expected_contributions.min(axis=0)

    np.testing.assert_allclose(result.contributions, expected_contributions)
    np.testing.assert_allclose(result.raw_scores, expected_scores)
    np.testing.assert_allclose(result.scores, expected_scores)
    np.testing.assert_allclose(
        result.normalized_scores,
        expected_scores,
    )


def test_src_saturated_score_matches_frozen_rms_definition():
    rewards = np.array(
        [
            [0.0, 1.0, 4.0],
            [0.0, 2.0, 2.0],
        ]
    )
    epsilon = 1e-8
    result = _compute(rewards, score_type="saturated", epsilon=epsilon)
    centered_rewards = rewards - rewards.mean(axis=1, keepdims=True)
    centered_rewards /= np.sqrt(np.mean(np.square(centered_rewards), axis=1, keepdims=True))
    scalar_rewards = rewards.sum(axis=0)
    scalar_centered = scalar_rewards - scalar_rewards.mean()
    scalar_centered /= np.sqrt(np.mean(np.square(scalar_centered)))
    saturation = scalar_centered / (np.abs(scalar_centered) + 1.0 + epsilon)
    expected_contributions = centered_rewards * saturation[None, :]
    expected_scores = expected_contributions.min(axis=0)

    np.testing.assert_allclose(result.contributions, expected_contributions)
    np.testing.assert_allclose(result.saturated_scores, expected_scores)
    np.testing.assert_allclose(result.scores, expected_scores)
    np.testing.assert_allclose(result.normalized_scores, expected_scores)


def test_src_saturated_score_suppresses_near_zero_scalar_contrast():
    delta = 1e-9
    rewards = np.array(
        [
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0 + delta, 1.0 - delta],
        ]
    )
    result = _compute(rewards, score_type="saturated")
    scalar_rewards = rewards.sum(axis=0)
    scalar_centered = scalar_rewards - scalar_rewards.mean()
    scalar_rms = np.sqrt(np.mean(np.square(scalar_centered)))

    assert abs(scalar_centered[1]) < 1e-8 * scalar_rms
    assert abs(result.saturated_scores[1]) < 1e-8


def test_src_saturated_score_approaches_sign_profile_for_large_scalar_contrast():
    rewards = np.zeros((2, 100), dtype=np.float64)
    rewards[:, -1] = 100.0
    result = _compute(rewards, score_type="saturated")
    centered_rewards = rewards - rewards.mean(axis=1, keepdims=True)
    centered_rewards /= np.sqrt(np.mean(np.square(centered_rewards), axis=1, keepdims=True))
    scalar_rewards = rewards.sum(axis=0)
    scalar_centered = scalar_rewards - scalar_rewards.mean()
    scalar_centered /= np.sqrt(np.mean(np.square(scalar_centered)))
    saturation = scalar_centered[-1] / (abs(scalar_centered[-1]) + 1.0 + 1e-8)
    sign_profile = centered_rewards[:, -1] * saturation

    np.testing.assert_allclose(
        result.contributions[:, -1],
        sign_profile,
        rtol=0.11,
    )


def test_src_score_type_switch_preserves_both_diagnostics():
    rewards = np.array([[0.0, 0.4, 1.0], [0.0, 0.8, 0.9]])
    raw = _compute(rewards, score_type="raw")
    saturated = _compute(rewards, score_type="saturated")

    np.testing.assert_allclose(raw.raw_scores, saturated.raw_scores)
    np.testing.assert_allclose(raw.saturated_scores, saturated.saturated_scores)
    np.testing.assert_allclose(raw.scores, raw.raw_scores)
    np.testing.assert_allclose(saturated.scores, saturated.saturated_scores)
    assert not np.allclose(raw.probabilities, saturated.probabilities)


def test_src_weighted_centering_diagnostics_remain_sign_based():
    rewards = np.array([[0.0, 1.0, 4.0], [0.0, 2.0, 2.0]])
    result = _compute(rewards, score_type="saturated")
    scalar_rewards = rewards.sum(axis=0)

    weighted_reward_means = rewards @ result.probabilities
    weighted_centered_rewards = rewards - weighted_reward_means[:, None]
    weighted_centered_rewards /= np.sqrt(
        result.probabilities @ np.square(weighted_centered_rewards).T
    )[:, None]
    weighted_scalar_centered = scalar_rewards - result.weighted_means[0]
    np.testing.assert_allclose(
        result.weighted_contributions,
        weighted_centered_rewards * np.sign(weighted_scalar_centered)[None, :],
    )


def test_src_interpolation_zero_recovers_uniform_group_relative_update():
    rewards = np.array([[0.0, 0.4, 1.0], [0.1, 0.8, 0.9]])
    result = _compute(rewards, interpolation=0.0)
    scalar_rewards = rewards.sum(axis=0)
    baseline = (scalar_rewards - scalar_rewards.mean()) / np.sqrt(scalar_rewards.var() + 1e-8)

    np.testing.assert_allclose(result.probabilities, np.full(3, 1.0 / 3.0))
    np.testing.assert_allclose(result.loss_multipliers, np.ones(3))
    np.testing.assert_allclose(result.effective_advantages, baseline)


def test_src_degenerate_group_uses_uniform_weights_and_zero_advantages():
    result = _compute(np.ones((2, 3)))

    assert result.degenerate_scalar_contrast.all()
    np.testing.assert_allclose(result.probabilities, np.full(3, 1.0 / 3.0))
    np.testing.assert_allclose(result.effective_advantages, np.zeros(3))


def test_src_never_mixes_prompt_groups():
    rewards = np.array(
        [
            [0.0, 1.0, 100.0, 101.0],
            [0.0, 1.0, 100.0, 101.0],
        ]
    )
    result = _compute(rewards, groups=np.array([0, 0, 1, 1]))

    np.testing.assert_allclose(result.probabilities[:2], result.probabilities[2:])
    assert result.probabilities[:2].sum() == pytest.approx(1.0)
    assert result.probabilities[2:].sum() == pytest.approx(1.0)


def test_src_requires_two_positive_active_rewards():
    rewards = np.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="at least two active rewards"):
        compute_src_reweight(
            reward_matrix=rewards,
            weight_matrix=np.array([[1.0, 1.0], [0.0, 0.0]]),
            applicable=np.ones_like(rewards, dtype=bool),
            group_indices=np.array([0, 0]),
            interpolation=0.5,
            temperature=1.0,
            epsilon=1e-8,
            degeneracy_threshold=1e-12,
        )


def test_src_rejects_invalid_score_type():
    with pytest.raises(ValueError, match="score_type"):
        compute_src_reweight(
            reward_matrix=np.array([[0.0, 1.0], [0.0, 1.0]]),
            weight_matrix=np.ones((2, 2)),
            applicable=np.ones((2, 2), dtype=bool),
            group_indices=np.array([0, 0]),
            interpolation=0.5,
            temperature=1.0,
            epsilon=1e-8,
            degeneracy_threshold=1e-12,
            score_type="unknown",
        )


def _src_config(trainer_type: str = "grpo", reward_count: int = 2):
    return {
        "data": {
            "datasets": [
                {
                    "name": "default",
                    "dataset_dir": "data",
                    "train": {},
                }
            ]
        },
        "train": {
            "trainer_type": trainer_type,
            "group_size": 2,
            "sample_weighting": "src",
            "advantage_aggregation": "sum",
        },
        "rewards": [
            {
                "name": f"reward_{index}",
                "reward_model": "CLIP",
                "weight": 1.0,
            }
            for index in range(reward_count)
        ],
    }


def test_src_config_accepts_shared_linear_advantage_trainer():
    args = Arguments.from_dict(_src_config())

    assert args.training_args.sample_weighting == "src"
    assert args.training_args.src_score_type == "saturated"


@pytest.mark.parametrize("trainer_type", ["ga_grpo_guard", "ga_nft"])
def test_src_config_accepts_ga_trainers(trainer_type):
    args = Arguments.from_dict(_src_config(trainer_type=trainer_type))

    assert args.training_args.sample_weighting == "src"
    assert args.training_args.trainer_type == trainer_type


def test_src_config_accepts_raw_score_type():
    config = _src_config()
    config["train"]["src_score_type"] = "raw"

    args = Arguments.from_dict(config)

    assert args.training_args.src_score_type == "raw"


def test_src_config_rejects_invalid_score_type():
    config = _src_config()
    config["train"]["src_score_type"] = "unknown"

    with pytest.raises(ValueError, match="src_score_type"):
        Arguments.from_dict(config)


def test_src_config_accepts_on_policy_nft_consumer():
    args = Arguments.from_dict(_src_config(trainer_type="nft"))

    assert args.training_args.sample_weighting == "src"
    assert args.training_args.off_policy is False


def test_src_config_accepts_off_policy_nft_consumer():
    config = _src_config(trainer_type="nft")
    config["train"]["off_policy"] = True

    args = Arguments.from_dict(config)

    assert args.training_args.sample_weighting == "src"
    assert args.training_args.off_policy is True


def test_src_config_still_rejects_off_policy_awm_consumer():
    config = _src_config(trainer_type="awm")
    config["train"]["off_policy"] = True

    with pytest.raises(ValueError, match="requires on-policy rollout groups"):
        Arguments.from_dict(config)


def test_src_config_requires_two_rewards_per_training_source():
    with pytest.raises(ValueError, match="at least two active rewards"):
        Arguments.from_dict(_src_config(reward_count=1))


class NoReduceAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    process_index = 0

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        raise AssertionError(f"SRC path unexpectedly called rank reduce({reduction})")


class Float32ReduceAccelerator(NoReduceAccelerator):
    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        assert tensor.dtype != torch.float64
        return tensor


@pytest.mark.parametrize("sampler_type", ["distributed_k_repeat", "group_contiguous"])
def test_advantage_processor_stores_src_weights_and_logs_without_rank_reduce(sampler_type):
    samples = [BaseSample(prompt="prompt", _unique_id=7) for _ in range(3)]
    processor = AdvantageProcessor(
        accelerator=NoReduceAccelerator(),
        reward_weights={"quality": {"default": 1.0}, "safety": {"default": 1.0}},
        group_size=3,
        global_std=True,
        sampler_type=sampler_type,
        verbose=False,
        sample_weighting="src",
        src_reweight_interpolation=0.6,
        src_reweight_temperature=0.7,
    )

    advantages = processor.compute_advantages(
        samples,
        rewards={
            "quality": torch.tensor([0.0, 0.4, 1.0]),
            "safety": torch.tensor([0.0, 0.8, 0.9]),
        },
        aggregation_func="sum",
    )
    expected_src = _compute(np.array([[0.0, 0.4, 1.0], [0.0, 0.8, 0.9]]))
    np.testing.assert_allclose(
        advantages.cpu().numpy(), expected_src.effective_advantages, rtol=1e-6, atol=1e-6
    )
    metrics = processor.pop_advantage_metrics()

    assert advantages.dtype == torch.float32
    assert all("sample_weight" in sample.extra_kwargs for sample in samples)
    assert metrics["train/sample_weight_mean"] == pytest.approx(1.0)
    assert metrics["train/src_probability_sum_error_max"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["train/src_weighted_centering_error_max"] == pytest.approx(0.0, abs=1e-6)
    assert "train/src_raw_score_mean" in metrics
    assert "train/src_raw_score_min" in metrics
    assert "train/src_raw_score_max" in metrics
    assert "train/src_saturated_score_mean" in metrics
    assert "train/src_saturated_score_min" in metrics
    assert "train/src_saturated_score_max" in metrics
    assert "train/src_uniform_advantage_mean" in metrics
    assert len(metrics["train/src_groups"]) == 1
    assert "raw_scores" in metrics["train/src_groups"][0]
    assert "saturated_scores" in metrics["train/src_groups"][0]
    assert "train/src_conflict_ratio_quality" in metrics


@pytest.mark.parametrize("consumer", ["linear_advantage", "nft"])
def test_advantage_processor_supports_src_on_ga_survivor_population(consumer):
    samples = [
        BaseSample(
            prompt="prompt",
            _unique_id=23,
            extra_kwargs={"is_crossover_child": index == 2},
        )
        for index in range(3)
    ]
    rewards = {
        "quality": torch.tensor([0.0, 0.4, 1.0]),
        "safety": torch.tensor([0.0, 0.8, 0.9]),
    }
    processor = AdvantageProcessor(
        accelerator=Float32ReduceAccelerator() if consumer == "nft" else NoReduceAccelerator(),
        reward_weights={"quality": {"default": 1.0}, "safety": {"default": 1.0}},
        group_size=3,
        global_std=True,
        sampler_type="group_contiguous",
        verbose=False,
        sample_weighting="src",
        sample_weighting_consumer=consumer,
        src_reweight_interpolation=0.6,
        src_reweight_temperature=0.7,
    )
    processor._ga_enabled = True
    processor._child_in_norm = True

    advantages = processor.compute_advantages(samples, rewards, aggregation_func="sum")
    expected_src = _compute(np.array([[0.0, 0.4, 1.0], [0.0, 0.8, 0.9]]))

    if consumer == "linear_advantage":
        np.testing.assert_allclose(
            advantages.cpu().numpy(), expected_src.effective_advantages, rtol=1e-6, atol=1e-6
        )
    else:
        np.testing.assert_allclose(
            advantages.cpu().numpy(), expected_src.uniform_advantages, rtol=1e-6, atol=1e-6
        )
    np.testing.assert_allclose(
        [sample.extra_kwargs["sample_weight"].item() for sample in samples],
        expected_src.loss_multipliers,
        rtol=1e-6,
    )


def test_globally_gathered_baseline_metrics_are_not_rank_reduced_again():
    samples = [BaseSample(prompt="prompt", _unique_id=11) for _ in range(3)]
    processor = AdvantageProcessor(
        accelerator=NoReduceAccelerator(),
        reward_weights={"quality": {"default": 1.0}},
        group_size=3,
        global_std=False,
        sampler_type="group_contiguous",
        verbose=False,
    )

    processor.compute_advantages(
        samples,
        rewards={"quality": torch.tensor([0.0, 0.5, 1.0])},
        aggregation_func="sum",
    )
    metrics = processor.pop_advantage_metrics()

    assert metrics["train/reward_quality_mean"] == pytest.approx(0.5)
    assert metrics["train/reward_zero_std_ratio"] == pytest.approx(0.0)


def test_nft_consumer_keeps_outer_weight_separate_and_uses_baseline_global_std():
    samples = [BaseSample(prompt="prompt", _unique_id=13) for _ in range(3)]
    rewards = np.array(
        [
            [0.0, 0.4, 1.0],
            [0.0, 0.8, 0.9],
        ]
    )
    processor = AdvantageProcessor(
        accelerator=Float32ReduceAccelerator(),
        reward_weights={"quality": {"default": 1.0}, "safety": {"default": 1.0}},
        group_size=3,
        global_std=True,
        sampler_type="group_contiguous",
        verbose=False,
        sample_weighting="src",
        sample_weighting_consumer="nft",
        src_reweight_interpolation=0.6,
        src_reweight_temperature=0.7,
    )

    advantages = processor.compute_advantages(
        samples,
        rewards={
            "quality": torch.tensor(rewards[0]),
            "safety": torch.tensor(rewards[1]),
        },
        aggregation_func="sum",
    )
    src_result = _compute(rewards)
    scalar_rewards = rewards.sum(axis=0)
    expected = (scalar_rewards - scalar_rewards.mean()) / scalar_rewards.std()
    metrics = processor.pop_advantage_metrics()

    np.testing.assert_allclose(advantages.cpu().numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(
        [sample.extra_kwargs["sample_weight"].item() for sample in samples],
        src_result.loss_multipliers,
        rtol=1e-6,
    )
    assert "train/src_weighted_contribution_quality_mean" in metrics
    assert "weighted_contributions" in metrics["train/src_groups"][0]


def test_nft_src_interpolation_zero_recovers_baseline_advantage_and_unit_mass():
    samples = [BaseSample(prompt="prompt", _unique_id=17) for _ in range(3)]
    rewards = {
        "quality": torch.tensor([0.0, 0.4, 1.0]),
        "safety": torch.tensor([0.0, 0.8, 0.9]),
    }
    common_kwargs = {
        "accelerator": Float32ReduceAccelerator(),
        "reward_weights": {"quality": {"default": 1.0}, "safety": {"default": 1.0}},
        "group_size": 3,
        "global_std": True,
        "sampler_type": "group_contiguous",
        "verbose": False,
    }
    baseline_processor = AdvantageProcessor(**common_kwargs)
    src_processor = AdvantageProcessor(
        **common_kwargs,
        sample_weighting="src",
        sample_weighting_consumer="nft",
        src_reweight_interpolation=0.0,
    )

    baseline = baseline_processor.compute_advantages(
        samples,
        rewards,
        aggregation_func="sum",
        store_to_samples=False,
    )
    src = src_processor.compute_advantages(
        samples,
        rewards,
        aggregation_func="sum",
        store_to_samples=True,
    )

    torch.testing.assert_close(src, baseline, check_dtype=False)
    assert src.dtype == torch.float32
    torch.testing.assert_close(
        torch.stack([sample.extra_kwargs["sample_weight"] for sample in samples]),
        torch.ones(3),
    )


def test_nft_src_prompt_normalizer_preserves_baseline_contract_at_zero_interpolation():
    samples = [BaseSample(prompt="prompt", _unique_id=19) for _ in range(3)]
    rewards = {
        "quality": torch.tensor([0.0, 0.4, 1.0]),
        "safety": torch.tensor([0.0, 0.8, 0.9]),
    }
    common_kwargs = {
        "accelerator": NoReduceAccelerator(),
        "reward_weights": {"quality": {"default": 1.0}, "safety": {"default": 1.0}},
        "group_size": 3,
        "global_std": False,
        "sampler_type": "group_contiguous",
        "verbose": False,
    }
    baseline_processor = AdvantageProcessor(**common_kwargs)
    src_processor = AdvantageProcessor(
        **common_kwargs,
        sample_weighting="src",
        sample_weighting_consumer="nft",
        src_reweight_interpolation=0.0,
    )

    baseline = baseline_processor.compute_advantages(
        samples,
        rewards,
        aggregation_func="sum",
        store_to_samples=False,
    )
    src = src_processor.compute_advantages(
        samples,
        rewards,
        aggregation_func="sum",
        store_to_samples=False,
    )

    torch.testing.assert_close(src, baseline, check_dtype=False)
