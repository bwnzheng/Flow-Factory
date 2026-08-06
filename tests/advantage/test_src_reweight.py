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
        epsilon=1e-8,
        degeneracy_threshold=1e-12,
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
    assert result.probabilities @ result.weighted_advantages == pytest.approx(0.0, abs=1e-8)
    assert result.probabilities @ np.square(result.weighted_advantages) == pytest.approx(
        1.0, rel=1e-6
    )
    np.testing.assert_allclose(
        result.effective_advantages,
        result.loss_multipliers * result.weighted_advantages,
    )
    assert result.lower_bound_reweighted[0] >= result.lower_bound_uniform[0]


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
            "stddev_reweighting": False,
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


def test_src_config_rejects_nonlinear_advantage_consumer():
    with pytest.raises(ValueError, match="only exact for trainers"):
        Arguments.from_dict(_src_config(trainer_type="nft"))


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
    metrics = processor.pop_advantage_metrics()

    assert advantages.dtype == torch.float32
    assert all("sample_weight" in sample.extra_kwargs for sample in samples)
    assert metrics["train/sample_weight_mean"] == pytest.approx(1.0)
    assert metrics["train/src_probability_sum_error_max"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["train/src_weighted_centering_error_max"] == pytest.approx(0.0, abs=1e-6)
    assert len(metrics["train/src_groups"]) == 1
    assert "train/src_conflict_ratio_quality" in metrics


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
