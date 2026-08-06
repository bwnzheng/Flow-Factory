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
import torch

from flow_factory.hparams import (
    CrossoverArguments,
    CrossoverGRPOGuardTrainingArguments,
    CrossoverNFTTrainingArguments,
)
from flow_factory.hparams.args import Arguments
from flow_factory.samples import BaseSample
from flow_factory.trainers.crossover.genetic_algorithm import (
    GeneticAlgorithm,
    _format_covariance_selection_log,
)
from flow_factory.trainers.crossover_grpo_guard import CrossoverGRPOGuardTrainer
from flow_factory.trainers.crossover_nft import CrossoverNFTTrainer
from flow_factory.trainers.grpo import GRPOGuardTrainer
from flow_factory.trainers.nft import DiffusionNFTTrainer


def _dense_parent(num_steps: int = 4) -> BaseSample:
    return BaseSample(
        timesteps=torch.arange(num_steps + 1, dtype=torch.float32),
        all_latents=torch.arange(num_steps + 1, dtype=torch.float32).view(-1, 1, 1),
        latent_index_map=torch.arange(num_steps + 1),
        log_probs=torch.arange(num_steps, dtype=torch.float32),
        log_prob_index_map=torch.arange(num_steps),
        extra_kwargs={
            "callback_index_map": torch.arange(num_steps),
            "next_latents_mean": torch.arange(num_steps, dtype=torch.float32).view(-1, 1, 1),
        },
    )


def test_crossover_arguments_expose_only_effective_selection_controls():
    args = CrossoverArguments()

    assert args.survivor_score == "advantage"
    assert not hasattr(args, "covariance_reward_bounds")
    assert not hasattr(args, "include_parents")
    assert not hasattr(args, "selective_crossover")
    assert not hasattr(args, "pareto_filter")
    assert not hasattr(args, "child_warmup_epochs")


def test_crossover_rejects_non_local_group_sampler():
    config = SimpleNamespace(
        training_args=SimpleNamespace(
            trainer_type="crossover-nft",
            crossover=SimpleNamespace(enabled=True),
        ),
        data_args=SimpleNamespace(sampler_type="distributed_k_repeat"),
        reward_args=[],
        eval_reward_args=[],
    )

    with pytest.raises(ValueError, match="requires sampler_type='group_contiguous'"):
        Arguments._resolve_sampler_type(config)


def test_grpo_crossover_sample_batch_does_not_require_sample_side_effect(monkeypatch):
    trainer = CrossoverGRPOGuardTrainer.__new__(CrossoverGRPOGuardTrainer)
    trainer._crossover_enabled = True
    trainer.epoch = 0
    trainer.training_args = SimpleNamespace(
        crossover=CrossoverArguments(step=2),
        seed=42,
        num_inference_steps=4,
    )
    trainer.adapter = SimpleNamespace(
        scheduler=SimpleNamespace(train_timesteps=torch.tensor([1, 3]))
    )
    samples = [_dense_parent()]
    captured_kwargs = {}

    def base_sample_batch(self, batch, reward_buffer=None, **kwargs):
        captured_kwargs.update(kwargs)
        return samples

    monkeypatch.setattr(
        GRPOGuardTrainer,
        "sample_batch",
        base_sample_batch,
    )

    result = trainer.sample_batch(
        {"prompt": ["test prompt"]},
        _crossover_rollout=True,
        compute_log_prob=True,
        trajectory_indices=list(range(5)),
    )

    assert result is samples
    assert result[0].extra_kwargs["_cxo_step"] == 2
    assert not hasattr(trainer, "_max_sde")
    assert "_crossover_rollout" not in captured_kwargs


def test_grpo_crossover_sample_marks_training_rollout():
    trainer = CrossoverGRPOGuardTrainer.__new__(CrossoverGRPOGuardTrainer)
    trainer._crossover_enabled = True
    trainer.epoch = 3
    trainer.training_args = SimpleNamespace(num_inference_steps=4, seed=42)
    seed_calls = []
    trainer.adapter = SimpleNamespace(scheduler=SimpleNamespace(set_seed=seed_calls.append))
    trainer.reward_buffer = object()
    captured = {}
    trainer.generate_samples = lambda **kwargs: captured.update(kwargs) or []

    assert trainer.sample() == []
    assert seed_calls == [45]
    assert captured["_crossover_rollout"] is True
    assert captured["trajectory_indices"] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("crossover_enabled", "compute_log_prob"),
    [(True, False), (False, True)],
)
def test_grpo_crossover_sample_batch_bypasses_non_crossover_rollouts(
    monkeypatch,
    crossover_enabled,
    compute_log_prob,
):
    trainer = CrossoverGRPOGuardTrainer.__new__(CrossoverGRPOGuardTrainer)
    trainer._crossover_enabled = crossover_enabled
    marker = [BaseSample()]
    monkeypatch.setattr(
        GRPOGuardTrainer,
        "sample_batch",
        lambda self, batch, reward_buffer=None, **kwargs: marker,
    )

    result = trainer.sample_batch(
        {"prompt": ["eval prompt"]},
        compute_log_prob=compute_log_prob,
        trajectory_indices=None,
    )

    assert result is marker
    assert "_cxo_step" not in result[0].extra_kwargs


def test_ga_rejects_unknown_survivor_score():
    training_args = SimpleNamespace(
        crossover=SimpleNamespace(survivor_score="largest"),
        advantage_aggregation="gdpo",
        num_inference_steps=4,
        group_size=2,
    )

    with pytest.raises(ValueError, match="survivor_score"):
        GeneticAlgorithm(
            crossover_strategy=None,
            adapter=None,
            accelerator=None,
            autocast=None,
            training_args=training_args,
            reward_buffer=None,
        )


def test_ga_rejects_unknown_advantage_aggregation():
    training_args = SimpleNamespace(
        crossover=SimpleNamespace(survivor_score="advantage"),
        advantage_aggregation="unknown",
        num_inference_steps=4,
        group_size=2,
    )

    with pytest.raises(ValueError, match="advantage_aggregation"):
        GeneticAlgorithm(
            crossover_strategy=None,
            adapter=None,
            accelerator=None,
            autocast=None,
            training_args=training_args,
            reward_buffer=None,
        )


@pytest.mark.parametrize(
    "training_args_cls",
    [CrossoverGRPOGuardTrainingArguments, CrossoverNFTTrainingArguments],
)
def test_cov_per_sample_config_requires_sum_aggregation(training_args_cls):
    with pytest.raises(ValueError, match="cov_per_sample.*requires.*advantage_aggregation: sum"):
        training_args_cls(
            advantage_aggregation="gdpo",
            crossover=CrossoverArguments(survivor_score="cov_per_sample"),
        )


def test_ga_cov_per_sample_requires_sum_aggregation_for_direct_callers():
    training_args = SimpleNamespace(
        crossover=SimpleNamespace(survivor_score="cov_per_sample"),
        advantage_aggregation="gdpo",
        trainer_type="crossover-nft",
        num_inference_steps=4,
        group_size=2,
    )

    with pytest.raises(ValueError, match="cov_per_sample.*requires.*advantage_aggregation='sum'"):
        GeneticAlgorithm(
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


def test_ga_moves_each_template_before_stacking_mixed_devices():
    class PromptAdapter:
        def forward(self, prompt_embeds):
            return prompt_embeds

    cpu_sample = BaseSample(prompt_embeds=torch.ones(2))
    other_device_sample = BaseSample(prompt_embeds=torch.empty(2, device="meta"))
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._adapter = PromptAdapter()

    with pytest.raises(RuntimeError):
        BaseSample.stack([cpu_sample, other_device_sample])

    batch = ga._stack_templates_on_device(
        [cpu_sample, other_device_sample],
        torch.device("meta"),
    )

    assert batch["prompt_embeds"].device.type == "meta"
    assert batch["prompt_embeds"].shape == (2, 2)
    assert cpu_sample.prompt_embeds.device.type == "cpu"
    assert other_device_sample.prompt_embeds.device.type == "meta"


def test_grpo_child_factory_uses_trainer_public_state():
    class DecodeAdapter:
        def __init__(self):
            self.decode_calls = 0

        def decode_latents(self, latents):
            self.decode_calls += 1
            return None

    trainer = CrossoverGRPOGuardTrainer.__new__(CrossoverGRPOGuardTrainer)
    trainer.training_args = SimpleNamespace(
        num_inference_steps=4,
        crossover=SimpleNamespace(strategy="uniform"),
    )
    trainer.adapter = DecodeAdapter()
    trainer._compute_boundary_statistics = lambda parent, latent, step: {
        "log_prob": torch.tensor(2.5),
        "next_latents_mean": torch.tensor([[20.5]]),
    }
    parent = _dense_parent()
    denoise_output = (
        torch.tensor([[[50.0]]]),
        [torch.tensor([[[30.0]]]), torch.tensor([[[40.0]]])],
        [torch.tensor([3.0]), torch.tensor([4.0])],
        {
            "next_latents_mean": [
                torch.tensor([[[30.5]]]),
                torch.tensor([[[40.5]]]),
            ]
        },
    )

    children = trainer._grpo_child_factory(
        templates=[parent],
        child_latents=torch.tensor([[[20.0]]]),
        cxo_step=2,
        denoise_output=denoise_output,
        ctx=SimpleNamespace(strategy_name="uniform", gen_idx=0, gid=123),
    )

    assert trainer.adapter.decode_calls == 1
    assert len(children) == 1
    assert children[0].all_latents[:, 0, 0].tolist() == [0.0, 1.0, 20.0, 30.0, 40.0]


def test_nft_crossover_sample_batch_bypasses_eval(monkeypatch):
    trainer = CrossoverNFTTrainer.__new__(CrossoverNFTTrainer)
    trainer._crossover_enabled = True
    marker = [BaseSample()]
    monkeypatch.setattr(
        DiffusionNFTTrainer,
        "sample_batch",
        lambda self, batch, reward_buffer=None, **kwargs: marker,
    )

    result = trainer.sample_batch(
        {"prompt": ["eval prompt"]},
        compute_log_prob=False,
        trajectory_indices=None,
    )

    assert result is marker
    assert "_cxo_step" not in result[0].extra_kwargs


def test_nft_crossover_sample_batch_consumes_training_marker(monkeypatch):
    trainer = CrossoverNFTTrainer.__new__(CrossoverNFTTrainer)
    trainer._crossover_enabled = True
    trainer.epoch = 0
    trainer.training_args = SimpleNamespace(
        crossover=CrossoverArguments(step=2),
        seed=42,
        num_inference_steps=4,
    )
    marker = [BaseSample()]
    captured_kwargs = {}

    def base_sample_batch(self, batch, reward_buffer=None, **kwargs):
        captured_kwargs.update(kwargs)
        return marker

    monkeypatch.setattr(DiffusionNFTTrainer, "sample_batch", base_sample_batch)

    result = trainer.sample_batch(
        {"prompt": ["train prompt"]},
        _crossover_rollout=True,
        compute_log_prob=False,
        trajectory_indices=[2, 4],
    )

    assert result is marker
    assert result[0].extra_kwargs["_cxo_step"] == 2
    assert "_crossover_rollout" not in captured_kwargs


def test_nft_crossover_sample_marks_training_rollout():
    trainer = CrossoverNFTTrainer.__new__(CrossoverNFTTrainer)
    trainer._crossover_enabled = True
    trainer.training_args = SimpleNamespace(
        num_inference_steps=4,
        crossover=CrossoverArguments(step=2),
    )
    trainer.reward_buffer = object()
    captured = {}
    trainer.generate_samples = lambda **kwargs: captured.update(kwargs) or []

    assert trainer.sample() == []
    assert captured["_crossover_rollout"] is True
    assert captured["trajectory_indices"] == [2, 4]


def test_grpo_child_trajectory_places_intervention_and_uses_identity_maps():
    trainer = CrossoverGRPOGuardTrainer.__new__(CrossoverGRPOGuardTrainer)
    trainer.training_args = SimpleNamespace(crossover=SimpleNamespace(strategy="uniform"))
    parent = _dense_parent()
    crossover_latent = torch.tensor([[20.0]])
    post_latents = [torch.tensor([[30.0]]), torch.tensor([[40.0]])]
    post_log_probs = [torch.tensor(3.0), torch.tensor(4.0)]
    post_callbacks = {"next_latents_mean": [torch.tensor([[30.5]]), torch.tensor([[40.5]])]}
    boundary = {
        "log_prob": torch.tensor(2.5),
        "next_latents_mean": torch.tensor([[20.5]]),
    }

    child = trainer._build_child(
        parent=parent,
        post_latents=post_latents,
        post_log_probs=post_log_probs,
        post_callbacks=post_callbacks,
        image=None,
        cxo_step=2,
        num_steps=4,
        cxo_latent=crossover_latent,
        boundary=boundary,
    )

    assert child.all_latents.shape == (5, 1, 1)
    assert child.all_latents[:, 0, 0].tolist() == [0.0, 1.0, 20.0, 30.0, 40.0]
    assert child.log_probs.tolist() == [0.0, 2.5, 3.0, 4.0]
    assert child.next_latents_mean[:, 0, 0].tolist() == [0.0, 20.5, 30.5, 40.5]
    assert torch.equal(child.latent_index_map, torch.arange(5))
    assert torch.equal(child.log_prob_index_map, torch.arange(4))
    assert torch.equal(child.callback_index_map, torch.arange(4))


def test_grpo_parent_statistics_are_densified_for_mixed_parent_child_batches():
    parent = _dense_parent()
    parent.log_probs = torch.tensor([11.0, 33.0])
    parent.log_prob_index_map = torch.tensor([-1, 0, -1, 1])

    CrossoverGRPOGuardTrainer._densify_parent_maps(parent, num_steps=4)

    assert parent.log_probs.tolist() == [0.0, 11.0, 0.0, 33.0]
    assert torch.equal(parent.log_prob_index_map, torch.arange(4))
    assert torch.equal(parent.latent_index_map, torch.arange(5))
    assert torch.equal(parent.callback_index_map, torch.arange(4))


def test_ga_advantage_uses_group_source_weight():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._advantage_aggregation = "gdpo"
    ga._reward_weights = {"quality": {"source_a": 1.0, "source_b": 3.0}}
    rewards = {"quality": np.array([0.0, 1.0], dtype=np.float32)}

    advantage_a = ga._compute_advantage(rewards, ["quality"], "source_a")
    advantage_b = ga._compute_advantage(rewards, ["quality"], "source_b")

    np.testing.assert_allclose(advantage_b, advantage_a * 3.0)


def test_ga_advantage_follows_configured_aggregation():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._reward_weights = {
        "quality": {"default": 1.0},
        "safety": {"default": 2.0},
    }
    rewards = {
        "quality": np.array([0.0, 10.0, 11.0], dtype=np.float32),
        "safety": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    ga._advantage_aggregation = "sum"
    sum_advantage = ga._compute_advantage(rewards, ["quality", "safety"], None)
    weighted_sum = rewards["quality"] + 2.0 * rewards["safety"]
    expected_sum = (weighted_sum - weighted_sum.mean()) / weighted_sum.std()

    ga._advantage_aggregation = "gdpo"
    gdpo_advantage = ga._compute_advantage(rewards, ["quality", "safety"], None)
    quality_advantage = (rewards["quality"] - rewards["quality"].mean()) / rewards["quality"].std()
    safety_advantage = (rewards["safety"] - rewards["safety"].mean()) / rewards["safety"].std()
    expected_gdpo = quality_advantage + 2.0 * safety_advantage

    np.testing.assert_allclose(sum_advantage, expected_sum)
    np.testing.assert_allclose(gdpo_advantage, expected_gdpo)
    assert not np.allclose(sum_advantage, gdpo_advantage)


@pytest.mark.parametrize("aggregation", ["sum", "gdpo"])
def test_ga_advantage_zero_centers_constant_rewards(aggregation):
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._advantage_aggregation = aggregation
    ga._reward_weights = {"quality": {"default": 1.0}}
    rewards = {"quality": np.array([3.0, 3.0, 3.0], dtype=np.float32)}

    advantage = ga._compute_advantage(rewards, ["quality"], None)

    np.testing.assert_array_equal(advantage, np.zeros(3, dtype=np.float32))


def test_survivor_score_controls_pareto_trimming():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._group_size = 2
    ga._advantage_aggregation = "gdpo"
    ga._reward_weights = {"quality": {"default": 1.0}}
    population = [_dense_parent(), _dense_parent()]
    children = [_dense_parent(), _dense_parent()]
    rewards = {
        "quality": np.array([0.0, 1.0], dtype=np.float32),
    }
    child_rewards = {
        "quality": np.array([-10.0, 2.0], dtype=np.float32),
    }

    ga._survivor_score = "advantage"
    _, high_rewards, _ = ga._select_survivors(
        population, children, rewards, child_rewards, ["quality"], ["quality"], None
    )
    ga._survivor_score = "abs_advantage"
    _, extreme_rewards, _ = ga._select_survivors(
        population, children, rewards, child_rewards, ["quality"], ["quality"], None
    )

    assert high_rewards["quality"].tolist() == [2.0, 1.0]
    assert extreme_rewards["quality"].tolist() == [2.0, -10.0]


def test_ga_distributed_stats_reduce_uses_float32():
    class DtypeCheckingAccelerator:
        num_processes = 2
        device = torch.device("cpu")

        def __init__(self):
            self.reduced_dtype = None

        def reduce(self, tensor, reduction):
            self.reduced_dtype = tensor.dtype
            assert reduction == "sum"
            assert tensor.dtype == torch.float32
            return tensor * self.num_processes

    accelerator = DtypeCheckingAccelerator()
    stats = GeneticAlgorithm.reduce_stats(
        ga_acc={"n_groups": 1, "gen0_count": 1},
        ga_samples=[],
        accelerator=accelerator,
    )

    assert accelerator.reduced_dtype == torch.float32
    assert stats["ga/n_groups"] == 2


def test_cov_per_sample_console_log_uses_selection_specific_metrics():
    line = _format_covariance_selection_log(
        {
            "branch": "cov_per_sample",
            "elite_id": 3,
            "frozen_score": 0.25,
            "lower_bound": 0.2,
            "approximation_gap": 0.05,
            "score": -0.1,
            "scalar_variance": 1.5,
            "degenerate_scalar_contrast": False,
        },
        n_pop=2,
    )

    assert "selection=cov_per_sample" in line
    assert "elite=child:3" in line
    assert "frozen_J=0.25" in line
    assert "lower_bound=0.2" in line
    assert "gap=0.05" in line
    assert "true_J=-0.1" in line
    assert "degenerate_scalar_contrast=False" in line
    assert "degenerate_fallback" not in line


def test_group_covariance_console_log_remains_backward_compatible():
    line = _format_covariance_selection_log(
        {
            "branch": "prune",
            "score": 0.3,
            "scalar_variance": 2.0,
            "degenerate_fallback": False,
        },
        n_pop=2,
    )

    assert line == "covariance=prune J=0.3 variance=2 degenerate_fallback=False"


def test_cov_per_sample_stats_reduce_to_float32_platform_metrics():
    class DtypeCheckingAccelerator:
        num_processes = 2
        device = torch.device("cpu")

        def reduce(self, tensor, reduction):
            assert reduction == "sum"
            assert tensor.dtype == torch.float32
            return tensor * self.num_processes

    stats = GeneticAlgorithm.reduce_stats(
        ga_acc={
            "n_groups": 2,
            "gen0_count": 2,
            "gen0_cov_per_sample_count": 2,
            "gen0_cov_per_sample_frozen_score_sum": 3.0,
            "gen0_cov_per_sample_lower_bound_sum": 2.0,
            "gen0_cov_per_sample_approximation_gap_sum": 1.0,
            "gen0_cov_per_sample_true_score_sum": 0.5,
            "gen0_cov_per_sample_true_score_count": 1,
            "gen0_cov_per_sample_scalar_variance_sum": 4.0,
            "gen0_cov_per_sample_elite_child_count": 1,
            "gen0_cov_per_sample_degenerate_count": 1,
        },
        ga_samples=[],
        accelerator=DtypeCheckingAccelerator(),
    )

    prefix = "ga/gen0/cov_per_sample"
    assert stats[f"{prefix}/frozen_score"] == pytest.approx(1.5)
    assert stats[f"{prefix}/lower_bound"] == pytest.approx(1.0)
    assert stats[f"{prefix}/approximation_gap"] == pytest.approx(0.5)
    assert stats[f"{prefix}/true_score"] == pytest.approx(0.5)
    assert stats[f"{prefix}/scalar_variance"] == pytest.approx(2.0)
    assert stats[f"{prefix}/elite_child_rate"] == pytest.approx(0.5)
    assert stats[f"{prefix}/degenerate_scalar_contrast_rate"] == pytest.approx(0.5)
