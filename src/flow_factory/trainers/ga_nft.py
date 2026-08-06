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

# src/flow_factory/trainers/ga_nft.py
"""
GA NFT — DiffusionNFT trainer with Genetic Algorithm augmentation.

Parents generated during ``sample()`` store crossover-step latents.  In
``prepare_feedback()``, a per-group genetic algorithm evolves the
population: select parents by advantage, crossover + mutation, preserve
Pareto candidates, and trim with the configured survivor score.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List

import torch

from ..hparams import GANFTTrainingArguments
from ..samples import BaseSample
from ..utils.base import create_generator
from ..utils.logger_utils import setup_logger
from .crossover import (
    GeneticAlgorithm,
    create_crossover_strategy,
    sample_crossover_step,
)
from .nft import DiffusionNFTTrainer

logger = setup_logger(__name__)


class GANFTTrainer(DiffusionNFTTrainer):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: GANFTTrainingArguments
        cxo_args = self.training_args.ga
        self._crossover_enabled = cxo_args.enabled
        if self._crossover_enabled:
            if self.reward_processor._groupwise_models:
                raise ValueError(
                    "Crossover trainers currently support pointwise rewards only; "
                    "groupwise rewards require candidate-population-wide rescoring."
                )
            strategy = create_crossover_strategy(
                name=cxo_args.strategy,
                augmentation_factor=cxo_args.augmentation_factor,
                **cxo_args.strategy_kwargs,
            )
            offspring_mode = getattr(cxo_args, "offspring_mode", "crossover")
            self._ga = GeneticAlgorithm(
                crossover_strategy=strategy,
                adapter=self.adapter,
                accelerator=self.accelerator,
                autocast=self.autocast,
                training_args=self.training_args,
                reward_buffer=self.reward_buffer,
                parent_ratio=getattr(cxo_args, "parent_ratio", 0.25),
                mutation_std=getattr(cxo_args, "mutation_std", 0.0),
                evolution_generations=getattr(cxo_args, "evolution_generations", 1),
                offspring_mode=offspring_mode,
                reward_weights=self.advantage_processor.reward_weights,
                seed=self.training_args.seed,
            )
            if getattr(cxo_args, "log_rewards", True):
                self.advantage_processor._log_crossover_rewards = True
            self.advantage_processor._child_in_norm = True
            logger.info(
                f"GA NFT: offspring_mode={offspring_mode} "
                f"strategy={cxo_args.strategy} "
                f"advantage_aggregation({self._ga._advantage_aggregation}) "
                f"ga.survivor_score({self._ga._survivor_score}) "
                f"survivor_selection_aggregation("
                f"{self._ga._survivor_selection_aggregation}) "
                f"parent_ratio={self._ga._parent_ratio} "
                f"mutation_std={self._ga._mutation_std} "
                f"generations={self._ga._n_generations}"
            )

    # =========================== Sampling ==================================

    def sample(self) -> List[BaseSample]:
        if not self._crossover_enabled:
            return self.generate_samples(
                reward_buffer=self.reward_buffer,
                compute_log_prob=False,
                trajectory_indices=[-1],
            )

        # Union of all possible crossover step positions — same pattern as
        # GRPO-Guard.  Enables batched inference via generate_samples().
        num_steps = self.training_args.num_inference_steps
        cxo_cfg = self.training_args.ga
        lo = (
            int(cxo_cfg.step_range[0] * num_steps)
            if cxo_cfg.step_sampling != "fixed"
            else (
                int(cxo_cfg.step * num_steps) if isinstance(cxo_cfg.step, float) else cxo_cfg.step
            )
        )
        hi = int(cxo_cfg.step_range[1] * num_steps) if cxo_cfg.step_sampling != "fixed" else lo
        lo, hi = max(1, lo), min(num_steps - 1, hi)
        ext_idx = sorted(set(range(lo, hi + 1)) | {num_steps})

        # Reuse the standard sampling pipeline — generate_samples() handles
        # adapter.rollout(), dataloader.set_epoch(), the inference loop,
        # metadata injection, and CPU offloading.  Per-prompt cxo_step
        # assignment is injected via the sample_batch() override below.
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=ext_idx,
            _crossover_rollout=True,
        )

    def sample_batch(
        self, batch: Dict[str, Any], reward_buffer=None, **extra_inference_kwargs
    ) -> List[BaseSample]:
        """Like the base implementation, but also assigns per-prompt cxo_step."""
        crossover_rollout = bool(extra_inference_kwargs.pop("_crossover_rollout", False))
        if not self._crossover_enabled or not crossover_rollout:
            return super().sample_batch(
                batch,
                reward_buffer=reward_buffer,
                **extra_inference_kwargs,
            )

        cxo_cfg = self.training_args.ga
        base_seed = self.training_args.seed + self.epoch
        num_steps = self.training_args.num_inference_steps

        prompts = batch.get("prompt")
        B = len(prompts) if prompts is not None and isinstance(prompts, list) else 1

        cxo_steps: List[int] = []
        for i in range(B):
            p = prompts[i] if prompts is not None and isinstance(prompts, list) else str(i)
            h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            gen = create_generator((base_seed + h) % (2**31), device="cpu")
            cxo_steps.append(sample_crossover_step(cxo_cfg, num_steps, generator=gen))

        # Standard batched inference (uses trajectory_indices from extra_inference_kwargs)
        samples = super().sample_batch(batch, reward_buffer=reward_buffer, **extra_inference_kwargs)

        # The data sampler has already materialized all K repeated rows. Adapter
        # inference returns one sample per row, so assignments stay row-aligned.
        for s, step in zip(samples, cxo_steps):
            s.extra_kwargs["_cxo_step"] = step

        return samples

    # =========================== Feedback =================================

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        device = self.accelerator.device
        rank = self.accelerator.process_index
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        if self._crossover_enabled:
            logger.info(
                f"[rank {rank}] prepare_feedback: calling GA evolve "
                f"({len(set(s.unique_id for s in samples))} groups)"
            )
            applicable = GeneticAlgorithm.build_applicable_mask(samples, sorted(rewards.keys()))
            t_ga = time.time()
            with self.sampling_context():
                evolved_samples, evolved_rewards, ga_acc, ga_selection_events = self._ga.evolve(
                    parent_samples=samples,
                    parent_rewards=rewards,
                    applicable=applicable,
                    epoch=self.epoch,
                    verbose=self.log_args.verbose,
                )
            t_ga = time.time() - t_ga
            logger.info(
                f"[rank {rank}] prepare_feedback: GA returned "
                f"{len(evolved_samples)} evolved samples in {t_ga:.1f}s"
            )

            # Barrier: prevent cross-epoch drift accumulation.
            # With imbalanced per-group reward compute, evolve() time varies
            # across ranks.  This barrier caps the drift to a single epoch's
            # max time difference.  If HCCL_EXEC_TIMEOUT is too low (default
            # 300 s), fast ranks may be killed by the watchdog while waiting.
            # Increase to e.g. 1800 s via: export HCCL_EXEC_TIMEOUT=1800
            self.accelerator.wait_for_everyone()

            # Reduce GA stats across ranks
            ga_stats = GeneticAlgorithm.reduce_stats(ga_acc, ga_selection_events, self.accelerator)
            if ga_stats and self.accelerator.is_main_process:
                self.log_data(ga_stats, step=self.step)

            # GA output replaces original group entirely
            samples[:] = evolved_samples
            rewards = {k: v.to(device) for k, v in evolved_rewards.items()}

        logger.info(f"[rank {rank}] prepare_feedback: calling compute_advantages")
        self.compute_advantages(samples, rewards, store_to_samples=True)
        logger.info(f"[rank {rank}] prepare_feedback: compute_advantages done")
        stats = self.advantage_processor.pop_all_stats()
        if stats:
            self.log_data(stats, step=self.step)
