# Copyright 2026 Bowen-Zheng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

# src/flow_factory/trainers/crossover_nft.py
"""
CrossoverNFT — DiffusionNFT trainer with Genetic Algorithm augmentation.

Parents generated during ``sample()`` store crossover-step latents.  In
``prepare_feedback()``, a per-group genetic algorithm evolves the
population: select top parents by advantage, crossover + mutation, filter
by Pareto expansion, trim by |advantage|.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm as tqdm_

from ..hparams import CrossoverNFTTrainingArguments
from ..samples import BaseSample
from ..utils.base import create_generator
from ..utils.logger_utils import setup_logger
from .crossover import (
    GeneticAlgorithm,
    compute_pareto_mask,
    create_crossover_strategy,
    sample_crossover_step,
)
from .nft import DiffusionNFTTrainer

tqdm = tqdm_.tqdm
logger = setup_logger(__name__)


class CrossoverNFTTrainer(DiffusionNFTTrainer):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: CrossoverNFTTrainingArguments
        cxo_args = self.training_args.crossover
        self._crossover_enabled = cxo_args.enabled
        if self._crossover_enabled:
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
                f"CrossoverNFT GA: offspring_mode={offspring_mode} "
                f"strategy={cxo_args.strategy} "
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
        cxo_cfg = self.training_args.crossover
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
        )

    def sample_batch(
        self, batch: Dict[str, Any], reward_buffer=None, **extra_inference_kwargs
    ) -> List[BaseSample]:
        """Like the base implementation, but also assigns per-prompt cxo_step."""
        cxo_cfg = self.training_args.crossover
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

        # Assign per-prompt cxo_step to each of the K group members
        K = self.training_args.group_size
        expanded_steps = [s for s in cxo_steps for _ in range(K)]
        for s, step in zip(samples, expanded_steps):
            s.extra_kwargs["_cxo_step"] = step

        return samples

    # =========================== Feedback =================================

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        device = self.accelerator.device
        rank = self.accelerator.process_index
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        if self._crossover_enabled:
            logger.info(f"[rank {rank}] prepare_feedback: calling GA evolve")
            applicable = GeneticAlgorithm.build_applicable_mask(samples, sorted(rewards.keys()))
            evolved_samples, evolved_rewards, ga_acc, ga_samples = self._ga.evolve(
                parent_samples=samples,
                parent_rewards=rewards,
                applicable=applicable,
                epoch=self.epoch,
                verbose=self.log_args.verbose,
            )
            # Barrier: ensure all ranks finish GA before reduction
            self.accelerator.wait_for_everyone()
            logger.info(
                f"[rank {rank}] prepare_feedback: GA returned "
                f"{len(evolved_samples)} evolved samples"
            )

            # Reduce GA stats across ranks
            ga_stats = GeneticAlgorithm.reduce_stats(ga_acc, ga_samples, self.accelerator)
            if ga_stats and self.accelerator.is_main_process:
                self.log_data(ga_stats, step=self.step)

            # GA output replaces original group entirely
            samples[:] = evolved_samples
            rewards = {k: v.to(device) for k, v in evolved_rewards.items()}

        self.advantage_processor._child_advantage_scale = 1.0

        logger.info(f"[rank {rank}] prepare_feedback: calling compute_advantages")
        self.compute_advantages(samples, rewards, store_to_samples=True)
        logger.info(f"[rank {rank}] prepare_feedback: compute_advantages done")
        stats = self.advantage_processor.pop_all_stats()
        if stats:
            self.log_data(stats, step=self.step)
        self.compute_advantages(samples, rewards, store_to_samples=True)
        logger.info(f"[rank {rank}] prepare_feedback: compute_advantages done")
        stats = self.advantage_processor.pop_all_stats()
        if stats:
            self.log_data(stats, step=self.step)
