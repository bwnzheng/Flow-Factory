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
                reward_weights=self.advantage_processor.reward_weights,
                seed=self.training_args.seed,
            )
            if getattr(cxo_args, "log_rewards", True):
                self.advantage_processor._log_crossover_rewards = True
            self.advantage_processor._child_in_norm = True
            logger.info(
                f"CrossoverNFT GA: strategy={cxo_args.strategy} "
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
            evolved_samples, evolved_rewards, ga_acc, ga_samples = self._ga.evolve(
                parent_samples=samples,
                parent_rewards=rewards,
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
            ga_stats = self._reduce_ga_stats(ga_acc, ga_samples)
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

    # ======================================================================
    # GA stats reduction
    # ======================================================================

    def _reduce_ga_stats(
        self,
        ga_acc: Dict[str, Any],
        ga_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Reduce GA accumulator across ranks and build final stats dict."""
        num_ranks = self.accelerator.num_processes
        if num_ranks <= 1:
            return self._build_ga_stats(ga_acc, ga_samples)

        # Pack all float values into a tensor for a single reduce call
        float_keys = [k for k, v in ga_acc.items() if isinstance(v, float)]
        int_keys = [k for k, v in ga_acc.items() if isinstance(v, int)]
        all_keys = float_keys + int_keys
        t = torch.tensor(
            [float(ga_acc[k]) for k in all_keys],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        t = self.accelerator.reduce(t, reduction="sum")
        for i, k in enumerate(all_keys):
            if k in int_keys:
                ga_acc[k] = int(t[i].item())
            else:
                ga_acc[k] = t[i].item()

        return self._build_ga_stats(ga_acc, ga_samples)

    @staticmethod
    def _build_ga_stats(
        ga_acc: Dict[str, Any],
        ga_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute final metrics from reduced accumulators."""
        stats: Dict[str, Any] = {"ga/n_groups": ga_acc["n_groups"]}
        max_gen = 0
        while f"gen{max_gen}_count" in ga_acc:
            max_gen += 1

        reward_keys = set()
        for k in ga_acc:
            if k.startswith("gen0_") and k.endswith("_pop_sum"):
                reward_keys.add(k[len("gen0_"):-len("_pop_sum")])

        for gen in range(max_gen):
            count = ga_acc[f"gen{gen}_count"]
            if count == 0:
                continue
            p = f"ga/gen{gen}"
            for prefix, key in [
                ("n_replaced", "n_replaced"),
                ("n_children", "n_children"),
                ("n_children_kept", "n_children_kept"),
                ("n_pareto_parents", "n_pareto_parents"),
                ("n_pareto_children", "n_pareto_children"),
                ("n_filled", "n_filled"),
            ]:
                stats[f"{p}/{prefix}"] = round(ga_acc[f"gen{gen}_{key}"] / max(count, 1), 2)

            for rk in sorted(reward_keys):
                for prefix, sum_key, sum_sq_key in [
                    ("pop_mean", "pop_sum", "pop_sum_sq"),
                    ("child_mean", "child_sum", "child_sum_sq"),
                    ("new_mean", "new_sum", "new_sum_sq"),
                ]:
                    s = ga_acc[f"gen{gen}_{rk}_{sum_key}"]
                    sq = ga_acc[f"gen{gen}_{rk}_{sum_sq_key}"]
                    n_eff = count if "child" not in sum_key else max(count, 1)
                    mean = s / max(n_eff, 1)
                    var = max(sq / max(n_eff, 1) - mean**2, 0.0)
                    if "mean" in prefix:
                        stats[f"{p}/{rk}/{prefix}"] = round(mean, 6)
                    else:
                        stats[f"{p}/{rk}/{prefix.replace('mean', 'std')}"] = round(var**0.5, 6)

        # Per-sample reward records for JSONL
        if ga_samples:
            stats["ga/samples"] = ga_samples

        return stats
        self.compute_advantages(samples, rewards, store_to_samples=True)
        logger.info(f"[rank {rank}] prepare_feedback: compute_advantages done")
        stats = self.advantage_processor.pop_all_stats()
        if stats:
            self.log_data(stats, step=self.step)

