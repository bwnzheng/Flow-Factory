# Copyright 2026 Bowen-Zheng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

# src/flow_factory/trainers/crossover_grpo_guard.py
"""
CrossoverGRPOGuard — GRPO-Guard trainer with Genetic Algorithm augmentation.

Parents generated during ``sample()`` store crossover-step latents + full
trajectory.  In ``prepare_feedback()``, a per-group genetic algorithm evolves
the population: select top parents by advantage, crossover + mutation, filter
by Pareto expansion, trim by |advantage|.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm as tqdm_

from ..hparams import CrossoverGRPOGuardTrainingArguments
from ..samples import BaseSample
from ..utils.base import create_generator
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices
from .crossover import (
    GeneticAlgorithm,
    create_crossover_strategy,
    sample_crossover_step,
)
from .grpo import GRPOGuardTrainer

tqdm = tqdm_.tqdm
logger = setup_logger(__name__)


class CrossoverGRPOGuardTrainer(GRPOGuardTrainer):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: CrossoverGRPOGuardTrainingArguments
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
                denoise_kwargs={
                    "compute_log_prob": True,
                    "collect_trajectory": True,
                    "extra_call_back_kwargs": ["next_latents_mean"],
                    "collect_callbacks": True,
                },
                child_factory=self._grpo_child_factory,
            )
            if getattr(cxo_args, "log_rewards", True):
                self.advantage_processor._log_crossover_rewards = True
            logger.info(
                f"CrossoverGRPOGuard GA: offspring_mode={offspring_mode} "
                f"strategy={cxo_args.strategy} "
                f"parent_ratio={self._ga._parent_ratio} "
                f"mutation_std={self._ga._mutation_std} "
                f"generations={self._ga._n_generations}"
            )

    # =========================== Sampling ==================================

    def sample(self) -> List[BaseSample]:
        if not self._crossover_enabled:
            return super().sample()

        num_steps = self.training_args.num_inference_steps
        cxo_cfg = self.training_args.crossover
        base_seed = self.training_args.seed + self.epoch

        # SDE seed: epoch-level (original GRPO).  All parents share the same
        # train_timesteps so optimize can iterate uniformly.
        self.adapter.scheduler.set_seed(base_seed)
        train_ts = self.adapter.scheduler.train_timesteps
        train_idx = compute_trajectory_indices(
            train_timestep_indices=train_ts, num_inference_steps=num_steps
        )
        self._max_sde = int(train_ts.max().item()) if train_ts.numel() > 0 else num_steps

        # Union of all possible cxo steps from step_range → uniform all_latents dim-0
        lo = (
            int(cxo_cfg.step_range[0] * num_steps)
            if cxo_cfg.step_sampling != "fixed"
            else (
                int(cxo_cfg.step * num_steps) if isinstance(cxo_cfg.step, float) else cxo_cfg.step
            )
        )
        hi = int(cxo_cfg.step_range[1] * num_steps) if cxo_cfg.step_sampling != "fixed" else lo
        lo, hi = max(1, lo), min(num_steps - 1, hi)
        ext_idx = sorted(set(train_idx) | {0} | set(range(lo, hi + 1)))

        # Reuse the standard sampling pipeline — generate_samples() handles
        # adapter.rollout(), dataloader.set_epoch(), the inference loop,
        # metadata injection, and CPU offloading.  Per-prompt cxo_step
        # assignment is injected via the sample_batch() override below.
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=ext_idx,
            extra_call_back_kwargs=["next_latents_mean"],
        )

    def sample_batch(
        self, batch: Dict[str, Any], reward_buffer=None, **extra_inference_kwargs
    ) -> List[BaseSample]:
        """Like the base implementation, but also assigns per-prompt cxo_step."""
        cxo_cfg = self.training_args.crossover
        base_seed = self.training_args.seed + self.epoch
        num_steps = self.training_args.num_inference_steps
        max_sde = self._max_sde

        prompts = batch.get("prompt")
        B = len(prompts) if prompts is not None and isinstance(prompts, list) else 1

        cxo_steps: List[int] = []
        for i in range(B):
            p = prompts[i] if prompts is not None and isinstance(prompts, list) else str(i)
            h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            gen = create_generator((base_seed + h) % (2**31), device="cpu")
            raw = sample_crossover_step(cxo_cfg, num_steps, generator=gen)
            step = min(raw, max_sde - 1) if max_sde > 1 else raw
            cxo_steps.append(max(1, step))

        # Standard batched inference
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
            logger.info(
                f"[rank {rank}] prepare_feedback: calling GA evolve "
                f"({len(set(s.unique_id for s in samples))} groups)"
            )
            applicable = GeneticAlgorithm.build_applicable_mask(samples, sorted(rewards.keys()))
            t_ga = time.time()
            evolved_samples, evolved_rewards, ga_acc, ga_samples = self._ga.evolve(
                parent_samples=samples,
                parent_rewards=rewards,
                applicable=applicable,
                epoch=self.epoch,
                verbose=self.show_progress_bar,
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

            ga_stats = GeneticAlgorithm.reduce_stats(ga_acc, ga_samples, self.accelerator)
            if ga_stats and self.accelerator.is_main_process:
                self.log_data(ga_stats, step=self.step)
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
    # GRPO-Guard child factory (with trajectory merging)
    # ======================================================================

    def _grpo_child_factory(
        self,
        template: BaseSample,
        child_latents: torch.Tensor,
        cxo_step: int,
        denoise_output: tuple,
        ctx: Any,
    ) -> List[BaseSample]:
        """Child factory for GRPO-Guard — merges parent pre-cxo trajectory
        with child post-cxo trajectory to create full-trajectory children."""
        device = child_latents.device
        num_steps = self._num_steps
        finals, post_lat, post_lp, post_cb = denoise_output
        n_children = child_latents.shape[0]
        cross_latents_cpu = child_latents.detach().cpu()
        children: List[BaseSample] = []

        for m in range(n_children):
            imgs = self._adapter.decode_latents(finals[m : m + 1])
            child_post_lat = [lat[m : m + 1] for lat in post_lat]
            child_post_lp = [lp[m : m + 1] for lp in post_lp]
            child_post_cb = (
                {k: [cb[m : m + 1] for cb in v] for k, v in post_cb.items()} if post_cb else None
            )
            child = self._build_child(
                parent=template,
                post_latents=child_post_lat,
                post_log_probs=child_post_lp,
                post_callbacks=child_post_cb,
                image=imgs,
                cxo_step=cxo_step,
                num_steps=num_steps,
                cxo_latent=cross_latents_cpu[m],
            )
            child.extra_kwargs["crossover_strategy"] = ctx.strategy_name
            child.extra_kwargs["generation"] = ctx.gen_idx
            child._unique_id = ctx.gid
            children.append(child)

        return children

    # ======================================================================
    # Child trajectory construction
    # ======================================================================

    def _build_child(
        self,
        parent,
        post_latents,
        post_log_probs,
        post_callbacks,
        image,
        cxo_step,
        num_steps,
        cxo_latent=None,
    ):
        device = post_latents[0].device
        T, T1 = num_steps, num_steps + 1
        cb_map = parent.extra_kwargs.get("callback_index_map", torch.arange(T, device=device))

        def _merge_cb(key):
            pv = parent.extra_kwargs.get(key)
            cl = post_callbacks.get(key, []) if post_callbacks else []
            if pv is None and not cl:
                return None
            merged = []
            for si in range(T):
                if si < cxo_step and pv is not None:
                    pi = int(cb_map[si])
                    merged.append(
                        pv[pi]
                        if pi >= 0
                        else (torch.zeros_like(cl[0]) if cl else torch.tensor(0.0, device=device))
                    )
                elif (si - cxo_step) < len(cl):
                    merged.append(cl[si - cxo_step])
                else:
                    merged.append(
                        torch.zeros_like(merged[-1]) if merged else torch.tensor(0.0, device=device)
                    )
            return torch.stack(merged) if merged else None

        # all_latents
        pm = parent.latent_index_map
        al_list, lm = [], torch.full((T1,), -1, dtype=torch.long, device=device)
        pos = 0
        for si in range(T1):
            if si < cxo_step:
                pi = int(pm[si])
                if pi >= 0:
                    al_list.append(parent.all_latents[pi])
                    lm[si] = pos
                    pos += 1
            elif si == cxo_step and post_latents:
                al_list.append(post_latents[0])
                lm[si] = pos
                pos += 1
            else:
                j = si - cxo_step - 1
                if j >= 0 and j < len(post_latents):
                    al_list.append(post_latents[j])
                    lm[si] = pos
                    pos += 1
        merged_al = torch.stack(al_list) if al_list else torch.empty(0, device=device)

        # log_probs
        lpm = parent.log_prob_index_map
        lp_list, lpm2 = [], torch.full((T,), -1, dtype=torch.long, device=device)
        pos = 0
        for si in range(T):
            if si < cxo_step:
                pi = int(lpm[si])
                if pi >= 0:
                    lp_list.append(parent.log_probs[pi])
                    lpm2[si] = pos
                    pos += 1
            else:
                j = si - cxo_step
                if j >= 0 and j < len(post_log_probs):
                    lp_list.append(post_log_probs[j])
                    lpm2[si] = pos
                    pos += 1
        merged_lp = torch.stack(lp_list) if lp_list else torch.empty(0, device=device)

        # Inherit all parent fields via to_dict/from_dict
        parent_dict = parent.to_dict()
        parent_dict["all_latents"] = merged_al
        parent_dict["latent_index_map"] = lm
        parent_dict["log_probs"] = merged_lp
        parent_dict["log_prob_index_map"] = lpm2
        parent_dict["image"] = image
        parent_dict["_unique_id"] = None
        parent_dict["applicable_rewards"] = set()

        extra = parent_dict.get("extra_kwargs", {})
        extra["is_crossover_child"] = True
        extra["crossover_step"] = cxo_step
        extra["crossover_strategy"] = self.training_args.crossover.strategy
        if cxo_latent is not None:
            extra["_cxo_latent"] = cxo_latent.detach().cpu()
        merged_nlm = _merge_cb("next_latents_mean")
        if merged_nlm is not None:
            extra["next_latents_mean"] = merged_nlm
        parent_dict["extra_kwargs"] = extra

        child = type(parent).from_dict(parent_dict)
        return child
