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

# src/flow_factory/trainers/crossover_grpo_guard.py
"""
CrossoverGRPOGuard — GRPO-Guard trainer with Genetic Algorithm augmentation.

Parents generated during ``sample()`` store crossover-step latents + full
trajectory.  In ``prepare_feedback()``, a per-group genetic algorithm evolves
the population: select parents by advantage, crossover + mutation, preserve
Pareto candidates, and trim with the configured survivor score.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List

import torch

from ..hparams import CrossoverGRPOGuardTrainingArguments
from ..samples import BaseSample
from ..utils.base import create_generator, filter_kwargs, move_tensors_to_device
from ..utils.logger_utils import setup_logger
from .crossover import (
    GeneticAlgorithm,
    create_crossover_strategy,
    sample_crossover_step,
)
from .grpo import GRPOGuardTrainer

logger = setup_logger(__name__)


class CrossoverGRPOGuardTrainer(GRPOGuardTrainer):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: CrossoverGRPOGuardTrainingArguments
        cxo_args = self.training_args.crossover
        self._crossover_enabled = cxo_args.enabled
        if self._crossover_enabled:
            if self.adapter.scheduler.dynamics_type != "Flow-SDE":
                raise ValueError(
                    "crossover-grpo-guard requires scheduler.dynamics_type='Flow-SDE'; "
                    f"got {self.adapter.scheduler.dynamics_type!r}."
                )
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
            self.advantage_processor._child_in_norm = True
            logger.info(
                f"CrossoverGRPOGuard GA: offspring_mode={offspring_mode} "
                f"strategy={cxo_args.strategy} "
                f"advantage_aggregation({self._ga._advantage_aggregation}) "
                f"crossover.survivor_score({self._ga._survivor_score}) "
                f"survivor_selection_aggregation("
                f"{self._ga._survivor_selection_aggregation}) "
                f"parent_ratio={self._ga._parent_ratio} "
                f"mutation_std={self._ga._mutation_std} "
                f"generations={self._ga._n_generations}"
            )

    # =========================== Sampling ==================================

    def sample(self) -> List[BaseSample]:
        if not self._crossover_enabled:
            return super().sample()

        num_steps = self.training_args.num_inference_steps
        base_seed = self.training_args.seed + self.epoch

        # SDE seed: epoch-level (original GRPO).  All parents share the same
        # train_timesteps so optimize can iterate uniformly.
        self.adapter.scheduler.set_seed(base_seed)

        # Store a dense parent trajectory. Child crossover steps can differ within
        # one optimize batch, while BaseSample treats trajectory index maps as
        # shared fields; identity maps keep every child batch-safe and auditable.
        trajectory_indices = list(range(num_steps + 1))
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
            extra_call_back_kwargs=["next_latents_mean"],
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

        cxo_cfg = self.training_args.crossover
        base_seed = self.training_args.seed + self.epoch
        num_steps = self.training_args.num_inference_steps
        train_ts = self.adapter.scheduler.train_timesteps
        max_sde = int(train_ts.max().item()) if train_ts.numel() > 0 else num_steps

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

        # The data sampler has already materialized all K repeated rows. Adapter
        # inference returns one sample per row, so assignments stay row-aligned.
        for s, step in zip(samples, cxo_steps):
            self._densify_parent_maps(s, num_steps)
            s.extra_kwargs["_cxo_step"] = step

        return samples

    @staticmethod
    def _densify_parent_maps(sample: BaseSample, num_steps: int) -> None:
        """Convert sparse rollout statistics to batch-safe identity maps."""
        device = sample.all_latents.device
        log_prob_dtype = (
            sample.log_probs.dtype if sample.log_probs is not None else sample.all_latents.dtype
        )
        dense_log_probs = torch.zeros(
            num_steps,
            device=device,
            dtype=log_prob_dtype,
        )
        if sample.log_probs is not None and sample.log_prob_index_map is not None:
            for step_idx in range(num_steps):
                compact_idx = int(sample.log_prob_index_map[step_idx])
                if compact_idx >= 0:
                    dense_log_probs[step_idx] = sample.log_probs[compact_idx]
        sample.log_probs = dense_log_probs
        sample.log_prob_index_map = torch.arange(num_steps, dtype=torch.long, device=device)
        sample.latent_index_map = torch.arange(num_steps + 1, dtype=torch.long, device=device)
        if "next_latents_mean" in sample.extra_kwargs:
            sample.extra_kwargs["callback_index_map"] = torch.arange(
                num_steps, dtype=torch.long, device=device
            )

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
            evolved_samples, evolved_rewards, ga_acc, ga_selection_events = self._ga.evolve(
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

            ga_stats = GeneticAlgorithm.reduce_stats(ga_acc, ga_selection_events, self.accelerator)
            if ga_stats and self.accelerator.is_main_process:
                self.log_data(ga_stats, step=self.step)
            samples[:] = evolved_samples
            rewards = {k: v.to(device) for k, v in evolved_rewards.items()}

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
        templates: List[BaseSample],
        child_latents: torch.Tensor,
        cxo_step: int,
        denoise_output: tuple,
        ctx: Any,
    ) -> List[BaseSample]:
        """Child factory for GRPO-Guard — merges parent pre-cxo trajectory
        with child post-cxo trajectory to create full-trajectory children."""
        num_steps = self.training_args.num_inference_steps
        finals, post_lat, post_lp, post_cb = denoise_output
        n_children = child_latents.shape[0]
        children: List[BaseSample] = []

        for m in range(n_children):
            template = templates[m]
            imgs = self.adapter.decode_latents(finals[m : m + 1])
            child_post_lat = [lat[m] for lat in post_lat]
            child_post_lp = [lp[m] for lp in post_lp]
            child_post_cb = (
                {k: [cb[m] for cb in v] for k, v in post_cb.items()} if post_cb else None
            )
            boundary = self._compute_boundary_statistics(template, child_latents[m], cxo_step)
            child = self._build_child(
                parent=template,
                post_latents=child_post_lat,
                post_log_probs=child_post_lp,
                post_callbacks=child_post_cb,
                image=imgs,
                cxo_step=cxo_step,
                num_steps=num_steps,
                cxo_latent=child_latents[m],
                boundary=boundary,
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
        boundary=None,
    ):
        device = cxo_latent.device
        T, T1 = num_steps, num_steps + 1
        cb_map = parent.extra_kwargs.get("callback_index_map", torch.arange(T, device=device))

        def _merge_cb(key):
            pv = parent.extra_kwargs.get(key)
            cl = post_callbacks.get(key, []) if post_callbacks else []
            if pv is None and not cl:
                return None
            merged = []
            for si in range(T):
                if boundary is not None and si == cxo_step - 1:
                    merged.append(boundary[key])
                elif si < cxo_step and pv is not None:
                    pi = int(cb_map[si])
                    merged.append(
                        pv[pi].to(device)
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
        al_list = []
        for si in range(T1):
            if si < cxo_step:
                pi = int(pm[si])
                if pi < 0:
                    raise RuntimeError(f"Missing parent latent at trajectory position {si}.")
                al_list.append(parent.all_latents[pi].to(device))
            elif si == cxo_step:
                al_list.append(cxo_latent)
            else:
                j = si - cxo_step - 1
                al_list.append(post_latents[j])
        merged_al = torch.stack(al_list)
        lm = torch.arange(T1, dtype=torch.long, device=device)

        # log_probs
        lpm = parent.log_prob_index_map
        lp_list = []
        for si in range(T):
            if boundary is not None and si == cxo_step - 1:
                lp_list.append(boundary["log_prob"])
            elif si < cxo_step:
                pi = int(lpm[si])
                if pi >= 0:
                    lp_list.append(parent.log_probs[pi].to(device))
                else:
                    lp_list.append(torch.zeros((), device=device, dtype=merged_al.dtype))
            else:
                j = si - cxo_step
                lp_list.append(post_log_probs[j])
        merged_lp = torch.stack(lp_list)
        lpm2 = torch.arange(T, dtype=torch.long, device=device)

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
            extra["callback_index_map"] = torch.arange(T, dtype=torch.long, device=device)
        parent_dict["extra_kwargs"] = extra

        child = type(parent).from_dict(parent_dict)
        return child

    def _compute_boundary_statistics(
        self,
        parent: BaseSample,
        crossover_latent: torch.Tensor,
        cxo_step: int,
    ) -> Any:
        """Recompute old-policy statistics for the intervention boundary."""
        if cxo_step <= 0:
            return None

        boundary_index = cxo_step - 1
        parent_index = int(parent.latent_index_map[boundary_index])
        device = self.accelerator.device
        latents = parent.all_latents[parent_index].unsqueeze(0).to(device)
        next_latents = crossover_latent.unsqueeze(0).to(device)
        t = parent.timesteps[boundary_index].to(device)
        t_next = parent.timesteps[boundary_index + 1].to(device)
        noise_level = self.adapter.scheduler.get_noise_level_for_timestep(t)
        compute_log_prob = bool(boundary_index in self.adapter.scheduler.train_timesteps.tolist())
        batch = BaseSample.stack([parent])
        forward_inputs = {
            **self.training_args,
            **batch,
            "t": t,
            "t_next": t_next,
            "latents": latents,
            "next_latents": next_latents,
            "compute_log_prob": compute_log_prob,
            "noise_level": noise_level,
            "return_kwargs": ["log_prob", "next_latents_mean"],
        }
        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
        forward_inputs = move_tensors_to_device(forward_inputs, device)
        with self.autocast():
            output = self.adapter.forward(**forward_inputs)
        log_prob = (
            output.log_prob[0]
            if compute_log_prob
            else torch.zeros((), device=latents.device, dtype=latents.dtype)
        )
        return {
            "log_prob": log_prob.detach(),
            "next_latents_mean": output.next_latents_mean[0].detach(),
        }
