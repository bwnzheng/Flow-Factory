# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

# src/flow_factory/trainers/crossover_grpo_guard.py
"""
CrossoverGRPOGuard — GRPO-Guard trainer with global-group crossover.

Parents generated during ``sample()`` store crossover-step latents + full
trajectory.  In ``prepare_feedback()``, rewards are gathered globally,
non-dominated parents identified per group, and crossover children generated
with equal share across all ranks.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm as tqdm_

from ..hparams import CrossoverGRPOGuardTrainingArguments
from ..samples import BaseSample
from ..utils.base import create_generator, filter_kwargs
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices
from .crossover import (
    compute_pareto_mask,
    create_crossover_strategy,
    run_denoising_phase,
    sample_crossover_step,
    select_non_dominated_parents,
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
            self._crossover_strategy = create_crossover_strategy(
                name=cxo_args.strategy,
                augmentation_factor=cxo_args.augmentation_factor,
                **cxo_args.strategy_kwargs,
            )
            self._selective = getattr(cxo_args, "selective_crossover", False)
            self._include_parents = getattr(cxo_args, "include_parents", True)
            if getattr(cxo_args, "pareto_filter", False):
                self.advantage_processor._pareto_enabled = True
            if getattr(cxo_args, "log_rewards", True):
                self.advantage_processor._log_crossover_rewards = True
            logger.info(
                f"CrossoverGRPOGuard: strategy={cxo_args.strategy} selective={self._selective} "
                f"include_parents={self._include_parents}"
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
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        if self._crossover_enabled:
            children, child_rewards = self._crossover_augment(samples, rewards)
            if children:
                if self._include_parents:
                    samples.extend(children)
                    rewards = {
                        k: torch.cat([rewards[k].to(device), child_rewards[k].to(device)], dim=0)
                        for k in rewards
                    }
                else:
                    samples[:] = children
                    rewards = {k: v.to(device) for k, v in child_rewards.items()}

                # Sort by unique_id so children are interleaved with their
                # parent groups — advantage grouping is keyed on unique_id,
                # and logging (train_samples) now covers children evenly.
                uids = [s.unique_id for s in samples]
                perm = sorted(range(len(samples)), key=lambda i: uids[i])
                samples[:] = [samples[i] for i in perm]
                rewards = {k: v[perm].to(device) for k, v in rewards.items()}

        # ---- Child advantage warmup ----
        warmup_epochs = getattr(self.training_args.crossover, "child_advantage_warmup_epochs", 0)
        if warmup_epochs > 0 and self._crossover_enabled:
            scale = min(1.0, self.epoch / max(warmup_epochs, 1))
            self.advantage_processor._child_advantage_scale = scale
        else:
            self.advantage_processor._child_advantage_scale = 1.0

        self.compute_advantages(samples, rewards, store_to_samples=True)
        stats = self.advantage_processor.pop_all_stats()
        if stats:
            self.log_data(stats, step=self.step)

    # ======================================================================
    # Crossover augmentation
    # ======================================================================

    @torch.no_grad()
    def _crossover_augment(
        self, parent_samples: List[BaseSample], parent_rewards: Dict[str, torch.Tensor]
    ) -> tuple:
        device = self.accelerator.device
        num_steps = self.training_args.num_inference_steps
        rank = self.accelerator.process_index
        world = self.accelerator.num_processes
        B = len(parent_samples)

        # 1. Gather rewards + unique_ids globally
        reward_keys = sorted(parent_rewards.keys())
        packed = torch.stack(
            [torch.as_tensor(parent_rewards[k], device=device) for k in reward_keys]
            + [
                torch.as_tensor(
                    [s.unique_id for s in parent_samples], dtype=torch.float32, device=device
                )
            ],
            dim=1,
        )
        gathered = self.accelerator.gather(packed)
        g_rewards = {k: gathered[:, i].cpu().numpy() for i, k in enumerate(reward_keys)}
        g_ids = gathered[:, -1].cpu().numpy().astype(np.int64)

        # 2. Non-dominated per group (global)
        if self._selective:
            nondom = select_non_dominated_parents(
                {k: torch.from_numpy(v) for k, v in g_rewards.items()}, g_ids.tolist()
            )
        else:
            nondom = np.ones(len(g_ids), dtype=bool)

        # 3. All_gather (latent, gid, step) of non-dominated parents.
        #    Prompt data is fetched from each child's local parent in step 5.
        local_nondom = nondom[rank * B : (rank + 1) * B]
        local_latents, local_gids, local_steps = [], [], []
        for i, s in enumerate(parent_samples):
            if local_nondom[i]:
                step = s.extra_kwargs["_cxo_step"]
                idx = int(s.latent_index_map[step])
                local_latents.append(s.all_latents[idx].to(device))
                local_gids.append(s.unique_id)
                local_steps.append(step)

        cnt_t = torch.tensor([len(local_latents)], device=device)
        max_cnt = max(self.accelerator.gather(cnt_t).max().item(), 1)
        dummy = (
            torch.zeros_like(local_latents[0]) if local_latents else torch.zeros(1, device=device)
        )
        padded_l = (
            (local_latents + [dummy] * (max_cnt - len(local_latents)))
            if local_latents
            else [dummy] * max_cnt
        )
        all_lat = self.accelerator.gather(torch.stack(padded_l))
        all_gid = (
            self.accelerator.gather(
                torch.tensor(
                    local_gids + [-1] * (max_cnt - len(local_gids)), dtype=torch.long, device=device
                )
            )
            .cpu()
            .tolist()
        )
        all_step = (
            self.accelerator.gather(
                torch.tensor(
                    local_steps + [-1] * (max_cnt - len(local_steps)),
                    dtype=torch.long,
                    device=device,
                )
            )
            .cpu()
            .tolist()
        )
        all_cnt = self.accelerator.gather(cnt_t).cpu().tolist()

        entries = []
        off = 0
        for cnt in all_cnt:
            for j in range(cnt):
                entries.append((all_lat[off + j], all_gid[off + j], all_step[off + j]))
            off += max_cnt

        # 4. Per-group crossover — each rank only processes children from its
        #    own local groups so the actual parent Sample is always available.
        groups: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        for latent, gid, step in entries:
            groups[(gid, step)].append(latent)
        gid_to_parent_local: Dict[int, BaseSample] = {}
        for s in parent_samples:
            gid_to_parent_local.setdefault(s.unique_id, s)

        my_child_latents: List[torch.Tensor] = []
        my_child_steps: List[int] = []
        my_child_gids: List[int] = []
        my_child_metas: List[Dict[str, Any]] = []
        my_child_parents: List[BaseSample] = []

        for (gid, step), gl in groups.items():
            parent = gid_to_parent_local.get(gid)
            if parent is None or len(gl) < 2:
                continue
            gen = create_generator(self.training_args.seed + self.epoch + gid, device="cpu")
            out = self._crossover_strategy.crossover(torch.stack(gl), generator=gen)
            Mg = out.child_latents.shape[0]
            my_child_latents.append(out.child_latents)
            my_child_steps.extend([step] * Mg)
            my_child_gids.extend([gid] * Mg)
            my_child_parents.extend([parent] * Mg)
            for m in range(Mg):
                meta: Dict[str, Any] = {
                    "parent_i": int(out.parent_indices_i[m]),
                    "parent_j": int(out.parent_indices_j[m]),
                }
                for mk, mv in out.metadata.items():
                    meta[mk] = mv[m] if isinstance(mv, list) and len(mv) == Mg else mv
                my_child_metas.append(meta)

        if not my_child_latents:
            empty = {k: torch.zeros(0, device=device) for k in reward_keys}
            return [], empty

        my_child_latents_t = torch.cat(my_child_latents, dim=0).float()

        # 5. Local denoising — each child uses its own parent for trajectory
        #    and prompt data, no template needed.
        _p0 = parent_samples[0]
        timesteps = _p0.timesteps.to(device)

        my_children: List[BaseSample] = []
        for ci in tqdm(
            range(len(my_child_latents_t)),
            desc=f"Child denoising (rank {rank})",
            disable=not self.show_progress_bar,
        ):
            gid = my_child_gids[ci]
            step = my_child_steps[ci]
            parent = my_child_parents[ci]

            pooled = getattr(parent, "pooled_prompt_embeds", None)
            child_batch = {
                k: getattr(parent, k).to(device)
                for k in ("prompt_embeds", "prompt_ids")
                if getattr(parent, k, None) is not None
            }
            if pooled is not None:
                child_batch["pooled_prompt_embeds"] = pooled.to(device)
            missing = [
                k for k in ("prompt_embeds", "pooled_prompt_embeds") if k not in child_batch
            ]
            if missing:
                raise RuntimeError(
                    f"Child gid={gid} (ci={ci}) missing required prompt field(s) "
                    f"{missing} from parent group. "
                    f"Check that the parent sample stores pooled_prompt_embeds."
                )
            final, post_lat, post_lp, post_cb = run_denoising_phase(
                adapter=self.adapter,
                accelerator=self.accelerator,
                autocast_ctx=self.autocast,
                latents=my_child_latents_t[ci : ci + 1],
                timesteps=timesteps,
                start_idx=step,
                end_idx=num_steps,
                batch=child_batch,
                training_args=self.training_args,
                compute_log_prob=True,
                collect_trajectory=True,
                extra_call_back_kwargs=["next_latents_mean"],
                collect_callbacks=True,
            )
            imgs = self.adapter.decode_latents(final)
            child = self._build_child(
                parent=parent,
                post_latents=post_lat,
                post_log_probs=post_lp,
                post_callbacks=post_cb,
                image=imgs,
                cxo_step=step,
                num_steps=num_steps,
            )
            child.extra_kwargs["crossover_meta"] = my_child_metas[ci]
            child._unique_id = gid
            my_children.append(child)

        child_rewards_dict = self.reward_buffer.rp.compute_rewards(
            my_children, store_to_samples=False, split="pointwise"
        )
        my_children, child_rewards_dict = self._filter_children(
            my_children, my_child_gids, child_rewards_dict, g_rewards, g_ids
        )
        child_rewards = {
            k: torch.as_tensor(v, device=device) for k, v in child_rewards_dict.items()
        }

        logger.info(
            f"Crossover: {len(groups)} groups, {len(entries)} parents "
            f"→ {len(my_child_latents_t)} raw children, {len(my_children)} kept after filter (rank {rank})"
        )
        return my_children, child_rewards

    # ======================================================================
    # Child filtering

    def _filter_children(self, children, child_gids, child_rewards_dict, g_rewards, g_ids):
        """Keep only non-dominated children within each parent group."""
        if not children:
            return children, child_rewards_dict
        reward_keys = list(g_rewards.keys())
        child_gids_arr = np.array(child_gids, dtype=np.int64)
        keep = np.ones(len(children), dtype=bool)
        for gid in np.unique(child_gids_arr):
            c_idx = np.where(child_gids_arr == gid)[0]
            p_idx = np.where(g_ids == gid)[0]
            if len(p_idx) == 0 or len(c_idx) == 0:
                continue
            p_mat = np.stack([g_rewards[k][p_idx] for k in reward_keys], axis=1)
            c_mat = np.array([[child_rewards_dict[k][i] for k in reward_keys] for i in c_idx])
            combined = np.vstack([p_mat, c_mat])
            pareto = compute_pareto_mask(combined)
            keep[c_idx] = pareto[len(p_idx) :]
        filtered_children = [c for i, c in enumerate(children) if keep[i]]
        filtered_rewards = {k: v[keep] for k, v in child_rewards_dict.items()}
        return filtered_children, filtered_rewards

    # ======================================================================
    # Child trajectory construction
    # ======================================================================

    def _build_child(
        self, parent, post_latents, post_log_probs, post_callbacks, image, cxo_step, num_steps
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

        # Inherit all parent fields (including model-specific ones) via
        # to_dict/from_dict, overriding only the fields that differ.
        parent_dict = parent.to_dict()
        parent_dict["all_latents"] = merged_al
        parent_dict["latent_index_map"] = lm
        parent_dict["log_probs"] = merged_lp
        parent_dict["log_prob_index_map"] = lpm2
        parent_dict["image"] = image
        parent_dict["_unique_id"] = None  # recompute from child content
        parent_dict["applicable_rewards"] = set()

        extra = parent_dict.get("extra_kwargs", {})
        extra["is_crossover_child"] = True
        extra["crossover_step"] = cxo_step
        extra["crossover_strategy"] = self.training_args.crossover.strategy
        merged_nlm = _merge_cb("next_latents_mean")
        if merged_nlm is not None:
            extra["next_latents_mean"] = merged_nlm
        parent_dict["extra_kwargs"] = extra

        child = type(parent).from_dict(parent_dict)
        return child
