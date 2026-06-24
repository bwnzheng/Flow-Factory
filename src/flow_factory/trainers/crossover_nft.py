# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

# src/flow_factory/trainers/crossover_nft.py
"""
CrossoverNFT — DiffusionNFT trainer with global-group crossover augmentation.

Parents generated during ``sample()`` store crossover-step latents.  In
``prepare_feedback()``, rewards are gathered globally, non-dominated parents
identified per group, and crossover children generated with equal share across
all ranks.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm as tqdm_

from ..hparams import CrossoverNFTTrainingArguments
from ..samples import BaseSample
from ..utils.base import create_generator, filter_kwargs
from ..utils.logger_utils import setup_logger
from .crossover import (
    compute_pareto_mask,
    create_crossover_strategy,
    run_denoising_phase,
    sample_crossover_step,
    select_non_dominated_parents,
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
                f"CrossoverNFT: strategy={cxo_args.strategy} selective={self._selective} "
                f"include_parents={self._include_parents}"
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
    # Crossover augmentation (global gather → per-group → distributed)
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

        # 1. Gather rewards + unique_ids globally (same pattern as collect_group_rewards)
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

        # 2. Non-dominated parents per group (global)
        if self._selective:
            nondom = select_non_dominated_parents(
                {k: torch.from_numpy(v) for k, v in g_rewards.items()}, g_ids.tolist()
            )
        else:
            nondom = np.ones(len(g_ids), dtype=bool)

        # 3. All_gather (latent, gid, step) of non-dominated parents
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

        entries = []  # (latent, gid, step)
        off = 0
        for cnt in all_cnt:
            for j in range(cnt):
                entries.append((all_lat[off + j], all_gid[off + j], all_step[off + j]))
            off += max_cnt

        # 4. Per-group crossover
        groups: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        for latent, gid, step in entries:
            groups[(gid, step)].append(latent)

        child_latents, child_steps, child_gids, child_metas = [], [], [], []
        for (gid, step), gl in groups.items():
            if len(gl) < 2:
                continue
            gen = create_generator(self.training_args.seed + self.epoch + gid, device="cpu")
            out = self._crossover_strategy.crossover(torch.stack(gl), generator=gen)
            Mg = out.child_latents.shape[0]
            child_latents.append(out.child_latents)
            child_steps.extend([step] * Mg)
            child_gids.extend([gid] * Mg)
            # Per-child provenance: parent indices + strategy params / sampled values
            for m in range(Mg):
                meta: Dict[str, Any] = {
                    "parent_i": int(out.parent_indices_i[m]),
                    "parent_j": int(out.parent_indices_j[m]),
                }
                for mk, mv in out.metadata.items():
                    meta[mk] = mv[m] if isinstance(mv, list) and len(mv) == Mg else mv
                child_metas.append(meta)

        if not child_latents:
            empty = {k: torch.zeros(0, device=device) for k in reward_keys}
            return [], empty

        child_latents = torch.cat(child_latents, dim=0).float()
        M = child_latents.shape[0]

        # 5. Distributed denoising: each rank generates ceil(M/W) children
        chunk = int(np.ceil(M / world))
        my_children: List[BaseSample] = []
        tpl = parent_samples[0]
        n_stored = tpl.all_latents.shape[0]
        timesteps = tpl.timesteps.to(device)
        batch = {
            k: getattr(tpl, k, None).to(device)
            for k in ("prompt_embeds", "prompt_ids", "pooled_prompt_embeds")
            if getattr(tpl, k, None) is not None
        }

        for i in tqdm(
            range(chunk),
            desc=f"Child denoising (rank {rank})",
            disable=not self.show_progress_bar,
        ):
            ci = min(rank * chunk + i, M - 1)
            step = child_steps[ci]
            with self.sampling_context():
                final, _, _, _ = run_denoising_phase(
                    adapter=self.adapter,
                    accelerator=self.accelerator,
                    autocast_ctx=self.autocast,
                    latents=child_latents[ci : ci + 1],
                    timesteps=timesteps,
                    start_idx=step,
                    end_idx=num_steps,
                    batch=batch,
                    training_args=self.training_args,
                    compute_log_prob=False,
                    collect_trajectory=False,
                )
            imgs = self.adapter.decode_latents(final)
            al = final.expand(n_stored, *final.shape[1:]).clone()
            lmap = torch.full((num_steps + 1,), -1, dtype=torch.long, device=device)
            lmap[-1] = n_stored - 1
            # Inherit all parent fields (including model-specific ones like
            # pooled_prompt_embeds) via to_dict/from_dict, overriding only
            # the fields that differ for the child.
            parent_dict = tpl.to_dict()
            parent_dict["all_latents"] = al
            parent_dict["latent_index_map"] = lmap
            parent_dict["image"] = imgs
            parent_dict["log_probs"] = None
            parent_dict["log_prob_index_map"] = None
            parent_dict["_unique_id"] = child_gids[ci]  # assign to actual parent group
            parent_dict["applicable_rewards"] = set()

            extra = parent_dict.get("extra_kwargs", {})
            extra["is_crossover_child"] = True
            extra["crossover_step"] = step
            extra["crossover_strategy"] = self.training_args.crossover.strategy
            extra["crossover_meta"] = child_metas[ci]
            parent_dict["extra_kwargs"] = extra

            child = type(tpl).from_dict(parent_dict)
            my_children.append(child)

        child_rewards_dict = self.reward_buffer.rp.compute_rewards(
            my_children, store_to_samples=False, split="pointwise"
        )
        # Build local_child_gids matching the same ci formula used in the
        # denoising loop above.  _filter_children uses these to index into
        # my_children / child_rewards_dict (both local, length ≈ chunk).
        local_child_gids = [
            child_gids[min(rank * chunk + i, M - 1)] for i in range(len(my_children))
        ]
        # Keep only non-dominated children (per group, vs parents)
        my_children, child_rewards_dict = self._filter_children(
            my_children, local_child_gids, child_rewards_dict, g_rewards, g_ids
        )
        child_rewards = {
            k: torch.as_tensor(v, device=device) for k, v in child_rewards_dict.items()
        }

        logger.info(
            f"Crossover: {len(groups)} groups, {len(entries)} parents "
            f"→ {M} raw children, {len(my_children)} kept after filter (rank {rank})"
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
            # Build combined matrix: parents first, then children
            p_mat = np.stack([g_rewards[k][p_idx] for k in reward_keys], axis=1)
            c_mat = np.array([[child_rewards_dict[k][i] for k in reward_keys] for i in c_idx])
            combined = np.vstack([p_mat, c_mat])
            pareto = compute_pareto_mask(combined)
            # Children are the last len(c_idx) entries
            keep[c_idx] = pareto[len(p_idx) :]
        filtered_children = [c for i, c in enumerate(children) if keep[i]]
        filtered_rewards = {k: v[keep] for k, v in child_rewards_dict.items()}
        return filtered_children, filtered_rewards
