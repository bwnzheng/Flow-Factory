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
from typing import Any, Dict, List, Tuple

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
                # ---- Child count warmup: limit children in early epochs ----
                warmup_epochs = getattr(
                    self.training_args.crossover, "child_warmup_epochs", 0
                )
                if warmup_epochs > 0 and len(children) > 0:
                    ratio = min(1.0, self.epoch / max(warmup_epochs, 1))
                    target = max(1, int(len(children) * ratio))
                    if target < len(children):
                        idx = torch.randperm(len(children))[:target].tolist()
                        children = [children[i] for i in idx]
                        child_rewards = {k: v[idx] for k, v in child_rewards.items()}

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

                # Remove dominated parents so total sample count stays at the
                # original parent count — keeps gradient_accumulation_steps
                # correct and avoids wasted / lost gradients.
                self._remove_dominated_parents(samples, rewards)

        self.advantage_processor._child_advantage_scale = 1.0

        self.compute_advantages(samples, rewards, store_to_samples=True)
        stats = self.advantage_processor.pop_all_stats()
        if stats:
            self.log_data(stats, step=self.step)

        # Per-group parent removal ensures all ranks have the same sample
        # count — no padding needed.

    # ======================================================================
    # Crossover augmentation (local per-group → local denoising)
    # ======================================================================

    @torch.no_grad()
    def _crossover_augment(
        self, parent_samples: List[BaseSample], parent_rewards: Dict[str, torch.Tensor]
    ) -> tuple:
        device = self.accelerator.device
        num_steps = self.training_args.num_inference_steps
        rank = self.accelerator.process_index
        reward_keys = sorted(parent_rewards.keys())

        # 1. Build local reward arrays + unique_ids for Pareto & child filtering.
        #    In group_contiguous mode all K copies of a prompt share the same
        #    rank, so every group is complete locally — no communication needed.
        local_g_rewards = {
            k: torch.as_tensor(v).cpu().numpy() for k, v in parent_rewards.items()
        }
        local_g_ids = np.array([s.unique_id for s in parent_samples], dtype=np.int64)

        # 2. Non-dominated parents per group (local).
        if self._selective:
            nondom = select_non_dominated_parents(
                {k: torch.from_numpy(v) for k, v in local_g_rewards.items()},
                local_g_ids.tolist(),
            )
        else:
            nondom = np.ones(len(parent_samples), dtype=bool)

        # 3. Collect local non-dominated parent latents — no all_gather needed.
        gid_to_parent_local: Dict[int, BaseSample] = {}
        for s in parent_samples:
            gid_to_parent_local.setdefault(s.unique_id, s)

        entries: List[Tuple[torch.Tensor, int, int]] = []
        for i, s in enumerate(parent_samples):
            if nondom[i]:
                step = s.extra_kwargs["_cxo_step"]
                idx = int(s.latent_index_map[step])
                entries.append((s.all_latents[idx].to(device), s.unique_id, step))

        # 4. Per-group crossover (local only).
        groups: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        for latent, gid, step in entries:
            groups[(gid, step)].append(latent)

        # child_groups: (gid, step, parent, latents, metas) for each group
        # that produces children; latents are stacked → batch denoising.
        child_groups: List[Tuple[int, int, BaseSample, torch.Tensor, List[Dict[str, Any]]]] = []

        for (gid, step), gl in groups.items():
            parent = gid_to_parent_local.get(gid)
            if parent is None or len(gl) < 2:
                continue
            gen = create_generator(self.training_args.seed + self.epoch + gid, device="cpu")
            out = self._crossover_strategy.crossover(torch.stack(gl), generator=gen)
            Mg = out.child_latents.shape[0]
            metas: List[Dict[str, Any]] = []
            for m in range(Mg):
                meta: Dict[str, Any] = {
                    "parent_i": int(out.parent_indices_i[m]),
                    "parent_j": int(out.parent_indices_j[m]),
                }
                for mk, mv in out.metadata.items():
                    meta[mk] = mv[m] if isinstance(mv, list) and len(mv) == Mg else mv
                metas.append(meta)
            child_groups.append((gid, step, parent, out.child_latents.float(), metas))

        if not child_groups:
            empty = {k: torch.zeros(0, device=device) for k in reward_keys}
            return [], empty

        # 5. Batched denoising — children from the same (gid, step) share
        #    prompt data and denoising start, so they are batched together.
        _p0 = parent_samples[0]
        sample_cls = type(_p0)
        n_stored = _p0.all_latents.shape[0]
        timesteps = _p0.timesteps.to(device)
        _shared_extra = dict(_p0.extra_kwargs) if _p0.extra_kwargs else {}

        my_children: List[BaseSample] = []
        my_child_gids_flat: List[int] = []
        total_children = sum(lat.shape[0] for _, _, _, lat, _ in child_groups)

        for gid, step, parent, child_latents_batch, metas in tqdm(
            child_groups,
            desc=f"Child denoising (rank {rank})",
            disable=not self.log_args.verbose,
            position=rank,
        ):
            child_batch = {
                k: getattr(parent, k).to(device)
                for k in ("prompt_embeds", "pooled_prompt_embeds", "prompt_ids")
                if getattr(parent, k, None) is not None
            }
            with self.sampling_context():
                finals, _, _, _ = run_denoising_phase(
                    adapter=self.adapter,
                    accelerator=self.accelerator,
                    autocast_ctx=self.autocast,
                    latents=child_latents_batch,
                    timesteps=timesteps,
                    start_idx=step,
                    end_idx=num_steps,
                    batch=child_batch,
                    training_args=self.training_args,
                    compute_log_prob=False,
                    collect_trajectory=False,
                )
            Mg = child_latents_batch.shape[0]
            for m in range(Mg):
                final = finals[m : m + 1]
                imgs = self.adapter.decode_latents(final)
                al = final.expand(n_stored, *final.shape[1:]).clone()
                lmap = torch.full((num_steps + 1,), -1, dtype=torch.long, device=device)
                lmap[-1] = n_stored - 1

                extra = dict(_shared_extra)
                extra.update(
                    is_crossover_child=True,
                    crossover_step=step,
                    crossover_strategy=self.training_args.crossover.strategy,
                    crossover_meta=metas[m],
                )
                pooled = getattr(parent, "pooled_prompt_embeds", None)
                if pooled is not None:
                    extra["pooled_prompt_embeds"] = pooled
                child = sample_cls(
                    timesteps=timesteps,
                    all_latents=al,
                    latent_index_map=lmap,
                    image=imgs,
                    log_probs=None,
                    log_prob_index_map=None,
                    prompt=parent.prompt,
                    prompt_ids=parent.prompt_ids,
                    prompt_embeds=parent.prompt_embeds,
                    negative_prompt=parent.negative_prompt,
                    _unique_id=gid,
                    applicable_rewards=set(),
                    extra_kwargs=extra,
                )
                my_children.append(child)
                my_child_gids_flat.append(gid)

        child_rewards_dict = self.reward_buffer.rp.compute_rewards(
            my_children, store_to_samples=False, split="pointwise"
        )
        my_children, child_rewards_dict = self._filter_children(
            my_children, my_child_gids_flat, child_rewards_dict, local_g_rewards, local_g_ids
        )
        child_rewards = {
            k: torch.as_tensor(v, device=device) for k, v in child_rewards_dict.items()
        }

        logger.info(
            f"Crossover: {len(groups)} groups, {len(entries)} parents "
            f"→ {total_children} raw children, {len(my_children)} kept after filter (rank {rank})"
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

    # ======================================================================
    # Dominated parent removal

    def _remove_dominated_parents(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> None:
        """Remove dominated parents so total count stays at original parent count.

        For each group, removes the N lowest-reward parents where N equals the
        number of children in that group.  Parents on the Pareto front are
        removed last — dominated parents are removed first.

        This keeps ``len(samples)`` close to the original parent count,
        preventing gradient accumulation steps from becoming misaligned.
        """
        if not samples:
            return

        device = rewards[next(iter(rewards.keys()))].device
        reward_keys = sorted(rewards.keys())

        # Build masks
        is_child = torch.tensor(
            [s.extra_kwargs.get("is_crossover_child", False) for s in samples],
            dtype=torch.bool, device=device,
        )
        uids = torch.tensor(
            [s.unique_id for s in samples], dtype=torch.long, device=device,
        )

        # Aggregated reward (simple sum for ranking)
        agg = torch.zeros(len(samples), device=device)
        for k in reward_keys:
            agg += rewards[k].to(device)

        remove = torch.zeros(len(samples), dtype=torch.bool, device=device)
        child_gids = uids[is_child].unique()

        for gid in child_gids:
            c_mask = is_child & (uids == gid)
            p_mask = ~is_child & (uids == gid)
            n_children = int(c_mask.sum().item())
            n_parents = int(p_mask.sum().item())

            if n_children == 0 or n_parents <= 1:
                # Keep at least 1 parent per group
                continue

            n_remove = min(n_children, n_parents - 1)
            if n_remove <= 0:
                continue

            p_indices = p_mask.nonzero(as_tuple=True)[0]
            p_rewards = agg[p_indices]

            # Compute Pareto mask on parents to remove dominated first
            p_rewards_np = torch.stack([rewards[k][p_indices] for k in reward_keys], dim=1).cpu().float().numpy()
            p_pareto = compute_pareto_mask(p_rewards_np)
            p_pareto_t = torch.from_numpy(p_pareto).to(device)

            # Order: dominated parents first, then non-dominated (by lowest reward)
            dominated_idx = p_indices[~p_pareto_t]
            nondom_idx = p_indices[p_pareto_t]

            # Sort each subset by reward ascending (remove worst first)
            dominated_sorted = dominated_idx[agg[dominated_idx].argsort()]
            nondom_sorted = nondom_idx[agg[nondom_idx].argsort()]

            # Build ordered removal list
            removal_order = torch.cat([dominated_sorted, nondom_sorted])
            for idx in removal_order[:n_remove]:
                remove[idx] = True

        keep = ~remove
        keep_indices = keep.nonzero(as_tuple=True)[0].tolist()
        if len(keep_indices) < len(samples):
            samples[:] = [samples[i] for i in keep_indices]
            for k in rewards:
                rewards[k] = rewards[k][keep_indices].to(device)

            n_removed = int(remove.sum().item())
            logger.info(
                f"Crossover: removed {n_removed} dominated parents to balance "
                f"{int(is_child.sum().item())} children (total now {len(samples)})"
            )

    # ======================================================================
    # Sample redistribution for balanced training
    # ======================================================================