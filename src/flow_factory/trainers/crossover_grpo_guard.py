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
from typing import Any, Dict, List, Optional, Tuple

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
                # ---- Child count warmup: limit children in early epochs ----
                warmup_epochs = getattr(self.training_args.crossover, "child_warmup_epochs", 0)
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

        # Build child_groups with batched latents for batch denoising.
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

        # 5. Multi-generation evolutionary loop.
        #    Each generation: denoise → evaluate → select layer-0 survivors →
        #    re-crossover with parent latents (elitism) for next generation.
        _p0 = parent_samples[0]
        timesteps = _p0.timesteps.to(device)
        n_generations = max(1, getattr(self.training_args.crossover, "evolution_generations", 1))

        current_groups = child_groups
        all_children: List[BaseSample] = []
        all_child_gids: List[int] = []
        total_raw = 0
        total_filter_stats = {"layer0": 0, "dominated_by_all": 0, "discarded": 0}

        for gen_idx in range(n_generations):
            gen_children: List[BaseSample] = []
            gen_gids: List[int] = []
            n_raw = sum(lat.shape[0] for _, _, _, lat, _ in current_groups)

            for gid, step, parent, child_latents_batch, metas in tqdm(
                current_groups,
                desc=f"Child denoising gen {gen_idx} (rank {rank})",
                disable=not self.show_progress_bar,
            ):
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
                        f"Child gid={gid} missing required prompt field(s) "
                        f"{missing} from parent group. "
                        f"Check that the parent sample stores pooled_prompt_embeds."
                    )

                Mg = child_latents_batch.shape[0]
                finals, post_lat, post_lp, post_cb = run_denoising_phase(
                    adapter=self.adapter,
                    accelerator=self.accelerator,
                    autocast_ctx=self.autocast,
                    latents=child_latents_batch,
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

                # Slice batched outputs per-child for individual Sample construction.
                cross_latents = child_latents_batch.detach().cpu()
                for m in range(Mg):
                    imgs = self.adapter.decode_latents(finals[m : m + 1])
                    child_post_lat = [lat[m : m + 1] for lat in post_lat]
                    child_post_lp = [lp[m : m + 1] for lp in post_lp]
                    child_post_cb = (
                        {k: [cb[m : m + 1] for cb in v] for k, v in post_cb.items()}
                        if post_cb
                        else None
                    )
                    child = self._build_child(
                        parent=parent,
                        post_latents=child_post_lat,
                        post_log_probs=child_post_lp,
                        post_callbacks=child_post_cb,
                        image=imgs,
                        cxo_step=step,
                        num_steps=num_steps,
                        cxo_latent=cross_latents[m],
                    )
                    child.extra_kwargs["crossover_meta"] = metas[m]
                    child._unique_id = gid
                    gen_children.append(child)
                    gen_gids.append(gid)

            # Sync device after denoising before reward computation
            if current_groups:
                if device.type == "npu" and hasattr(torch, "npu"):
                    torch.npu.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()

            # Evaluate this generation
            if gen_children:
                gen_rewards_dict = self.reward_buffer.rp.compute_rewards(
                    gen_children, store_to_samples=False, split="pointwise"
                )
                if device.type == "npu" and hasattr(torch, "npu"):
                    torch.npu.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
            else:
                gen_rewards_dict = {k: np.array([]) for k in reward_keys}

            n_before = len(gen_children)
            gen_kept, gen_kept_rewards = self._filter_children(
                gen_children, gen_gids, gen_rewards_dict, g_rewards, g_ids
            )

            # Build next generation from layer-0 survivors
            if gen_idx < n_generations - 1:
                gen_layer0, _, fstats = self._filter_children_layer0(
                    gen_children, gen_gids, gen_rewards_dict, g_rewards, g_ids
                )
                for k in total_filter_stats:
                    total_filter_stats[k] += fstats[k]
                current_groups = (
                    self._build_child_crossover_groups(
                        gen_layer0, [c.unique_id for c in gen_layer0], gid_to_parent_local
                    )
                    if gen_layer0
                    else []
                )
                if not current_groups:
                    break
                logger.info(
                    f"Evolution gen {gen_idx}: {n_before} children → "
                    f"{fstats['layer0']} layer-0, {fstats['discarded']} discarded "
                    f"({len(current_groups)} groups for next) (rank {rank})"
                )
            else:
                logger.info(
                    f"Evolution gen {gen_idx} (final): {n_before} children → "
                    f"{len(gen_kept)} kept after filter (rank {rank})"
                )

            all_children.extend(gen_kept)
            all_child_gids.extend([c.unique_id for c in gen_kept])
            total_raw += n_raw

        # Clean up internal keys before reward computation
        for c in all_children:
            c.extra_kwargs.pop("_cxo_latent", None)

        if all_children:
            child_rewards_dict = self.reward_buffer.rp.compute_rewards(
                all_children, store_to_samples=False, split="pointwise"
            )
            child_rewards = {
                k: torch.as_tensor(v, device=device) for k, v in child_rewards_dict.items()
            }
        else:
            child_rewards = {k: torch.zeros(0, device=device) for k in reward_keys}

        logger.info(
            f"Crossover: {len(groups)} groups, {len(entries)} parents "
            f"→ {total_raw} raw children ({n_generations} gens), {len(all_children)} kept "
            f"(layer0={total_filter_stats['layer0']}, "
            f"dominated_by_all={total_filter_stats['dominated_by_all']}, "
            f"discarded={total_filter_stats['discarded']}) (rank {rank})"
        )

        # Sync device stream before downstream gather in compute_advantages
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        return all_children, child_rewards

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

    def _filter_children_layer0(self, children, child_gids, child_rewards_dict, g_rewards, g_ids):
        """Keep only layer-0 (non-dominated) children for next-gen re-crossover."""
        if not children:
            return children, child_gids, {"layer0": 0, "dominated_by_all": 0, "discarded": 0}
        reward_keys = list(g_rewards.keys())
        child_gids_arr = np.array(child_gids, dtype=np.int64)
        keep = np.zeros(len(children), dtype=bool)
        n_layer0 = 0
        n_discarded = 0
        for gid in np.unique(child_gids_arr):
            c_idx = np.where(child_gids_arr == gid)[0]
            p_idx = np.where(g_ids == gid)[0]
            if len(p_idx) == 0:
                keep[c_idx] = True
                n_layer0 += len(c_idx)
                continue
            p_mat = np.stack([g_rewards[k][p_idx] for k in reward_keys], axis=1)
            c_mat = np.array([[child_rewards_dict[k][i] for k in reward_keys] for i in c_idx])
            combined = np.vstack([p_mat, c_mat])
            n_p = len(p_idx)
            all_pareto = compute_pareto_mask(combined)
            c_pareto = all_pareto[n_p:]
            keep[c_idx] = c_pareto
            n_layer0 += int(c_pareto.sum())
            n_discarded += int((~c_pareto).sum())
        filtered_children = [c for i, c in enumerate(children) if keep[i]]
        filtered_gids = [g for i, g in enumerate(child_gids) if keep[i]]
        stats = {"layer0": n_layer0, "dominated_by_all": 0, "discarded": n_discarded}
        return filtered_children, filtered_gids, stats

    def _build_child_crossover_groups(
        self,
        children: List[BaseSample],
        child_gids: List[int],
        gid_to_parent: Dict[int, BaseSample],
    ) -> List[Tuple[int, int, BaseSample, torch.Tensor, List[Dict]]]:
        """Re-crossover surviving children's latents for the next generation.

        Groups children by (gid, step), adds parent latent for elitism,
        crosses the pool, and returns new child_groups for denoising.
        """
        device = self.accelerator.device
        latent_groups: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
        for child, gid in zip(children, child_gids):
            step = child.extra_kwargs.get("crossover_step")
            latent = child.extra_kwargs.get("_cxo_latent")
            if step is not None and latent is not None:
                latent_groups[(gid, step)].append(latent.to(device))

        new_groups: List[Tuple[int, int, BaseSample, torch.Tensor, List[Dict]]] = []
        for (gid, step), gl in latent_groups.items():
            parent = gid_to_parent.get(gid)
            if parent is None:
                continue
            # Elitism: include original parent latent so good genes are never lost
            parent_idx = int(parent.latent_index_map[step])
            parent_latent = parent.all_latents[parent_idx].to(device)
            gl = [parent_latent] + gl
            if len(gl) < 2:
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
            latents = out.child_latents.float()
            new_groups.append((gid, step, parent, latents, metas))

        return new_groups

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
        if cxo_latent is not None:
            extra["_cxo_latent"] = cxo_latent.detach().cpu()
        merged_nlm = _merge_cb("next_latents_mean")
        if merged_nlm is not None:
            extra["next_latents_mean"] = merged_nlm
        parent_dict["extra_kwargs"] = extra

        child = type(parent).from_dict(parent_dict)
        return child
