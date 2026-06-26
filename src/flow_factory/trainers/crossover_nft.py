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
from ..utils.base import create_generator
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
            self.advantage_processor._child_in_norm = True
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
                # parent groups — advantage grouping is keyed on unique_id.
                uids = [s.unique_id for s in samples]
                perm = sorted(range(len(samples)), key=lambda i: uids[i])
                samples[:] = [samples[i] for i in perm]
                rewards = {k: v[perm].to(device) for k, v in rewards.items()}

                # Remove neutral parents BEFORE advantage computation so
                # advantages are computed once on the final trimmed set.
                self._remove_neutral_parents(samples, rewards)

        self.advantage_processor._child_advantage_scale = 1.0

        self.compute_advantages(samples, rewards, store_to_samples=True)
        stats = self.advantage_processor.pop_all_stats()
        if stats:
            self.log_data(stats, step=self.step)

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
        cxo_cfg = self.training_args.crossover

        # 1. Build local reward arrays + unique_ids for Pareto & child filtering.
        #    In group_contiguous mode all K copies of a prompt share the same
        #    rank, so every group is complete locally — no communication needed.
        local_g_rewards = {k: torch.as_tensor(v).cpu().numpy() for k, v in parent_rewards.items()}
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
            # Apply mutation to initial crossover latents
            latents = out.child_latents.float()
            mutation_std = getattr(cxo_cfg, "mutation_std", 0.0)
            if mutation_std > 0:
                latents = latents + torch.randn_like(latents) * mutation_std
            child_groups.append((gid, step, parent, latents, metas))

        if not child_groups:
            empty = {k: torch.zeros(0, device=device) for k in reward_keys}
            return [], empty

        # 5. Multi-generation evolutionary loop.
        #    Evaluate with ODE (deterministic) for unbiased fitness comparison.
        #    Selection: layer-0 latents survive.  Mutation: Gaussian noise on latents.
        _p0 = parent_samples[0]
        sample_cls = type(_p0)
        n_stored = _p0.all_latents.shape[0]
        timesteps = _p0.timesteps.to(device)
        _shared_extra = dict(_p0.extra_kwargs) if _p0.extra_kwargs else {}
        n_generations = max(1, getattr(cxo_cfg, "evolution_generations", 1))

        # Current generation's latent groups: (gid, step, parent, latents, metas)
        current_groups: List[Tuple[int, int, BaseSample, torch.Tensor, List[Dict]]] = child_groups
        all_children: List[BaseSample] = []
        total_raw = 0
        total_filter_stats = {"layer0": 0, "dominated_by_all": 0, "discarded": 0}

        for gen_idx in range(n_generations):
            gen_children: List[BaseSample] = []
            gen_gids: List[int] = []
            n_raw = sum(lat.shape[0] for _, _, _, lat, _ in current_groups)

            for gid, step, parent, child_latents_batch, metas in tqdm(
                current_groups,
                desc=f"Child denoising gen {gen_idx} (rank {rank})",
                disable=not self.log_args.verbose,
                position=rank,
            ):
                child_batch = {
                    k: getattr(parent, k).to(device)
                    for k in ("prompt_embeds", "pooled_prompt_embeds", "prompt_ids")
                    if getattr(parent, k, None) is not None
                }
                Mg = child_latents_batch.shape[0]
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
                cross_latents = child_latents_batch.detach().cpu()
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
                        crossover_strategy=cxo_cfg.strategy,
                        crossover_meta=metas[m],
                        generation=gen_idx,
                    )
                    pooled = getattr(parent, "pooled_prompt_embeds", None)
                    if pooled is not None:
                        extra["pooled_prompt_embeds"] = pooled
                    extra["_cxo_latent"] = cross_latents[m]
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
                    gen_children.append(child)
                    gen_gids.append(gid)

            # Sync NPU/CUDA after denoising before reward computation
            if current_groups:
                if device.type == "npu" and hasattr(torch, "npu"):
                    torch.npu.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()

            # Evaluate this generation (handle empty for ranks with no children)
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
            # Full filter for training: keep layer-0 + dominated_by_all
            gen_kept, _, fstats = self._filter_children(
                gen_children, gen_gids, gen_rewards_dict, local_g_rewards, local_g_ids
            )
            for k in total_filter_stats:
                total_filter_stats[k] += fstats[k]

            # Build next generation from layer-0 children (including their latents)
            if gen_idx < n_generations - 1:
                gen_layer0, _, _ = self._filter_children_layer0(
                    gen_children, gen_gids, gen_rewards_dict, local_g_rewards, local_g_ids
                )
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
                    f"{fstats['layer0']} layer-0 + {fstats['dominated_by_all']} dominated "
                    f"= {len(gen_kept)} kept ({len(current_groups)} groups for next) "
                    f"(rank {rank})"
                )
            else:
                logger.info(
                    f"Evolution gen {gen_idx} (final): {n_before} children → "
                    f"{fstats['layer0']} layer-0 + {fstats['dominated_by_all']} dominated "
                    f"= {len(gen_kept)} kept, {fstats['discarded']} discarded (rank {rank})"
                )

            all_children.extend(gen_kept)
            total_raw += n_raw

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
        # Clean up internal keys that would break BaseSample.stack (parents
        # don't have them → mixed None/tensor in the same field).
        for c in all_children:
            c.extra_kwargs.pop("_cxo_latent", None)

        # Sync device stream — ensure all async denoising + reward ops
        # complete before the gather in downstream compute_advantages.
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        return all_children, child_rewards

    # ======================================================================
    # Child filtering

    def _filter_children(self, children, child_gids, child_rewards_dict, g_rewards, g_ids):
        """Keep frontier children and fully-dominated children.

        Layer 0 (non-dominated on combined parent+child set) expands the
        Pareto frontier → positive signal.  Children dominated by ALL parents
        (every parent >= child in every dimension, at least one >) → negative
        signal.  Middle children are discarded.
        """
        empty_stats = {"layer0": 0, "dominated_by_all": 0, "discarded": 0}
        if not children:
            return children, child_rewards_dict, empty_stats
        reward_keys = list(g_rewards.keys())
        child_gids_arr = np.array(child_gids, dtype=np.int64)
        keep = np.zeros(len(children), dtype=bool)
        n_layer0 = n_dominated_by_all = n_discarded = 0
        for gid in np.unique(child_gids_arr):
            c_idx = np.where(child_gids_arr == gid)[0]
            p_idx = np.where(g_ids == gid)[0]
            if len(p_idx) == 0 or len(c_idx) == 0:
                keep[c_idx] = True
                n_layer0 += len(c_idx)
                continue
            p_mat = np.stack([g_rewards[k][p_idx] for k in reward_keys], axis=1)  # (n_p, R)
            c_mat = np.array(
                [[child_rewards_dict[k][i] for k in reward_keys] for i in c_idx]
            )  # (n_c, R)
            combined = np.vstack([p_mat, c_mat])
            n_p = len(p_idx)

            # Layer 0: non-dominated on combined set
            all_pareto = compute_pareto_mask(combined)
            c_pareto = all_pareto[n_p:]  # (n_c,)

            # Children dominated by ALL parents:
            #   for every parent p: p >= c in all dims AND p > c in at least one dim
            #   → dominated by each parent individually
            dominated_by_all = np.ones(len(c_idx), dtype=bool)
            for p in p_mat:
                dom_by_p = (p >= c_mat).all(axis=1) & (p > c_mat).any(axis=1)
                dominated_by_all &= dom_by_p

            g_keep = c_pareto | dominated_by_all
            keep[c_idx] = g_keep
            n_layer0 += int(c_pareto.sum())
            n_dominated_by_all += int(dominated_by_all.sum())
            n_discarded += int((~g_keep).sum())

        filtered_children = [c for i, c in enumerate(children) if keep[i]]
        stats = {
            "layer0": n_layer0,
            "dominated_by_all": n_dominated_by_all,
            "discarded": n_discarded,
        }
        return filtered_children, {}, stats

    def _filter_children_layer0(self, children, child_gids, child_rewards_dict, g_rewards, g_ids):
        """Keep only layer-0 (non-dominated) children for re-crossover."""
        if not children:
            return (
                children,
                child_rewards_dict,
                {"layer0": 0, "dominated_by_all": 0, "discarded": 0},
            )
        reward_keys = list(g_rewards.keys())
        child_gids_arr = np.array(child_gids, dtype=np.int64)
        keep = np.zeros(len(children), dtype=bool)
        n_layer0 = 0
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
        filtered_children = [c for i, c in enumerate(children) if keep[i]]
        stats = {"layer0": n_layer0, "dominated_by_all": 0, "discarded": 0}
        return filtered_children, {}, stats

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
        # Group children's latents by (gid, step)
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
            # Elitism: include original parent latents so good genes never lost
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
            # Apply mutation after re-crossover
            mutation_std = getattr(self.training_args.crossover, "mutation_std", 0.0)
            if mutation_std > 0:
                latents = latents + torch.randn_like(latents) * mutation_std
            new_groups.append((gid, step, parent, latents, metas))

        return new_groups

    # ======================================================================
    # Neutral parent removal

    def _remove_neutral_parents(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> None:
        """Remove parents with |advantage| closest to 0 per group.

        Computes a lightweight per-group normalized advantage (weighted
        reward, parent-only mean/std).  For each group, removes N parents
        closest to zero advantage (dominated first, then non-dominated),
        where N equals the number of children in that group.

        Called BEFORE ``compute_advantages`` so the official advantage
        computation runs on the final trimmed set.
        """
        if not samples:
            return

        device = rewards[next(iter(rewards.keys()))].device
        reward_keys = sorted(rewards.keys())

        # Build masks
        is_child = torch.tensor(
            [s.extra_kwargs.get("is_crossover_child", False) for s in samples],
            dtype=torch.bool,
            device=device,
        )
        uids = torch.tensor(
            [s.unique_id for s in samples],
            dtype=torch.long,
            device=device,
        )

        # Weighted aggregated reward (matches GDPO weights)
        rw = self.advantage_processor.reward_weights
        agg = torch.zeros(len(samples), device=device)
        for k in reward_keys:
            w = next(iter(rw[k].values()))
            agg += rewards[k].to(device) * w

        remove = torch.zeros(len(samples), dtype=torch.bool, device=device)
        child_gids = uids[is_child].unique()

        for gid in child_gids:
            c_mask = is_child & (uids == gid)
            p_mask = ~is_child & (uids == gid)
            n_children = int(c_mask.sum().item())
            n_parents = int(p_mask.sum().item())

            if n_children == 0 or n_parents <= 1:
                continue

            n_remove = min(n_children, n_parents - 1)
            if n_remove <= 0:
                continue

            p_indices = p_mask.nonzero(as_tuple=True)[0]
            c_indices = c_mask.nonzero(as_tuple=True)[0]

            # Lightweight per-group normalized advantage (parent-only mean/std)
            p_agg = agg[p_indices]
            group_mean = p_agg.mean()
            group_std = p_agg.std()
            p_adv = (p_agg - group_mean) / (group_std + 1e-6)

            # Pareto mask on parents+children combined
            all_indices = torch.cat([p_indices, c_indices])
            all_rewards_np = (
                torch.stack([rewards[k][all_indices] for k in reward_keys], dim=1)
                .cpu()
                .float()
                .numpy()
            )
            all_pareto = compute_pareto_mask(all_rewards_np)
            all_pareto_t = torch.from_numpy(all_pareto).to(device)
            p_pareto_t = all_pareto_t[:n_parents]

            # Order: dominated first, then non-dominated.
            # Within each: sort by |advantage| ascending — closest to 0 removed first.
            dom_mask = ~p_pareto_t
            nondom_mask = p_pareto_t

            dominated_by_adv = p_indices[dom_mask][p_adv[dom_mask].abs().argsort()]
            nondom_by_adv = p_indices[nondom_mask][p_adv[nondom_mask].abs().argsort()]

            removal_order = torch.cat([dominated_by_adv, nondom_by_adv])
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
                f"Crossover: removed {n_removed} neutral parents to balance "
                f"{int(is_child.sum().item())} children (total now {len(samples)})"
            )

    # ======================================================================
    # Sample redistribution for balanced training
    # ======================================================================
