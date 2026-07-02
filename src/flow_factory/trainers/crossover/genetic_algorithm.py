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

# src/flow_factory/trainers/crossover/genetic_algorithm.py
"""
Genetic Algorithm for per-group population evolution.

Replaces the old Pareto-parent crossover + multi-generation re-crossover
with a true GA: select top parents by advantage, crossover + mutation,
filter by Pareto expansion, trim by |advantage| to maintain group size K.

Usage::

    ga = GeneticAlgorithm(
        crossover_strategy=strategy,
        parent_ratio=0.25,
        mutation_std=0.05,
        evolution_generations=3,
        reward_weights=advantage_processor.reward_weights,
        adapter=adapter,
        accelerator=accelerator,
        autocast=autocast,
        training_args=training_args,
        reward_buffer=reward_buffer,
        seed=42,
    )
    evolved_samples, evolved_rewards = ga.evolve(
        parent_samples=samples,
        parent_rewards=rewards,
        epoch=epoch,
    )
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm as tqdm_

from ...samples import BaseSample
from ...utils.logger_utils import setup_logger
from .abc import BaseCrossover
from .pareto import compute_pareto_mask
from .sampling import run_denoising_phase

tqdm = tqdm_.tqdm
logger = setup_logger(__name__)


# ============================================================================
# Helpers
# ============================================================================


@dataclass
class _EvolveCtx:
    """Immutable context shared across groups and generations."""

    sample_cls: type
    n_stored: int
    shared_extra: Dict[str, Any]
    strategy_name: str
    gid: int = 0
    gen_idx: int = 0


def _resolve_cxo_step(sample: BaseSample, num_steps: int) -> int:
    """Extract the crossover step from a sample's extra_kwargs."""
    step = sample.extra_kwargs.get("_cxo_step")
    if step is not None:
        return step
    step = sample.extra_kwargs.get("crossover_step")
    if step is not None:
        return step
    return num_steps // 2


# ============================================================================
# Genetic Algorithm
# ============================================================================


class GeneticAlgorithm:
    """Per-group genetic algorithm for latent-space population evolution.

    Each group (K samples sharing a prompt) evolves independently:

    1. Compute advantage → select top *parent_ratio* as parents
    2. Crossover parent latents + Gaussian mutation → M children
    3. Denoise children → compute rewards
    4. Merge population → keep non-dominated (Pareto front expanders)
    5. Fill back to K by keeping dominated samples with largest |advantage|

    Args:
        crossover_strategy: Pluggable crossover strategy.
        parent_ratio: Fraction of group selected as parents (0–1).
        mutation_std: Gaussian noise stddev applied to child latents.
        evolution_generations: Number of GA generations.
        reward_weights: ``{reward_key: {source: weight}}`` dict.
        adapter: Model adapter for denoising.
        accelerator: HF Accelerate instance.
        autocast: Mixed-precision autocast context.
        training_args: Training arguments (for num_inference_steps, etc.).
        reward_buffer: Reward buffer for computing child rewards.
        seed: Base random seed.
    """

    def __init__(
        self,
        crossover_strategy: BaseCrossover,
        adapter: Any,
        accelerator: Any,
        autocast: Any,
        training_args: Any,
        reward_buffer: Any,
        parent_ratio: float = 0.25,
        mutation_std: float = 0.0,
        evolution_generations: int = 1,
        reward_weights: Optional[Dict[str, Dict[str, float]]] = None,
        seed: int = 42,
    ) -> None:
        # Strategy
        self._strategy = crossover_strategy
        self._parent_ratio = max(0.0, min(1.0, float(parent_ratio)))
        self._mutation_std = float(mutation_std)
        self._n_generations = max(1, int(evolution_generations))
        self._reward_weights = reward_weights or {}

        # Environment (constant across epochs)
        self._adapter = adapter
        self._accelerator = accelerator
        self._autocast = autocast
        self._training_args = training_args
        self._reward_buffer = reward_buffer
        self._seed = seed

        # Derived constants
        self._num_steps: int = training_args.num_inference_steps
        self._group_size: int = training_args.group_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._accelerator.device

    @torch.no_grad()
    def evolve(
        self,
        parent_samples: List[BaseSample],
        parent_rewards: Dict[str, torch.Tensor],
        epoch: int,
        verbose: bool = True,
    ) -> Tuple[List[BaseSample], Dict[str, torch.Tensor], Dict[str, Any], List[Dict[str, Any]]]:
        """Run GA on all groups and return the evolved population.

        Returns:
            ``(evolved_samples, evolved_rewards, ga_stats, ga_samples)``.
            *ga_stats* is a dict of per-generation accumulators (to be
            reduced across ranks).  *ga_samples* is a list of per-sample
            reward records.
        """
        t_start = time.time()
        reward_keys = sorted(parent_rewards.keys())
        rank = self._accelerator.process_index
        device = self.device

        # Group samples by unique_id
        gid_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, s in enumerate(parent_samples):
            gid_to_indices[s.unique_id].append(i)

        local_g_rewards = {
            k: torch.as_tensor(v).cpu().numpy() for k, v in parent_rewards.items()
        }

        # Pre-compute shared context
        _p0 = parent_samples[0]
        ctx = _EvolveCtx(
            sample_cls=type(_p0),
            n_stored=_p0.all_latents.shape[0],
            shared_extra=dict(_p0.extra_kwargs) if _p0.extra_kwargs else {},
            strategy_name=getattr(
                getattr(self._training_args, "crossover", None), "strategy", "unknown"
            ),
        )

        all_evolved: List[BaseSample] = []
        all_evolved_rewards: Dict[str, List[float]] = {k: [] for k in reward_keys}

        # Accumulate stats locally on this rank.
        # Counts: summed across groups, then reduced across ranks.
        # Reward moments: (sum, sum_sq, count) per (gen, key) for weighted averaging.
        acc: Dict[str, Any] = {"n_groups": 0}
        for gen in range(self._n_generations):
            acc[f"gen{gen}_count"] = 0
            for k in reward_keys:
                acc[f"gen{gen}_{k}_pop_sum"] = 0.0
                acc[f"gen{gen}_{k}_pop_sum_sq"] = 0.0
                acc[f"gen{gen}_{k}_child_sum"] = 0.0
                acc[f"gen{gen}_{k}_child_sum_sq"] = 0.0
                acc[f"gen{gen}_{k}_new_sum"] = 0.0
                acc[f"gen{gen}_{k}_new_sum_sq"] = 0.0
            acc[f"gen{gen}_n_replaced"] = 0
            acc[f"gen{gen}_n_children"] = 0
            acc[f"gen{gen}_n_children_kept"] = 0
            acc[f"gen{gen}_n_pareto_parents"] = 0
            acc[f"gen{gen}_n_pareto_children"] = 0
            acc[f"gen{gen}_n_filled"] = 0
        # Per-sample rewards: list of (gen, gid, sample_idx, rewards_dict) records
        ga_samples: List[Dict[str, Any]] = []

        gid_items = sorted(gid_to_indices.items())
        if verbose and rank == 0:
            gid_items = list(
                tqdm(gid_items, desc=f"GA evolve (rank {rank})", position=rank)
            )

        for gid, indices in gid_items:
            population = [parent_samples[i] for i in indices]
            pop_rewards = {k: local_g_rewards[k][indices].copy() for k in reward_keys}
            acc["n_groups"] += 1

            for gen_idx in range(self._n_generations):
                ctx.gid = gid
                ctx.gen_idx = gen_idx
                population, pop_rewards, stats = self._run_generation(
                    population=population,
                    pop_rewards=pop_rewards,
                    reward_keys=reward_keys,
                    epoch=epoch,
                    ctx=ctx,
                )
                if stats is None:
                    break

                # ---- Log to console ----
                rw_lines = "  ".join(
                    f"{k}: pop {stats['pop_rewards'][k]['mean']:.3f}→"
                    f"{stats['new_rewards'][k]['mean']:.3f}"
                    f" | child {stats['child_rewards'][k]['mean']:.3f}"
                    for k in reward_keys
                )
                logger.info(
                    f"[rank {rank}] GA gid={gid} gen={gen_idx}: "
                    f"pop={stats['n_pop']} "
                    f"replaced={stats['n_replaced']}/{stats['n_pop']} "
                    f"(children_kept={stats['n_children_kept']}/{stats['n_children']}, "
                    f"pareto={stats['n_pareto_parents']}+{stats['n_pareto_children']}, "
                    f"filled={stats['n_filled']}) | "
                    f"{rw_lines}"
                )

                # ---- Accumulate aggregate stats ----
                acc[f"gen{gen_idx}_count"] += 1
                acc[f"gen{gen_idx}_n_replaced"] += stats["n_replaced"]
                acc[f"gen{gen_idx}_n_children"] += stats["n_children"]
                acc[f"gen{gen_idx}_n_children_kept"] += stats["n_children_kept"]
                acc[f"gen{gen_idx}_n_pareto_parents"] += stats["n_pareto_parents"]
                acc[f"gen{gen_idx}_n_pareto_children"] += stats["n_pareto_children"]
                acc[f"gen{gen_idx}_n_filled"] += stats["n_filled"]
                n_pop = float(stats["n_pop"])
                for k in reward_keys:
                    pop_m = stats["pop_rewards"][k]["mean"]
                    pop_s = stats["pop_rewards"][k]["std"]
                    child_m = stats["child_rewards"][k]["mean"]
                    child_s = stats["child_rewards"][k]["std"]
                    new_m = stats["new_rewards"][k]["mean"]
                    new_s = stats["new_rewards"][k]["std"]
                    # Accumulate moments: sum, sum_sq
                    acc[f"gen{gen_idx}_{k}_pop_sum"] += pop_m * n_pop
                    acc[f"gen{gen_idx}_{k}_pop_sum_sq"] += (pop_s**2 + pop_m**2) * n_pop
                    n_child = float(stats["n_children"])
                    if n_child > 0:
                        acc[f"gen{gen_idx}_{k}_child_sum"] += child_m * n_child
                        acc[f"gen{gen_idx}_{k}_child_sum_sq"] += (child_s**2 + child_m**2) * n_child
                    acc[f"gen{gen_idx}_{k}_new_sum"] += new_m * n_pop
                    acc[f"gen{gen_idx}_{k}_new_sum_sq"] += (new_s**2 + new_m**2) * n_pop

                # ---- Record per-sample rewards ----
                for si in range(len(population)):
                    ga_samples.append({
                        "gen": gen_idx,
                        "gid": int(gid),
                        "rank": int(rank),
                        "sample_idx": si,
                        "is_child": bool(population[si].extra_kwargs.get("is_crossover_child", False)),
                        "rewards": {k: float(pop_rewards[k][si]) for k in reward_keys},
                    })

            all_evolved.extend(population)
            for k in reward_keys:
                all_evolved_rewards[k].extend(pop_rewards[k].tolist())

        # Clean up internal key
        for s in all_evolved:
            s.extra_kwargs.pop("_cxo_latent", None)

        elapsed = time.time() - t_start
        logger.info(
            f"[rank {rank}] GA: {len(gid_to_indices)} groups → "
            f"{len(all_evolved)} evolved samples, elapsed {elapsed:.1f}s"
        )

        evolved_rewards_tensors = {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            for k, v in all_evolved_rewards.items()
        }
        return all_evolved, evolved_rewards_tensors, acc, ga_samples

    # ------------------------------------------------------------------
    # Generation step
    # ------------------------------------------------------------------

    def _run_generation(
        self,
        population: List[BaseSample],
        pop_rewards: Dict[str, np.ndarray],
        reward_keys: List[str],
        epoch: int,
        ctx: _EvolveCtx,
    ) -> Tuple[
        List[BaseSample],
        Dict[str, np.ndarray],
        Optional[Dict[str, Any]],
    ]:
        """One GA generation: select → crossover → denoise → evaluate → filter.

        Returns ``(new_population, new_rewards, stats)``.  *stats* is None
        when there aren't enough parents.
        """
        n_pop = len(population)

        # 1. Compute advantage
        adv = self._compute_advantage(pop_rewards, reward_keys)

        # 2. Select parents (non-dominated first, then by advantage)
        parent_idx, n_parents = self._select_parents(adv, pop_rewards, reward_keys)
        if parent_idx is None:
            return population, pop_rewards, None

        # 3. Crossover + mutation
        device = self.device
        parent_latents = torch.stack(
            [self._get_crossover_latent(population[pi], device) for pi in parent_idx]
        )
        child_latents = self._crossover_and_mutate(
            parent_latents, epoch + ctx.gid + ctx.gen_idx
        )

        # 4. Denoise → child samples
        cxo_step = _resolve_cxo_step(population[0], self._num_steps)
        children = self._denoise_and_create_children(
            child_latents=child_latents,
            cxo_step=cxo_step,
            template=population[0],
            ctx=ctx,
        )

        # 5. Evaluate children
        child_rewards_dict_raw = self._reward_buffer.rp.compute_rewards(
            children, store_to_samples=False, split="pointwise"
        )
        child_rewards_dict = {
            k: v.cpu().numpy() for k, v in child_rewards_dict_raw.items()
        }
        self._device_sync()

        # 6. Select survivors
        population, pop_rewards, stats = self._select_survivors(
            population=population,
            children=children,
            pop_rewards=pop_rewards,
            child_rewards=child_rewards_dict,
            pop_adv=adv,
            reward_keys=reward_keys,
        )

        return population, pop_rewards, stats

    # ------------------------------------------------------------------
    # Step 1–2: Parent selection + crossover + mutation
    # ------------------------------------------------------------------

    def _select_parents(
        self,
        adv: np.ndarray,
        pop_rewards: Dict[str, np.ndarray],
        reward_keys: List[str],
    ) -> Tuple[Optional[np.ndarray], int]:
        """Select parents: non-dominated first, then by advantage."""
        n_parents = max(2, int(len(adv) * self._parent_ratio))
        if n_parents < 2:
            return None, 0

        # Pareto mask on current population
        stack = np.stack(
            [pop_rewards[k].astype(np.float32) for k in reward_keys], axis=1
        )
        pareto = compute_pareto_mask(stack)

        # Non-dominated first, sorted by advantage descending
        nondom_idx = np.where(pareto)[0]
        nondom_idx = nondom_idx[np.argsort(adv[nondom_idx])[::-1]]

        selected = list(nondom_idx[:n_parents])

        # Fill remaining from dominated, by advantage descending
        if len(selected) < n_parents:
            dom_idx = np.where(~pareto)[0]
            dom_idx = dom_idx[np.argsort(adv[dom_idx])[::-1]]
            selected.extend(dom_idx[:n_parents - len(selected)])

        return np.array(selected), n_parents

    def _crossover_and_mutate(
        self, parent_latents: torch.Tensor, rng_seed: int
    ) -> torch.Tensor:
        """Apply crossover strategy + Gaussian mutation."""
        gen_rng = torch.Generator()
        gen_rng.manual_seed(rng_seed)
        out = self._strategy.crossover(parent_latents, generator=gen_rng)
        child_latents = out.child_latents.float()
        if self._mutation_std > 0:
            child_latents = (
                child_latents
                + torch.randn_like(child_latents) * self._mutation_std
            )
        return child_latents

    # ------------------------------------------------------------------
    # Step 3: Denoise → child samples
    # ------------------------------------------------------------------

    def _denoise_and_create_children(
        self,
        child_latents: torch.Tensor,
        cxo_step: int,
        template: BaseSample,
        ctx: _EvolveCtx,
    ) -> List[BaseSample]:
        """Denoise child latents to images and wrap as BaseSample objects."""
        device = self.device

        child_batch = {
            k: getattr(template, k).to(device)
            for k in ("prompt_embeds", "pooled_prompt_embeds", "prompt_ids")
            if getattr(template, k, None) is not None
        }

        timesteps = template.timesteps.to(device)
        finals, _, _, _ = run_denoising_phase(
            adapter=self._adapter,
            accelerator=self._accelerator,
            autocast_ctx=self._autocast,
            latents=child_latents,
            timesteps=timesteps,
            start_idx=cxo_step,
            end_idx=self._num_steps,
            batch=child_batch,
            training_args=self._training_args,
            compute_log_prob=False,
            collect_trajectory=False,
        )

        self._device_sync()

        n_children = child_latents.shape[0]
        cross_latents_cpu = child_latents.detach().cpu()
        children: List[BaseSample] = []

        for m in range(n_children):
            final = finals[m : m + 1]
            imgs = self._adapter.decode_latents(final)
            al = final.expand(ctx.n_stored, *final.shape[1:]).clone()
            lmap = torch.full(
                (self._num_steps + 1,), -1, dtype=torch.long, device=device
            )
            lmap[-1] = ctx.n_stored - 1

            extra = dict(ctx.shared_extra)
            extra.update(
                is_crossover_child=True,
                crossover_step=cxo_step,
                crossover_strategy=ctx.strategy_name,
                generation=ctx.gen_idx,
            )
            pooled = getattr(template, "pooled_prompt_embeds", None)
            if pooled is not None:
                extra["pooled_prompt_embeds"] = pooled
            extra["_cxo_latent"] = cross_latents_cpu[m]

            child = ctx.sample_cls(
                timesteps=timesteps,
                all_latents=al,
                latent_index_map=lmap,
                image=imgs,
                log_probs=None,
                log_prob_index_map=None,
                prompt=template.prompt,
                prompt_ids=template.prompt_ids,
                prompt_embeds=template.prompt_embeds,
                negative_prompt=template.negative_prompt,
                _unique_id=ctx.gid,
                applicable_rewards=set(),
                extra_kwargs=extra,
            )
            children.append(child)

        return children

    # ------------------------------------------------------------------
    # Step 4: Select survivors (Pareto + |advantage| trim)
    # ------------------------------------------------------------------

    def _select_survivors(
        self,
        population: List[BaseSample],
        children: List[BaseSample],
        pop_rewards: Dict[str, np.ndarray],
        child_rewards: Dict[str, np.ndarray],
        pop_adv: np.ndarray,
        reward_keys: List[str],
    ) -> Tuple[
        List[BaseSample],
        Dict[str, np.ndarray],
        Dict[str, Any],
    ]:
        """Merge population + children, keep non-dominated, trim to K."""
        n_pop = len(population)
        n_children = len(children)

        # ---- Reward stats before replacement ----
        pop_rw_stats = {
            k: {"mean": float(pop_rewards[k].mean()), "std": float(pop_rewards[k].std())}
            for k in reward_keys
        }

        # Compute child advantages
        child_adv = self._compute_advantage(child_rewards, reward_keys)

        # Merge
        combined_adv = np.concatenate([pop_adv, child_adv])
        combined_rewards: Dict[str, np.ndarray] = {}
        for k in reward_keys:
            combined_rewards[k] = np.concatenate([pop_rewards[k], child_rewards[k]])

        # ---- Child reward stats ----
        child_rw_stats = {
            k: {"mean": float(child_rewards[k].mean()), "std": float(child_rewards[k].std())}
            for k in reward_keys
        }

        # Pareto mask
        stack = np.stack(
            [combined_rewards[k].astype(np.float32) for k in reward_keys], axis=1
        )
        pareto = compute_pareto_mask(stack)

        # Keep non-dominated
        keep = pareto.copy()
        n_pareto = int(keep.sum())

        # Fill to group_size with dominated by |advantage|
        K = self._group_size
        n_filled = 0
        if n_pareto < K:
            dominated = ~keep
            dom_idx = np.where(dominated)[0]
            if len(dom_idx) > 0:
                dom_adv_abs = np.abs(combined_adv[dom_idx])
                n_fill = min(K - n_pareto, len(dom_idx))
                fill_order = dom_idx[dom_adv_abs.argsort()[::-1][:n_fill]]
                keep[fill_order] = True
                n_filled = n_fill

        # Trim to K
        keep_indices = np.where(keep)[0][:K]
        n_keep_final = len(keep_indices)

        # Build new population
        combined_pop = population + children
        new_population = [combined_pop[ci] for ci in keep_indices]
        new_rewards = {
            k: combined_rewards[k][keep_indices].copy() for k in reward_keys
        }

        assert n_keep_final == n_pop, (
            f"GA population size mismatch: {n_pop} in, {n_keep_final} out. "
            f"n_pareto={n_pareto}, n_filled={n_filled}, "
            f"n_children={n_children}, group_size={K}"
        )

        # ---- Breakdown of survivors ----
        n_parents_kept = int((keep_indices < n_pop).sum())
        n_children_kept = n_keep_final - n_parents_kept
        n_pop_replaced = n_pop - n_parents_kept
        n_pareto_children = int(pareto[n_pop:].sum())
        n_pareto_parents = int(pareto[:n_pop].sum())

        # ---- Reward stats after replacement ----
        new_rw_stats = {
            k: {"mean": float(new_rewards[k].mean()), "std": float(new_rewards[k].std())}
            for k in reward_keys
        }

        stats = {
            "n_pop": n_pop,
            "n_keep": n_keep_final,
            "n_parents_kept": n_parents_kept,
            "n_replaced": n_pop_replaced,
            "n_children": n_children,
            "n_children_kept": n_children_kept,
            "n_pareto_parents": n_pareto_parents,
            "n_pareto_children": n_pareto_children,
            "n_filled": n_filled,
            "pop_rewards": pop_rw_stats,
            "child_rewards": child_rw_stats,
            "new_rewards": new_rw_stats,
        }
        return new_population, new_rewards, stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_crossover_latent(sample: BaseSample, device: torch.device) -> torch.Tensor:
        """Get a sample's latent at its crossover step.

        Original parents: ``all_latents[latent_index_map[step]]``.
        Children from previous generations: ``extra_kwargs['_cxo_latent']``.
        """
        cxo_latent = sample.extra_kwargs.get("_cxo_latent")
        if cxo_latent is not None:
            return cxo_latent.to(device)
        step = sample.extra_kwargs.get("_cxo_step") or sample.extra_kwargs.get(
            "crossover_step"
        )
        if step is not None and hasattr(sample, "latent_index_map"):
            idx = int(sample.latent_index_map[step])
            return sample.all_latents[idx].to(device)
        return sample.all_latents[-1].to(device)

    def _compute_advantage(
        self,
        rewards_dict: Dict[str, np.ndarray],
        reward_keys: List[str],
    ) -> np.ndarray:
        """Per-group GDPO-style advantage. Local only, no cross-rank comm."""
        if not reward_keys:
            return np.array([])

        n = len(next(iter(rewards_dict.values())))
        if n == 0:
            return np.array([])

        agg = np.zeros(n, dtype=np.float32)
        for key in reward_keys:
            w = next(iter(self._reward_weights.get(key, {"default": 1.0}).values()))
            vals = rewards_dict[key].astype(np.float32)
            mean = vals.mean()
            std = vals.std()
            if std > 1e-8 and n > 1:
                vals = (vals - mean) / std
            agg += vals * w

        return agg.astype(np.float32)

    def _device_sync(self) -> None:
        """Synchronize CUDA/NPU stream."""
        device = self.device
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
