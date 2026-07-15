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
    applicable = GeneticAlgorithm.build_applicable_mask(samples, rewards)
    evolved_samples, evolved_rewards = ga.evolve(
        parent_samples=samples,
        parent_rewards=rewards,
        applicable=applicable,
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
    2. Generate children by *offspring_mode*:

       - ``"crossover"`` — crossover parent latents + optional Gaussian mutation
       - ``"resample"``  — pure random noise, no parents involved
       - ``"mutation"``  — clone a single parent + Gaussian mutation, no crossover
    3. Denoise children → compute rewards
    4. Merge population → keep non-dominated (Pareto front expanders)
    5. Fill back to K by keeping dominated samples with largest |advantage|

    Args:
        crossover_strategy: Pluggable crossover strategy (used only in
            ``offspring_mode="crossover"``).
        offspring_mode: How to generate children:
            ``"crossover"``, ``"resample"``, or ``"mutation"``.
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
        offspring_mode: str = "crossover",
        reward_weights: Optional[Dict[str, Dict[str, float]]] = None,
        seed: int = 42,
        denoise_kwargs: Optional[Dict[str, Any]] = None,
        child_factory: Optional[callable] = None,
    ) -> None:
        # Strategy
        self._strategy = crossover_strategy
        self._parent_ratio = max(0.0, min(1.0, float(parent_ratio)))
        self._mutation_std = float(mutation_std)
        self._n_generations = max(1, int(evolution_generations))
        self._offspring_mode = offspring_mode
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

        # Denoising and child creation
        self._denoise_kwargs = denoise_kwargs or {}
        self._child_factory = child_factory or self._default_child_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._accelerator.device

    # ------------------------------------------------------------------
    # Applicable mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_applicable_mask(
        samples: List[BaseSample],
        reward_keys: List[str],
    ) -> np.ndarray:
        """Build ``(R, S)`` boolean applicable mask from ``sample.applicable_rewards``.

        This is the authoritative source of truth for which reward model
        applies to which sample — the same mask used by
        :class:`AdvantageProcessor` for reward aggregation.

        Args:
            samples: All parent samples (any ordering / group assignment).
            reward_keys: Ordered list of reward names (matching axis 0).

        Returns:
            ``(R, S)`` boolean array where ``mask[r, s]`` is True iff
            ``reward_keys[r]`` is in ``samples[s].applicable_rewards``.
        """
        R, S = len(reward_keys), len(samples)
        mask = np.zeros((R, S), dtype=bool)
        if R == 0 or S == 0:
            return mask
        rk_to_idx = {rk: i for i, rk in enumerate(reward_keys)}
        for s_idx, s in enumerate(samples):
            for rk in s.applicable_rewards:
                idx = rk_to_idx.get(rk)
                if idx is not None:
                    mask[idx, s_idx] = True
        return mask

    @torch.no_grad()
    def evolve(
        self,
        parent_samples: List[BaseSample],
        parent_rewards: Dict[str, torch.Tensor],
        epoch: int,
        applicable: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Tuple[List[BaseSample], Dict[str, torch.Tensor], Dict[str, Any], List[Dict[str, Any]]]:
        """Run GA on all groups and return the evolved population.

        Args:
            parent_samples: All parent samples across all groups on this rank.
            parent_rewards: ``{reward_name: tensor(S,)}`` — per-reward scores
                for every parent sample (NaN at non-applicable positions).
            epoch: Current training epoch (used as RNG seed component).
            applicable: Optional ``(R, S)`` boolean mask from
                :meth:`build_applicable_mask`.  When provided, per-group valid
                reward keys are derived from this mask; when ``None``, the
                GA falls back to treating *all* global reward keys as valid
                (single-source / homogeneous training).

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

        local_g_rewards = {k: torch.as_tensor(v).cpu().numpy() for k, v in parent_rewards.items()}

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
        ga_samples: List[Dict[str, Any]] = []

        gid_items = sorted(gid_to_indices.items())
        if verbose and rank == 0:
            gid_items = list(tqdm(gid_items, desc=f"GA evolve (rank {rank})", position=rank))

        for gid, indices in gid_items:
            population = [parent_samples[i] for i in indices]
            pop_rewards = {k: local_g_rewards[k][indices].copy() for k in reward_keys}
            acc["n_groups"] += 1

            # ---- Determine valid reward keys for this group -------------
            # All samples in a group share the same source, so we consult
            # the first sample's applicable_rewards (set by RewardProcessor).
            if applicable is not None:
                # Applicable mask supplied: use it to derive per-group validity.
                # A reward is valid for this group if it applies to *any*
                # sample (all share source → either all or none apply).
                group_applicable = applicable[:, indices]
                valid_reward_keys = [
                    rk for r_idx, rk in enumerate(reward_keys) if group_applicable[r_idx].any()
                ]
            else:
                # Legacy / single-source path: all global reward keys are valid.
                valid_reward_keys = list(reward_keys)

            for gen_idx in range(self._n_generations):
                ctx.gid = gid
                ctx.gen_idx = gen_idx
                population, pop_rewards, stats = self._run_generation(
                    population=population,
                    pop_rewards=pop_rewards,
                    reward_keys=reward_keys,
                    valid_reward_keys=valid_reward_keys,
                    epoch=epoch,
                    ctx=ctx,
                )
                if stats is None:
                    break

                # ---- Log to console ----
                _logged_keys = sorted(
                    set(stats["pop_rewards"].keys())
                    | set(stats["child_rewards"].keys())
                    | set(stats["new_rewards"].keys())
                )
                rw_lines = "  ".join(
                    f"{k}: pop {stats['pop_rewards'][k]['mean']:.3f}→"
                    f"{stats['new_rewards'][k]['mean']:.3f}"
                    f" | child {stats['child_rewards'][k]['mean']:.3f}"
                    for k in _logged_keys
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
                for k in _logged_keys:
                    pop_m = stats["pop_rewards"][k]["mean"]
                    pop_s = stats["pop_rewards"][k]["std"]
                    child_m = stats["child_rewards"][k]["mean"]
                    child_s = stats["child_rewards"][k]["std"]
                    new_m = stats["new_rewards"][k]["mean"]
                    new_s = stats["new_rewards"][k]["std"]
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
                    ga_samples.append(
                        {
                            "gen": gen_idx,
                            "gid": int(gid),
                            "rank": int(rank),
                            "sample_idx": si,
                            "is_child": bool(
                                population[si].extra_kwargs.get("is_crossover_child", False)
                            ),
                            "rewards": {k: float(pop_rewards[k][si]) for k in reward_keys},
                        }
                    )

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
        valid_reward_keys: List[str],
        epoch: int,
        ctx: _EvolveCtx,
    ) -> Tuple[
        List[BaseSample],
        Dict[str, np.ndarray],
        Optional[Dict[str, Any]],
    ]:
        """One GA generation: select → crossover → denoise → evaluate → filter.

        *reward_keys* is the global list (used for dict keys).  *valid_reward_keys*
        is the subset that actually applies to this group's source; only these
        participate in advantage computation and Pareto filtering.

        Returns ``(new_population, new_rewards, stats)``.  *stats* is None
        when there aren't enough parents.
        """
        # 1. Compute advantage (only on valid reward dimensions)
        adv = self._compute_advantage(pop_rewards, valid_reward_keys)

        # 2–3. Generate children by offspring mode
        device = self.device

        if self._offspring_mode == "resample":
            # ---- Resample: pure random noise, no parents ----
            template_latent = self._get_crossover_latent(population[0], device)
            n_children = self._strategy.num_children(len(population))
            child_latents = self._resample_children(
                batch_size=n_children,
                latent_shape=template_latent.shape,
                dtype=template_latent.dtype,
                rng_seed=epoch + ctx.gid + ctx.gen_idx,
            )
        elif self._offspring_mode == "mutation":
            # ---- Mutation-only: clone single parent + noise, no crossover ----
            parent_idx, n_parents = self._select_parents(adv, pop_rewards, valid_reward_keys)
            if parent_idx is None:
                return population, pop_rewards, None
            parent_latents = torch.stack(
                [self._get_crossover_latent(population[pi], device) for pi in parent_idx]
            )
            child_latents = self._mutate_only(parent_latents, epoch + ctx.gid + ctx.gen_idx)
        else:
            # ---- Crossover (default): two-parent crossover + optional mutation ----
            parent_idx, n_parents = self._select_parents(adv, pop_rewards, valid_reward_keys)
            if parent_idx is None:
                return population, pop_rewards, None
            parent_latents = torch.stack(
                [self._get_crossover_latent(population[pi], device) for pi in parent_idx]
            )
            child_latents = self._crossover_and_mutate(
                parent_latents, epoch + ctx.gid + ctx.gen_idx
            )

        # 4. Denoise → child samples.
        #    Crossover / mutation start from cxo_step (mid-denoising).
        #    Resample starts from step 0 (full noise → full denoising,
        #    same as original sampling).
        if self._offspring_mode == "resample":
            denoise_start = 0
        else:
            denoise_start = _resolve_cxo_step(population[0], self._num_steps)

        children = self._denoise_and_create_children(
            child_latents=child_latents,
            cxo_step=denoise_start,
            template=population[0],
            ctx=ctx,
        )

        # 5. Evaluate children
        child_rewards_dict_raw = self._reward_buffer.rp.compute_rewards(
            children, store_to_samples=False, split="pointwise"
        )
        child_rewards_dict = {k: v.cpu().numpy() for k, v in child_rewards_dict_raw.items()}
        self._device_sync()

        # 6. Select survivors (advantage computed on merged set internally)
        population, pop_rewards, stats = self._select_survivors(
            population=population,
            children=children,
            pop_rewards=pop_rewards,
            child_rewards=child_rewards_dict,
            reward_keys=reward_keys,
            valid_reward_keys=valid_reward_keys,
        )

        return population, pop_rewards, stats

    # ------------------------------------------------------------------
    # Step 1–2: Parent selection + crossover + mutation
    # ------------------------------------------------------------------

    def _select_parents(
        self,
        adv: np.ndarray,
        pop_rewards: Dict[str, np.ndarray],
        valid_reward_keys: List[str],
    ) -> Tuple[Optional[np.ndarray], int]:
        """Select parents: non-dominated first (on valid dimensions), then by advantage."""
        if not valid_reward_keys:
            return None, 0

        n_parents = max(2, int(len(adv) * self._parent_ratio))
        if n_parents < 2:
            return None, 0

        # Pareto mask on current population — only valid reward dimensions
        stack = np.stack([pop_rewards[k].astype(np.float32) for k in valid_reward_keys], axis=1)
        pareto = compute_pareto_mask(stack)

        # Non-dominated first, sorted by advantage descending
        nondom_idx = np.where(pareto)[0]
        nondom_idx = nondom_idx[np.argsort(adv[nondom_idx])[::-1]]

        selected = list(nondom_idx[:n_parents])

        # Fill remaining from dominated, by advantage descending
        if len(selected) < n_parents:
            dom_idx = np.where(~pareto)[0]
            dom_idx = dom_idx[np.argsort(adv[dom_idx])[::-1]]
            selected.extend(dom_idx[: n_parents - len(selected)])

        return np.array(selected), n_parents

    def _crossover_and_mutate(self, parent_latents: torch.Tensor, rng_seed: int) -> torch.Tensor:
        """Apply crossover strategy + Gaussian mutation."""
        gen_rng = torch.Generator()
        gen_rng.manual_seed(rng_seed)
        out = self._strategy.crossover(parent_latents, generator=gen_rng)
        child_latents = out.child_latents.float()
        if self._mutation_std > 0:
            child_latents = child_latents + torch.randn_like(child_latents) * self._mutation_std
        return child_latents

    def _resample_children(
        self,
        batch_size: int,
        latent_shape: torch.Size,
        dtype: torch.dtype,
        rng_seed: int,
    ) -> torch.Tensor:
        """Generate children from pure random noise (no parents involved).

        Args:
            batch_size: Number of children ``M``.
            latent_shape: Per-sample shape ``(C, H, W)`` or ``(L, D)``.
            dtype: Data type of the latents.
            rng_seed: Seed for reproducibility.

        Returns:
            Random noise tensor of shape ``(M, *latent_shape)``.
        """
        device = self.device
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(rng_seed)
        shape = (batch_size, *latent_shape)
        child_latents = torch.randn(shape, device=device, dtype=dtype, generator=gen_rng)
        return child_latents

    def _mutate_only(self, parent_latents: torch.Tensor, rng_seed: int) -> torch.Tensor:
        """Clone a single parent + Gaussian mutation (no crossover).

        Each child is a noisy copy of one randomly selected parent.
        The ``mutation_std`` controls the noise magnitude; a warning is
        emitted if it is zero (children would be identical clones).

        Args:
            parent_latents: Parent latents, shape ``(K, *latent_dims)``.
            rng_seed: Seed for reproducibility.

        Returns:
            Mutated child latents of shape ``(M, *latent_dims)``.
        """
        K = parent_latents.shape[0]
        M = self._strategy.num_children(K)
        gen_rng = torch.Generator()
        gen_rng.manual_seed(rng_seed)
        # Randomly select one parent for each child
        pick = torch.randint(0, K, (M,), generator=gen_rng)
        child_latents = parent_latents[pick].clone()
        # Mutation
        std = self._mutation_std
        if std <= 0:
            std = 0.05
            logger.warning(
                f"offspring_mode='mutation' but mutation_std={self._mutation_std}. "
                f"Falling back to mutation_std={std} to avoid producing identical clones."
            )
        noise = torch.randn_like(child_latents)
        child_latents = child_latents + noise * std
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
        """Denoise child latents and create samples via child_factory."""
        device = self.device

        # Build the denoising batch from the template sample.
        # Must also check extra_kwargs because subclasses like SD3_5Sample
        # declare pooled_prompt_embeds as a direct dataclass field (default
        # None), which shadows the extra_kwargs fallback of __getattr__.
        child_batch: Dict[str, Any] = {}
        for k in ("prompt_embeds", "pooled_prompt_embeds", "prompt_ids"):
            val = getattr(template, k, None)
            if val is None:
                val = template.extra_kwargs.get(k)
            if val is not None:
                child_batch[k] = val.to(device)
            elif k == "pooled_prompt_embeds":
                logger.warning(
                    f"_denoise_and_create_children: template is missing "
                    f"'{k}'.  The adapter may fail if it requires it."
                )

        timesteps = template.timesteps.to(device)
        raw = run_denoising_phase(
            adapter=self._adapter,
            accelerator=self._accelerator,
            autocast_ctx=self._autocast,
            latents=child_latents,
            timesteps=timesteps,
            start_idx=cxo_step,
            end_idx=self._num_steps,
            batch=child_batch,
            training_args=self._training_args,
            compute_log_prob=self._denoise_kwargs.get("compute_log_prob", False),
            collect_trajectory=self._denoise_kwargs.get("collect_trajectory", False),
            extra_call_back_kwargs=self._denoise_kwargs.get("extra_call_back_kwargs"),
            collect_callbacks=self._denoise_kwargs.get("collect_callbacks", False),
        )

        self._device_sync()

        return self._child_factory(
            template=template,
            child_latents=child_latents,
            cxo_step=cxo_step,
            denoise_output=raw,
            ctx=ctx,
        )

    def _default_child_factory(
        self,
        template: BaseSample,
        child_latents: torch.Tensor,
        cxo_step: int,
        denoise_output: tuple,
        ctx: _EvolveCtx,
    ) -> List[BaseSample]:
        """Default child factory — NFT-style (no log_probs, no trajectory).

        Uses ``template.to_dict()`` to inherit all fields from the parent,
        then overrides only the fields that differ.  This is the same
        pattern used by ``_grpo_child_factory`` and guarantees that
        parent ↔ child field parity is always maintained.
        """
        device = child_latents.device
        finals, _, _, _ = denoise_output
        n_children = child_latents.shape[0]
        cross_latents_cpu = child_latents.detach().cpu()
        children: List[BaseSample] = []

        for m in range(n_children):
            final = finals[m : m + 1]
            imgs = self._adapter.decode_latents(final)
            al = final.expand(ctx.n_stored, *final.shape[1:]).clone()
            lmap = torch.full((ctx.n_stored,), -1, dtype=torch.long, device=device)
            lmap[-1] = ctx.n_stored - 1

            # Inherit everything from the template, then override.
            child_dict = template.to_dict()
            child_dict["all_latents"] = al
            child_dict["latent_index_map"] = lmap
            child_dict["image"] = imgs
            child_dict["log_probs"] = None
            child_dict["log_prob_index_map"] = None
            child_dict["applicable_rewards"] = set()
            child_dict["_unique_id"] = ctx.gid

            extra = child_dict.get("extra_kwargs", {})
            extra["is_crossover_child"] = True
            extra["crossover_step"] = cxo_step
            extra["crossover_strategy"] = ctx.strategy_name
            extra["generation"] = ctx.gen_idx
            extra["_cxo_latent"] = cross_latents_cpu[m]
            child_dict["extra_kwargs"] = extra

            child = type(template).from_dict(child_dict)
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
        reward_keys: List[str],
        valid_reward_keys: List[str],
    ) -> Tuple[
        List[BaseSample],
        Dict[str, np.ndarray],
        Dict[str, Any],
    ]:
        """Merge population + children, compute unified advantage, trim to K.

        Advantage is computed *after* merging so all K+M samples share the
        same normalization (combined mean/std).  Pareto and |advantage|
        trimming use only *valid_reward_keys*; *reward_keys* is the full
        global set for dict iteration and stats bookkeeping.
        """
        n_pop = len(population)
        n_children = len(children)

        # Merge rewards first
        combined_rewards: Dict[str, np.ndarray] = {}
        for k in reward_keys:
            combined_rewards[k] = np.concatenate([pop_rewards[k], child_rewards[k]])

        # Compute advantage on the FULL combined set (unified normalization)
        combined_adv = self._compute_advantage(combined_rewards, valid_reward_keys)

        # ---- Reward stats before replacement ----
        pop_rw_stats = {
            k: {"mean": float(pop_rewards[k].mean()), "std": float(pop_rewards[k].std())}
            for k in valid_reward_keys
        }
        child_rw_stats = {
            k: {"mean": float(child_rewards[k].mean()), "std": float(child_rewards[k].std())}
            for k in valid_reward_keys
        }

        # ---- Pareto mask (valid dimensions only) ----
        if valid_reward_keys:
            stack = np.stack(
                [combined_rewards[k].astype(np.float32) for k in valid_reward_keys],
                axis=1,
            )
            pareto = compute_pareto_mask(stack)
        else:
            pareto = np.ones(len(combined_adv), dtype=bool)

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
        new_rewards = {k: combined_rewards[k][keep_indices].copy() for k in reward_keys}

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

        # ---- Reward stats after replacement (valid dimensions only) ----
        new_rw_stats = {
            k: {
                "mean": float(new_rewards[k].mean()),
                "std": float(new_rewards[k].std()),
            }
            for k in valid_reward_keys
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
        step = sample.extra_kwargs.get("_cxo_step") or sample.extra_kwargs.get("crossover_step")
        if step is not None and hasattr(sample, "latent_index_map"):
            idx = int(sample.latent_index_map[step])
            return sample.all_latents[idx].to(device)
        return sample.all_latents[-1].to(device)

    def _compute_advantage(
        self,
        rewards_dict: Dict[str, np.ndarray],
        valid_reward_keys: List[str],
    ) -> np.ndarray:
        """Per-group GDPO-style advantage (valid dimensions only).

        Only *valid_reward_keys* participate; all-NaN columns from
        non-applicable rewards are already excluded by the caller.
        """
        if not valid_reward_keys:
            if not rewards_dict:
                return np.array([])
            n = len(next(iter(rewards_dict.values())))
            return np.zeros(n, dtype=np.float32)

        n = len(rewards_dict[valid_reward_keys[0]])
        if n == 0:
            return np.array([])

        agg = np.zeros(n, dtype=np.float32)
        for key in valid_reward_keys:
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

    # ------------------------------------------------------------------
    # Stats reduction (shared across trainers)
    # ------------------------------------------------------------------

    @staticmethod
    def reduce_stats(
        ga_acc: Dict[str, Any],
        ga_samples: List[Dict[str, Any]],
        accelerator: Any,
    ) -> Dict[str, Any]:
        """Reduce per-rank GA accumulators across ranks, build final stats."""
        num_ranks = accelerator.num_processes

        max_gen = 0
        while f"gen{max_gen}_count" in ga_acc:
            max_gen += 1

        reward_keys = sorted(
            {
                k[len("gen0_") : -len("_pop_sum")]
                for k in ga_acc
                if k.startswith("gen0_") and k.endswith("_pop_sum")
            }
        )

        count_keys = ["n_groups"]
        for gen in range(max_gen):
            count_keys.append(f"gen{gen}_count")
            for key in [
                "n_replaced",
                "n_children",
                "n_children_kept",
                "n_pareto_parents",
                "n_pareto_children",
                "n_filled",
            ]:
                count_keys.append(f"gen{gen}_{key}")
            for rk in reward_keys:
                for suffix in [
                    "pop_sum",
                    "pop_sum_sq",
                    "child_sum",
                    "child_sum_sq",
                    "new_sum",
                    "new_sum_sq",
                ]:
                    count_keys.append(f"gen{gen}_{rk}_{suffix}")

        values = [float(ga_acc.get(k, 0)) for k in count_keys]
        t = torch.tensor(values, device=accelerator.device, dtype=torch.float32)

        if num_ranks > 1:
            t = accelerator.reduce(t, reduction="sum")

        reduced: Dict[str, float] = {}
        for i, k in enumerate(count_keys):
            reduced[k] = t[i].item()

        stats: Dict[str, Any] = {"ga/n_groups": int(reduced["n_groups"])}
        for gen in range(max_gen):
            count = reduced[f"gen{gen}_count"]
            if count == 0:
                continue
            p = f"ga/gen{gen}"
            for key in [
                "n_replaced",
                "n_children",
                "n_children_kept",
                "n_pareto_parents",
                "n_pareto_children",
                "n_filled",
            ]:
                stats[f"{p}/{key}"] = round(reduced[f"gen{gen}_{key}"] / count, 2)

            for rk in reward_keys:
                for prefix, sum_key, sum_sq_key in [
                    ("pop_mean", "pop_sum", "pop_sum_sq"),
                    ("child_mean", "child_sum", "child_sum_sq"),
                    ("new_mean", "new_sum", "new_sum_sq"),
                ]:
                    s = reduced[f"gen{gen}_{rk}_{sum_key}"]
                    sq = reduced[f"gen{gen}_{rk}_{sum_sq_key}"]
                    n_eff = count
                    if "child" in sum_key:
                        n_child = max(reduced.get(f"gen{gen}_n_children", 1.0), 1.0)
                        n_eff = n_child
                    mean = s / max(n_eff, 1.0)
                    var = max(sq / max(n_eff, 1.0) - mean**2, 0.0)
                    if "mean" in prefix:
                        stats[f"{p}/{rk}/{prefix}"] = round(mean, 6)
                    else:
                        stats[f"{p}/{rk}/{prefix.replace('mean', 'std')}"] = round(var**0.5, 6)

        if ga_samples:
            stats["ga/samples"] = ga_samples

        return stats
