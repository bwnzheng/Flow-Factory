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
filter by Pareto expansion, then apply the configured fixed-size environmental
selection rule to maintain group size K.

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

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm as tqdm_

from ...samples import BaseSample
from ...utils.base import filter_kwargs, move_tensors_to_device
from ...utils.logger_utils import setup_logger
from .abc import BaseCrossover
from .covariance import (
    population_covariance,
    select_covariance_guided_group,
    select_sample_wise_covariance_group,
)
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


def _format_covariance_selection_log(selection: Dict[str, Any], n_pop: int) -> str:
    """Format strategy-specific covariance selection diagnostics."""
    if selection["branch"] != "cov_per_sample":
        return (
            f"covariance={selection['branch']} "
            f"J={selection['score']} variance={selection['scalar_variance']:.6g} "
            f"degenerate_fallback={selection['degenerate_fallback']}"
        )

    elite_id = int(selection["elite_id"])
    elite_origin = "child" if elite_id >= n_pop else "parent"
    true_score = selection["score"]
    true_score_text = "None" if true_score is None else f"{true_score:.6g}"
    return (
        f"selection=cov_per_sample elite={elite_origin}:{elite_id} "
        f"frozen_J={selection['frozen_score']:.6g} "
        f"lower_bound={selection['lower_bound']:.6g} "
        f"gap={selection['approximation_gap']:.6g} "
        f"true_J={true_score_text} variance={selection['scalar_variance']:.6g} "
        f"degenerate_scalar_contrast={selection['degenerate_scalar_contrast']}"
    )


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
    5. Fill or trim back to K with ``crossover.survivor_score``

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
        self._advantage_aggregation = training_args.advantage_aggregation
        if self._advantage_aggregation not in {"sum", "gdpo"}:
            raise ValueError(
                "Crossover genetic evolution requires advantage_aggregation to be "
                f"'sum' or 'gdpo'; got {self._advantage_aggregation!r}."
            )
        self._survivor_score = training_args.crossover.survivor_score
        if self._survivor_score not in {
            "advantage",
            "abs_advantage",
            "covariance",
            "cov_per_sample",
        }:
            raise ValueError(
                "crossover.survivor_score must be 'advantage', 'abs_advantage', "
                "'covariance', or 'cov_per_sample'; "
                f"got {self._survivor_score!r}."
            )
        if self._survivor_score == "cov_per_sample" and self._advantage_aggregation != "sum":
            raise ValueError(
                "crossover.survivor_score='cov_per_sample' requires "
                "advantage_aggregation='sum' because its frozen contribution score is defined "
                "against the weighted-sum scalar policy direction; "
                f"got advantage_aggregation={self._advantage_aggregation!r}."
            )
        if self._survivor_score in {"covariance", "cov_per_sample"} and not self._reward_weights:
            raise ValueError(
                f"crossover.survivor_score={self._survivor_score!r} requires configured "
                "reward weights."
            )
        self._survivor_selection_aggregation = (
            "weighted_sum"
            if self._survivor_score in {"covariance", "cov_per_sample"}
            else self._advantage_aggregation
        )
        trainer_type = str(getattr(training_args, "trainer_type", "")).lower()
        self._covariance_objective = (
            "locally_linear_nft" if trainer_type == "crossover-nft" else "standardized_grpo"
        )

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
            if self._survivor_score == "cov_per_sample":
                acc[f"gen{gen}_cov_per_sample_count"] = 0
                acc[f"gen{gen}_cov_per_sample_frozen_score_sum"] = 0.0
                acc[f"gen{gen}_cov_per_sample_lower_bound_sum"] = 0.0
                acc[f"gen{gen}_cov_per_sample_approximation_gap_sum"] = 0.0
                acc[f"gen{gen}_cov_per_sample_true_score_sum"] = 0.0
                acc[f"gen{gen}_cov_per_sample_true_score_count"] = 0
                acc[f"gen{gen}_cov_per_sample_scalar_variance_sum"] = 0.0
                acc[f"gen{gen}_cov_per_sample_elite_child_count"] = 0
                acc[f"gen{gen}_cov_per_sample_degenerate_count"] = 0
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
                    source=population[0].source,
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
                selection_line = ""
                if "covariance_selection" in stats:
                    selection = stats["covariance_selection"]
                    selection_line = " | " + _format_covariance_selection_log(
                        selection, stats["n_pop"]
                    )
                if (
                    "covariance_selection" in stats
                    and stats["covariance_selection"]["branch"] == "cov_per_sample"
                ):
                    population_line = (
                        f"children_kept={stats['n_children_kept']}/{stats['n_children']}"
                    )
                else:
                    population_line = (
                        f"children_kept={stats['n_children_kept']}/{stats['n_children']}, "
                        f"pareto={stats['n_pareto_parents']}+{stats['n_pareto_children']}, "
                        f"filled={stats['n_filled']}"
                    )
                logger.info(
                    f"[rank {rank}] GA gid={gid} gen={gen_idx}: "
                    f"pop={stats['n_pop']} "
                    f"replaced={stats['n_replaced']}/{stats['n_pop']} "
                    f"({population_line}) | "
                    f"{rw_lines}{selection_line}"
                )

                # ---- Accumulate aggregate stats ----
                acc[f"gen{gen_idx}_count"] += 1
                acc[f"gen{gen_idx}_n_replaced"] += stats["n_replaced"]
                acc[f"gen{gen_idx}_n_children"] += stats["n_children"]
                acc[f"gen{gen_idx}_n_children_kept"] += stats["n_children_kept"]
                acc[f"gen{gen_idx}_n_pareto_parents"] += stats["n_pareto_parents"]
                acc[f"gen{gen_idx}_n_pareto_children"] += stats["n_pareto_children"]
                acc[f"gen{gen_idx}_n_filled"] += stats["n_filled"]
                if (
                    "covariance_selection" in stats
                    and stats["covariance_selection"]["branch"] == "cov_per_sample"
                ):
                    selection = stats["covariance_selection"]
                    prefix = f"gen{gen_idx}_cov_per_sample"
                    acc[f"{prefix}_count"] += 1
                    acc[f"{prefix}_frozen_score_sum"] += selection["frozen_score"]
                    acc[f"{prefix}_lower_bound_sum"] += selection["lower_bound"]
                    acc[f"{prefix}_approximation_gap_sum"] += selection["approximation_gap"]
                    acc[f"{prefix}_scalar_variance_sum"] += selection["scalar_variance"]
                    if selection["score"] is not None:
                        acc[f"{prefix}_true_score_sum"] += selection["score"]
                        acc[f"{prefix}_true_score_count"] += 1
                    acc[f"{prefix}_elite_child_count"] += int(
                        selection["elite_id"] >= stats["n_pop"]
                    )
                    acc[f"{prefix}_degenerate_count"] += int(
                        selection["degenerate_scalar_contrast"]
                    )
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
                    sample_record = {
                        "gen": gen_idx,
                        "gid": int(gid),
                        "rank": int(rank),
                        "sample_idx": si,
                        "is_child": bool(
                            population[si].extra_kwargs.get("is_crossover_child", False)
                        ),
                        "rewards": {k: float(pop_rewards[k][si]) for k in reward_keys},
                    }
                    if si == 0 and "covariance_selection" in stats:
                        sample_record["selection_diagnostics"] = stats["covariance_selection"]
                    ga_samples.append(sample_record)

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
        source: Optional[str],
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
        adv = self._compute_advantage(pop_rewards, valid_reward_keys, source)

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
                rng_seed=self._operation_seed(epoch, ctx.gid, ctx.gen_idx, "resample"),
            )
            prefix_indices = None
            secondary_indices = None
        elif self._offspring_mode == "mutation":
            # ---- Mutation-only: clone single parent + noise, no crossover ----
            parent_idx, n_parents = self._select_parents(adv, pop_rewards, valid_reward_keys)
            if parent_idx is None:
                return population, pop_rewards, None
            parent_latents = torch.stack(
                [self._get_crossover_latent(population[pi], device) for pi in parent_idx]
            )
            child_latents, selected_parent_indices = self._mutate_only(
                parent_latents,
                self._operation_seed(epoch, ctx.gid, ctx.gen_idx, "mutation"),
            )
            prefix_indices = parent_idx[selected_parent_indices]
            secondary_indices = None
        else:
            # ---- Crossover (default): two-parent crossover + optional mutation ----
            parent_idx, n_parents = self._select_parents(adv, pop_rewards, valid_reward_keys)
            if parent_idx is None:
                return population, pop_rewards, None
            parent_latents = torch.stack(
                [self._get_crossover_latent(population[pi], device) for pi in parent_idx]
            )
            child_latents, selected_parent_indices, selected_secondary_indices = (
                self._crossover_and_mutate(
                    parent_latents,
                    self._operation_seed(epoch, ctx.gid, ctx.gen_idx, "crossover"),
                )
            )
            prefix_indices = parent_idx[selected_parent_indices]
            secondary_indices = parent_idx[selected_secondary_indices]

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
            population=population,
            prefix_indices=prefix_indices,
            ctx=ctx,
        )
        for child_idx, child in enumerate(children):
            child.extra_kwargs["primary_parent_index"] = (
                int(prefix_indices[child_idx]) if prefix_indices is not None else None
            )
            child.extra_kwargs["secondary_parent_index"] = (
                int(secondary_indices[child_idx]) if secondary_indices is not None else None
            )
            child.extra_kwargs["offspring_mode"] = self._offspring_mode

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
            source=source,
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

    def _crossover_and_mutate(
        self, parent_latents: torch.Tensor, rng_seed: int
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Apply crossover strategy + Gaussian mutation."""
        gen_rng = torch.Generator()
        gen_rng.manual_seed(rng_seed)
        out = self._strategy.crossover(parent_latents, generator=gen_rng)
        child_latents = out.child_latents
        if self._mutation_std > 0:
            device_gen = torch.Generator(device=child_latents.device)
            device_gen.manual_seed(rng_seed + 1)
            noise = torch.randn(
                child_latents.shape,
                device=child_latents.device,
                dtype=child_latents.dtype,
                generator=device_gen,
            )
            child_latents = child_latents + noise * self._mutation_std
        return (
            child_latents,
            out.parent_indices_i.detach().cpu().numpy(),
            out.parent_indices_j.detach().cpu().numpy(),
        )

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

    def _mutate_only(
        self, parent_latents: torch.Tensor, rng_seed: int
    ) -> Tuple[torch.Tensor, np.ndarray]:
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
        gen_rng = torch.Generator(device=parent_latents.device)
        gen_rng.manual_seed(rng_seed)
        # Randomly select one parent for each child
        pick = torch.randint(0, K, (M,), device=parent_latents.device, generator=gen_rng)
        child_latents = parent_latents[pick].clone()
        # Mutation
        std = self._mutation_std
        if std <= 0:
            std = 0.05
            logger.warning(
                f"offspring_mode='mutation' but mutation_std={self._mutation_std}. "
                f"Falling back to mutation_std={std} to avoid producing identical clones."
            )
        noise = torch.randn(
            child_latents.shape,
            device=child_latents.device,
            dtype=child_latents.dtype,
            generator=gen_rng,
        )
        child_latents = child_latents + noise * std
        return child_latents, pick.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Step 3: Denoise → child samples
    # ------------------------------------------------------------------

    def _denoise_and_create_children(
        self,
        child_latents: torch.Tensor,
        cxo_step: int,
        population: List[BaseSample],
        prefix_indices: Optional[np.ndarray],
        ctx: _EvolveCtx,
    ) -> List[BaseSample]:
        """Denoise child latents and create samples via child_factory."""
        device = self.device
        templates = (
            [population[int(i)] for i in prefix_indices]
            if prefix_indices is not None
            else [population[0] for _ in range(child_latents.shape[0])]
        )

        # Reuse the exact optimize-time stacking contract so every adapter-specific
        # condition (negative CFG embeds, edit latents, connector embeds, etc.) is
        # propagated instead of maintaining a fragile field allowlist. Move each
        # template before stacking because multi-generation populations can mix
        # CPU-offloaded parents with device-resident children.
        child_batch = self._stack_templates_on_device(templates, device)

        timesteps = templates[0].timesteps.to(device)
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
            templates=templates,
            child_latents=child_latents,
            cxo_step=cxo_step,
            denoise_output=raw,
            ctx=ctx,
        )

    def _stack_templates_on_device(
        self,
        templates: List[BaseSample],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Move filtered template fields before applying sample stacking rules."""
        if not templates:
            raise ValueError("Cannot stack an empty crossover template list.")

        sample_cls = type(templates[0])
        template_dicts = []
        for template in templates:
            forward_kwargs = filter_kwargs(self._adapter.forward, **template.to_dict())
            template_dicts.append(move_tensors_to_device(forward_kwargs, device))

        all_keys = set()
        for template_dict in template_dicts:
            all_keys.update(template_dict.keys())

        return {
            key: sample_cls._stack_values(
                key,
                [template_dict.get(key) for template_dict in template_dicts],
            )
            for key in all_keys
        }

    def _default_child_factory(
        self,
        templates: List[BaseSample],
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
            template = templates[m]
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
    # Step 4: Select survivors
    # ------------------------------------------------------------------

    def _select_survivors(
        self,
        population: List[BaseSample],
        children: List[BaseSample],
        pop_rewards: Dict[str, np.ndarray],
        child_rewards: Dict[str, np.ndarray],
        reward_keys: List[str],
        valid_reward_keys: List[str],
        source: Optional[str],
    ) -> Tuple[
        List[BaseSample],
        Dict[str, np.ndarray],
        Dict[str, Any],
    ]:
        """Merge population + children, compute unified advantage, trim to K.

        Advantage is computed *after* merging so all K+M samples share the
        same normalization (combined mean/std). Selection uses
        ``crossover.survivor_score``. Pareto, covariance, and advantage
        computation use only *valid_reward_keys*; *reward_keys* is the full
        global set for dict iteration and stats bookkeeping.
        """
        n_pop = len(population)
        n_children = len(children)

        # Merge rewards first
        combined_rewards: Dict[str, np.ndarray] = {}
        for k in reward_keys:
            combined_rewards[k] = np.concatenate([pop_rewards[k], child_rewards[k]])

        # Compute advantage on the FULL combined set (unified normalization)
        combined_adv = self._compute_advantage(combined_rewards, valid_reward_keys, source)

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

        pareto_indices = np.where(pareto)[0]
        K = self._group_size
        covariance_stats = None
        if self._survivor_score in {"covariance", "cov_per_sample"}:
            reward_matrix, weights = self._prepare_covariance_inputs(
                combined_rewards, valid_reward_keys, source
            )
            if self._survivor_score == "covariance":
                covariance_fallback = np.abs(
                    self._compute_weighted_sum_advantage(
                        combined_rewards, valid_reward_keys, source
                    )
                )
                selection = select_covariance_guided_group(
                    reward_matrix=reward_matrix,
                    weights=weights,
                    target_size=K,
                    objective=self._covariance_objective,
                    fallback_scores=covariance_fallback,
                    candidate_ids=np.arange(len(combined_adv)),
                )
                sample_wise_stats = {}
                branch = selection.branch
                degenerate_fallback = selection.degenerate_fallback
            else:
                selection = select_sample_wise_covariance_group(
                    reward_matrix=reward_matrix,
                    weights=weights,
                    target_size=K,
                    objective=self._covariance_objective,
                    candidate_ids=np.arange(len(combined_adv)),
                )
                sample_scores = selection.sample_scores
                sample_wise_stats = {
                    "elite_id": selection.elite_index,
                    "pool_mean_rewards": sample_scores.pool_mean.tolist(),
                    "scalar_rewards": sample_scores.scalar_rewards.tolist(),
                    "scalar_advantages": sample_scores.scalar_advantages.tolist(),
                    "sample_contributions": sample_scores.contribution_matrix.tolist(),
                    "sample_fitness": sample_scores.fitness.tolist(),
                    "frozen_contribution_vector": selection.frozen_contribution_vector.tolist(),
                    "frozen_score": selection.frozen_score,
                    "lower_bound": selection.lower_bound,
                    "approximation_gap": selection.frozen_score - selection.lower_bound,
                    "degenerate_scalar_contrast": sample_scores.degenerate_scalar_contrast,
                }
                branch = "cov_per_sample"
                degenerate_fallback = False
            keep_indices = selection.selected_indices
            n_filled = (
                max(0, K - len(pareto_indices)) if self._survivor_score == "covariance" else 0
            )
            before_covariance = population_covariance(reward_matrix)
            group_score = selection.group_score
            finite_score = np.isfinite(group_score.score)
            covariance_stats = {
                "branch": branch,
                "selected_ids": keep_indices.tolist(),
                "rejected_ids": selection.rejected_indices.tolist(),
                "raw_rewards": np.stack(
                    [combined_rewards[key] for key in valid_reward_keys], axis=1
                ).tolist(),
                "covariance_before": before_covariance.tolist(),
                "covariance_after": group_score.covariance.tolist(),
                "contribution_vector": (
                    group_score.contribution_vector.tolist() if finite_score else None
                ),
                "score": group_score.score if finite_score else None,
                "scalar_variance": group_score.scalar_variance,
                "mean_scalar_reward": group_score.mean_scalar_reward,
                "normalized_covariance_conflict": (
                    max(0.0, -group_score.score) if finite_score else None
                ),
                "degenerate_fallback": degenerate_fallback,
                "selection_aggregation": self._survivor_selection_aggregation,
                "policy_advantage_aggregation": self._advantage_aggregation,
                **sample_wise_stats,
            }
        else:
            score = combined_adv if self._survivor_score == "advantage" else np.abs(combined_adv)
            dominated_indices = np.where(~pareto)[0]
            pareto_order = pareto_indices[np.argsort(score[pareto_indices])[::-1]]
            if len(pareto_order) >= K:
                keep_indices = pareto_order[:K]
                n_filled = 0
            else:
                dominated_order = dominated_indices[np.argsort(score[dominated_indices])[::-1]]
                n_filled = min(K - len(pareto_order), len(dominated_order))
                keep_indices = np.concatenate([pareto_order, dominated_order[:n_filled]])
        n_pareto = len(pareto_indices)
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
        if covariance_stats is not None:
            stats["covariance_selection"] = covariance_stats
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
        source: Optional[str],
    ) -> np.ndarray:
        """Compute selection advantages with the configured aggregation.

        Only *valid_reward_keys* participate; all-NaN columns from
        non-applicable rewards are already excluded by the caller.  This
        mirrors the local-group ordering produced by ``AdvantageProcessor``:
        ``sum`` aggregates raw weighted rewards before normalization, while
        ``gdpo`` normalizes each reward before weighted aggregation.
        """
        if not valid_reward_keys:
            if not rewards_dict:
                return np.array([])
            n = len(next(iter(rewards_dict.values())))
            return np.zeros(n, dtype=np.float32)

        n = len(rewards_dict[valid_reward_keys[0]])
        if n == 0:
            return np.array([])

        if self._advantage_aggregation == "sum":
            advantages = self._compute_weighted_sum_advantage(
                rewards_dict, valid_reward_keys, source
            )
        else:
            advantages = np.zeros(n, dtype=np.float64)
            for key in valid_reward_keys:
                values = rewards_dict[key].astype(np.float64)
                weight = self._get_reward_weight(key, source)
                advantages += self._normalize_group_values(values) * weight

        return advantages.astype(np.float32)

    def _compute_weighted_sum_advantage(
        self,
        rewards_dict: Dict[str, np.ndarray],
        valid_reward_keys: List[str],
        source: Optional[str],
    ) -> np.ndarray:
        """Compute group-normalized weighted-sum rewards for selection."""
        if not valid_reward_keys:
            if not rewards_dict:
                return np.array([])
            return np.zeros(len(next(iter(rewards_dict.values()))), dtype=np.float32)

        aggregated = np.zeros(len(rewards_dict[valid_reward_keys[0]]), dtype=np.float64)
        for key in valid_reward_keys:
            aggregated += rewards_dict[key].astype(np.float64) * self._get_reward_weight(
                key, source
            )
        return self._normalize_group_values(aggregated).astype(np.float32)

    def _prepare_covariance_inputs(
        self,
        rewards_dict: Dict[str, np.ndarray],
        valid_reward_keys: List[str],
        source: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stack raw rewards and resolve source-aware covariance weights."""
        if not valid_reward_keys:
            raise ValueError("Covariance survivor selection requires applicable rewards.")

        reward_columns = []
        weights = []
        for key in valid_reward_keys:
            values = rewards_dict[key].astype(np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Covariance survivor selection received non-finite reward {key!r}."
                )
            reward_columns.append(values)
            weights.append(self._get_reward_weight(key, source))

        weight_vector = np.asarray(weights, dtype=np.float64)
        if np.any(weight_vector < 0) or np.count_nonzero(weight_vector > 0) < 2:
            raise ValueError(
                "Covariance survivor selection requires at least two positive reward weights "
                "and does not support negative weights."
            )
        return np.stack(reward_columns, axis=1), weight_vector

    def _get_reward_weight(self, key: str, source: Optional[str]) -> float:
        """Resolve one reward's scalarization weight for the current source."""
        weight_map = self._reward_weights.get(key, {"default": 1.0})
        if source is not None and source in weight_map:
            return float(weight_map[source])
        if "default" in weight_map:
            return float(weight_map["default"])
        if len(weight_map) == 1:
            return float(next(iter(weight_map.values())))
        raise ValueError(
            f"Missing reward weight for reward={key!r}, source={source!r}; "
            f"available sources={sorted(weight_map)}."
        )

    @staticmethod
    def _normalize_group_values(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Return zero-mean group values with a numerically safe standard deviation."""
        mean = float(np.mean(values))
        std = max(float(np.std(values)), eps)
        return (values - mean) / std

    def _operation_seed(self, epoch: int, gid: int, generation: int, operation: str) -> int:
        """Derive a stable non-colliding seed for one GA operation."""
        payload = f"{self._seed}:{epoch}:{gid}:{generation}:{operation}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)

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
            covariance_prefix = f"gen{gen}_cov_per_sample"
            if f"{covariance_prefix}_count" in ga_acc:
                for suffix in [
                    "count",
                    "frozen_score_sum",
                    "lower_bound_sum",
                    "approximation_gap_sum",
                    "true_score_sum",
                    "true_score_count",
                    "scalar_variance_sum",
                    "elite_child_count",
                    "degenerate_count",
                ]:
                    count_keys.append(f"{covariance_prefix}_{suffix}")

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

            covariance_prefix = f"gen{gen}_cov_per_sample"
            covariance_count = reduced.get(f"{covariance_prefix}_count", 0.0)
            if covariance_count > 0:
                metric_prefix = f"{p}/cov_per_sample"
                for metric in [
                    "frozen_score",
                    "lower_bound",
                    "approximation_gap",
                    "scalar_variance",
                ]:
                    stats[f"{metric_prefix}/{metric}"] = round(
                        reduced[f"{covariance_prefix}_{metric}_sum"] / covariance_count,
                        6,
                    )
                true_score_count = reduced[f"{covariance_prefix}_true_score_count"]
                if true_score_count > 0:
                    stats[f"{metric_prefix}/true_score"] = round(
                        reduced[f"{covariance_prefix}_true_score_sum"] / true_score_count,
                        6,
                    )
                stats[f"{metric_prefix}/elite_child_rate"] = round(
                    reduced[f"{covariance_prefix}_elite_child_count"] / covariance_count,
                    6,
                )
                stats[f"{metric_prefix}/degenerate_scalar_contrast_rate"] = round(
                    reduced[f"{covariance_prefix}_degenerate_count"] / covariance_count,
                    6,
                )

        if ga_samples:
            stats["ga/samples"] = ga_samples

        return stats
