# Copyright 2026 Jayce-Ping
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

# src/flow_factory/advantage/advantage_processor.py
"""
Communication-aware Advantage Processor.

Extracts advantage computation logic from GRPOTrainer into a standalone,
reusable component.  Automatically selects the communication strategy based
on the resolved sampler type:

- ``distributed_k_repeat``: gather rewards + unique_ids across ranks →
  global grouping → scatter back to local rank.
- ``group_contiguous``: all K copies already reside on the same rank →
  skip all cross-rank communication for advantage computation.  Training log
  metrics are computed via mode-aware ``_metric_*`` helpers that transparently
  select between plain NumPy (post-gather global arrays) and ``utils.dist``
  reductions (local shards) so logging always reflects global statistics.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator

from ..rewards import RewardProcessor
from ..samples import BaseSample
from ..utils.dist import global_tensor_stats_batch, global_zero_std_ratio
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class AdvantageProcessor:
    """Communication-aware advantage computation processor.

    Parameters
    ----------
    accelerator : Accelerator
        HuggingFace Accelerator instance for distributed ops.
    reward_weights : dict[str, dict[str, float]]
        Mapping from reward name to per-dataset weights
        (``{reward_name: {dataset_name: weight}}``).  Resolved by
        ``Arguments._resolve_reward_weights`` from scalar or dict form.
    group_size : int
        Number of repeated samples per unique prompt (K).
    global_std : bool
        If ``True``, normalise advantages using the global std across all
        groups; otherwise use per-group std.
    sampler_type : str
        One of ``"distributed_k_repeat"`` or ``"group_contiguous"``.
        Determines whether cross-rank communication is needed.
    verbose : bool
        Whether to emit progress information.

    Notes
    -----
    After :meth:`compute_advantages` with ``'sum'`` or ``'gdpo'``, call
    :meth:`pop_advantage_metrics` once to retrieve training metrics (including
    ``train_samples``) for ``log_data``. Custom callables leave an empty metrics
    snapshot. This class does not perform logging itself.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        reward_weights: Dict[str, Dict[str, float]],
        group_size: int,
        global_std: bool = True,
        sampler_type: str = "distributed_k_repeat",
        verbose: bool = True,
        source_id_to_name: Optional[List[str]] = None,
        max_log_samples: Optional[int] = None,
        pareto_config: Optional[Dict[str, Any]] = None,
        stddev_reweighting: bool = False,
        stddev_ema_decay: float = 0.99,
    ):
        self.accelerator = accelerator
        self.reward_weights = reward_weights
        self.group_size = group_size
        self.global_std = global_std
        self.sampler_type = sampler_type
        self.verbose = verbose
        self.stddev_reweighting = stddev_reweighting
        self.stddev_ema_decay = stddev_ema_decay
        self.max_log_samples = max_log_samples
        self._source_id_to_name = source_id_to_name or []

        self.group_on_same_rank = sampler_type == "group_contiguous"
        self._pending_advantage_metrics: Optional[Dict[str, Any]] = None

        # Per-reward EMA of mean std across groups (for stddev reweighting).
        self._stddev_ema: Dict[str, float] = {}

        # Crossover / Pareto state
        self._pareto_enabled = pareto_config is not None and pareto_config.get("enabled", False)
        self._log_crossover_rewards: bool = (
            False  # set by crossover trainers via log_rewards option
        )
        self._pending_crossover_stats: Optional[Dict[str, Any]] = None
        self._pending_pareto_stats: Optional[Dict[str, Any]] = None
        self._child_advantage_scale: float = 1.0  # set by crossover trainers for warmup
        self._child_in_norm: bool = (
            False  # set by crossover trainers to include children in mean/std
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pop_advantage_metrics(self) -> Dict[str, Any]:
        """Return and clear metrics from the last ``sum`` / ``gdpo`` advantage pass.

        Call once per :meth:`compute_advantages` when using built-in aggregation.
        Returns an empty dict if nothing was produced (e.g. custom callable only,
        or no prior computation).
        """
        out = dict(self._pending_advantage_metrics or {})
        self._pending_advantage_metrics = None
        return out

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
    ) -> torch.Tensor:
        """Compute per-sample advantages.

        Parameters
        ----------
        samples : list[BaseSample]
            Samples on the current rank.
        rewards : dict[str, Tensor]
            Per-reward-model reward tensors aligned with *samples*.
        store_to_samples : bool
            Write computed advantages into ``sample.extra_kwargs['advantage']``.
        aggregation_func : str or callable
            ``'sum'`` for weighted-sum GRPO, ``'gdpo'`` for GDPO-style, or a
            custom ``callable(processor, samples, rewards, store_to_samples)``.

        Returns
        -------
        Tensor
            Advantages for the local rank, shape ``(len(samples),)``.
        """
        self._pending_advantage_metrics = None
        aggregation_func = aggregation_func or "gdpo"
        if aggregation_func == "sum":
            return self.compute_weighted_sum(samples, rewards, store_to_samples)
        elif aggregation_func == "gdpo":
            return self.compute_gdpo(samples, rewards, store_to_samples)
        elif callable(aggregation_func):
            adv = aggregation_func(self, samples, rewards, store_to_samples)
            if self._pending_advantage_metrics is None:
                self._pending_advantage_metrics = {}
            return adv
        else:
            raise ValueError(
                f"Unsupported advantage aggregation method: {aggregation_func}. "
                "Supported: ['sum', 'gdpo'] "
                "or a callable function that takes (processor, samples, rewards, store_to_samples) as inputs."
            )

    # ------------------------------------------------------------------
    # Communication layer
    # ------------------------------------------------------------------

    def collect_group_rewards(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Collect rewards, group indices, and source IDs.

        ``group_contiguous``: no communication; arrays are local ``(B,)``.
        ``distributed_k_repeat``: rewards + ``unique_id`` + ``source_id``
        are packed into a single ``(B, N+2)`` tensor and gathered with
        one ``accelerator.gather()`` call. Arrays are global ``(W*B,)``.

        Returns:
            collected_rewards: ``{reward_name: np.ndarray}``
            group_indices: integer array mapping each sample to its group
            gathered_source_ids: integer array of source IDs (``-1`` = legacy)
        """
        if self.group_on_same_rank:
            collected_rewards = {
                key: torch.as_tensor(value).cpu().numpy() for key, value in rewards.items()
            }
            unique_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
            _unique_ids, group_indices = np.unique(unique_ids, return_inverse=True)
            source_ids = np.array(
                [s.source_id if s.source_id is not None else -1 for s in samples],
                dtype=np.int64,
            )
            return collected_rewards, group_indices, source_ids
        else:
            rewards = {
                key: torch.as_tensor(value).to(self.accelerator.device)
                for key, value in rewards.items()
            }
            reward_keys = list(rewards.keys())
            device = self.accelerator.device
            unique_ids = torch.tensor(
                [s.unique_id for s in samples],
                dtype=torch.int64,
                device=device,
            )
            local_source_ids = torch.tensor(
                [s.source_id if s.source_id is not None else -1 for s in samples],
                dtype=torch.int64,
                device=device,
            )
            # Pack: [reward_0, ..., reward_{N-1}, unique_id, source_id]
            columns = [rewards[k].view(-1).float() for k in reward_keys]
            columns.append(unique_ids.float())
            columns.append(local_source_ids.float())
            packed = torch.stack(columns, dim=1)  # (B, N+2)

            gathered = self.accelerator.gather(packed).cpu().numpy()  # (W*B, N+2)

            collected_rewards = {key: gathered[:, i] for i, key in enumerate(reward_keys)}
            gathered_ids = gathered[:, -2].astype(np.int64)
            _unique_ids, group_indices = np.unique(gathered_ids, return_inverse=True)
            source_ids = gathered[:, -1].astype(np.int64)
            return collected_rewards, group_indices, source_ids

    def _gather_uneven(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor that may have different sizes across ranks.

        Pads to ``max_len`` across ranks, calls :meth:`Accelerator.gather`,
        then strips per-rank padding and concatenates only valid rows.
        """
        rank = self.accelerator.process_index
        local_n = tensor.shape[0]
        n_t = torch.tensor([local_n], device=self.accelerator.device)
        logger.debug(f"[rank {rank}] _gather_uneven: gather sizes (local_n={local_n})")
        all_n = self.accelerator.gather(n_t).cpu().tolist()
        max_n = max(all_n)
        logger.debug(f"[rank {rank}] _gather_uneven: sizes gathered, max_n={max_n}, all_n={all_n}")
        if local_n < max_n:
            pad = torch.zeros(
                max_n - local_n,
                *tensor.shape[1:],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, pad], dim=0)
        logger.debug(
            f"[rank {rank}] _gather_uneven: gather data "
            f"(shape={tensor.shape}, dtype={tensor.dtype})"
        )
        gathered = self.accelerator.gather(tensor)  # (W * max_n, ...)
        logger.debug(f"[rank {rank}] _gather_uneven: data gathered, shape={gathered.shape}")
        parts = []
        for rank_i, rank_n in enumerate(all_n):
            if rank_n > 0:
                parts.append(gathered[rank_i * max_n : rank_i * max_n + rank_n])
        return torch.cat(parts, dim=0) if parts else gathered[:0]

    def _gather_for_logging(
        self,
        samples: List[BaseSample],
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        advantages: Optional[np.ndarray] = None,
        applicable: Optional[np.ndarray] = None,
    ) -> Tuple[
        List[str], Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """Gather prompts, rewards, group_indices, advantages, and applicable for logging."""
        device = self.accelerator.device
        reward_keys = list(gathered_rewards.keys())
        R = len(reward_keys)

        # Gather prompts (pad both str-length and batch-count dims)
        rank = self.accelerator.process_index
        prompts_bytes = [s.prompt.encode("utf-8") for s in samples]
        max_str = max((len(b) for b in prompts_bytes), default=0)
        max_str_t = torch.tensor([max_str], dtype=torch.long, device=device)
        logger.debug(
            f"[rank {rank}] _gather_for_logging: gather max_str (local_max={max_str}, "
            f"local_n={len(samples)})"
        )
        synced = self.accelerator.gather(max_str_t)
        global_max_str = int(synced.max().item()) if synced.numel() > 0 else 0
        logger.debug(
            f"[rank {rank}] _gather_for_logging: max_str gathered, global_max={global_max_str}"
        )
        if global_max_str > 0:
            str_padded = [list(b.ljust(global_max_str, b"\x00")) for b in prompts_bytes] or [
                [0] * global_max_str
            ]
            t = (
                self._gather_uneven(torch.tensor(str_padded, dtype=torch.uint8, device=device))
                .cpu()
                .numpy()
            )
            all_prompts = [bytes(row).rstrip(b"\x00").decode("utf-8") for row in t]
        else:
            all_prompts = [""] * len(samples)

        if self.group_on_same_rank:
            # Use unique_id (globally unique) for grouping, not group_indices (local)
            unique_ids = torch.tensor(
                [s.unique_id for s in samples], dtype=torch.float32, device=device
            )

            columns = [
                torch.tensor(gathered_rewards[k], dtype=torch.float32, device=device)
                for k in reward_keys
            ]
            columns.append(unique_ids)
            if advantages is not None:
                columns.append(torch.tensor(advantages, dtype=torch.float32, device=device))
            if applicable is not None:
                for r in range(R):
                    columns.append(torch.tensor(applicable[r], dtype=torch.float32, device=device))
            packed = torch.stack(columns, dim=1)

            gathered = self._gather_uneven(packed).cpu().numpy()

            all_rewards = {k: gathered[:, i] for i, k in enumerate(reward_keys)}
            all_unique_ids = gathered[:, len(reward_keys)].astype(np.int64)
            col = len(reward_keys) + 1
            all_advantages = gathered[:, col].astype(np.float64) if advantages is not None else None
            if applicable is not None:
                col += 1
                all_applicable = gathered[:, col : col + R].astype(bool).T
            else:
                all_applicable = None
        else:
            all_rewards = gathered_rewards
            all_unique_ids = group_indices
            all_advantages = advantages
            all_applicable = applicable

        return all_prompts, all_rewards, all_unique_ids, all_advantages, all_applicable

    def build_source_aware_matrices(
        self,
        samples: List[BaseSample],
        reward_keys: List[str],
        gathered_source_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``(R, S)`` applicability mask and weight matrix locally.

        Uses ``applicable_rewards`` from local samples (``group_contiguous``)
        or derives applicability from ``gathered_source_ids`` + config-level
        ``_datasets_resolved`` (``distributed_k_repeat``). Weight matrix
        is computed from ``gathered_source_ids`` + ``reward_weights`` with
        zero communication.

        Args:
            samples: Local samples (used in ``group_contiguous`` path).
            reward_keys: Ordered list of reward names.
            gathered_source_ids: Source IDs from ``collect_group_rewards``.

        Returns:
            Tuple of ``(applicable, weight_matrix)`` both shape ``(R, S)``.
        """
        R = len(reward_keys)
        S = len(gathered_source_ids)

        if self.group_on_same_rank:
            local_mask = np.zeros((R, len(samples)), dtype=bool)
            for j, s in enumerate(samples):
                applicable = s.applicable_rewards
                has_source = s.source is not None or s.source_id is not None
                if not applicable and not has_source:
                    local_mask[:, j] = True
                else:
                    for i, name in enumerate(reward_keys):
                        local_mask[i, j] = name in applicable
            sources = [s.source for s in samples]
            weight_matrix = self._weights_from_sources(reward_keys, sources)
            return local_mask, weight_matrix

        # Distributed: derive applicability from gathered source_ids +
        # config-level reward routing (no communication needed).
        source_names = [
            self._source_id_to_name[sid] if 0 <= sid < len(self._source_id_to_name) else None
            for sid in gathered_source_ids
        ]
        applicable = np.zeros((R, S), dtype=bool)
        for j, src in enumerate(source_names):
            if src is None:
                applicable[:, j] = True
            else:
                for i, key in enumerate(reward_keys):
                    per_ds = self.reward_weights[key]
                    applicable[i, j] = src in per_ds

        weight_matrix = self._weights_from_sources(reward_keys, source_names)
        return applicable, weight_matrix

    def _weights_from_sources(
        self,
        reward_keys: List[str],
        sources: List[Optional[str]],
    ) -> np.ndarray:
        """Build ``(R, S)`` weight matrix from source names (no communication)."""
        R = len(reward_keys)
        S = len(sources)
        matrix = np.ones((R, S), dtype=np.float64)
        for r_idx, key in enumerate(reward_keys):
            per_ds = self.reward_weights[key]
            default_w = next(iter(per_ds.values()))
            for s_idx, src in enumerate(sources):
                if src is not None and src in per_ds:
                    matrix[r_idx, s_idx] = per_ds[src]
                else:
                    matrix[r_idx, s_idx] = default_w
        return matrix

    def _compute_stddev_weights(
        self,
        stack: np.ndarray,
        weight_matrix: np.ndarray,
        group_indices: np.ndarray,
        applicable: np.ndarray,
        reward_keys: List[str],
        norm_mask: Optional[np.ndarray] = None,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Reweight *weight_matrix* by per-reward within-group relative std.

        For each reward *r* and group *g*:

        1. Compute ``std_{r,g}`` (population std of reward *r* over samples
           in group *g* that are both applicable and pass *norm_mask*).
        2. Compute ``mean_std_r = mean_g(std_{r,g})`` — averaged only over
           groups with ≥2 applicable samples.  Update an EMA of this value
           (decay ``self.stddev_ema_decay``) and use the EMA for normalisation
           so that the reference scale is stable across steps.
        3. ``relative_std_{r,g} = std_{r,g} / ema_mean_std_r``
        4. ``effective_w_{r,g} = base_weight_{r,g} * relative_std_{r,g}``

        When the EMA for a reward has not been initialised yet (first step),
        the current *mean_std_r* is used directly.

        Args:
            stack: ``(R, S)`` raw reward values.
            weight_matrix: ``(R, S)`` base per-reward per-sample weights.
            group_indices: ``(S,)`` integer group ids.
            applicable: ``(R, S)`` boolean applicability mask.
            reward_keys: Ordered list of reward names (used as EMA dict keys).
            norm_mask: Optional ``(S,)`` boolean — only samples passing this
                mask participate in std computation.
            eps: Floor for near-zero stds.

        Returns:
            ``(R, S)`` effective weight matrix with stddev reweighting applied.
        """
        R, S = stack.shape
        num_groups = group_indices.max() + 1

        # --- 1. Per-reward per-group std via vectorised bincount ---
        # Shape (R, G): std of each reward within each group.
        group_stds = np.zeros((R, num_groups), dtype=np.float64)
        # valid_mask[r, g]: group g has >=2 applicable samples for reward r.
        valid_mask = np.zeros((R, num_groups), dtype=bool)

        for r in range(R):
            values = stack[r]
            stat_mask = applicable[r].copy()
            if norm_mask is not None:
                stat_mask = stat_mask & norm_mask

            # Bincount-based group std (same pattern as _group_normalize).
            masked_vals = np.where(stat_mask, values, 0.0)
            counts = np.bincount(
                group_indices,
                weights=stat_mask.astype(np.float64),
                minlength=num_groups,
            )
            sums = np.bincount(group_indices, weights=masked_vals, minlength=num_groups)
            safe_counts = np.maximum(counts, 1.0)
            means = sums / safe_counts
            residuals = np.where(stat_mask, values - means[group_indices], 0.0)
            sq_sums = np.bincount(group_indices, weights=residuals**2, minlength=num_groups)
            stds = np.sqrt(sq_sums / safe_counts)

            # Groups with <2 valid samples have undefined std → 0.
            valid = counts >= 2.0
            stds[~valid] = 0.0
            valid_mask[r] = valid
            group_stds[r] = np.maximum(stds, eps)

        # --- 2. Current-step mean std, EMA update, normalisation reference ---
        current_means = np.array(
            [group_stds[r, valid_mask[r]].mean() if valid_mask[r].any() else eps for r in range(R)],
            dtype=np.float64,
        )

        ema_means = np.empty(R, dtype=np.float64)
        for r in range(R):
            key = reward_keys[r]
            cur = float(current_means[r])
            prev = self._stddev_ema.get(key)
            if prev is None:
                # First step: seed the EMA directly.
                self._stddev_ema[key] = cur
                ema_means[r] = cur
            else:
                ema = self.stddev_ema_decay * prev + (1.0 - self.stddev_ema_decay) * cur
                self._stddev_ema[key] = ema
                ema_means[r] = ema
        ema_means = np.maximum(ema_means, eps)

        # --- 3. Relative std & effective weight ---
        # relative_std_{r,g} = std_{r,g} / ema_mean_std_r
        relative_stds = group_stds / ema_means[:, None]  # (R, G)

        # Base weight per (reward, group)
        first_of_group = np.array(
            [np.where(group_indices == g)[0][0] for g in range(num_groups)],
            dtype=np.int64,
        )
        base_w_g = weight_matrix[:, first_of_group]  # (R, G)

        effective_w_g = base_w_g * relative_stds  # (R, G)

        # --- 4. Expand to (R, S) and zero out non-applicable ---
        effective_weights = effective_w_g[:, group_indices]  # (R, S)
        effective_weights[~applicable] = 0.0
        return effective_weights

    def _to_local(
        self,
        values: np.ndarray,
    ) -> torch.Tensor:
        """Convert collected values back to a local-rank tensor.

        When ``group_on_same_rank`` is ``True`` the array is already local and
        is simply converted.  Otherwise the array spans all ranks and is sliced
        to this rank's portion.
        """
        if not self.group_on_same_rank:
            values = (
                torch.as_tensor(values)
                .reshape(self.accelerator.num_processes, -1, *values.shape[1:])[
                    self.accelerator.process_index
                ]
                .to(self.accelerator.device)
            )
        else:
            values = torch.as_tensor(values).to(self.accelerator.device)
        return values

    def _global_mean_std(self, values: np.ndarray) -> tuple:
        """Compute global mean and std for *values*.

        When ``group_on_same_rank`` is ``True`` the array only contains
        local-rank data, so we all-reduce ``(count, sum, sum_sq)`` in a
        single call to obtain the true global statistics.  Otherwise the
        array already spans all ranks (post-gather) and we compute
        directly with NumPy — no communication needed.
        """
        if self.group_on_same_rank:
            t = torch.tensor(
                [float(len(values)), float(np.sum(values)), float(np.sum(values**2))],
                device=self.accelerator.device,
            )
            t = self.accelerator.reduce(t, reduction="sum")  # 1 call, 3 scalars
            n, s, ss = t[0].item(), t[1].item(), t[2].item()
            mean = s / n
            std = max(max(ss / n - mean**2, 0.0) ** 0.5, 1e-6)
        else:
            mean = float(np.mean(values))
            std = max(float(np.std(values)), 1e-6)
        return mean, std

    # ------------------------------------------------------------------
    # Batched metric reduction (mode-aware)
    # ------------------------------------------------------------------

    def _batch_reduce_stats(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute global ``{min, max, mean, std}`` for each named array.

        When ``group_on_same_rank`` the arrays are local shards and require
        cross-rank reduction via :func:`dm.global_tensor_stats_batch` (3
        all-reduce calls total, regardless of the number of arrays).

        Otherwise the arrays already span all ranks (post-gather) and stats
        are computed locally with plain NumPy.
        """
        if self.group_on_same_rank:
            tensors = {
                k: torch.from_numpy(np.asarray(v, dtype=np.float64)) for k, v in arrays.items()
            }
            return global_tensor_stats_batch(self.accelerator, tensors)

        out: Dict[str, Dict[str, float]] = {}
        for k, v in arrays.items():
            v = np.asarray(v, dtype=np.float64)
            if len(v) == 0:
                out[k] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            else:
                out[k] = {
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "mean": float(np.mean(v)),
                    "std": max(float(np.std(v)), 1e-8),
                }
        return out

    def _metric_zero_std_ratio(self, rewards: np.ndarray, group_indices: np.ndarray) -> float:
        """Fraction of groups with near-zero std — global-reduced when ``group_on_same_rank``."""
        if self.group_on_same_rank:
            return global_zero_std_ratio(self.accelerator, rewards, group_indices)
        return RewardProcessor.compute_group_zero_std_ratio(rewards, group_indices)

    @staticmethod
    def _group_normalize(
        values: np.ndarray,
        group_indices: np.ndarray,
        mask: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Per-group zero-mean unit-variance normalization (vectorized).

        Args:
            values: ``(S,)`` array of values to normalize.
            group_indices: ``(S,)`` integer group assignments.
            mask: ``(S,)`` boolean; only masked-in positions participate in
                mean / std computation. ``None`` means all positions participate.
                All samples receive normalized output regardless of *mask*.
            eps: Minimum std to avoid division by zero.

        Returns:
            ``(S,)`` normalized values for all samples.
        """
        S = len(values)
        num_groups = group_indices.max() + 1
        if mask is None:
            mask = np.ones(S, dtype=bool)

        # Statistics from *mask* only.
        masked_vals = np.where(mask, values, 0.0)
        counts = np.bincount(group_indices, weights=mask.astype(np.float64), minlength=num_groups)
        sums = np.bincount(group_indices, weights=masked_vals, minlength=num_groups)
        safe_counts = np.maximum(counts, 1.0)
        means = sums / safe_counts

        residuals = np.where(mask, values - means[group_indices], 0.0)
        sq_sums = np.bincount(group_indices, weights=residuals**2, minlength=num_groups)
        stds = np.sqrt(sq_sums / safe_counts)
        stds = np.maximum(stds, eps)

        # Normalize all values using mask-derived statistics.
        return (values - means[group_indices]) / stds[group_indices]

    @staticmethod
    def _group_normalize_with_mask(
        values: np.ndarray,
        group_indices: np.ndarray,
        keep_mask: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Like :meth:`_group_normalize` but restricted to a keep mask.

        Only *keep_mask* = True samples participate in mean / std computation.
        All samples still receive an output.
        """
        if keep_mask is None:
            return AdvantageProcessor._group_normalize(values, group_indices, mask=None, eps=eps)
        return AdvantageProcessor._group_normalize(values, group_indices, mask=keep_mask, eps=eps)

    @staticmethod
    def _assert_no_nan_at_applicable(
        stack: np.ndarray,
        reward_keys: List[str],
        applicable: np.ndarray,
        label: str = "",
    ) -> None:
        """Assert no NaN/Inf at applicable positions — loud failure on reward bugs.

        Args:
            stack: ``(R, S)`` float64 array of all rewards.
            reward_keys: Ordered list of reward names matching stack axis 0.
            applicable: ``(R, S)`` boolean applicability mask.
            label: Optional prefix for the error message.
        """
        nan_mask = ~np.isfinite(stack)
        bug_positions = nan_mask & applicable
        if bug_positions.any():
            r_idx, s_idx = np.where(bug_positions)
            offenders = sorted({reward_keys[i] for i in r_idx})
            prefix = f"{label}: " if label else ""
            raise RuntimeError(
                f"{prefix}NaN/Inf reward at APPLICABLE positions for reward(s) "
                f"{offenders} (sample indices {sorted(set(s_idx.tolist()))[:10]}"
                f"{'...' if len(s_idx) > 10 else ''}). "
                "This is a reward-model bug, not a routing miss; "
                "aggregation refuses to silently mask it."
            )

    # ------------------------------------------------------------------
    # Strategy: weighted sum (default GRPO)
    # ------------------------------------------------------------------

    def compute_weighted_sum(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
    ) -> torch.Tensor:
        """Compute advantages using the weighted-sum GRPO strategy.

        This is the standard GRPO advantage computation.  Each reward model's
        scores are multiplied by its configured weight and summed into a single
        aggregated reward per sample.  Advantages are then group-normalised
        (subtract per-group mean, divide by std).

        **Source-aware aggregation** (plan §6.4): the per-sample
        applicability matrix from :meth:`build_source_aware_matrices` is
        the authoritative source of truth.  NaN at applicable positions
        is asserted to be a model bug (loud failure); NaN at
        non-applicable positions is honored as "this reward doesn't
        contribute to this sample".  Samples with NO applicable reward
        raise -- a misconfigured `RewardArguments.applicable_datasets` shouldn't
        silently produce zero advantages.

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments.
        2. **Aggregate** — compute
           ``r_agg[i] = sum_k(reward_k[i] * weight_k * applicable_k_i)``.
           NaN values at non-applicable positions are zero-weighted; NaN
           at applicable positions raises.
        3. **Group-normalise** — for each group *g*:
           ``advantage[i] = (r_agg[i] - mean(r_agg[g])) / std``
           where *std* is either the global std across all samples (when
           ``global_std=True``) or the per-group std (when ``global_std=False``).
        4. **To-local** — convert back to local-rank tensor via
           :meth:`_to_local`.
        5. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.
        """
        gathered_rewards, group_indices, source_ids = self.collect_group_rewards(samples, rewards)
        reward_keys = list(gathered_rewards.keys())

        # ---- Crossover stats (before any filtering) ----
        if self._log_crossover_rewards:
            self._build_crossover_stats(gathered_rewards, group_indices, samples)

        # ---- Pareto filtering & child-mask (crossover-only paths) ----
        # Non-crossover training: none of these gates activate, norm_mask stays
        # None → all samples participate in mean/std (original behavior).
        pareto_mask: Optional[np.ndarray] = None
        norm_mask: Optional[np.ndarray] = None
        child_mask: Optional[np.ndarray] = None
        crossover_active = self._pareto_enabled or self._log_crossover_rewards
        if crossover_active:
            child_mask = self._build_child_mask(samples, group_indices)
            if self._pareto_enabled:
                pareto_mask = self._filter_pareto(gathered_rewards, group_indices, child_mask)
            # Build norm_mask.  When _child_in_norm is True, children
            # participate in mean/std alongside parents.
            if pareto_mask is not None:
                norm_mask = pareto_mask
            else:
                norm_mask = None
            if not self._child_in_norm and child_mask is not None:
                norm_mask = norm_mask & ~child_mask if norm_mask is not None else ~child_mask
            # Safety: if norm_mask would exclude everything, fall back to all.
            if norm_mask is not None and not norm_mask.any():
                norm_mask = np.ones(len(group_indices), dtype=bool)

        # ---- Per-child reward details (after filtering decision) ----
        if self._log_crossover_rewards:
            keep_mask = pareto_mask if pareto_mask is not None else None
            self._log_child_details(gathered_rewards, group_indices, samples, keep_mask)

        applicable, weight_matrix = self.build_source_aware_matrices(
            samples, reward_keys, source_ids
        )

        # Build stack and validate: NaN at applicable position == reward-model bug.
        stack = np.stack(
            [gathered_rewards[k].astype(np.float64) for k in reward_keys], axis=0
        )  # (R, S)
        self._assert_no_nan_at_applicable(stack, reward_keys, applicable)

        # ---- Stddev reweighting (optional) ----
        if self.stddev_reweighting:
            weight_matrix = self._compute_stddev_weights(
                stack,
                weight_matrix,
                group_indices,
                applicable,
                reward_keys=reward_keys,
                norm_mask=norm_mask,
            )

        # Aggregate: weighted sum over applicable rewards only.
        contrib = np.where(applicable, stack, 0.0) * weight_matrix
        if pareto_mask is not None:
            contrib[:, ~pareto_mask] = 0.0  # dominated samples contribute nothing
        aggregated_rewards = contrib.sum(axis=0)  # (S,)

        # Per-sample applicable weight sum -> sanity check.
        weight_per_s = (applicable * weight_matrix).sum(axis=0)  # (S,)
        if pareto_mask is not None:
            # Only check kept samples
            kept_weight = weight_per_s[pareto_mask]
        else:
            kept_weight = weight_per_s
        if (kept_weight == 0).any():
            bad = np.where(
                (weight_per_s == 0)
                & (
                    pareto_mask
                    if pareto_mask is not None
                    else np.ones_like(weight_per_s, dtype=bool)
                )
            )[0].tolist()
            raise RuntimeError(
                "AdvantageProcessor: samples at indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''} have NO applicable "
                "reward (weight_sum == 0). Check that "
                "`RewardArguments.applicable_datasets` covers every training source — "
                "at least one reward must apply to every source."
            )

        # Group-normalise (vectorized via bincount).  When *norm_mask* is None
        # (non-crossover training), all samples participate in mean/std.
        # When set, children and/or dominated samples are excluded.
        if self.global_std:
            values_for_std = (
                aggregated_rewards if norm_mask is None else aggregated_rewards[norm_mask]
            )
            _, std = self._global_mean_std(values_for_std)
            num_groups = group_indices.max() + 1
            kept_w = (
                np.ones(len(group_indices), dtype=np.float64)
                if norm_mask is None
                else norm_mask.astype(np.float64)
            )
            sums = np.bincount(
                group_indices, weights=aggregated_rewards * kept_w, minlength=num_groups
            )
            counts = np.bincount(group_indices, weights=kept_w, minlength=num_groups)
            means = sums / np.maximum(counts, 1)
            advantages = (aggregated_rewards - means[group_indices]) / std
        else:
            # norm_mask restricts statistics (mean/std), but all samples
            # (including children) receive properly normalized output.
            advantages = self._group_normalize_with_mask(
                aggregated_rewards, group_indices, keep_mask=norm_mask
            )

        all_prompts, all_rewards, all_unique_ids, all_advantages, all_applicable = (
            self._gather_for_logging(
                samples,
                gathered_rewards,
                group_indices,
                advantages=advantages,
                applicable=applicable,
            )
        )

        # When group_on_same_rank, aggregated_rewards is local but all_unique_ids is
        # gathered. Gather aggregated_rewards so shapes match in log-data builder.
        if self.group_on_same_rank:
            agg_t = torch.tensor(
                aggregated_rewards, dtype=torch.float32, device=self.accelerator.device
            )
            aggregated_rewards = self._gather_uneven(agg_t).cpu().numpy()

        # stat_mask needs to match gathered rewards in group_contiguous mode.
        if child_mask is not None and self.group_on_same_rank:
            flag = torch.tensor(child_mask.astype(np.float32), device=self.accelerator.device)
            log_child_mask = self._gather_uneven(flag).cpu().numpy().astype(bool)
        else:
            log_child_mask = child_mask
        log_stat_mask = ~log_child_mask if log_child_mask is not None else None

        self._pending_advantage_metrics = self._build_weighted_sum_log_data(
            all_rewards,
            all_unique_ids,
            aggregated_rewards,
            all_advantages,
            samples,
            all_prompts,
            applicable=all_applicable,
            reward_keys=reward_keys,
            stat_mask=log_stat_mask,
        )

        # ---- Log stddev effective weights (optional) ----
        if self.stddev_reweighting:
            for r_idx, key in enumerate(reward_keys):
                w = weight_matrix[r_idx]
                w_pos = w[w > 0]
                if len(w_pos) > 0:
                    self._pending_advantage_metrics[f"train/stddev_effective_weight_{key}_mean"] = (
                        float(np.mean(w_pos))
                    )
                    self._pending_advantage_metrics[f"train/stddev_effective_weight_{key}_std"] = (
                        float(np.std(w_pos))
                    )

        # Scale child advantages for warmup (logged values remain unscaled).
        if child_mask is not None and self._child_advantage_scale != 1.0:
            advantages[child_mask] *= self._child_advantage_scale

        # Scatter & store
        advantages = self._to_local(advantages)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv

        # Mark dominated local samples
        if pareto_mask is not None:
            self._mark_dominated_samples(samples, pareto_mask)

        return advantages

    def compute_gdpo(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
    ) -> torch.Tensor:
        """Compute advantages using the GDPO (Group-wise DPO) strategy.

        Unlike :meth:`compute_weighted_sum`, which first aggregates all
        rewards into a single scalar then normalises, GDPO normalises each
        reward **independently** within its group before combining.  This
        prevents a single high-variance reward from dominating the advantage
        signal.

        **Source-aware aggregation**: per-reward group statistics are
        computed only over applicable group members.  Under the
        homogeneous-batch design (plan §6.7) a reward is either
        applicable to ALL K samples of a group or to NONE — so GDPO's
        per-(reward, group) normalisation either fires or is skipped
        entirely for that pair.  Mixed applicability within a group is
        an asserted error (caught upstream in
        ``_compute_groupwise_group``).

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments; also gather the
           per-(reward, sample) applicability matrix.
        2. **Per-reward, per-group, per-applicable normalisation**.
        3. **Combine** — sum per-reward normalised contributions.
        4. **Batch normalisation** — compute global mean and std and
           normalise.
        5. **To-local** — convert back to local-rank tensor.
        6. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.
        """
        gathered_rewards, group_indices, source_ids = self.collect_group_rewards(samples, rewards)
        reward_keys = list(gathered_rewards.keys())

        # ---- Crossover stats ----
        if self._log_crossover_rewards:
            self._build_crossover_stats(gathered_rewards, group_indices, samples)

        # ---- Pareto filtering & child-mask (crossover-only paths) ----
        # Non-crossover training: none of these gates activate, norm_mask stays
        # None → all samples participate in mean/std (original behavior).
        pareto_mask: Optional[np.ndarray] = None
        norm_mask: Optional[np.ndarray] = None
        child_mask: Optional[np.ndarray] = None
        crossover_active = self._pareto_enabled or self._log_crossover_rewards
        if crossover_active:
            child_mask = self._build_child_mask(samples, group_indices)
            if self._pareto_enabled:
                pareto_mask = self._filter_pareto(gathered_rewards, group_indices, child_mask)
            # Build norm_mask.  When _child_in_norm is True, children
            # participate in mean/std alongside parents.
            if pareto_mask is not None:
                norm_mask = pareto_mask
            else:
                norm_mask = None
            if not self._child_in_norm and child_mask is not None:
                norm_mask = norm_mask & ~child_mask if norm_mask is not None else ~child_mask
            # Safety: if norm_mask would exclude everything, fall back to all.
            if norm_mask is not None and not norm_mask.any():
                norm_mask = np.ones(len(group_indices), dtype=bool)

        # ---- Per-child reward details (after filtering decision) ----
        if self._log_crossover_rewards:
            keep_mask = pareto_mask if pareto_mask is not None else None
            self._log_child_details(gathered_rewards, group_indices, samples, keep_mask)

        applicable, weight_matrix = self.build_source_aware_matrices(
            samples, reward_keys, source_ids
        )

        # Build stack and validate: NaN at applicable position == reward-model bug.
        stack = np.stack([gathered_rewards[k].astype(np.float64) for k in reward_keys], axis=0)
        self._assert_no_nan_at_applicable(stack, reward_keys, applicable, label="GDPO")

        # ---- Stddev reweighting (optional) ----
        if self.stddev_reweighting:
            weight_matrix = self._compute_stddev_weights(
                stack,
                weight_matrix,
                group_indices,
                applicable,
                reward_keys=reward_keys,
                norm_mask=norm_mask,
            )

        # Per-reward group-wise normalisation.  When *norm_mask* is None
        # (non-crossover training), all applicable samples participate in both
        # statistics and output.
        all_reward_advantages = []
        for r_idx, key in enumerate(reward_keys):
            reward_array = gathered_rewards[key].astype(np.float64)
            r_applicable = (
                applicable[r_idx] & norm_mask if norm_mask is not None else applicable[r_idx]
            )
            reward_adv = self._group_normalize(reward_array, group_indices, mask=r_applicable)
            # Zero out positions where this reward isn't applicable.
            reward_adv[~applicable[r_idx]] = 0.0
            all_reward_advantages.append(reward_adv * weight_matrix[r_idx])

        # Combine and batch normalise.
        weight_per_s = (applicable * weight_matrix).sum(axis=0)
        if pareto_mask is not None:
            kept_w = weight_per_s[pareto_mask]
        else:
            kept_w = weight_per_s
        if (kept_w == 0).any():
            bad = np.where(kept_w == 0)[0].tolist()
            raise RuntimeError(
                "GDPO: samples at indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''} have NO applicable "
                "reward. Check `RewardArguments.applicable_datasets` coverage."
            )

        combined_advantages = np.sum(all_reward_advantages, axis=0)
        values_for_bn = combined_advantages if norm_mask is None else combined_advantages[norm_mask]
        bn_mean, bn_std = self._global_mean_std(values_for_bn)
        advantages = (combined_advantages - bn_mean) / bn_std

        all_prompts, all_rewards, all_unique_ids, all_advantages, all_applicable = (
            self._gather_for_logging(
                samples,
                gathered_rewards,
                group_indices,
                advantages=advantages,
                applicable=applicable,
            )
        )

        # stat_mask needs to match gathered rewards in group_contiguous mode.
        if child_mask is not None and self.group_on_same_rank:
            flag = torch.tensor(child_mask.astype(np.float32), device=self.accelerator.device)
            log_child_mask = self._gather_uneven(flag).cpu().numpy().astype(bool)
        else:
            log_child_mask = child_mask
        log_stat_mask = ~log_child_mask if log_child_mask is not None else None

        self._pending_advantage_metrics = self._build_gdpo_log_data(
            all_rewards,
            all_unique_ids,
            all_advantages,
            bn_mean,
            bn_std,
            samples,
            all_prompts,
            applicable=all_applicable,
            reward_keys=reward_keys,
            all_reward_advantages=all_reward_advantages,
            stat_mask=log_stat_mask,
        )

        # ---- Log stddev effective weights (optional) ----
        if self.stddev_reweighting:
            for r_idx, key in enumerate(reward_keys):
                w = weight_matrix[r_idx]
                w_pos = w[w > 0]
                if len(w_pos) > 0:
                    self._pending_advantage_metrics[f"train/stddev_effective_weight_{key}_mean"] = (
                        float(np.mean(w_pos))
                    )
                    self._pending_advantage_metrics[f"train/stddev_effective_weight_{key}_std"] = (
                        float(np.std(w_pos))
                    )

        # Scale child advantages for warmup (logged values remain unscaled).
        if child_mask is not None and self._child_advantage_scale != 1.0:
            advantages[child_mask] *= self._child_advantage_scale

        # Scatter & store (GDPO)
        advantages = self._to_local(advantages)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv

        # Mark dominated local samples
        if pareto_mask is not None:
            self._mark_dominated_samples(samples, pareto_mask)

        return advantages

    # ------------------------------------------------------------------
    # Crossover stats + Pareto filtering
    # ------------------------------------------------------------------

    def pop_all_stats(self) -> Dict[str, Any]:
        """Return merged crossover, Pareto, and advantage metrics for logging.

        Call once per :meth:`compute_advantages`.  Returns an empty dict if
        nothing was produced.
        """
        out: Dict[str, Any] = {}
        if self._pending_crossover_stats:
            out.update(self._pending_crossover_stats)
            self._pending_crossover_stats = None
        if self._pending_pareto_stats:
            out.update(self._pending_pareto_stats)
            self._pending_pareto_stats = None
        adv = self.pop_advantage_metrics()
        if adv:
            out.update(adv)
        return out

    def _log_child_details(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        samples: List[BaseSample],
        pareto_mask: Optional[np.ndarray],
    ) -> None:
        """Log per-child reward details (kept vs discarded) for JSONL / pkl.

        Gathers child data across ranks only in distributed mode, matching
        the logic in :meth:`_build_child_mask`.
        """
        local_mask = np.array(
            [s.extra_kwargs.get("is_crossover_child", False) for s in samples],
            dtype=bool,
        )
        gathered_len = len(group_indices)
        if len(local_mask) < gathered_len:
            # Distributed mode: gather child flags and metadata across ranks.
            local_flag = torch.tensor(local_mask.astype(np.float32), device=self.accelerator.device)
            child_mask = self.accelerator.gather(local_flag).cpu().numpy().astype(bool)
            samples = self._gather_sample_meta(samples, gathered_len)
        else:
            child_mask = local_mask

        if not child_mask.any():
            return

        keep_mask = pareto_mask if pareto_mask is not None else np.ones(len(child_mask), dtype=bool)
        self._build_child_reward_details(
            gathered_rewards, child_mask, keep_mask, samples, group_indices
        )

    def _gather_sample_meta(
        self, samples: List[BaseSample], gathered_len: int
    ) -> List[Dict[str, Any]]:
        """Gather per-sample crossover metadata (prompt, step, strategy) across ranks.

        Returns a list of *gathered_len* lightweight dicts so that
        ``_build_child_reward_details`` can read metadata for every sample
        in the global reward array.
        """
        device = self.accelerator.device
        B = len(samples)

        # -- crossover_step: int → tensor --
        steps = torch.tensor(
            [s.extra_kwargs.get("crossover_step", -1) for s in samples],
            dtype=torch.long,
            device=device,
        )
        all_steps = self.accelerator.gather(steps).cpu().tolist()

        # -- crossover_strategy: short string → padded tensor --
        strat_bytes = [
            (s.extra_kwargs.get("crossover_strategy") or "").encode("utf-8") for s in samples
        ]
        max_sl = max((len(b) for b in strat_bytes), default=0)
        max_sl_t = torch.tensor([max_sl], dtype=torch.long, device=device)
        global_max_sl = int(self.accelerator.gather(max_sl_t).max().item())
        spadded = torch.tensor(
            [b + b"\x00" * (global_max_sl - len(b)) for b in strat_bytes],
            dtype=torch.uint8,
            device=device,
        )
        all_strat = [
            bytes(row).rstrip(b"\x00").decode("utf-8") or None
            for row in self.accelerator.gather(spadded).cpu().numpy()
        ]

        # -- prompt: string → padded tensor (same as _gather_for_logging) --
        prompt_bytes = [(s.prompt or "").encode("utf-8") for s in samples]
        max_pl = max((len(b) for b in prompt_bytes), default=0)
        max_pl_t = torch.tensor([max_pl], dtype=torch.long, device=device)
        global_max_pl = int(self.accelerator.gather(max_pl_t).max().item())
        ppadded = torch.tensor(
            [b + b"\x00" * (global_max_pl - len(b)) for b in prompt_bytes],
            dtype=torch.uint8,
            device=device,
        )
        all_prompts = [
            bytes(row).rstrip(b"\x00").decode("utf-8") or None
            for row in self.accelerator.gather(ppadded).cpu().numpy()
        ]

        # Assemble lightweight dict per global sample
        return [
            {
                "prompt": all_prompts[i],
                "crossover_step": all_steps[i],
                "crossover_strategy": all_strat[i],
            }
            for i in range(gathered_len)
        ]

    def _build_crossover_stats(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        samples: List[BaseSample],
    ) -> None:
        """Build per-reward statistics separately for parent vs child samples.

        Requires that crossover children have ``is_crossover_child = True``
        in their ``extra_kwargs``.
        """
        # Only build stats when there are tagged child samples.
        has_tag = any(s.extra_kwargs.get("is_crossover_child", False) for s in samples)
        if not has_tag:
            self._pending_crossover_stats = None
            return

        child_mask = self._build_child_mask(samples, group_indices)
        parent_mask = ~child_mask

        stats: Dict[str, Any] = {}
        for key in sorted(gathered_rewards.keys()):
            arr = gathered_rewards[key]
            valid = ~np.isnan(arr)
            # Parents
            p_mask = parent_mask & valid
            if p_mask.any():
                stats[f"crossover/parent_{key}_mean"] = float(arr[p_mask].mean())
                stats[f"crossover/parent_{key}_std"] = float(arr[p_mask].std())
            # Children
            c_mask = child_mask & valid
            if c_mask.any():
                stats[f"crossover/child_{key}_mean"] = float(arr[c_mask].mean())
                stats[f"crossover/child_{key}_std"] = float(arr[c_mask].std())
            # Fraction of children exceeding their parent (approximate)
            if p_mask.any() and c_mask.any():
                parent_mean = arr[p_mask].mean()
                better = (arr[c_mask] > parent_mean).sum()
                stats[f"crossover/child_better_{key}"] = float(better) / max(1, c_mask.sum())

        self._pending_crossover_stats = stats

    def _build_child_reward_details(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        child_mask: np.ndarray,
        pareto_mask: np.ndarray,
        samples: List[BaseSample],
        group_indices: np.ndarray,
    ) -> None:
        """Record per-child rewards with keep/discard status.

        * Per-child list → ``crossover/children_rewards`` (JSONL / pkl).
        * Per-reward scalar aggregates → ``crossover/child_kept_{rw}_mean`` etc.
          (TensorBoard / WandB).

        Uses *group_indices* to determine each child's prompt group rather
        than relying on the child's ``prompt`` field, which may be inherited
        from an unrelated template parent during crossover generation.
        """
        if not child_mask.any():
            return

        if self._pending_crossover_stats is None:
            self._pending_crossover_stats = {}

        child_indices = np.where(child_mask)[0]
        parent_mask = ~child_mask
        reward_keys = sorted(gathered_rewards.keys())

        # Build group_idx → sequential prompt_idx mapping using PARENTS only.
        # Children may carry an incorrect prompt (inherited from the denoising
        # template), but group_indices is always authoritative.
        group_to_prompt_idx: Dict[int, int] = {}
        for gi, is_parent in zip(group_indices, parent_mask):
            if is_parent and gi not in group_to_prompt_idx:
                group_to_prompt_idx[int(gi)] = len(group_to_prompt_idx)

        # ---- Per-child records (prompt index + crossover provenance) ----
        child_records: List[Dict[str, Any]] = []
        for ci in child_indices:
            meta = samples[ci]
            if isinstance(meta, dict):
                cxo_step = meta["crossover_step"]
                cxo_strategy = meta["crossover_strategy"]
                cxo_meta = None  # gathered dicts don't carry full crossover_meta
            else:
                cxo_step = meta.extra_kwargs.get("crossover_step")
                cxo_strategy = meta.extra_kwargs.get("crossover_strategy")
                cxo_meta = meta.extra_kwargs.get("crossover_meta")

            gi = int(group_indices[ci])
            record: Dict[str, Any] = {
                "kept": bool(pareto_mask[ci]),
                "prompt_idx": group_to_prompt_idx.get(gi, -1),
                "rewards": {k: float(gathered_rewards[k][ci]) for k in reward_keys},
            }
            if cxo_step is not None:
                record["crossover_step"] = cxo_step
            if cxo_strategy is not None:
                record["crossover_strategy"] = cxo_strategy
            if cxo_meta is not None:
                record["crossover_meta"] = cxo_meta
            child_records.append(record)
        self._pending_crossover_stats["crossover/children_rewards"] = child_records
        self._pending_crossover_stats["crossover/child_kept_total"] = sum(
            1 for r in child_records if r["kept"]
        )
        self._pending_crossover_stats["crossover/child_disc_total"] = sum(
            1 for r in child_records if not r["kept"]
        )

        # ---- Per-reward mean summaries (mean is reward-specific; count is not) ----
        for key in reward_keys:
            arr = gathered_rewards[key]
            child_kept = arr[child_mask & pareto_mask]
            child_disc = arr[child_mask & ~pareto_mask]

            nk, nd = len(child_kept), len(child_disc)
            self._pending_crossover_stats[f"crossover/child_kept_{key}_mean"] = (
                float(child_kept.mean()) if nk > 0 else 0.0
            )
            self._pending_crossover_stats[f"crossover/child_disc_{key}_mean"] = (
                float(child_disc.mean()) if nd > 0 else 0.0
            )

    def _build_child_mask(
        self,
        samples: List[BaseSample],
        group_indices: np.ndarray,
    ) -> np.ndarray:
        """Build a boolean mask identifying crossover child samples.

        Returns ``(S,)`` bool array where ``True`` = child sample.
        Handles both ``group_contiguous`` (local) and distributed modes.
        """
        child_mask = np.array(
            [s.extra_kwargs.get("is_crossover_child", False) for s in samples],
            dtype=bool,
        )
        gathered_len = len(group_indices)
        if len(child_mask) < gathered_len:
            local_flag = torch.tensor(child_mask.astype(np.float32), device=self.accelerator.device)
            gathered_flag = self.accelerator.gather(local_flag).cpu().numpy()
            child_mask = gathered_flag.astype(bool)
        return child_mask

    def _filter_pareto(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        child_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply Pareto filtering and return a keep mask.

        Args:
            gathered_rewards: Per-reward scores gathered across all ranks.
            group_indices: ``(S,)`` integer group assignments.
            child_mask: ``(S,)`` boolean — ``True`` for child samples
                (pre-computed by caller via :meth:`_build_child_mask`).

        Returns:
            ``(S,)`` bool — ``True`` for non-dominated (keep).
        """
        from ..trainers.crossover.pareto import filter_by_group

        parent_mask = ~child_mask
        pareto_mask, stats = filter_by_group(gathered_rewards, group_indices, parent_mask)
        self._pending_pareto_stats = {f"pareto/{k}": v for k, v in stats.items()}

        return pareto_mask

    def _mark_dominated_samples(
        self,
        samples: List[BaseSample],
        pareto_mask: np.ndarray,
    ) -> None:
        """Set ``_is_dominated`` flag on local samples that were filtered out."""
        # pareto_mask is gathered size (S,).  In distributed mode we need
        # to scatter back to local ranks.
        if self.group_on_same_rank:
            local_mask = pareto_mask
        else:
            world_size = self.accelerator.num_processes
            local_size = len(samples)
            rank = self.accelerator.process_index
            local_mask = pareto_mask[rank * local_size : (rank + 1) * local_size]

        for i, sample in enumerate(samples):
            sample.extra_kwargs["_is_dominated"] = not bool(local_mask[i])

    # ------------------------------------------------------------------
    # Log payloads (trainers pass to ``log_data``)
    # ------------------------------------------------------------------

    def _build_base_log_stats(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        applicable: Optional[np.ndarray],
        reward_keys: Optional[List[str]],
        stat_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, bool]]]:
        """Shared boilerplate for both log-data builders.

        Returns (stat_arrays, r_applicable) where stat_arrays is ready
        for ``_batch_reduce_stats`` and r_applicable maps each reward
        key to its boolean mask over gathered samples.

        When *stat_mask* is provided, reward statistics are further
        restricted (e.g., parents only for comparable ``train/reward_*``
        metrics).
        """
        keys_sorted = sorted(gathered_rewards.keys())
        if applicable is not None and reward_keys is not None:
            r_applicable = {k: applicable[reward_keys.index(k)] for k in keys_sorted}
        else:
            r_applicable = {k: np.ones(len(gathered_rewards[k]), dtype=bool) for k in keys_sorted}

        stat_arrays: Dict[str, np.ndarray] = {}
        for key in keys_sorted:
            mask_k = r_applicable[key]
            if stat_mask is not None:
                mask_k = mask_k & stat_mask
            masked_rewards = gathered_rewards[key][mask_k]
            masked_gids = group_indices[mask_k]
            stat_arrays[f"reward_{key}"] = masked_rewards
            group_means, group_stds = RewardProcessor.compute_group_reward_stats(
                masked_rewards, masked_gids
            )
            stat_arrays[f"reward_{key}_g_stds"] = group_stds
            stat_arrays[f"reward_{key}_g_means"] = group_means

        return stat_arrays, r_applicable

    def _unpack_per_reward_log_data(
        self,
        all_stats: Dict[str, Dict[str, float]],
        gathered_rewards: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Unpack per-reward stats common to both log-data builders."""
        _log_data: Dict[str, Any] = {}
        keys_sorted = sorted(gathered_rewards.keys())
        for key in keys_sorted:
            reward_stats = all_stats[f"reward_{key}"]
            _log_data[f"train/reward_{key}_mean"] = reward_stats["mean"]
            _log_data[f"train/reward_{key}_std"] = reward_stats["std"]
            group_std_stats = all_stats[f"reward_{key}_g_stds"]
            group_mean_stats = all_stats[f"reward_{key}_g_means"]
            _log_data[f"train/reward_{key}_group_std_mean"] = group_std_stats["mean"]
            _log_data[f"train/reward_{key}_group_std_max"] = group_std_stats["max"]
            _log_data[f"train/reward_{key}_group_std_min"] = group_std_stats["min"]
            _log_data[f"train/reward_{key}_group_mean_std"] = group_mean_stats["std"]
        return _log_data

    def _build_weighted_sum_log_data(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        aggregated_rewards: np.ndarray,
        advantages: np.ndarray,
        samples: List[BaseSample],
        all_prompts: List[str],
        applicable: Optional[np.ndarray] = None,
        reward_keys: Optional[List[str]] = None,
        stat_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        stat_arrays, r_applicable = self._build_base_log_stats(
            gathered_rewards, group_indices, applicable, reward_keys, stat_mask=stat_mask
        )

        stat_arrays["reward_agg"] = aggregated_rewards
        agg_group_means, agg_group_stds = RewardProcessor.compute_group_reward_stats(
            aggregated_rewards, group_indices
        )
        stat_arrays["reward_agg_g_stds"] = agg_group_stds
        stat_arrays["reward_agg_g_means"] = agg_group_means
        stat_arrays["adv"] = advantages
        stat_arrays["adv_abs"] = np.abs(advantages)

        all_stats = self._batch_reduce_stats(stat_arrays)

        _log_data = self._unpack_per_reward_log_data(all_stats, gathered_rewards)
        _log_data["train/reward_mean"] = all_stats["reward_agg"]["mean"]
        _log_data["train/reward_std"] = all_stats["reward_agg"]["std"]

        agg_group_std_stats = all_stats["reward_agg_g_stds"]
        agg_group_mean_stats = all_stats["reward_agg_g_means"]
        _log_data["train/reward_group_std_mean"] = agg_group_std_stats["mean"]
        _log_data["train/reward_group_std_max"] = agg_group_std_stats["max"]
        _log_data["train/reward_group_mean_std"] = agg_group_mean_stats["std"]

        # Zero-std ratio (count-based; requires a separate all-reduce)
        _log_data["train/reward_zero_std_ratio"] = self._metric_zero_std_ratio(
            aggregated_rewards, group_indices
        )

        # Unpack advantage stats
        adv_stats = all_stats["adv"]
        _log_data["train/adv_min"] = adv_stats["min"]
        _log_data["train/adv_max"] = adv_stats["max"]
        _log_data["train/adv_abs_mean"] = all_stats["adv_abs"]["mean"]

        pos = advantages > 0
        neg = advantages < 0
        zero = advantages == 0
        n = len(advantages)
        _log_data["train/adv_pos_ratio"] = float(pos.sum() / n)
        _log_data["train/adv_neg_ratio"] = float(neg.sum() / n)
        _log_data["train/adv_zero_ratio"] = float(zero.sum() / n)
        _log_data["train/adv_pos_sum"] = float(advantages[pos].sum())

        self._add_train_samples(_log_data, samples)
        return self._finalize_log_data(_log_data, gathered_rewards, group_indices, all_prompts)

    def _build_gdpo_log_data(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        advantages: np.ndarray,
        bn_mean: float,
        bn_std: float,
        samples: List[BaseSample],
        all_prompts: List[str],
        applicable: Optional[np.ndarray] = None,
        reward_keys: Optional[List[str]] = None,
        all_reward_advantages: Optional[List[np.ndarray]] = None,
        stat_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        stat_arrays, r_applicable = self._build_base_log_stats(
            gathered_rewards, group_indices, applicable, reward_keys, stat_mask=stat_mask
        )

        stat_arrays["adv"] = advantages
        stat_arrays["adv_abs"] = np.abs(advantages)

        all_stats = self._batch_reduce_stats(stat_arrays)

        _log_data = self._unpack_per_reward_log_data(all_stats, gathered_rewards)

        keys_sorted = sorted(gathered_rewards.keys())
        for key in keys_sorted:
            mask_k = r_applicable[key]
            _log_data[f"train/reward_{key}_zero_std_ratio"] = self._metric_zero_std_ratio(
                gathered_rewards[key][mask_k], group_indices[mask_k]
            )

        adv_stats = all_stats["adv"]
        _log_data.update(
            {
                "train/batch_norm_mean": bn_mean,
                "train/batch_norm_std": bn_std,
                "train/adv_min": adv_stats["min"],
                "train/adv_max": adv_stats["max"],
                "train/adv_abs_mean": all_stats["adv_abs"]["mean"],
            }
        )
        self._add_train_samples(_log_data, samples)
        if all_reward_advantages is not None and reward_keys is not None:
            for r_idx, name in enumerate(reward_keys):
                adv = all_reward_advantages[r_idx]
                pos = adv > 0
                neg = adv < 0
                zero = adv == 0
                n = len(adv)
                _log_data[f"train/reward_{name}_adv_pos_ratio"] = float(pos.sum() / n)
                _log_data[f"train/reward_{name}_adv_neg_ratio"] = float(neg.sum() / n)
                _log_data[f"train/reward_{name}_adv_zero_ratio"] = float(zero.sum() / n)
                _log_data[f"train/reward_{name}_adv_pos_sum"] = float(adv[pos].sum())

        return self._finalize_log_data(_log_data, gathered_rewards, group_indices, all_prompts)

    def _group_rewards_by_prompt(
        self,
        prompts: List[str],
        group_indices: np.ndarray,
        gathered_rewards: Dict[str, np.ndarray],
    ) -> list:
        """Return per-prompt reward groups for offline analysis."""
        groups = {}
        for i, prompt in enumerate(prompts):
            gid = int(group_indices[i])
            if gid not in groups:
                groups[gid] = {
                    "prompt": prompt,
                    "rewards": {name: [] for name in gathered_rewards},
                }
            for name in gathered_rewards:
                groups[gid]["rewards"][name].append(float(gathered_rewards[name][i]))
        return list(groups.values())

    def _add_train_samples(self, log_data: Dict[str, Any], samples: List[BaseSample]) -> None:
        """Split samples into parents and children, each capped separately.

        Grouped by ``unique_id`` so filenames include group id, e.g.
        ``train_samples_g42_0_image.png``.
        """
        cap = self.max_log_samples
        parents: Dict[str, List[BaseSample]] = {}
        children: Dict[str, List[BaseSample]] = {}
        n_parents = 0
        n_children = 0
        for s in samples:
            gid = str(s.unique_id)
            if s.extra_kwargs.get("is_crossover_child", False):
                if cap is not None and n_children >= cap:
                    continue
                children.setdefault(gid, []).append(s)
                n_children += 1
            else:
                if cap is not None and n_parents >= cap:
                    continue
                parents.setdefault(gid, []).append(s)
                n_parents += 1
        log_data["train_samples"] = parents
        if children:
            log_data["train_child_samples"] = children

    def _finalize_log_data(
        self,
        _log_data: Dict[str, Any],
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        all_prompts: List[str],
    ) -> Dict[str, Any]:
        """Common tail for both log-data builders: percentiles + per-prompt groups."""
        for name in gathered_rewards:
            arr = gathered_rewards[name]
            for q in [0, 25, 50, 75, 100]:
                _log_data[f"train/reward_{name}_p{q}"] = float(np.percentile(arr, q))

        groups = self._group_rewards_by_prompt(all_prompts, group_indices, gathered_rewards)
        _log_data["train/prompts"] = [g["prompt"] for g in groups]
        _log_data["train/rewards"] = groups
        return _log_data
