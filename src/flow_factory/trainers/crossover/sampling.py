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

# src/flow_factory/trainers/crossover/sampling.py
"""
Shared sampling utilities for crossover trainers.

Provides a reusable denoising-loop helper and crossover-step resolution
so that both coupled (CrossoverGRPOGuard) and decoupled (CrossoverNFT)
trainers can share the same low-level logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator

from ...models.abc import BaseAdapter
from ...utils.base import filter_kwargs

# Avoid circular import; CrossoverArguments is referenced only via Any/type hints
_CrossoverArguments = Any

# ============================================================================
# Step resolution
# ============================================================================


def resolve_crossover_step(
    step_spec: Union[float, int],
    num_inference_steps: int,
) -> int:
    """Convert a crossover step specification to a concrete step index.

    Args:
        step_spec:
            - ``float`` in ``(0, 1)`` → fraction of *num_inference_steps*
              (e.g., 0.5 means the halfway point).
            - ``int`` → absolute step index (0-based within the denoising
              schedule).
        num_inference_steps: Total number of inference steps (T).

    Returns:
        Integer step index in ``[1, num_inference_steps - 1]``.  Crossover
        is never applied at step 0 (pure noise) or at the last step.
    """
    if isinstance(step_spec, float):
        if not 0.0 < step_spec < 1.0:
            raise ValueError(f"Float crossover step must be in (0, 1), got {step_spec}")
        step = max(1, int(num_inference_steps * step_spec))
    else:
        step = int(step_spec)

    # Clamp to valid range: at least step 1, at most T-1
    step = max(1, min(step, num_inference_steps - 1))
    return step


def sample_crossover_step(
    crossover_args: _CrossoverArguments,
    num_inference_steps: int,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample a crossover step index according to the configured strategy.

    Args:
        crossover_args: :class:`CrossoverArguments` instance.
        num_inference_steps: Total number of inference steps (T).
        generator: Optional RNG for reproducibility.

    Returns:
        Integer step index in ``[1, num_inference_steps - 1]``.
    """
    sampling = getattr(crossover_args, "step_sampling", "fixed")

    if sampling == "fixed":
        step_spec = crossover_args.step
    elif sampling == "uniform":
        lo, hi = crossover_args.step_range
        # torch.rand does not accept non-CPU generators; always use CPU.
        cpu_gen = torch.Generator()
        if generator is not None:
            cpu_gen.manual_seed(generator.initial_seed())
        frac = torch.rand(1, generator=cpu_gen).item()
        step_spec = lo + (hi - lo) * frac
    else:
        raise ValueError(f"Unknown step_sampling: {sampling}.  Options: 'fixed', 'uniform'.")

    return resolve_crossover_step(step_spec, num_inference_steps)


# ============================================================================
# Denoising loop
# ============================================================================


def run_denoising_phase(
    adapter: BaseAdapter,
    accelerator: Accelerator,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    start_idx: int,
    end_idx: int,
    batch: Dict[str, Any],
    training_args: Any,
    *,
    compute_log_prob: bool = True,
    collect_trajectory: bool = False,
    extra_call_back_kwargs: Optional[List[str]] = None,
    collect_callbacks: bool = False,
    autocast_ctx: Any = None,
) -> Tuple[
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    Optional[Dict[str, List[torch.Tensor]]],
]:
    """Run a contiguous segment of the denoising loop.

    Iterates from ``timesteps[start_idx]`` to ``timesteps[end_idx - 1]``
    (inclusive start, exclusive end), calling ``adapter.forward()`` for each
    step.  Optionally collects intermediate latents, log-probabilities, and
    callback values (e.g., ``next_latents_mean``).

    Args:
        adapter: The model adapter providing ``forward()`` and ``cast_latents()``.
        accelerator: The HuggingFace Accelerator instance (used for autocast).
        latents: Initial latents at position *start_idx*, shape ``(B, *latent_dims)``.
        timesteps: 1-D tensor of scheduler timesteps for the full schedule
            ``[0, 1000]`` scale.
        start_idx: First step index to execute (inclusive).
        end_idx: Last step index + 1 (exclusive).  ``end_idx - start_idx``
            steps are run.
        batch: The raw batch dict containing prompt embeddings, metadata, etc.
        training_args: Training arguments dataclass (fields are spread into
            forward kwargs via ``filter_kwargs``).
        compute_log_prob: If ``True``, request log-probability from the
            scheduler step (only meaningful for SDE steps with noise).
        collect_trajectory: If ``True``, collect a list of latents after each
            step (position *i+1*) and per-step log-probabilities.
        extra_call_back_kwargs: Optional extra keys to request from
            ``adapter.forward()`` via ``return_kwargs``.
        collect_callbacks: If ``True``, also collect per-step callback values
            (one list per key) and return them as the 4th tuple element.

    Returns:
        Tuple of ``(final_latents, latents_list, log_probs_list, callbacks)``.

        - *final_latents*: Latents after the last executed step.
        - *latents_list*: List of tensors, or empty if *collect_trajectory* False.
        - *log_probs_list*: List of tensors, or empty if *collect_trajectory* False.
        - *callbacks*: ``{key: [tensor_per_step]}`` if *collect_callbacks* True,
          else ``None``.
    """
    latents_list: List[torch.Tensor] = []
    log_probs_list: List[torch.Tensor] = []
    callbacks: Optional[Dict[str, List[torch.Tensor]]] = (
        {k: [] for k in extra_call_back_kwargs}
        if (collect_callbacks and extra_call_back_kwargs)
        else None
    )
    device = accelerator.device
    scheduler = adapter.scheduler

    if extra_call_back_kwargs is None:
        extra_call_back_kwargs = []

    for i in range(start_idx, end_idx):
        t = timesteps[i]
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0, device=device)

        # Determine whether SDE noise is active at this step
        current_noise_level = scheduler.get_noise_level_for_timestep(t)
        compute_lp = compute_log_prob and (current_noise_level > 0)

        # Build forward kwargs — same pattern as the GRPO optimize loop
        return_kwargs = ["next_latents"]
        if compute_lp:
            return_kwargs.append("log_prob")
        return_kwargs.extend(extra_call_back_kwargs)

        forward_inputs = {
            **training_args,
            "t": t,
            "t_next": t_next,
            "latents": latents,
            "compute_log_prob": compute_lp,
            "noise_level": current_noise_level,
            "return_kwargs": return_kwargs,
            **batch,
        }
        forward_inputs = filter_kwargs(adapter.forward, **forward_inputs)

        ac = autocast_ctx() if autocast_ctx is not None else accelerator.autocast()
        with ac:
            output = adapter.forward(**forward_inputs)

        # Advance latents
        latents = output.next_latents

        # Optionally collect trajectory
        if collect_trajectory:
            latents_list.append(latents)
            if compute_lp and output.log_prob is not None:
                log_probs_list.append(output.log_prob)
            else:
                log_probs_list.append(
                    torch.zeros(latents.shape[0], device=device, dtype=latents.dtype)
                )

        # Optionally collect callback values
        if callbacks is not None:
            for key in callbacks:
                val = getattr(output, key, None)
                if val is not None:
                    callbacks[key].append(val.detach())

    return latents, latents_list, log_probs_list, callbacks
