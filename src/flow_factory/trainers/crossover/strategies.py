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

# src/flow_factory/trainers/crossover/strategies.py
"""
Built-in crossover strategies for intermediate denoising states.

Each strategy implements :class:`BaseCrossover` and can be selected via the
``crossover.strategy`` YAML field.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .abc import BaseCrossover, CrossoverOutput


def _cpu_generator(generator: Optional[torch.Generator] = None) -> torch.Generator:
    """Return a CPU generator with the same seed as *generator* (if given)."""
    cpu_gen = torch.Generator()
    if generator is not None:
        cpu_gen.manual_seed(generator.initial_seed())
    return cpu_gen


def _device_generator(
    generator: Optional[torch.Generator] = None, device: torch.device = None
) -> torch.Generator:
    """Return a generator on *device* with the same seed as *generator*."""
    if device is not None and device.type != "cpu":
        dev_gen = torch.Generator(device=device)
    else:
        dev_gen = torch.Generator()
    if generator is not None:
        dev_gen.manual_seed(generator.initial_seed())
    return dev_gen


# ============================================================================
# Strategy implementations
# ============================================================================


class UniformCrossover(BaseCrossover):
    r"""Random convex combination of two distinct parents.

    For each child, two distinct parents :math:`P_i, P_j` are chosen uniformly
    at random, and an interpolation weight :math:`\alpha \sim U(0, 1)` is
    sampled:

    .. math::

        \text{child} = \alpha \cdot P_i + (1 - \alpha) \cdot P_j
    """

    def __init__(self, augmentation_factor: float = 2.0, **kwargs: Any) -> None:
        super().__init__(augmentation_factor)

    def crossover(
        self,
        parent_latents: torch.Tensor,
        parent_aux: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> CrossoverOutput:
        K = parent_latents.shape[0]
        M = self.num_children(K)
        device = parent_latents.device
        dtype = parent_latents.dtype
        gen_dev = _device_generator(generator, device)
        gen_cpu = _cpu_generator(generator)

        # Pick two distinct parents for each child
        idx_i = torch.randint(0, K, (M,), device=device, generator=gen_dev)
        idx_j = torch.randint(0, K, (M,), device=device, generator=gen_dev)
        same = idx_i == idx_j
        if same.any():
            idx_j[same] = (idx_i[same] + 1) % K

        # Per-element alpha (broadcast over all non-batch dims)
        alpha_shape = (M,) + (1,) * (parent_latents.ndim - 1)
        alpha = torch.rand(alpha_shape, device=device, dtype=dtype, generator=gen_dev)

        children = alpha * parent_latents[idx_i] + (1.0 - alpha) * parent_latents[idx_j]

        return CrossoverOutput(
            child_latents=children,
            parent_indices_i=idx_i,
            parent_indices_j=idx_j,
            metadata={"alpha": alpha.flatten().tolist()},
        )

    def extra_repr(self) -> str:
        return f"uniform, {super().extra_repr()}"


class ConvexCrossover(BaseCrossover):
    r"""Smooth convex combination with Beta-distributed weights.

    Instead of uniform :math:`\alpha`, this strategy samples
    :math:`\alpha \sim \text{Beta}(a, a)`.  When *a* is large (e.g., 5.0),
    children cluster near the average of the two parents; when *a* is small
    (e.g., 0.5), children tend toward one parent or the other.

    Args:
        augmentation_factor: Target M / K ratio.
        beta_concentration: Concentration parameter *a* for the symmetric
            Beta distribution.  Must be > 0.
    """

    def __init__(
        self, augmentation_factor: float = 2.0, beta_concentration: float = 0.5, **kwargs: Any
    ) -> None:
        super().__init__(augmentation_factor)
        self.beta_concentration = float(beta_concentration)
        if self.beta_concentration <= 0:
            raise ValueError(f"beta_concentration must be > 0, got {self.beta_concentration}")

    def crossover(
        self,
        parent_latents: torch.Tensor,
        parent_aux: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> CrossoverOutput:
        K = parent_latents.shape[0]
        M = self.num_children(K)
        device = parent_latents.device
        dtype = parent_latents.dtype
        gen_dev = _device_generator(generator, device)

        idx_i = torch.randint(0, K, (M,), device=device, generator=gen_dev)
        idx_j = torch.randint(0, K, (M,), device=device, generator=gen_dev)
        same = idx_i == idx_j
        if same.any():
            idx_j[same] = (idx_i[same] + 1) % K

        # Sample from Beta(a, a) via Kumaraswamy approximation (uses torch.rand).
        def _sample_beta(shape, a, gen):
            u = torch.rand(shape, device=device, dtype=dtype, generator=gen)
            inv_a = 1.0 / a
            return (1.0 - (1.0 - u).pow(inv_a)).pow(inv_a)

        alpha = _sample_beta((M,), self.beta_concentration, gen_dev)
        alpha = alpha.view(M, *([1] * (parent_latents.ndim - 1)))

        children = alpha * parent_latents[idx_i] + (1.0 - alpha) * parent_latents[idx_j]

        return CrossoverOutput(
            child_latents=children,
            parent_indices_i=idx_i,
            parent_indices_j=idx_j,
            metadata={"alpha": alpha.flatten().tolist()},
        )

    def extra_repr(self) -> str:
        return f"convex(beta_a={self.beta_concentration}), {super().extra_repr()}"


class BlockCrossover(BaseCrossover):
    """Block-level random mixing between two distinct parents.

    The latent is divided into 3D blocks of size
    ``block_size_c × block_size × block_size`` (for image latents) or
    ``block_size`` tokens (for sequence latents).  For each child, a per-block
    binary mask is sampled.  Where 1, the entire block comes from the primary
    parent; where 0, from the secondary parent.

    Args:
        augmentation_factor: Target M / K ratio.
        mixing_ratio: Fraction of blocks from the primary parent.
        block_size: Spatial block size (H and W).  Default: 8.
        block_size_c: Channel block size.  ``1`` = no channel splitting
            (all channels move together).  ``4`` = blocks of 4 channels.
            Default: 1.
    """

    def __init__(
        self,
        augmentation_factor: float = 2.0,
        mixing_ratio: float = 0.5,
        block_size: int = 8,
        block_size_c: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(augmentation_factor)
        self.mixing_ratio = float(mixing_ratio)
        self.block_size = int(block_size)
        self.block_size_c = int(block_size_c)
        if not 0.0 < self.mixing_ratio < 1.0:
            raise ValueError(f"mixing_ratio must be in (0, 1), got {self.mixing_ratio}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.block_size_c < 1:
            raise ValueError(f"block_size_c must be >= 1, got {self.block_size_c}")

    def crossover(
        self,
        parent_latents: torch.Tensor,
        parent_aux: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> CrossoverOutput:
        K = parent_latents.shape[0]
        M = self.num_children(K)
        device = parent_latents.device
        dtype = parent_latents.dtype
        gen_dev = _device_generator(generator, device)

        idx_i = torch.randint(0, K, (M,), device=device, generator=gen_dev)
        idx_j = torch.randint(0, K, (M,), device=device, generator=gen_dev)
        same = idx_i == idx_j
        if same.any():
            idx_j[same] = (idx_i[same] + 1) % K

        latent_shape = parent_latents.shape[1:]  # (*spatial_dims)

        if parent_latents.ndim == 4:
            # (C, H, W) image latent → 3D block mask
            C, H, W = latent_shape
            bc = max(1, C // self.block_size_c)
            bh = max(1, H // self.block_size)
            bw = max(1, W // self.block_size)
            # mask at block resolution: (M, bc, bh, bw)
            # interpolate treats (N, C, H, W) as 2 spatial dims.  We need
            # 3 spatial dims (bc, bh, bw) → unsqueeze to (M, 1, bc, bh, bw).
            mask = torch.rand(M, 1, bc, bh, bw, device=device, generator=gen_dev)
            mask = mask < self.mixing_ratio
            mask = mask.float()
            mask = torch.nn.functional.interpolate(mask, size=(C, H, W), mode="nearest").squeeze(1)
            mask = mask.bool()
        elif parent_latents.ndim == 3:
            # (L, C) sequence latent
            L, _ = latent_shape
            bl = max(1, L // self.block_size)
            mask = torch.rand(M, 1, bl, device=device, generator=gen_dev)
            mask = mask < self.mixing_ratio
            mask = mask.float()
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(-1), size=(L, 1), mode="nearest"
            ).squeeze(-1)
            mask = mask.bool()
        else:
            raise ValueError(
                f"BlockCrossover expects 3D (L, C) or 4D (C, H, W) latents, "
                f"got shape {parent_latents.shape}"
            )

        children = torch.where(mask, parent_latents[idx_i], parent_latents[idx_j])

        return CrossoverOutput(
            child_latents=children,
            parent_indices_i=idx_i,
            parent_indices_j=idx_j,
            metadata={
                "mixing_ratio": self.mixing_ratio,
                "block_size": self.block_size,
                "block_size_c": self.block_size_c,
            },
        )

    def extra_repr(self) -> str:
        return (
            f"block(mix={self.mixing_ratio}, block=({self.block_size_c},{self.block_size})), "
            f"{super().extra_repr()}"
        )


# ============================================================================
# Strategy registry & factory
# ============================================================================

_CROSSOVER_REGISTRY: Dict[str, type] = {
    "uniform": UniformCrossover,
    "convex": ConvexCrossover,
    "block": BlockCrossover,
}


def create_crossover_strategy(
    name: str,
    augmentation_factor: float = 2.0,
    **strategy_kwargs: Any,
) -> BaseCrossover:
    """Create a crossover strategy instance by name."""
    cls = _CROSSOVER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown crossover strategy '{name}'. "
            f"Available: {list(_CROSSOVER_REGISTRY.keys())}"
        )
    return cls(augmentation_factor=augmentation_factor, **strategy_kwargs)


def list_crossover_strategies() -> Dict[str, type]:
    """Return a copy of the crossover strategy registry."""
    return dict(_CROSSOVER_REGISTRY)
