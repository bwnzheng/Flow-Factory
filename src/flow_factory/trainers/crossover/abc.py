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

# src/flow_factory/trainers/crossover/abc.py
"""
Abstract base class for crossover strategies.

Provides the pluggable interface for augmenting intermediate denoising states
via crossover operations during training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class CrossoverOutput:
    """Output of a crossover operation.

    Attributes:
        child_latents: Tensor of shape ``(M, *latent_dims)`` — the M child latents
            produced by crossover.
        parent_indices: LongTensor of shape ``(M,)`` mapping each child to its
            primary parent index (0-based within the parent group).
        metadata: Optional dict of debug/logging information (e.g., alpha values,
            mixing ratios).
    """

    child_latents: torch.Tensor
    parent_indices: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCrossover(ABC):
    """Pluggable crossover strategy for intermediate denoising states.

    During training, K parent latents at a configurable crossover timestep are
    fed into :meth:`crossover` to produce M child latents (M ≥ 2).  Each child
    then continues denoising independently, effectively augmenting the training
    batch from K to (optionally) K + M samples.

    Subclasses implement specific combination strategies (uniform interpolation,
    block swapping, etc.).

    Args:
        augmentation_factor: Target ratio M / K.  The actual number of children
            is ``max(2, ceil(K * augmentation_factor))``.
    """

    def __init__(self, augmentation_factor: float = 2.0) -> None:
        self._augmentation_factor = float(augmentation_factor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def crossover(
        self,
        parent_latents: torch.Tensor,
        parent_aux: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> CrossoverOutput:
        """Apply crossover to a group of parent latents.

        Args:
            parent_latents: Latents at the same denoising step, shape
                ``(K, *latent_dims)`` where K is the number of parents.
            parent_aux: Optional auxiliary per-parent tensors (e.g., timestep,
                noise level).  Subclasses may ignore this.
            generator: Optional RNG for stochastic strategies.

        Returns:
            A :class:`CrossoverOutput` with M child latents.
        """
        ...

    @property
    def augmentation_factor(self) -> float:
        """The target M / K ratio."""
        return self._augmentation_factor

    def num_children(self, num_parents: int) -> int:
        """Return the number of children to produce for *num_parents* parents.

        Guaranteed to be at least 2.
        """
        return max(2, int(num_parents * self._augmentation_factor + 0.5))

    def extra_repr(self) -> str:
        """Human-readable string for logging / debugging."""
        return f"augmentation_factor={self._augmentation_factor}"
