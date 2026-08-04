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

"""Training arguments for crossover-augmented algorithms.

Provides a shared :class:`CrossoverArguments` dataclass consumed by both
the coupled (CrossoverGRPOGuard) and decoupled (CrossoverNFT) trainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Union

from ..abc import ArgABC
from ._base import TrainingArguments
from .grpo import GRPOTrainingArguments
from .nft import NFTTrainingArguments


@dataclass
class CrossoverArguments(ArgABC):
    """Configuration for intermediate denoising-state crossover.

    All fields are namespaced under the ``crossover:`` key in the YAML config
    so they do not collide with base training arguments.
    """

    enabled: bool = field(
        default=True,
        metadata={"help": "Whether to apply crossover augmentation.  Set to False to disable."},
    )
    step: Union[float, int] = field(
        default=0.5,
        metadata={
            "help": (
                "Crossover position.  float in (0, 1) → fraction of num_inference_steps; "
                "int → absolute step index (0-based).  Default: 0.5 (halfway)."
            )
        },
    )
    step_sampling: Literal["fixed", "uniform"] = field(
        default="fixed",
        metadata={
            "help": (
                "How to choose the crossover step.  'fixed' uses the value of `step`; "
                "'uniform' samples uniformly from `step_range` each batch."
            )
        },
    )
    step_range: tuple = field(
        default=(0.2, 0.8),
        metadata={"help": "Range for uniform step sampling as (min_frac, max_frac) in (0, 1)."},
    )
    offspring_mode: Literal["crossover", "resample", "mutation"] = field(
        default="crossover",
        metadata={
            "help": (
                "How to generate offspring.  Mutually exclusive modes:\n"
                "  'crossover' — crossover between two parents + optional mutation (default).\n"
                "  'resample' — pure random noise, no parents involved.\n"
                "  'mutation'  — clone a single parent + Gaussian mutation, no crossover.\n"
                "When 'resample' or 'mutation', the ``strategy`` field is ignored."
            )
        },
    )
    strategy: str = field(
        default="uniform",
        metadata={
            "help": (
                "Crossover strategy name.  Options: 'uniform', 'convex', 'block'.  "
                "Ignored when ``offspring_mode`` is not 'crossover'."
            )
        },
    )
    augmentation_factor: float = field(
        default=2.0,
        metadata={
            "help": "Target ratio M/K of children to parents.  ceil(K * factor) children are produced."
        },
    )
    strategy_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Additional keyword arguments forwarded to the crossover strategy.  "
                "Examples: {'beta_concentration': 0.5} for 'convex', "
                "{'mixing_ratio': 0.5} for 'block'."
            )
        },
    )
    log_rewards: bool = field(
        default=True,
        metadata={"help": "Log per-reward statistics separately for parent and child samples."},
    )
    survivor_score: Literal["advantage", "abs_advantage", "covariance", "cov_per_sample"] = field(
        default="advantage",
        metadata={
            "help": (
                "Environmental-selection score for the merged parent-offspring pool. "
                "'advantage' prefers high-fitness samples; 'abs_advantage' also preserves "
                "strong negative samples; 'covariance' greedily scores complete "
                "Pareto-guided groups; 'cov_per_sample' preserves one scalar elite and "
                "fills remaining slots by frozen-pool per-sample covariance contribution."
            )
        },
    )
    evolution_generations: int = field(
        default=1,
        metadata={
            "help": (
                "Number of evolutionary generations.  Generation 0 crosses parents; "
                "each subsequent generation crosses the layer-0 children from the "
                "previous generation.  Evaluation uses ODE (deterministic) for "
                "unbiased fitness comparison.  Default 1 = single generation."
            )
        },
    )
    mutation_std: float = field(
        default=0.0,
        metadata={
            "help": (
                "Standard deviation of Gaussian noise added to crossover latents "
                "as mutation.  Applied after each crossover operation (both initial "
                "and re-crossover between generations).  0 = no mutation.  "
                "Typical values: 0.01–0.1 (relative to latent scale)."
            )
        },
    )
    parent_ratio: float = field(
        default=0.25,
        metadata={
            "help": (
                "Fraction of group selected as parents in the genetic algorithm.  "
                "Samples are sorted by advantage, and the top ``parent_ratio * K`` "
                "are chosen as parents for crossover.  Clamped to at least 2.  "
                "Default: 0.25 (top quarter of the group)."
            )
        },
    )


# ============================================================================
# Algorithm-specific TrainingArguments
# ============================================================================


@dataclass
class CrossoverGRPOGuardTrainingArguments(GRPOTrainingArguments):
    """GRPO-Guard training arguments with crossover augmentation.

    Inherits all GRPO / GRPO-Guard hyperparameters (clip_range, kl_beta,
    advantage_aggregation, etc.) and adds a ``crossover`` namespace.
    """

    crossover: CrossoverArguments = field(default_factory=CrossoverArguments)

    @classmethod
    def from_dict(cls, args_dict: Dict[str, Any]) -> "CrossoverGRPOGuardTrainingArguments":
        if "crossover" in args_dict and isinstance(args_dict["crossover"], dict):
            args_dict = dict(args_dict)
            args_dict["crossover"] = CrossoverArguments.from_dict(args_dict["crossover"])
        return super().from_dict(args_dict)  # type: ignore[return-value]


@dataclass
class CrossoverNFTTrainingArguments(NFTTrainingArguments):
    """DiffusionNFT training arguments with crossover augmentation.

    Inherits all NFT hyperparameters (nft_beta, off_policy, time_sampling, etc.)
    and adds a ``crossover`` namespace.
    """

    crossover: CrossoverArguments = field(default_factory=CrossoverArguments)

    @classmethod
    def from_dict(cls, args_dict: Dict[str, Any]) -> "CrossoverNFTTrainingArguments":
        if "crossover" in args_dict and isinstance(args_dict["crossover"], dict):
            args_dict = dict(args_dict)
            args_dict["crossover"] = CrossoverArguments.from_dict(args_dict["crossover"])
        return super().from_dict(args_dict)  # type: ignore[return-value]
