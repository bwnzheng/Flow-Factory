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

# src/flow_factory/trainers/crossover/__init__.py
"""Crossover strategies and shared sampling utilities for augmenting
intermediate denoising states during RL training."""

from .abc import BaseCrossover, CrossoverOutput
from .covariance import (
    CovarianceGroupScore,
    CovarianceSelectionResult,
    SampleWiseContributionScore,
    SampleWiseCovarianceSelectionResult,
    covariance_group_score,
    population_covariance,
    sample_wise_covariance_contributions,
    select_covariance_guided_group,
    select_sample_wise_covariance_group,
)
from .genetic_algorithm import GeneticAlgorithm
from .pareto import (
    compute_pareto_mask,
    filter_by_group,
    select_non_dominated_parents,
)
from .sampling import (
    resolve_crossover_step,
    run_denoising_phase,
    sample_crossover_step,
)
from .strategies import (
    BlockCrossover,
    ConvexCrossover,
    UniformCrossover,
    create_crossover_strategy,
    list_crossover_strategies,
)

__all__ = [
    "BaseCrossover",
    "BlockCrossover",
    "ConvexCrossover",
    "CovarianceGroupScore",
    "CovarianceSelectionResult",
    "CrossoverOutput",
    "GeneticAlgorithm",
    "SampleWiseContributionScore",
    "SampleWiseCovarianceSelectionResult",
    "UniformCrossover",
    "compute_pareto_mask",
    "covariance_group_score",
    "create_crossover_strategy",
    "filter_by_group",
    "list_crossover_strategies",
    "population_covariance",
    "resolve_crossover_step",
    "run_denoising_phase",
    "sample_crossover_step",
    "sample_wise_covariance_contributions",
    "select_covariance_guided_group",
    "select_sample_wise_covariance_group",
    "select_non_dominated_parents",
]
