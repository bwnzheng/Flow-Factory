#!/usr/bin/env python3
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

"""Run offline reward-concordance analysis from saved run logs.

Usage::

    python -m tools.train_reward_analysis.analyze \\
        -c tools/train_reward_analysis/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tools.train_reward_analysis.metrics import (
    aggregate_group_metrics,
    compute_reward_concordance_metrics,
)
from tools.train_reward_analysis.plots import (
    plot_per_reward_conflict_score_trajectories,
    plot_per_reward_disagreement_trajectories,
    plot_reward_concordance_lower_bound_trajectories,
)
from tools.train_reward_analysis.reward_logs import (
    RewardGroup,
    SavedRewardWeightContext,
    load_saved_reward_weight_context,
    load_train_reward_groups,
)


@dataclass(frozen=True)
class RunSpec:
    """One saved run and optional fallback scalarization weights."""

    name: str
    label: str
    reward_weights: dict[str, float]


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for an offline-only multi-run analysis."""

    save_dir: str = "saves"
    runs: list[RunSpec] = field(default_factory=list)
    smoothing_window: int = 5
    output_dir: str = "analysis_output/train_reward_analysis"


def main() -> None:
    """Parse CLI arguments and write the experiment artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Path to an analysis YAML file.")
    args = parser.parse_args()
    config = _parse_config(args.config)
    _validate_config(config)
    rows, metadata = run_analysis(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, output_dir / "metrics.csv")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_per_reward_conflict_score_trajectories(
        rows,
        output_dir,
        smoothing_window=config.smoothing_window,
    )
    plot_per_reward_disagreement_trajectories(
        rows,
        output_dir,
        smoothing_window=config.smoothing_window,
    )
    plot_reward_concordance_lower_bound_trajectories(
        rows,
        output_dir,
        smoothing_window=config.smoothing_window,
    )
    print(
        "[Reward concordance] "
        f"runs={len(config.runs)} metric_rows={len(rows)} output={output_dir}"
    )


def run_analysis(config: AnalysisConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze configured saved runs and return CSV rows plus audit metadata."""
    rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    for run in config.runs:
        run_dir = Path(config.save_dir) / run.name
        rewards_dir = run_dir / "logs" / "rewards"
        step_groups = load_train_reward_groups(rewards_dir)
        saved_weight_context = load_saved_reward_weight_context(run_dir)
        groups_seen = 0
        weight_sources: dict[str, str] = {}

        by_step_combination: dict[tuple[int, tuple[str, ...]], list[RewardGroup]] = {}
        for step, groups in step_groups.items():
            for group in groups:
                key = (step, group.reward_names)
                by_step_combination.setdefault(key, []).append(group)

        for (step, reward_names), groups in sorted(by_step_combination.items()):
            weights, weight_source = _weights_for_group(run, reward_names, saved_weight_context)
            weight_sources["__".join(reward_names)] = weight_source
            metrics = [
                compute_reward_concordance_metrics(
                    group.rewards,
                    weights,
                )
                for group in groups
            ]
            aggregate = aggregate_group_metrics(metrics)
            groups_seen += len(groups)
            rows.extend(_metric_rows(run, step, reward_names, aggregate))

        run_metadata.append(
            {
                "run_name": run.name,
                "run_label": run.label,
                "reward_weights": run.reward_weights,
                "reward_weight_sources": weight_sources,
                "n_steps": len(step_groups),
                "n_groups": groups_seen,
            }
        )

    metadata = {
        "metric_version": 3,
        "source": "saved_train_reward_pickles_and_optional_media_run_context",
        "centering": "uniform_prompt_local_frozen_reward_mean",
        "natural_aggregation": "macro_average_over_prompt_groups",
        "plot_smoothing_window": config.smoothing_window,
        "metrics": {
            "per_reward_conflict_score": "mean_raw_weighted_reward_contribution",
            "per_reward_disagreement": "fraction_of_samples_with_negative_reward_scalar_alignment",
            "reward_concordance_lower_bound": (
                "mean_over_samples_of_the_minimum_raw_reward_contribution"
            ),
        },
        "runs": run_metadata,
    }
    return rows, metadata


def _parse_config(path: str | Path) -> AnalysisConfig:
    """Parse one concise YAML file and reject ambiguous weight specifications."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Analysis configuration must be a YAML mapping.")

    if "analysis" in raw:
        raise ValueError(
            "analysis is no longer supported: raw reward-concordance metrics have no neutral "
            "thresholds. Remove the analysis mapping."
        )
    output = raw.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping when present.")
    plot = raw.get("plot", {})
    if not isinstance(plot, dict):
        raise ValueError("plot must be a mapping when present.")
    global_weights = _parse_weight_mapping(raw.get("reward_weights", {}), "reward_weights")

    runs_raw = raw.get("runs", [])
    if not isinstance(runs_raw, list):
        raise ValueError("runs must be a list.")
    runs: list[RunSpec] = []
    for index, entry in enumerate(runs_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"runs[{index}] must be a mapping.")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"runs[{index}].name must be a non-empty string.")
        local_weights = _parse_weight_mapping(
            entry.get("reward_weights", {}),
            f"runs[{index}].reward_weights",
        )
        runs.append(
            RunSpec(
                name=name,
                label=str(entry.get("label", name)),
                reward_weights={**global_weights, **local_weights},
            )
        )

    return AnalysisConfig(
        save_dir=str(raw.get("save_dir", "saves")),
        runs=runs,
        smoothing_window=_parse_positive_odd_int(
            plot.get("smoothing_window", 5),
            "plot.smoothing_window",
        ),
        output_dir=str(output.get("dir", "analysis_output/train_reward_analysis")),
    )


def _validate_config(config: AnalysisConfig) -> None:
    if not config.runs:
        raise ValueError("Configure at least one run under runs.")
    names = [run.name for run in config.runs]
    if len(names) != len(set(names)):
        raise ValueError(f"Run names must be unique, got {names}.")
    if not config.output_dir:
        raise ValueError("output.dir must be non-empty.")


def _parse_weight_mapping(value: Any, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping from reward name to positive weight.")
    result: dict[str, float] = {}
    for name, raw_weight in value.items():
        reward_name = str(name)
        if not reward_name:
            raise ValueError(f"{field_name} cannot contain an empty reward name.")
        weight = _parse_positive_float(raw_weight, f"{field_name}.{reward_name}")
        result[reward_name] = weight
    return result


def _parse_positive_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be finite and strictly positive, got {value!r}.")
    return number


def _parse_positive_odd_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive odd integer, got {value!r}.")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive odd integer, got {value!r}.") from error
    if number != value or number < 1 or number % 2 == 0:
        raise ValueError(f"{field_name} must be a positive odd integer, got {value!r}.")
    return number


def _weights_for_group(
    run: RunSpec,
    reward_names: tuple[str, ...],
    saved_context: SavedRewardWeightContext | None,
) -> tuple[np.ndarray, str]:
    """Resolve one group's weights from saved context, then YAML as a fallback."""
    context_weights, context_source = _weights_from_saved_context(saved_context, reward_names)
    missing = [name for name in reward_names if name not in run.reward_weights]
    yaml_weights = (
        None
        if missing
        else np.asarray([run.reward_weights[name] for name in reward_names], dtype=np.float64)
    )
    if context_weights is not None:
        if yaml_weights is not None and not np.allclose(context_weights, yaml_weights):
            raise ValueError(
                f"Analysis-YAML weights disagree with the saved run context for {run.name!r}, "
                f"active rewards {reward_names}: YAML={yaml_weights.tolist()}, "
                f"saved={context_weights.tolist()} ({context_source})."
            )
        return context_weights, context_source
    if yaml_weights is not None:
        return yaml_weights, "analysis_yaml"
    raise ValueError(
        f"Cannot recover scalarization weights for run {run.name!r}, active rewards {reward_names}. "
        "Its reward PKL does not encode reward.weight and the saved media run context is absent or "
        "ambiguous. Supply complete reward_weights in the analysis YAML."
    )


def _weights_from_saved_context(
    saved_context: SavedRewardWeightContext | None,
    reward_names: tuple[str, ...],
) -> tuple[np.ndarray | None, str | None]:
    if saved_context is None:
        return None, None
    active = set(reward_names)
    candidates: list[tuple[str, np.ndarray]] = []
    for source, weights in saved_context.weights_by_source.items():
        if set(weights) != active:
            continue
        vector = np.asarray([weights[name] for name in reward_names], dtype=np.float64)
        if not np.isfinite(vector).all() or np.any(vector <= 0.0):
            raise ValueError(
                f"Saved run context has non-positive weight(s) for source {source!r}, "
                f"active rewards {reward_names}: {vector.tolist()}."
            )
        candidates.append((source, vector))
    if not candidates:
        return None, None
    first_source, first_weights = candidates[0]
    if all(np.allclose(weights, first_weights) for _, weights in candidates[1:]):
        sources = ",".join(source for source, _ in candidates)
        return first_weights, f"saved_media_run_context:{sources}"
    sources = [source for source, _ in candidates]
    raise ValueError(
        "Saved run context maps the same active reward set to different weights for sources "
        f"{sources}. Reward PKLs lack source IDs, so this group cannot be disambiguated safely."
    )


def _metric_rows(
    run: RunSpec,
    step: int,
    reward_names: tuple[str, ...],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_name": run.name,
        "run_label": run.label,
        "step": step,
        "reward_combination": "__".join(reward_names),
        "n_groups": metrics["n_groups"],
    }
    rows: list[dict[str, Any]] = []
    for reward_name, value in zip(
        reward_names,
        metrics["per_reward_conflict_score"],
    ):
        rows.append(
            {
                **common,
                "reward": reward_name,
                "metric": "per_reward_conflict_score",
                "value": float(value),
            }
        )
    for reward_name, value in zip(reward_names, metrics["per_reward_disagreement"]):
        rows.append(
            {
                **common,
                "reward": reward_name,
                "metric": "per_reward_disagreement",
                "value": float(value),
            }
        )
    rows.append(
        {
            **common,
            "reward": "",
            "metric": "reward_concordance_lower_bound",
            "value": float(metrics["reward_concordance_lower_bound"]),
        }
    )
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "run_name",
        "run_label",
        "step",
        "reward_combination",
        "n_groups",
        "reward",
        "metric",
        "value",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
