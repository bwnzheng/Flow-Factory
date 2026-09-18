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
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from tools.train_reward_analysis.figure_spec import SPEC_VERSION, read_spec
from tools.train_reward_analysis.metrics import (
    aggregate_group_metrics,
    compute_reward_concordance_metrics,
    compute_src_sample_weights,
    compute_weighted_advantage_sign_metrics,
)
from tools.train_reward_analysis.plots import (
    FigureOutput,
    build_figures,
    render_figure,
    write_figure_data,
)
from tools.train_reward_analysis.reward_logs import (
    RewardGroup,
    SavedRewardWeightContext,
    load_saved_reward_weight_context,
    load_train_reward_groups,
)

# The one source of truth for accepted cache modes: the YAML parser and the
# command-line override both validate against it, so the two cannot drift apart.
CACHE_MODES = ("regenerate", "reuse")
LEGACY_DATA_FILES = ("plot_data.json", "metrics.csv")


@dataclass(frozen=True)
class RunSpec:
    """One saved run and optional fallback scalarization weights."""

    name: str
    label: str
    reward_weights: dict[str, float]
    src_reweight: bool = False


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for an offline-only multi-run analysis."""

    save_dir: str = "saves"
    runs: list[RunSpec] = field(default_factory=list)
    smoothing_window: int = 5
    output_dir: str = "analysis_output/train_reward_analysis"
    plot_format: str = "png"
    cache_mode: str = "regenerate"
    src_interpolation: float = 0.8
    src_temperature: float = 0.5


def _build_parser() -> argparse.ArgumentParser:
    """Describe the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Path to an analysis YAML file.")
    parser.add_argument(
        "--cache-mode",
        choices=CACHE_MODES,
        default=None,
        help=(
            "Override output.cache_mode for this invocation, so redrawing from an existing "
            "set of per-figure JSON files needs no edit to the config file. Omit to use the "
            "config value."
        ),
    )
    return parser


def _apply_overrides(config: AnalysisConfig, args: argparse.Namespace) -> AnalysisConfig:
    """Layer command-line overrides on top of the parsed YAML configuration.

    Every override defaults to ``None``, meaning "not given on the command line",
    so an omitted flag leaves the config file's value untouched rather than
    silently replacing it with a default.
    """
    if args.cache_mode is not None:
        return replace(config, cache_mode=args.cache_mode)
    return config


def main() -> None:
    """Parse CLI arguments and write the experiment artifacts."""
    args = _build_parser().parse_args()
    config = _apply_overrides(_parse_config(args.config), args)
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_data = config.cache_mode == "regenerate"

    if config.cache_mode == "reuse":
        metadata = _read_metadata(output_dir)
        outputs = [
            (stem, read_spec(_spec_path(output_dir, stem)))
            for stem in _figure_stems(metadata, output_dir)
        ]
        print(f"[Reward concordance] Reusing {len(outputs)} figure specs from {output_dir}")
    else:
        rows, metadata = run_analysis(config)
        outputs = build_figures(rows, config.smoothing_window)
        metadata["figures"] = write_figure_data(outputs, output_dir)
        print(f"[Reward concordance] Wrote {len(outputs)} figure specs to {output_dir}")

    metadata["format_version"] = SPEC_VERSION
    metadata["plot_format"] = config.plot_format
    metadata["plot_workers"] = _plot_worker_count(len(outputs))
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _render_figures(config, outputs, output_dir)
    if remove_legacy_data:
        removed = _remove_legacy_data_files(output_dir)
        if removed:
            print(f"[Reward concordance] Removed legacy data files: {', '.join(removed)}")
    print(
        "[Reward concordance] "
        f"runs={len(config.runs)} figures={len(outputs)} output={output_dir}"
    )


def _spec_path(output_dir: Path, stem: str) -> Path:
    """Return the data file that sits beside one figure's image."""
    return output_dir / f"{stem}.json"


def _read_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No metadata.json in {output_dir}. Re-run with output.cache_mode: regenerate to "
            "write the figure data first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_legacy_data_files(output_dir: Path) -> list[str]:
    """Remove superseded aggregate data only after figure specs exist.

    The old files duplicate the per-figure points and can be much larger than
    the figures themselves. The exact-name list keeps migration scoped: images,
    metadata, and unrelated analysis artifacts are never touched.
    """
    removed = []
    for name in LEGACY_DATA_FILES:
        path = output_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def _figure_stems(metadata: dict[str, Any], output_dir: Path) -> list[str]:
    """List the figures the last regeneration wrote, refusing to guess.

    Reading the index rather than globbing means a figure whose data was deleted
    is reported instead of silently disappearing, and a leftover file from an
    older configuration is never redrawn.
    """
    stems = metadata.get("figures")
    if not stems:
        raise ValueError(
            f"{output_dir / 'metadata.json'} lists no figures, so there is nothing to redraw. "
            "Re-run with output.cache_mode: regenerate."
        )
    if not isinstance(stems, list) or not all(isinstance(stem, str) and stem for stem in stems):
        raise ValueError(
            f"{output_dir / 'metadata.json'} must list figure path stems as non-empty strings. "
            "Re-run with output.cache_mode: regenerate."
        )
    if len(stems) != len(set(stems)):
        raise ValueError(
            f"{output_dir / 'metadata.json'} lists duplicate figures. Re-run with "
            "output.cache_mode: regenerate."
        )
    missing = [stem for stem in stems if not _spec_path(output_dir, stem).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Figure data missing for {missing}. Re-run with output.cache_mode: regenerate."
        )
    return [str(stem) for stem in stems]


def _render_figure_task(task: tuple[Any, ...]) -> None:
    """Draw one figure inside a worker process."""
    spec, output_dir, stem, plot_format = task
    render_figure(spec, output_dir, stem, plot_format)


def _plot_worker_count(figure_count: int) -> int:
    """Pick how many worker processes the figure stage should use."""
    return max(1, min(figure_count, os.cpu_count() or 1))


def _render_figures(
    config: AnalysisConfig,
    outputs: Sequence[FigureOutput],
    output_dir: Path,
) -> None:
    """Render every figure, spreading them over worker processes.

    Matplotlib is imported by this module and keeps global state, so the pool is
    spawned rather than forked: forking a process that already loaded extension
    modules and started threads risks deadlocking the children. Worker startup
    costs a fresh interpreter import, which the parallel figures amortize.
    """
    tasks = [(spec, str(output_dir), stem, config.plot_format) for stem, spec in outputs]
    workers = _plot_worker_count(len(tasks))
    if workers == 1:
        for task in tasks:
            _render_figure_task(task)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        list(executor.map(_render_figure_task, tasks))


def run_analysis(config: AnalysisConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze configured saved runs and return CSV rows plus audit metadata."""
    rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    tasks = []
    for run in config.runs:
        run_dir = Path(config.save_dir) / run.name
        rewards_dir = run_dir / "logs" / "rewards"
        step_groups = load_train_reward_groups(rewards_dir)
        saved_weight_context = load_saved_reward_weight_context(run_dir)
        for step, groups in step_groups.items():
            tasks.append(
                (
                    run,
                    step,
                    groups,
                    saved_weight_context,
                    config.src_interpolation,
                    config.src_temperature,
                )
            )
        run_metadata.append(
            {
                "run_name": run.name,
                "run_label": run.label,
                "reward_weights": run.reward_weights,
                "n_steps": len(step_groups),
            }
        )

    workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_analyze_run_step, tasks))
    for result in results:
        rows.extend(result["rows"])
        for metadata in run_metadata:
            if metadata["run_name"] == result["run_name"]:
                metadata["n_groups"] = metadata.get("n_groups", 0) + result["n_groups"]
                metadata.setdefault("reward_weight_sources", {}).update(result["weight_sources"])
                break
    metadata = {
        "source": "saved_train_reward_pickles_and_optional_media_run_context",
        "centering": "uniform_prompt_local_frozen_reward_mean",
        "natural_aggregation": "macro_average_over_prompt_groups",
        "plot_smoothing_window": config.smoothing_window,
        "plot_format": config.plot_format,
        "analysis_workers": workers,
        "metrics": {
            "per_reward_conflict_score": "mean_standardized_weighted_reward_contribution",
            "per_reward_disagreement": "fraction_of_samples_with_negative_reward_scalar_alignment",
            "standardized_reward_covariance": "prompt-local population covariance of standardized reward pairs",
            "reward_concordance_lower_bound": "mean_over_samples_of_the_minimum standardized reward contribution",
            "sample_agreement_count_distribution": "prompt-group fraction of samples whose agreeing-reward count equals each value from 0 to n_rewards",
            "mean_agreement_count": "prompt-group mean agreeing-reward count per sample, exactly n_rewards minus the summed per-reward disagreement",
            "fully_concordant_sample_rate": "prompt-group fraction of samples that agree with the weighted scalar on every active reward",
            "positive_fully_concordant_sample_rate": "prompt-group fraction of samples above their group mean on every active reward",
            "negative_fully_concordant_sample_rate": "prompt-group fraction of samples below their group mean on every active reward",
            "per_reward_mean_reward": "prompt-group macro-average of the raw reward level, over the groups keeping that reward active",
        },
        "runs": run_metadata,
    }
    return rows, metadata


def _analyze_run_step(task: tuple[Any, ...]) -> dict[str, Any]:
    """Analyze one run and training step in a worker process."""
    run, step, groups, saved_weight_context, interpolation, temperature = task
    rows: list[dict[str, Any]] = []
    by_combination: dict[tuple[str, ...], list[RewardGroup]] = {}
    for group in groups:
        by_combination.setdefault(group.reward_names, []).append(group)
    weight_sources = {}
    for reward_names, combination_groups in sorted(by_combination.items()):
        weights, weight_source = _weights_for_group(run, reward_names, saved_weight_context)
        weight_sources["__".join(reward_names)] = weight_source
        dataset = _dataset_from_weight_source(weight_source)
        metrics = [
            compute_reward_concordance_metrics(
                group.rewards,
                weights,
            )
            for group in combination_groups
        ]
        aggregate = aggregate_group_metrics(metrics)
        sign_metrics = []
        for group in combination_groups:
            sample_weights = (
                compute_src_sample_weights(group.rewards, weights, interpolation, temperature)
                if run.src_reweight
                else np.ones(group.rewards.shape[0])
            )
            sign_metrics.append(
                compute_weighted_advantage_sign_metrics(group.rewards, weights, sample_weights)
            )
        if not run.src_reweight:
            for item in sign_metrics:
                item.pop("weight_ge_1_adv_positive")
                item.pop("weight_ge_1_adv_negative")
                item.pop("weight_lt_1_adv_positive")
                item.pop("weight_lt_1_adv_negative")
                item.pop("weight_ge_1_adv_zero")
                item.pop("weight_lt_1_adv_zero")
                item["_counts"].pop("weight_ge_1_adv_positive")
                item["_counts"].pop("weight_ge_1_adv_negative")
                item["_counts"].pop("weight_ge_1_adv_zero")
                item["_counts"].pop("weight_lt_1_adv_positive")
                item["_counts"].pop("weight_lt_1_adv_negative")
                item["_counts"].pop("weight_lt_1_adv_zero")
        else:
            for item in sign_metrics:
                item.pop("adv_positive")
                item.pop("adv_negative")
                item.pop("adv_zero")
                item["_counts"].pop("adv_positive")
                item["_counts"].pop("adv_negative")
                item["_counts"].pop("adv_zero")
        for metric_name in sign_metrics[0]:
            if metric_name == "_counts":
                continue
            values = np.mean([item[metric_name] for item in sign_metrics], axis=0)
            aggregate[metric_name] = values
        aggregate["_sign_counts"] = {
            metric_name: np.mean([item["_counts"][metric_name] for item in sign_metrics], axis=0)
            for metric_name in sign_metrics[0]["_counts"]
        }
        for metric_name, counts in aggregate["_sign_counts"].items():
            aggregate[f"sample_count_{metric_name}"] = counts
        rows.extend(_metric_rows(run, step, reward_names, aggregate, dataset))
    return {
        "run_name": run.name,
        "rows": rows,
        "n_groups": len(groups),
        "weight_sources": weight_sources,
    }


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
                src_reweight=_parse_src_reweight(entry, name),
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
        plot_format=_parse_plot_format(output.get("plot_format", "png")),
        cache_mode=_parse_cache_mode(output.get("cache_mode", "regenerate")),
        src_interpolation=float(raw.get("src_interpolation", 0.8)),
        src_temperature=float(raw.get("src_temperature", 0.5)),
    )


def _parse_src_reweight(entry: dict[str, Any], run_name: str) -> bool:
    """Resolve whether a run uses SRC-Reweight sample weighting."""
    if "src_reweight" in entry:
        return bool(entry["src_reweight"])
    if "src" in entry:
        return bool(entry["src"])
    identity = f"{run_name} {entry.get('label', '')}".lower()
    if "ga" in identity or "evolve" in identity:
        return False
    return "src" in identity


def _validate_config(config: AnalysisConfig) -> None:
    if not config.runs:
        raise ValueError("Configure at least one run under runs.")
    names = [run.name for run in config.runs]
    if len(names) != len(set(names)):
        raise ValueError(f"Run names must be unique, got {names}.")
    if not config.output_dir:
        raise ValueError("output.dir must be non-empty.")


def _parse_plot_format(value: Any) -> str:
    """Validate the configured matplotlib output format."""
    if not isinstance(value, str) or value.lower() not in {"png", "pdf"}:
        raise ValueError("output.plot_format must be either 'png' or 'pdf'.")
    return value.lower()


def _parse_cache_mode(value: Any) -> str:
    """Validate whether plot data should be regenerated or reused."""
    if not isinstance(value, str) or value.lower() not in CACHE_MODES:
        raise ValueError(f"output.cache_mode must be one of {list(CACHE_MODES)}.")
    return value.lower()


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
    dataset: str,
) -> list[dict[str, Any]]:
    common = {
        "run_name": run.name,
        "run_label": run.label,
        "dataset": dataset,
        "step": step,
        "reward_combination": "__".join(reward_names),
        "n_groups": metrics["n_groups"],
        "reward_pair": "",
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
    for reward_name, value in zip(reward_names, metrics["per_reward_bottleneck_rate"]):
        rows.append(
            {
                **common,
                "reward": reward_name,
                "metric": "per_reward_bottleneck_rate",
                "value": float(value),
            }
        )
    for metric_name in (
        "weight_ge_1_adv_positive",
        "weight_ge_1_adv_negative",
        "weight_ge_1_adv_zero",
        "weight_lt_1_adv_positive",
        "weight_lt_1_adv_negative",
        "weight_lt_1_adv_zero",
        "adv_positive",
        "adv_negative",
        "adv_zero",
    ):
        if metric_name in metrics:
            for reward_name, value in zip(reward_names, metrics[metric_name]):
                index = reward_names.index(reward_name)
                counts = metrics.get("_sign_counts", {}).get(metric_name, ())
                rows.append(
                    {
                        **common,
                        "reward": reward_name,
                        "metric": metric_name,
                        "value": float(value),
                        "sample_count": float(counts[index]) if len(counts) else float("nan"),
                    }
                )
    for metric_name in (
        "sample_count_weight_ge_1_adv_positive",
        "sample_count_weight_ge_1_adv_negative",
        "sample_count_weight_ge_1_adv_zero",
        "sample_count_weight_lt_1_adv_positive",
        "sample_count_weight_lt_1_adv_negative",
        "sample_count_weight_lt_1_adv_zero",
        "sample_count_adv_positive",
        "sample_count_adv_negative",
        "sample_count_adv_zero",
    ):
        if metric_name in metrics:
            for reward_name, value in zip(reward_names, metrics[metric_name]):
                rows.append(
                    {**common, "reward": reward_name, "metric": metric_name, "value": float(value)}
                )
    covariance = np.asarray(metrics["standardized_reward_covariance"], dtype=np.float64)
    for first_index, first_name in enumerate(reward_names):
        for second_index in range(first_index + 1, len(reward_names)):
            rows.append(
                {
                    **common,
                    "reward": "",
                    "reward_pair": f"{first_name}__{reward_names[second_index]}",
                    "metric": "standardized_reward_covariance",
                    "value": float(covariance[first_index, second_index]),
                }
            )
    rows.append(
        {
            **common,
            "reward": "",
            "reward_pair": "",
            "metric": "reward_concordance_lower_bound",
            "value": float(metrics["reward_concordance_lower_bound"]),
        }
    )
    agreement_distribution = np.asarray(
        metrics["sample_agreement_count_distribution"], dtype=np.float64
    )
    for agreeing_count, fraction in enumerate(agreement_distribution):
        rows.append(
            {
                **common,
                "reward": "",
                "reward_pair": f"count_{agreeing_count}",
                "metric": f"agreement_count_c{agreeing_count}",
                "value": float(fraction),
            }
        )
    rows.append(
        {
            **common,
            "reward": "",
            "reward_pair": "",
            "metric": "mean_agreement_count",
            "value": float(metrics["mean_agreement_count"]),
        }
    )
    rows.append(
        {
            **common,
            "reward": "",
            "reward_pair": "",
            "metric": "fully_concordant_sample_rate",
            "value": float(metrics["fully_concordant_sample_rate"]),
        }
    )
    for metric_name in (
        "positive_fully_concordant_sample_rate",
        "negative_fully_concordant_sample_rate",
    ):
        rows.append(
            {
                **common,
                "reward": "",
                "reward_pair": "",
                "metric": metric_name,
                "value": float(metrics[metric_name]),
            }
        )
    for reward_name, value in zip(reward_names, metrics["per_reward_mean_reward"]):
        rows.append(
            {
                **common,
                "reward": reward_name,
                "metric": "per_reward_mean_reward",
                "value": float(value),
            }
        )
    return rows


def _dataset_from_weight_source(weight_source: str) -> str:
    """Extract the saved dataset/source label used to resolve reward weights."""
    prefix = "saved_media_run_context:"
    if weight_source.startswith(prefix):
        source = weight_source[len(prefix) :].strip()
        if source:
            return source.replace(",", "+")
    return "unknown_dataset"


if __name__ == "__main__":
    main()
