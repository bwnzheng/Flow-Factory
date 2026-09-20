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

"""Generate fresh checkpoint rollouts and analyze prompt-local reward geometry.

Usage::

    python -m tools.eval_reward_analysis.analyze \\
        -c tools/eval_reward_analysis/jsr_runs.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
import yaml

from tools.eval_reward_analysis.jsr import (
    analyze_cached_results,
    build_reference_thresholds,
    compute_jsr,
)
from tools.eval_reward_analysis.metrics import (
    aggregate_group_metrics,
    compute_group_metrics,
)
from tools.eval_reward_analysis.plots import (
    AGREEMENT_COUNT_STEM,
    COVARIANCE_STEM,
    JSR_STEM,
    build_agreement_count_figure,
    build_covariance_figure,
    build_jsr_figure,
)
from tools.eval_reward_analysis.reward_scoring import score_reward
from tools.figures import (
    FIGURE_INDEX_NAME,
    FIGURE_MODES,
    FigureOutput,
    read_indexed_figures,
    render_figures,
    write_figure_data,
    write_figure_index,
)
from tools.model_inference import (
    EvaluationRunner,
    ParallelEvaluationRunner,
    resolve_device,
    run_evaluation_set,
)
from tools.train_reward_analysis.reward_logs import load_saved_reward_weight_context
from tools.utils import PromptRecord, load_prompt_records


@dataclass(frozen=True)
class ModelConfig:
    """Configure base-model loading and accelerator workers."""

    base_model: str
    dtype: str
    device: Optional[str]
    num_processes: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Configure repeated rollout generation and reward batching."""

    num_samples_per_prompt: int
    generation_batch_size: int
    reward_batch_size: int
    seed: int
    generation_kwargs: Dict[str, Any]


@dataclass(frozen=True)
class SourceConfig:
    """Configure one evaluation prompt source and reward suite."""

    name: str
    prompts_file: str
    prompt_key: str
    max_prompts: int
    rewards: List[Dict[str, Any]]


@dataclass(frozen=True)
class RunConfig:
    """Configure one saved LoRA checkpoint or the base model."""

    name: str
    label: str
    checkpoint: Optional[str]
    base_model_only: bool = False
    reward_weights: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class AnalysisConfig:
    """Store the validated checkpoint covariance experiment configuration."""

    model: ModelConfig
    evaluation: EvaluationConfig
    sources: List[SourceConfig]
    runs: List[RunConfig]
    output_dir: str
    plot_format: str = "png"
    jsr: Optional[Dict[str, Any]] = None
    covariance: Optional[Dict[str, Any]] = None
    agreement_count: Optional[Dict[str, Any]] = None
    jsr_output_dir: Optional[str] = None
    figure_mode: str = "regenerate"


def load_config(path: Union[str, Path]) -> AnalysisConfig:
    """Load and strictly validate one analysis YAML file.

    Args:
        path: Analysis YAML path.

    Returns:
        Validated checkpoint evaluation configuration.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Analysis config must be a YAML mapping.")
    _reject_unknown(
        raw,
        {
            "model",
            "evaluation",
            "sources",
            "runs",
            "output",
            "jsr",
            "covariance",
            "agreement_count",
        },
        "root",
    )
    model = _mapping(raw, "model")
    evaluation = _mapping(raw, "evaluation")
    output = _mapping(raw, "output")
    jsr_raw = raw.get("jsr")
    jsr = _parse_jsr(jsr_raw) if jsr_raw is not None else None
    covariance_raw = raw.get("covariance")
    covariance = (
        _parse_toggle(covariance_raw, "covariance")
        if covariance_raw is not None
        else {"enabled": True}
    )
    agreement_count_raw = raw.get("agreement_count")
    agreement_count = (
        _parse_toggle(agreement_count_raw, "agreement_count")
        if agreement_count_raw is not None
        else {"enabled": True}
    )
    _reject_unknown(output, {"dir", "cache_dir", "jsr_dir", "plot_format", "figure_mode"}, "output")
    _reject_unknown(model, {"base_model", "dtype", "device", "num_processes"}, "model")
    _reject_unknown(
        evaluation,
        {
            "num_samples_per_prompt",
            "generation_batch_size",
            "reward_batch_size",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "height",
            "width",
        },
        "evaluation",
    )
    num_processes = _positive_int(model.get("num_processes", 1), "model.num_processes")
    device_value = model.get("device")
    device = None if device_value in (None, "") else _nonempty_string(device_value, "model.device")
    if device is not None and device.split(":", maxsplit=1)[0] not in {"cuda", "npu", "cpu"}:
        raise ValueError("model.device must use cuda, npu, cpu, or null for auto-detection.")
    if num_processes > 1 and device is not None and ":" in device:
        raise ValueError(
            "model.device must be an accelerator type such as 'cuda' or 'npu' when "
            "model.num_processes > 1; indexed devices are only valid for one process."
        )
    if num_processes > 1 and device == "cpu":
        raise ValueError("model.num_processes > 1 requires CUDA or NPU accelerators.")
    sources_raw = raw.get("sources")
    runs_raw = raw.get("runs")
    if jsr is None and (not isinstance(sources_raw, list) or not sources_raw):
        raise ValueError("sources must be a non-empty list.")
    if jsr is None and (not isinstance(runs_raw, list) or not runs_raw):
        raise ValueError("runs must be a non-empty list.")
    sources_raw = sources_raw or []
    runs_raw = runs_raw or []
    sources = [_parse_source(item, index) for index, item in enumerate(sources_raw)]
    runs = [_parse_run(item, index) for index, item in enumerate(runs_raw)]
    _require_unique([source.name for source in sources], "source names")
    _require_unique([run.name for run in runs], "run names")
    dtype = _nonempty_string(model.get("dtype", "bfloat16"), "model.dtype")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported model.dtype: {dtype!r}.")
    return AnalysisConfig(
        model=ModelConfig(
            base_model=_nonempty_string(model.get("base_model"), "model.base_model"),
            dtype=dtype,
            device=device,
            num_processes=num_processes,
        ),
        evaluation=EvaluationConfig(
            num_samples_per_prompt=_minimum_int(
                evaluation.get("num_samples_per_prompt", 16),
                2,
                "evaluation.num_samples_per_prompt",
            ),
            generation_batch_size=_positive_int(
                evaluation.get("generation_batch_size", 1),
                "evaluation.generation_batch_size",
            ),
            reward_batch_size=_positive_int(
                evaluation.get("reward_batch_size", 16),
                "evaluation.reward_batch_size",
            ),
            seed=_integer(evaluation.get("seed", 42), "evaluation.seed"),
            generation_kwargs={
                key: evaluation[key]
                for key in (
                    "num_inference_steps",
                    "guidance_scale",
                    "height",
                    "width",
                )
                if key in evaluation
            },
        ),
        sources=sources,
        runs=runs,
        output_dir=_nonempty_string(output.get("cache_dir", output.get("dir")), "output.cache_dir"),
        plot_format=_plot_format(output.get("plot_format", "png")),
        jsr=jsr,
        covariance=covariance,
        agreement_count=agreement_count,
        jsr_output_dir=(
            _nonempty_string(output["jsr_dir"], "output.jsr_dir")
            if output.get("jsr_dir") is not None
            else None
        ),
        figure_mode=_figure_mode(output.get("figure_mode", "regenerate")),
    )


def run_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Run fresh inference, reward scoring, and prompt-local aggregation.

    Args:
        config: Validated analysis configuration.

    Returns:
        Top-level experiment metadata and summaries.
    """
    if _is_cached_jsr(config):
        result = analyze_cached_results(**config.jsr)
        output_root = _jsr_output_root(config)
        output_root.mkdir(parents=True, exist_ok=True)
        _write_figures(
            [
                build_jsr_figure(
                    {name: values["jsr"] for name, values in result["models"].items()},
                    result["q"],
                )
            ],
            output_root,
            config,
        )
        result["plot"] = f"{JSR_STEM}.{config.plot_format}"
        _write_json(output_root / "jsr_results.json", _json_safe_jsr(result))
        return {"schema_version": 1, "source": "cached_reward_records", "jsr": result}
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(config.model.device)
    experiment_summaries: List[Dict[str, Any]] = []
    for run in config.runs:
        if run.base_model_only:
            step = 0
        else:
            assert run.checkpoint is not None
            checkpoint = Path(run.checkpoint)
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
            step = _checkpoint_step(checkpoint)
        for source in config.sources:
            prompt_records = load_prompt_records(
                source.prompts_file, source.prompt_key, source.max_prompts
            )
            experiment_dir = output_root / run.name / source.name
            image_root = experiment_dir / "images"
            manifest_rows = _generate_images(config, run, step, source, prompt_records, image_root)
            reward_values: Dict[str, Dict[str, float]] = {}
            for reward in source.rewards:
                name = str(reward["name"])
                reward_values[name] = score_reward(
                    reward_config=reward,
                    manifest_rows=manifest_rows,
                    image_root=image_root,
                    prompt_records=prompt_records,
                    output_path=experiment_dir / "reward_scores" / f"{name}.jsonl",
                    device=resolved_device,
                    dtype=config.model.dtype,
                    num_processes=config.model.num_processes,
                    batch_size=config.evaluation.reward_batch_size,
                )
            # Agreement counts are the only statistics that compare rewards against
            # the weighted scalar, so nothing else forces weights to be resolved.
            if (config.agreement_count or {"enabled": True}).get("enabled", True):
                reward_weights, reward_weight_source = _resolve_reward_weights(config, run, source)
            else:
                reward_weights, reward_weight_source = None, "disabled"
            summary = _write_analysis_artifacts(
                config,
                run,
                source,
                step,
                prompt_records,
                manifest_rows,
                reward_values,
                experiment_dir,
                reward_weights,
                reward_weight_source,
                write_covariance=(config.covariance or {"enabled": True}).get("enabled", True),
            )
            experiment_summaries.append(summary)
    if config.jsr is not None:
        _write_run_jsr_results(config, experiment_summaries)
    metadata = {
        "schema_version": 1,
        "source": "fresh_checkpoint_or_base_rollouts_and_reward_model_forward",
        "num_processes": config.model.num_processes,
        "experiments": experiment_summaries,
    }
    return metadata


def _generate_images(
    config: AnalysisConfig,
    run: RunConfig,
    step: int,
    source: SourceConfig,
    prompt_records: List[PromptRecord],
    image_root: Path,
) -> List[Dict[str, Any]]:
    prompts = [record.prompt for record in prompt_records]
    runner: EvaluationRunner | ParallelEvaluationRunner
    if config.model.num_processes == 1:
        runner = EvaluationRunner(
            config.model.base_model, config.model.dtype, device=config.model.device
        )
    else:
        runner = ParallelEvaluationRunner(
            config.model.base_model,
            config.model.dtype,
            num_processes=config.model.num_processes,
            device=config.model.device,
        )
    try:
        run_evaluation_set(
            runner=runner,
            checkpoints=[(step, run.checkpoint)],
            prompts=prompts,
            output_dir=str(image_root),
            num_samples=config.evaluation.num_samples_per_prompt,
            generation_kwargs=config.evaluation.generation_kwargs,
            batch_size=config.evaluation.generation_batch_size,
            base_seed=config.evaluation.seed,
        )
    finally:
        runner.close()
    manifest_path = image_root / "manifest.jsonl"
    return [json.loads(line) for line in manifest_path.read_text().splitlines() if line]


def _write_analysis_artifacts(
    config: AnalysisConfig,
    run: RunConfig,
    source: SourceConfig,
    step: int,
    prompt_records: List[PromptRecord],
    manifest_rows: List[Dict[str, Any]],
    reward_values: Dict[str, Dict[str, float]],
    experiment_dir: Path,
    reward_weights: Optional[Dict[str, float]],
    reward_weight_source: str,
    write_covariance: bool = True,
) -> Dict[str, Any]:
    reward_names = [str(reward["name"]) for reward in source.rewards]
    rows_by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    sample_rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        key = _sample_key(row)
        prompt_index = int(row["prompt_index"])
        sample = {
            "run_name": run.name,
            "run_label": run.label,
            "checkpoint_step": step,
            "checkpoint_path": run.checkpoint or "base_model",
            "source": source.name,
            "prompt_index": prompt_index,
            "prompt": row["prompt"],
            "metadata": prompt_records[prompt_index].metadata,
            "sample_index": int(row["sample_index"]),
            "seed": int(row["seed"]),
            "image_path": str(Path("images") / row["image_path"]),
            "rewards": {name: reward_values[name][key] for name in reward_names},
        }
        sample_rows.append(sample)
        rows_by_prompt.setdefault(prompt_index, []).append(sample)
    _write_jsonl(experiment_dir / "samples.jsonl", sample_rows)

    prompt_metrics: List[Dict[str, Any]] = []
    group_metrics = []
    for prompt_index in sorted(rows_by_prompt):
        samples = sorted(rows_by_prompt[prompt_index], key=lambda item: item["sample_index"])
        matrix = np.asarray(
            [[sample["rewards"][name] for name in reward_names] for sample in samples],
            dtype=np.float64,
        )
        metric = compute_group_metrics(
            matrix,
            (
                np.asarray([reward_weights[name] for name in reward_names], dtype=np.float64)
                if reward_weights is not None
                else None
            ),
        )
        group_metrics.append(metric)
        prompt_metrics.append(
            {
                "run_name": run.name,
                "checkpoint_step": step,
                "source": source.name,
                "prompt_index": prompt_index,
                "prompt": samples[0]["prompt"],
                "reward_names": reward_names,
                "reward_matrix": matrix.tolist(),
                **_json_metrics(metric),
            }
        )
    _write_jsonl(experiment_dir / "prompt_metrics.jsonl", prompt_metrics)
    aggregate = aggregate_group_metrics(group_metrics)
    plots_dir = experiment_dir / "plots"
    outputs: List[FigureOutput] = []
    if write_covariance:
        outputs.append(
            build_covariance_figure(
                covariance=np.asarray(aggregate["standardized_covariance"]),
                reward_names=reward_names,
                title=f"Reward covariance: {run.label} checkpoint-{step} ({source.name})",
            )
        )
    if reward_weights is not None:
        # Fresh rollouts are not shaped by the training-time sample selector, so this
        # figure shows the agreement structure the checkpoint actually produces.
        outputs.append(
            build_agreement_count_figure(
                distribution=np.asarray(aggregate["agreement_count_distribution"]),
                title=f"Agreement count: {run.label} checkpoint-{step} ({source.name})",
            )
        )
    written = {stem for stem, _ in outputs}
    _write_figures(outputs, plots_dir, config)
    covariance_plot_path = (
        plots_dir / f"{COVARIANCE_STEM}.{config.plot_format}"
        if COVARIANCE_STEM in written
        else None
    )
    agreement_count_plot_path = (
        plots_dir / f"{AGREEMENT_COUNT_STEM}.{config.plot_format}"
        if AGREEMENT_COUNT_STEM in written
        else None
    )
    summary = {
        "run_name": run.name,
        "run_label": run.label,
        "checkpoint_step": step,
        "checkpoint_path": run.checkpoint or "base_model",
        "source": source.name,
        "reward_names": reward_names,
        "n_prompts": len(prompt_metrics),
        "samples_per_prompt": config.evaluation.num_samples_per_prompt,
        "reward_weights": (
            {name: float(reward_weights[name]) for name in reward_names}
            if reward_weights is not None
            else None
        ),
        "reward_weight_source": reward_weight_source,
        "covariance_plot": (
            str(covariance_plot_path.relative_to(experiment_dir)) if covariance_plot_path else None
        ),
        "agreement_count_plot": (
            str(agreement_count_plot_path.relative_to(experiment_dir))
            if agreement_count_plot_path
            else None
        ),
        **_json_metrics(aggregate),
    }
    _write_json(experiment_dir / "summary.json", summary)
    return summary


def _is_cached_jsr(config: AnalysisConfig) -> bool:
    """Whether the config selects the cached-record JSR entry.

    The YAML key is ``jsr.reference``, which the parser stores as
    ``reference_path``; reading the parsed key is what tells the two entries
    apart, since the run-based one carries ``reference_run`` instead.
    """
    return config.jsr is not None and "reference_path" in config.jsr


def _jsr_output_root(config: AnalysisConfig) -> Path:
    """Return the directory the configured JSR entry writes its figures into.

    The two JSR entries land in different places, which the reader has to mirror
    exactly: the cached-record entry writes into ``output.jsr_dir`` or the output
    directory itself, while the run-based entry nests one directory per source
    under ``output.jsr_dir``, defaulting to ``jsr`` inside the output directory.
    """
    if _is_cached_jsr(config):
        return Path(config.jsr_output_dir or config.output_dir)
    return Path(config.jsr_output_dir or (Path(config.output_dir) / "jsr"))


def _write_figures(
    outputs: Sequence[FigureOutput], plots_dir: Path, config: AnalysisConfig
) -> None:
    """Write one figures directory: the specs, their index, and the images.

    The index is what the reuse figure mode reads, so a directory is always
    redrawable from its own contents alone.
    """
    stems = write_figure_data(outputs, plots_dir)
    write_figure_index(plots_dir, stems, config.figure_mode, config.plot_format)
    render_figures(outputs, plots_dir, config.plot_format)


def _redraw_figures(config: AnalysisConfig) -> List[str]:
    """Redraw every figure an earlier run described, without re-analyzing anything.

    The directories come from the configuration, so this enumerates exactly the
    runs, sources, and JSR panels the config claims, and each one's own index
    decides what to draw. Nothing here reads images, reward caches, or metrics.
    """
    directories: List[Path] = []
    output_root = Path(config.output_dir)
    if _is_cached_jsr(config):
        # The cached-record entry computes one JSR figure from existing JSONL and
        # writes no per-run figures, so it is the whole list.
        directories.append(_jsr_output_root(config))
    else:
        for run in config.runs:
            for source in config.sources:
                directories.append(output_root / run.name / source.name / "plots")
        if config.jsr is not None and config.jsr.get("enabled", True):
            jsr_root = _jsr_output_root(config)
            if not jsr_root.is_dir():
                raise FileNotFoundError(
                    f"No JSR output directory at {jsr_root}. Re-run with the regenerate figure "
                    "mode to write the figure data first."
                )
            # One directory per source, plus "overall" when it was configured; a
            # directory without an index is not a JSR panel, so it is skipped
            # rather than treated as missing output.
            panels = [
                path
                for path in sorted(jsr_root.iterdir())
                if path.is_dir() and (path / FIGURE_INDEX_NAME).is_file()
            ]
            if not panels:
                raise FileNotFoundError(
                    f"No {FIGURE_INDEX_NAME} under {jsr_root}. Re-run with the regenerate figure "
                    "mode to write the figure data first."
                )
            directories.extend(panels)

    redrawn: List[str] = []
    for directory in directories:
        if not (directory / FIGURE_INDEX_NAME).is_file():
            raise FileNotFoundError(
                f"No {FIGURE_INDEX_NAME} in {directory}. Re-run with the regenerate figure mode "
                "to write the figure data first."
            )
        outputs = read_indexed_figures(directory)
        render_figures(outputs, directory, config.plot_format)
        redrawn.extend(str(directory / f"{stem}.{config.plot_format}") for stem, _ in outputs)
    return redrawn


def _write_run_jsr_results(config: AnalysisConfig, summaries: List[Dict[str, Any]]) -> None:
    """Compute JSR from generated run caches selected by the config section."""
    section = config.jsr or {}
    if not section.get("enabled", True):
        return
    reference_name = section.get("reference_run")
    comparison_names = section.get("comparison_runs", [])
    labels = {run.name: run.label for run in config.runs}
    if reference_name not in {run.name for run in config.runs}:
        raise ValueError(f"jsr.reference_run does not match any configured run: {reference_name}")
    all_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for source in config.sources:
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for run_name in [reference_name, *comparison_names]:
            sample_path = Path(config.output_dir) / run_name / source.name / "samples.jsonl"
            if not sample_path.is_file():
                raise FileNotFoundError(f"JSR requires complete run output: {sample_path}")
            by_name[run_name] = [
                json.loads(line)
                for line in sample_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        all_rows[source.name] = by_name
        out_dir = _jsr_output_root(config) / source.name
        rewards = [str(item["name"]) for item in source.rewards]
        q_grid = section.get("q_grid", [i / 100 for i in range(101)])
        thresholds = build_reference_thresholds(by_name[reference_name], rewards, q_grid)
        curves = {
            name: compute_jsr(rows, rewards, thresholds, q_grid)
            for name, rows in by_name.items()
            if name != reference_name
        }
        result = {
            "reference_run": reference_name,
            "q": q_grid,
            "thresholds": thresholds,
            "models": curves,
        }
        _write_json(out_dir / "jsr_results.json", _json_safe_jsr(result))
        _write_figures(
            [
                build_jsr_figure(
                    {labels.get(name, name): data["jsr"] for name, data in curves.items()},
                    q_grid,
                )
            ],
            out_dir,
            config,
        )
    if section.get("overall", False):
        _write_overall_jsr(config, all_rows, section)


def _write_overall_jsr(
    config: AnalysisConfig,
    source_rows: Dict[str, Dict[str, List[Dict[str, Any]]]],
    section: Dict[str, Any],
) -> None:
    """Write an all-source JSR panel with missing rewards treated as success."""
    reference_name = section["reference_run"]
    comparison_names = section["comparison_runs"]
    rewards = sorted({str(item["name"]) for source in config.sources for item in source.rewards})
    q_grid = section.get("q_grid", [i / 100 for i in range(101)])
    names = [reference_name, *comparison_names]
    combined: Dict[str, List[Dict[str, Any]]] = {name: [] for name in names}
    for source in config.sources:
        for name, rows in source_rows[source.name].items():
            for index, row in enumerate(rows):
                copied = dict(row)
                copied["prompt_id"] = (
                    f"{source.name}:{row.get('prompt_id', row.get('prompt_index'))}"
                )
                copied["image_id"] = (
                    f"{source.name}:{row.get('image_id', row.get('sample_index', index))}"
                )
                copied["rewards"] = dict(row.get("rewards", {}))
                combined[name].append(copied)
    thresholds = {}
    for reward in rewards:
        available = [row for row in combined[reference_name] if reward in row["rewards"]]
        if not available:
            raise ValueError(f"Overall JSR reward has no reference values: {reward}")
        thresholds[reward] = build_reference_thresholds(available, [reward], q_grid)[reward]
    curves = {}
    for name in comparison_names:
        rows = []
        for row in combined[name]:
            copied = dict(row)
            copied["rewards"] = {
                reward: row["rewards"].get(reward, float("inf")) for reward in rewards
            }
            rows.append(copied)
        curves[name] = compute_jsr(rows, rewards, thresholds, q_grid, allow_positive_inf=True)
    out_dir = _jsr_output_root(config) / "overall"
    result = {
        "reference_run": reference_name,
        "q": q_grid,
        "rewards": rewards,
        "missing_reward_policy": "default_success",
        "thresholds": thresholds,
        "models": curves,
    }
    _write_json(out_dir / "jsr_results.json", _json_safe_jsr(result))
    _write_figures(
        [
            build_jsr_figure(
                {
                    {run.name: run.label for run in config.runs}.get(name, name): data["jsr"]
                    for name, data in curves.items()
                },
                q_grid,
                title="Overall Joint Success Rate",
            )
        ],
        out_dir,
        config,
    )


def _parse_source(value: Any, index: int) -> SourceConfig:
    if not isinstance(value, dict):
        raise ValueError(f"sources[{index}] must be a mapping.")
    _reject_unknown(
        value, {"name", "prompts_file", "prompt_key", "max_prompts", "rewards"}, f"sources[{index}]"
    )
    rewards = value.get("rewards")
    if not isinstance(rewards, list) or len(rewards) < 2:
        raise ValueError(f"sources[{index}].rewards must contain at least two rewards.")
    normalized = []
    for reward_index, reward in enumerate(rewards):
        if not isinstance(reward, dict):
            raise ValueError(f"sources[{index}].rewards[{reward_index}] must be a mapping.")
        if (
            not str(reward.get("name", "")).strip()
            or not str(reward.get("reward_model", "")).strip()
        ):
            raise ValueError(
                f"sources[{index}].rewards[{reward_index}] requires name and reward_model."
            )
        normalized.append(dict(reward))
    _require_unique([str(reward["name"]) for reward in normalized], f"sources[{index}] rewards")
    return SourceConfig(
        name=_nonempty_string(value.get("name"), f"sources[{index}].name"),
        prompts_file=_nonempty_string(value.get("prompts_file"), f"sources[{index}].prompts_file"),
        prompt_key=_nonempty_string(
            value.get("prompt_key", "prompt"), f"sources[{index}].prompt_key"
        ),
        max_prompts=_minimum_int(value.get("max_prompts", 0), 0, f"sources[{index}].max_prompts"),
        rewards=normalized,
    )


def _parse_jsr(value: Any) -> Dict[str, Any]:
    """Validate the cached JSR analysis configuration."""
    if not isinstance(value, dict):
        raise ValueError("jsr must be a mapping.")
    _reject_unknown(
        value,
        {
            "enabled",
            "overall",
            "reference_run",
            "comparison_runs",
            "reference",
            "models",
            "rewards",
            "q_grid",
            "bootstrap_replicates",
            "bootstrap_seed",
        },
        "jsr",
    )
    if value.get("enabled", True) is False:
        return {"enabled": False}
    if "reference_run" in value:
        comparison_runs = value.get("comparison_runs")
        if not isinstance(comparison_runs, list) or not comparison_runs:
            raise ValueError("jsr.comparison_runs must be a non-empty list.")
        return {
            "enabled": True,
            "overall": bool(value.get("overall", False)),
            "reference_run": _nonempty_string(value["reference_run"], "jsr.reference_run"),
            "comparison_runs": [
                _nonempty_string(item, "jsr.comparison_runs item") for item in comparison_runs
            ],
            "q_grid": [float(item) for item in value.get("q_grid", [i / 100 for i in range(101)])],
            "bootstrap_replicates": _minimum_int(
                value.get("bootstrap_replicates", 0), 0, "jsr.bootstrap_replicates"
            ),
            "bootstrap_seed": _integer(value.get("bootstrap_seed", 0), "jsr.bootstrap_seed"),
        }
    reference = _nonempty_string(value.get("reference"), "jsr.reference")
    models = value.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("jsr.models must be a non-empty mapping of name to JSONL path.")
    models = {
        str(name): _nonempty_string(path, f"jsr.models[{name}]") for name, path in models.items()
    }
    rewards = value.get("rewards")
    if (
        not isinstance(rewards, list)
        or not rewards
        or not all(isinstance(item, str) and item.strip() for item in rewards)
    ):
        raise ValueError("jsr.rewards must be a non-empty list of names.")
    q_grid = value.get("q_grid", [i / 100 for i in range(101)])
    if not isinstance(q_grid, list) or not q_grid:
        raise ValueError("jsr.q_grid must be a non-empty list.")
    return {
        "reference_path": reference,
        "model_paths": models,
        "rewards": [item.strip() for item in rewards],
        "q_grid": [float(item) for item in q_grid],
        "bootstrap_replicates": _minimum_int(
            value.get("bootstrap_replicates", 0), 0, "jsr.bootstrap_replicates"
        ),
        "bootstrap_seed": _integer(value.get("bootstrap_seed", 0), "jsr.bootstrap_seed"),
    }


def _parse_toggle(value: Any, field: str) -> Dict[str, bool]:
    if not isinstance(value, dict) or set(value) - {"enabled"}:
        raise ValueError(f"{field} must contain only an enabled boolean.")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{field}.enabled must be a boolean.")
    return {"enabled": enabled}


def _parse_run(value: Any, index: int) -> RunConfig:
    if not isinstance(value, dict):
        raise ValueError(f"runs[{index}] must be a mapping.")
    _reject_unknown(
        value,
        {"name", "label", "checkpoint", "base_model_only", "reward_weights"},
        f"runs[{index}]",
    )
    name = _nonempty_string(value.get("name"), f"runs[{index}].name")
    checkpoint_value = value.get("checkpoint")
    base_model_only = value.get("base_model_only", False)
    if not isinstance(base_model_only, bool):
        raise ValueError(f"runs[{index}].base_model_only must be a boolean.")
    if bool(checkpoint_value) == base_model_only:
        raise ValueError(
            f"runs[{index}] must specify exactly one of checkpoint or base_model_only."
        )
    return RunConfig(
        name=name,
        label=_nonempty_string(value.get("label", name), f"runs[{index}].label"),
        checkpoint=(
            _nonempty_string(checkpoint_value, f"runs[{index}].checkpoint")
            if checkpoint_value
            else None
        ),
        base_model_only=base_model_only,
        reward_weights=_parse_reward_weights(value.get("reward_weights"), f"runs[{index}]"),
    )


def _parse_reward_weights(value: Any, field: str) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field}.reward_weights must be a non-empty mapping when present.")
    weights: Dict[str, float] = {}
    for name, weight in value.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{field}.reward_weights[{name!r}] must be a number.")
        if not np.isfinite(float(weight)) or float(weight) <= 0.0:
            raise ValueError(f"{field}.reward_weights[{name!r}] must be finite and positive.")
        weights[str(name)] = float(weight)
    return weights


def _run_context_path(checkpoint: Path) -> Optional[Path]:
    """Find the training run's media manifest above a checkpoint directory."""
    for parent in [checkpoint, *checkpoint.parents][:4]:
        candidate = parent / "logs" / "media.jsonl"
        if candidate.is_file():
            return candidate
    return None


def _saved_weights_for_source(
    run: RunConfig, source: SourceConfig, reward_names: List[str]
) -> tuple[Optional[Dict[str, float]], str]:
    """Read one checkpoint run's saved weights for a source, with a reason if absent."""
    assert run.checkpoint is not None
    context_path = _run_context_path(Path(run.checkpoint))
    if context_path is None:
        return None, f"no logs/media.jsonl was found above checkpoint {run.checkpoint!r}"
    context = load_saved_reward_weight_context(context_path.parent.parent)
    if context is None:
        return None, f"{context_path} has no run_context record"
    saved = context.weights_by_source.get(source.name)
    if saved is None:
        return None, (
            f"{context_path} has no saved weights for source {source.name!r}; available "
            f"sources are {sorted(context.weights_by_source)}"
        )
    if set(saved) != set(reward_names):
        raise ValueError(
            f"Saved weights for source {source.name!r} in run {run.name!r} cover "
            f"{sorted(saved)}, but the source config lists {sorted(reward_names)}."
        )
    return {name: float(saved[name]) for name in reward_names}, ""


def _inherit_reward_weights(
    config: AnalysisConfig, run: RunConfig, source: SourceConfig, reward_names: List[str]
) -> tuple[Dict[str, float], str]:
    """Take a base-model run's weights from the checkpoint runs it is compared with.

    The base model has no training log, so its agreeing counts can only sit on the
    same axis as the compared runs by borrowing their weights. Inheriting them
    beats making the config repeat the numbers by hand, which drifts silently.
    """
    inherited: Dict[str, Dict[str, float]] = {}
    for sibling in config.runs:
        if sibling.checkpoint is None:
            continue
        weights, _ = _saved_weights_for_source(sibling, source, reward_names)
        if weights is not None:
            inherited[sibling.name] = weights
    if not inherited:
        raise ValueError(
            f"Run {run.name!r} evaluates the base model and has no training log to inherit "
            f"scalarization weights from, and none of the configured checkpoint runs saved "
            f"weights for source {source.name!r} either; set runs[{run.name!r}].reward_weights "
            f"explicitly, or disable the agreement counts with agreement_count.enabled: false."
        )
    reference_name, reference = next(iter(inherited.items()))
    for name, weights in inherited.items():
        if weights != reference:
            raise ValueError(
                f"Configured runs disagree on the scalarization weights for source "
                f"{source.name!r}: {reference_name!r} uses {reference}, {name!r} uses "
                f"{weights}; set runs[{run.name!r}].reward_weights explicitly."
            )
    return reference, f"inherited_run_context:{','.join(sorted(inherited))}"


def _resolve_reward_weights(
    config: AnalysisConfig, run: RunConfig, source: SourceConfig
) -> tuple[Dict[str, float], str]:
    """Resolve the scalarization weights used to count agreeing rewards.

    The counts are only comparable to the training-side analysis when they use
    the weights the checkpoints were trained with, so an explicit override must be
    complete, a checkpoint run reads its own saved run context, and a base-model
    run borrows the weights of the checkpoint runs it is compared against.
    """
    reward_names = [str(reward["name"]) for reward in source.rewards]
    if run.reward_weights is not None:
        missing = [name for name in reward_names if name not in run.reward_weights]
        extra = [name for name in run.reward_weights if name not in reward_names]
        if missing or extra:
            raise ValueError(
                f"runs[{run.name!r}].reward_weights must cover exactly the {source.name!r} "
                f"rewards; missing={missing}, unexpected={extra}."
            )
        return {name: run.reward_weights[name] for name in reward_names}, "run_config"

    if run.checkpoint is None:
        return _inherit_reward_weights(config, run, source, reward_names)

    saved, reason = _saved_weights_for_source(run, source, reward_names)
    if saved is None:
        raise ValueError(
            f"Run {run.name!r} cannot supply weights for source {source.name!r}: {reason}; "
            f"set runs[{run.name!r}].reward_weights explicitly."
        )
    return saved, f"saved_run_context:{source.name}"


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix) or not path.name[len(prefix) :].isdigit():
        raise ValueError(f"Checkpoint directory must be named checkpoint-N: {path}")
    return int(path.name[len(prefix) :])


def _sample_key(row: Dict[str, Any]) -> str:
    return f"p{int(row['prompt_index'])}_s{int(row['sample_index'])}"


def _json_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "reward_mean": np.asarray(metrics["mean"]).tolist(),
        "covariance": np.asarray(metrics["covariance"]).tolist(),
        "standardized_covariance": np.asarray(metrics["standardized_covariance"]).tolist(),
        "correlation": np.asarray(metrics["correlation"]).tolist(),
        "negative_pairwise_correlation_ratio": float(
            metrics["negative_pairwise_correlation_ratio"]
        ),
        "mean_negative_pairwise_correlation": float(metrics["mean_negative_pairwise_correlation"]),
    }
    # The agreement counts, and with them every stat that needs weights, are absent
    # when the analysis runs without scalarization weights.
    if "agreement_count_distribution" in metrics:
        payload["agreement_count_distribution"] = np.asarray(
            metrics["agreement_count_distribution"]
        ).tolist()
        payload["mean_agreement_count"] = float(metrics["mean_agreement_count"])
        payload["fully_concordant_sample_rate"] = float(metrics["fully_concordant_sample_rate"])
    return payload


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe_jsr(value: Any) -> Any:
    """Encode the JSR-defined negative-infinity endpoint for strict JSON."""
    if isinstance(value, float):
        if np.isneginf(value):
            return "-inf"
        if not np.isfinite(value):
            raise ValueError("JSR output contains an invalid non-finite value.")
        return value
    if isinstance(value, dict):
        return {key: _json_safe_jsr(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_jsr(item) for item in value]
    return value


def _mapping(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _reject_unknown(value: Dict[str, Any], allowed: Set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown fields in {field}: {unknown}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    return value


def _positive_int(value: Any, field: str) -> int:
    return _minimum_int(value, 1, field)


def _minimum_int(value: Any, minimum: int, field: str) -> int:
    number = _integer(value, field)
    if number < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {number}.")
    return number


def _require_unique(values: List[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique, got {values}.")


def _plot_format(value: Any) -> str:
    """Validate the configured covariance plot format."""
    if not isinstance(value, str) or value.lower() not in {"png", "pdf"}:
        raise ValueError("output.plot_format must be either 'png' or 'pdf'.")
    return value.lower()


def _figure_mode(value: Any) -> str:
    """Validate whether the figure stage redraws from its data or rebuilds it.

    This governs the figures only. Generated images and reward scores are always
    reused when they exist, whatever this is set to.
    """
    if not isinstance(value, str) or value.lower() not in FIGURE_MODES:
        raise ValueError(
            f"output.figure_mode must be one of {list(FIGURE_MODES)}; it controls the figure "
            "stage only, never the image and reward caches."
        )
    return value.lower()


def main() -> None:
    """Parse CLI arguments and run the checkpoint covariance experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default=str(Path(__file__).with_name("default.yaml")),
        help="Path to the analysis YAML configuration.",
    )
    parser.add_argument(
        "--cached-reference", help="JSONL reward records for the shared reference model."
    )
    parser.add_argument(
        "--cached-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Cached comparison records; repeatable.",
    )
    parser.add_argument("--jsr-rewards", nargs="+", help="Reward names for cached JSR analysis.")
    parser.add_argument(
        "--q-grid", nargs="+", type=float, help="Reference percentile grid for cached JSR."
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--jsr-output", help="Write cached JSR result JSON to this path.")
    parser.add_argument(
        "--figures",
        choices=FIGURE_MODES,
        default=None,
        help=(
            "Override output.figure_mode for this invocation, so redrawing from an existing "
            "set of per-figure JSON files needs no edit to the config file. Omit to use the "
            "config value. This never affects generated images or reward scores: those caches "
            "are always reused when they exist."
        ),
    )
    args = parser.parse_args()
    if args.cached_reference:
        if not args.cached_model or not args.jsr_rewards or not args.q_grid:
            parser.error("cached JSR requires --cached-model, --jsr-rewards, and --q-grid")
        model_paths = {}
        for item in args.cached_model:
            if "=" not in item:
                parser.error("--cached-model must use NAME=PATH")
            name, path = item.split("=", 1)
            if not name or not path:
                parser.error("--cached-model must use NAME=PATH")
            model_paths[name] = path
        result = analyze_cached_results(
            args.cached_reference,
            model_paths,
            args.jsr_rewards,
            args.q_grid,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        rendered = json.dumps(_json_safe_jsr(result), ensure_ascii=False, indent=2, allow_nan=False)
        if args.jsr_output:
            output_path = Path(args.jsr_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return
    config = load_config(args.config)
    if args.figures is not None:
        config = replace(config, figure_mode=args.figures)
    if config.figure_mode == "reuse":
        redrawn = _redraw_figures(config)
        print(
            "[Evaluation reward analysis] "
            f"figures={len(redrawn)} mode=reuse output={config.output_dir}"
        )
        return
    result = run_analysis(config)
    # The cached-record entry returns one JSR result rather than per-run experiments.
    detail = (
        f"experiments={len(result['experiments'])}"
        if "experiments" in result
        else f"jsr={result['jsr']['plot']}"
    )
    print(f"[Evaluation reward analysis] {detail} output={config.output_dir}")


if __name__ == "__main__":
    main()
