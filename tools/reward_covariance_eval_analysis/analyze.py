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

"""Generate fresh checkpoint rollouts and analyze prompt-local reward geometry."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import yaml

from tools.model_inference import (
    EvaluationRunner,
    ParallelEvaluationRunner,
    resolve_device,
    run_evaluation_set,
)
from tools.reward_covariance_eval_analysis.metrics import (
    aggregate_group_metrics,
    compute_group_metrics,
)
from tools.reward_covariance_eval_analysis.plots import plot_covariance_matrix
from tools.reward_covariance_eval_analysis.reward_scoring import score_reward


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
    """Configure one saved LoRA checkpoint."""

    name: str
    label: str
    checkpoint: str


@dataclass(frozen=True)
class AnalysisConfig:
    """Store the validated checkpoint covariance experiment configuration."""

    model: ModelConfig
    evaluation: EvaluationConfig
    sources: List[SourceConfig]
    runs: List[RunConfig]
    output_dir: str
    plot_format: str = "png"


@dataclass(frozen=True)
class PromptRecord:
    """Store one prompt and its JSON-encoded reward metadata."""

    prompt: str
    metadata: str


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
    _reject_unknown(raw, {"model", "evaluation", "sources", "runs", "output"}, "root")
    model = _mapping(raw, "model")
    evaluation = _mapping(raw, "evaluation")
    output = _mapping(raw, "output")
    _reject_unknown(output, {"dir", "plot_format"}, "output")
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
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("sources must be a non-empty list.")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("runs must be a non-empty list.")
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
        output_dir=_nonempty_string(output.get("dir"), "output.dir"),
        plot_format=_plot_format(output.get("plot_format", "png")),
    )


def run_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Run fresh inference, reward scoring, and prompt-local aggregation.

    Args:
        config: Validated analysis configuration.

    Returns:
        Top-level experiment metadata and summaries.
    """
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(config.model.device)
    experiment_summaries: List[Dict[str, Any]] = []
    for run in config.runs:
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
            summary = _write_analysis_artifacts(
                config,
                run,
                source,
                step,
                prompt_records,
                manifest_rows,
                reward_values,
                experiment_dir,
            )
            experiment_summaries.append(summary)
    metadata = {
        "schema_version": 1,
        "source": "fresh_checkpoint_rollouts_and_reward_model_forward",
        "num_processes": config.model.num_processes,
        "experiments": experiment_summaries,
    }
    _write_json(output_root / "summary.json", metadata)
    return metadata


def load_prompt_records(
    path: Union[str, Path], prompt_key: str = "prompt", max_prompts: int = 0
) -> List[PromptRecord]:
    """Load text or JSONL prompts while preserving reward metadata.

    Args:
        path: Text or JSONL evaluation-set path.
        prompt_key: JSONL field containing the generation prompt.
        max_prompts: Maximum records to keep; zero keeps every record.

    Returns:
        Ordered prompt and metadata records.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {source_path}")
    records: List[PromptRecord] = []
    for line_number, line in enumerate(
        source_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        if source_path.suffix.lower() == ".jsonl":
            value = json.loads(line)
            prompt = value.get(prompt_key) if isinstance(value, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Missing non-empty {prompt_key!r} at {source_path}:{line_number}."
                )
            records.append(
                PromptRecord(prompt=prompt.strip(), metadata=json.dumps(value, ensure_ascii=False))
            )
        else:
            records.append(PromptRecord(prompt=line, metadata="{}"))
        if max_prompts and len(records) >= max_prompts:
            break
    if not records:
        raise ValueError(f"Prompt file contains no prompts: {source_path}")
    return records


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
            "checkpoint_path": run.checkpoint,
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
        metric = compute_group_metrics(matrix)
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
    covariance_plot_path = experiment_dir / "plots" / f"covariance_matrix.{config.plot_format}"
    plot_covariance_matrix(
        covariance=np.asarray(aggregate["standardized_covariance"]),
        reward_names=reward_names,
        output_path=covariance_plot_path,
        title=f"Reward covariance: {run.label} checkpoint-{step} ({source.name})",
    )
    summary = {
        "run_name": run.name,
        "run_label": run.label,
        "checkpoint_step": step,
        "source": source.name,
        "reward_names": reward_names,
        "n_prompts": len(prompt_metrics),
        "samples_per_prompt": config.evaluation.num_samples_per_prompt,
        "covariance_plot": str(covariance_plot_path.relative_to(experiment_dir)),
        **_json_metrics(aggregate),
    }
    _write_json(experiment_dir / "summary.json", summary)
    return summary


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


def _parse_run(value: Any, index: int) -> RunConfig:
    if not isinstance(value, dict):
        raise ValueError(f"runs[{index}] must be a mapping.")
    _reject_unknown(value, {"name", "label", "checkpoint"}, f"runs[{index}]")
    name = _nonempty_string(value.get("name"), f"runs[{index}].name")
    return RunConfig(
        name=name,
        label=_nonempty_string(value.get("label", name), f"runs[{index}].label"),
        checkpoint=_nonempty_string(value.get("checkpoint"), f"runs[{index}].checkpoint"),
    )


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix) or not path.name[len(prefix) :].isdigit():
        raise ValueError(f"Checkpoint directory must be named checkpoint-N: {path}")
    return int(path.name[len(prefix) :])


def _sample_key(row: Dict[str, Any]) -> str:
    return f"p{int(row['prompt_index'])}_s{int(row['sample_index'])}"


def _json_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reward_mean": np.asarray(metrics["mean"]).tolist(),
        "covariance": np.asarray(metrics["covariance"]).tolist(),
        "standardized_covariance": np.asarray(metrics["standardized_covariance"]).tolist(),
        "correlation": np.asarray(metrics["correlation"]).tolist(),
        "negative_pairwise_correlation_ratio": float(
            metrics["negative_pairwise_correlation_ratio"]
        ),
        "mean_negative_pairwise_correlation": float(metrics["mean_negative_pairwise_correlation"]),
    }


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


def main() -> None:
    """Parse CLI arguments and run the checkpoint covariance experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default=str(Path(__file__).with_name("default.yaml")),
        help="Path to the analysis YAML configuration.",
    )
    config = load_config(parser.parse_args().config)
    result = run_analysis(config)
    print(
        "[Reward covariance evaluation] "
        f"experiments={len(result['experiments'])} output={config.output_dir}"
    )


if __name__ == "__main__":
    main()
