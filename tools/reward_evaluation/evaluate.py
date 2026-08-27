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

"""Generate checkpoint rollouts and evaluate a configurable reward suite."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml

from tools.model_inference import (
    EvaluationRunner,
    ParallelEvaluationRunner,
    resolve_checkpoints,
    resolve_device,
    run_evaluation_set,
)
from tools.reward_covariance_eval_analysis.analyze import PromptRecord, load_prompt_records
from tools.reward_evaluation.scoring import score_reward


def _progress(message: str) -> None:
    """Emit evaluator progress immediately when stdout is redirected."""
    print(message, flush=True)


@dataclass(frozen=True)
class ModelConfig:
    """Configure base-model loading and local accelerator workers."""

    base_model: str
    dtype: str
    device: Optional[str]
    num_processes: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Configure deterministic rollout generation and reward batching."""

    num_samples_per_prompt: int
    generation_batch_size: int
    reward_batch_size: int
    seed: int
    generation_kwargs: Dict[str, Any]


@dataclass(frozen=True)
class SourceConfig:
    """Configure one prompt file and its registry reward suite."""

    name: str
    prompts_file: Optional[str]
    dataset_dir: Optional[str]
    split: str
    prompt_key: str
    max_prompts: int
    rewards: List[Dict[str, Any]]


@dataclass(frozen=True)
class RunConfig:
    """Configure one checkpoint directory or one explicit checkpoint."""

    name: str
    label: str
    checkpoint: Optional[str]
    checkpoint_dir: Optional[str]


@dataclass(frozen=True)
class EvaluationSuiteConfig:
    """Validated configuration for the standalone reward evaluator."""

    model: ModelConfig
    evaluation: EvaluationConfig
    sources: List[SourceConfig]
    runs: List[RunConfig]
    output_dir: str


def load_config(path: Union[str, Path]) -> EvaluationSuiteConfig:
    """Load and validate a reward-evaluation YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Evaluation config must be a YAML mapping.")
    _reject_unknown(raw, {"model", "evaluation", "sources", "runs", "output"}, "root")
    model = _mapping(raw, "model")
    evaluation = _mapping(raw, "evaluation")
    output = _mapping(raw, "output")
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
    device_value = model.get("device")
    device = None if device_value in (None, "") else _nonempty_string(device_value, "model.device")
    if device is not None and device.split(":", maxsplit=1)[0] not in {"cuda", "npu", "cpu"}:
        raise ValueError("model.device must use cuda, npu, cpu, or null for auto-detection.")
    num_processes = _positive_int(model.get("num_processes", 1), "model.num_processes")
    if num_processes > 1 and device is not None and ":" in device:
        raise ValueError(
            "model.device must be an accelerator type such as 'cuda' or 'npu' when "
            "model.num_processes > 1."
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
    return EvaluationSuiteConfig(
        model=ModelConfig(
            base_model=_nonempty_string(model.get("base_model"), "model.base_model"),
            dtype=dtype,
            device=device,
            num_processes=num_processes,
        ),
        evaluation=EvaluationConfig(
            num_samples_per_prompt=_positive_int(
                evaluation.get("num_samples_per_prompt", 4),
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
                for key in ("num_inference_steps", "guidance_scale", "height", "width")
                if key in evaluation
            },
        ),
        sources=sources,
        runs=runs,
        output_dir=_nonempty_string(output.get("dir"), "output.dir"),
    )


def run_evaluation(config: EvaluationSuiteConfig) -> Dict[str, Any]:
    """Generate images with ``tools.model_inference`` and score every reward."""
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(config.model.device)
    _progress(
        f"[Reward evaluation] start output={output_root.resolve()} device={resolved_device} "
        f"num_processes={config.model.num_processes}"
    )
    summaries: List[Dict[str, Any]] = []

    for run in config.runs:
        checkpoints = _resolve_run_checkpoints(run)
        _progress(
            f"[Reward evaluation] run={run.name} checkpoints={len(checkpoints)} "
            f"checkpoint_paths={[path for _, path in checkpoints]}"
        )
        for source in config.sources:
            prompt_records = load_prompt_records(
                _resolve_prompt_file(source), source.prompt_key, source.max_prompts
            )
            experiment_dir = output_root / run.name / source.name
            _progress(
                f"[Reward evaluation] source={source.name} prompts={len(prompt_records)} "
                f"experiment_dir={experiment_dir.resolve()}"
            )
            image_root = experiment_dir / "images"
            _progress(f"[Reward evaluation] generating source={source.name}")
            manifest_rows = _generate_images(config, checkpoints, prompt_records, image_root)
            _progress(
                f"[Reward evaluation] images ready source={source.name} "
                f"samples={len(manifest_rows)} manifest={image_root / 'manifest.jsonl'}"
            )
            for step, checkpoint_path in checkpoints:
                checkpoint_rows = [
                    row for row in manifest_rows if int(row["checkpoint_step"]) == step
                ]
                if not checkpoint_rows:
                    raise RuntimeError(
                        f"Inference manifest does not contain checkpoint step {step} "
                        f"for run {run.name!r} and source {source.name!r}."
                    )
                reward_values: Dict[str, Dict[str, float]] = {}
                for reward in source.rewards:
                    reward_name = str(reward["name"])
                    _progress(
                        f"[Reward evaluation] scoring source={source.name} "
                        f"checkpoint={step} reward={reward_name} samples={len(checkpoint_rows)}"
                    )
                    reward_values[reward_name] = score_reward(
                        reward_config=reward,
                        manifest_rows=checkpoint_rows,
                        image_root=image_root,
                        prompt_records=prompt_records,
                        output_path=experiment_dir / "reward_scores" / f"{reward_name}.jsonl",
                        device=resolved_device,
                        dtype=config.model.dtype,
                        num_processes=config.model.num_processes,
                        batch_size=config.evaluation.reward_batch_size,
                    )
                    _progress(
                        f"[Reward evaluation] scored source={source.name} "
                        f"checkpoint={step} reward={reward_name}"
                    )
                summary = _write_artifacts(
                    config,
                    run,
                    source,
                    step,
                    checkpoint_path,
                    prompt_records,
                    checkpoint_rows,
                    reward_values,
                    experiment_dir,
                )
                summaries.append(summary)
                _progress(
                    f"[Reward evaluation] checkpoint complete source={source.name} "
                    f"checkpoint={step} results={experiment_dir / 'checkpoint_results' / f'checkpoint-{step}.jsonl'}"
                )

    result = {
        "schema_version": 1,
        "source": "tools.model_inference + flow_factory.rewards.registry",
        "num_processes": config.model.num_processes,
        "experiments": summaries,
    }
    _write_json(output_root / "summary.json", result)
    _progress(f"[Reward evaluation] complete summary={output_root / 'summary.json'}")
    return result


def _generate_images(
    config: EvaluationSuiteConfig,
    checkpoints: List[Tuple[int, str]],
    prompt_records: List[PromptRecord],
    image_root: Path,
) -> List[Dict[str, Any]]:
    if config.model.num_processes == 1:
        runner = EvaluationRunner(config.model.base_model, config.model.dtype, config.model.device)
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
            checkpoints=checkpoints,
            prompts=[record.prompt for record in prompt_records],
            output_dir=str(image_root),
            num_samples=config.evaluation.num_samples_per_prompt,
            generation_kwargs=config.evaluation.generation_kwargs,
            batch_size=config.evaluation.generation_batch_size,
            base_seed=config.evaluation.seed,
        )
    finally:
        runner.close()
    manifest = image_root / "manifest.jsonl"
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


def _write_artifacts(
    config: EvaluationSuiteConfig,
    run: RunConfig,
    source: SourceConfig,
    step: int,
    checkpoint_path: str,
    prompt_records: List[PromptRecord],
    manifest_rows: List[Dict[str, Any]],
    reward_values: Dict[str, Dict[str, float]],
    experiment_dir: Path,
) -> Dict[str, Any]:
    reward_names = [str(reward["name"]) for reward in source.rewards]
    samples: List[Dict[str, Any]] = []
    for row in manifest_rows:
        key = _sample_key(row)
        prompt_index = int(row["prompt_index"])
        samples.append(
            {
                "run_name": run.name,
                "run_label": run.label,
                "checkpoint_step": step,
                "checkpoint_path": checkpoint_path,
                "source": source.name,
                "prompt_index": prompt_index,
                "prompt": row["prompt"],
                "metadata": prompt_records[prompt_index].metadata,
                "sample_index": int(row["sample_index"]),
                "seed": int(row["seed"]),
                "image_path": str(Path("images") / str(row["image_path"])),
                "rewards": {name: reward_values[name][key] for name in reward_names},
            }
        )
    checkpoint_results_dir = experiment_dir / "checkpoint_results"
    _write_jsonl(checkpoint_results_dir / f"checkpoint-{step}.jsonl", samples)
    _rebuild_results_jsonl(experiment_dir)
    reward_summary = {}
    for name in reward_names:
        values = np.asarray([sample["rewards"][name] for sample in samples], dtype=np.float64)
        reward_summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary = {
        "run_name": run.name,
        "run_label": run.label,
        "checkpoint_step": step,
        "checkpoint_path": checkpoint_path,
        "source": source.name,
        "reward_names": reward_names,
        "n_prompts": len(prompt_records),
        "n_samples": len(samples),
        "samples_per_prompt": config.evaluation.num_samples_per_prompt,
        "rewards": reward_summary,
    }
    _write_json(experiment_dir / "summary.json", summary)
    return summary


def _rebuild_results_jsonl(experiment_dir: Path) -> None:
    """Rebuild the source-level JSONL from completed checkpoint result files."""
    result_rows: List[Dict[str, Any]] = []
    checkpoint_results_dir = experiment_dir / "checkpoint_results"
    for path in sorted(checkpoint_results_dir.glob("checkpoint-*.jsonl")):
        result_rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    result_rows.sort(
        key=lambda row: (
            int(row["checkpoint_step"]),
            int(row["prompt_index"]),
            int(row["sample_index"]),
        )
    )
    _write_jsonl(experiment_dir / "results.jsonl", result_rows)


def _parse_source(value: Any, index: int) -> SourceConfig:
    if not isinstance(value, dict):
        raise ValueError(f"sources[{index}] must be a mapping.")
    _reject_unknown(
        value,
        {
            "name",
            "prompts_file",
            "dataset_dir",
            "split",
            "prompt_key",
            "max_prompts",
            "rewards",
        },
        f"sources[{index}]",
    )
    prompts_file_value = value.get("prompts_file")
    dataset_dir_value = value.get("dataset_dir")
    if bool(prompts_file_value) == bool(dataset_dir_value):
        raise ValueError(
            f"sources[{index}] must specify exactly one of prompts_file or dataset_dir."
        )
    rewards = value.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        raise ValueError(f"sources[{index}].rewards must be a non-empty list.")
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
        prompts_file=(
            _nonempty_string(prompts_file_value, f"sources[{index}].prompts_file")
            if prompts_file_value
            else None
        ),
        dataset_dir=(
            _nonempty_string(dataset_dir_value, f"sources[{index}].dataset_dir")
            if dataset_dir_value
            else None
        ),
        split=_nonempty_string(value.get("split", "test"), f"sources[{index}].split"),
        prompt_key=_nonempty_string(
            value.get("prompt_key", "prompt"), f"sources[{index}].prompt_key"
        ),
        max_prompts=_minimum_int(value.get("max_prompts", 0), 0, f"sources[{index}].max_prompts"),
        rewards=normalized,
    )


def _resolve_prompt_file(source: SourceConfig) -> str:
    """Resolve a source's explicit prompt file or training dataset split."""
    if source.prompts_file is not None:
        return source.prompts_file
    assert source.dataset_dir is not None
    dataset_dir = Path(source.dataset_dir)
    candidates = [dataset_dir / f"{source.split}.jsonl", dataset_dir / f"{source.split}.txt"]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"No evaluation split found for source {source.name!r}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _parse_run(value: Any, index: int) -> RunConfig:
    if not isinstance(value, dict):
        raise ValueError(f"runs[{index}] must be a mapping.")
    _reject_unknown(value, {"name", "label", "checkpoint", "checkpoint_dir"}, f"runs[{index}]")
    name = _nonempty_string(value.get("name"), f"runs[{index}].name")
    checkpoint_value = value.get("checkpoint")
    checkpoint_dir_value = value.get("checkpoint_dir")
    if bool(checkpoint_value) == bool(checkpoint_dir_value):
        raise ValueError(f"runs[{index}] must specify exactly one of checkpoint or checkpoint_dir.")
    return RunConfig(
        name=name,
        label=_nonempty_string(value.get("label", name), f"runs[{index}].label"),
        checkpoint=(
            _nonempty_string(checkpoint_value, f"runs[{index}].checkpoint")
            if checkpoint_value
            else None
        ),
        checkpoint_dir=(
            _nonempty_string(checkpoint_dir_value, f"runs[{index}].checkpoint_dir")
            if checkpoint_dir_value
            else None
        ),
    )


def _resolve_run_checkpoints(run: RunConfig) -> List[Tuple[int, str]]:
    """Resolve all checkpoints selected by one run configuration."""
    if run.checkpoint_dir is not None:
        return resolve_checkpoints(checkpoint_dir=run.checkpoint_dir)
    assert run.checkpoint is not None
    return resolve_checkpoints(checkpoint_path=run.checkpoint)


def _sample_key(row: Dict[str, Any]) -> str:
    step = row.get("checkpoint_step")
    checkpoint_prefix = f"c{int(step)}_" if step is not None else ""
    return f"{checkpoint_prefix}p{int(row['prompt_index'])}_s{int(row['sample_index'])}"


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary_path, path)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
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


def main() -> None:
    """Parse CLI arguments and run the reward evaluation suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default=str(Path(__file__).with_name("default.yaml")),
        help="Path to the reward evaluation YAML configuration.",
    )
    config = load_config(parser.parse_args().config)
    result = run_evaluation(config)
    print(
        "[Reward evaluation] "
        f"experiments={len(result['experiments'])} output={config.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
