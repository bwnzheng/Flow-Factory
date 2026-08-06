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

"""Load SD3.5 checkpoints and run reusable evaluation-set inference."""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler

Checkpoint = Tuple[int, str]
ManifestRow = Dict[str, Union[int, str]]

_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
_PIPELINE_LOAD_LOCK = threading.Lock()


def discover_checkpoints(checkpoint_dir: str) -> List[Checkpoint]:
    """Discover sorted ``checkpoint-N`` subdirectories.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories.

    Returns:
        Sorted ``(step, absolute_or_relative_path)`` tuples.

    Raises:
        FileNotFoundError: If ``checkpoint_dir`` does not exist.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints: List[Checkpoint] = []
    for name in os.listdir(checkpoint_dir):
        match = _CHECKPOINT_PATTERN.fullmatch(name)
        path = os.path.join(checkpoint_dir, name)
        if match and os.path.isdir(path):
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def resolve_checkpoints(
    checkpoint_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_step: Optional[int] = None,
) -> List[Checkpoint]:
    """Resolve either a checkpoint collection or one explicit checkpoint.

    Args:
        checkpoint_dir: Directory containing ``checkpoint-N`` subdirectories.
        checkpoint_path: Path to one checkpoint directory.
        checkpoint_step: Optional output step for a checkpoint whose basename is
            not ``checkpoint-N``.

    Returns:
        One or more sorted checkpoint tuples.

    Raises:
        ValueError: If selection arguments are ambiguous or a step cannot be inferred.
        FileNotFoundError: If a selected path does not exist.
    """
    if bool(checkpoint_dir) == bool(checkpoint_path):
        raise ValueError("Specify exactly one of checkpoint_dir or checkpoint_path.")

    if checkpoint_dir:
        if checkpoint_step is not None:
            raise ValueError("checkpoint_step is only valid with checkpoint_path.")
        checkpoints = discover_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise ValueError(f"No checkpoint-N subdirectories found in: {checkpoint_dir}")
        return checkpoints

    assert checkpoint_path is not None
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_step is None:
        match = _CHECKPOINT_PATTERN.fullmatch(os.path.basename(checkpoint_path.rstrip(os.sep)))
        if match is None:
            raise ValueError(
                "Cannot infer checkpoint step from path. Use checkpoint_step when the "
                f"directory is not named checkpoint-N: {checkpoint_path}"
            )
        checkpoint_step = int(match.group(1))
    if checkpoint_step < 0:
        raise ValueError(f"checkpoint_step must be non-negative, got {checkpoint_step}.")
    return [(checkpoint_step, checkpoint_path)]


def load_evaluation_prompts(
    evaluation_set: str,
    prompt_key: str = "prompt",
    max_prompts: int = 0,
) -> List[str]:
    """Load prompts from a text or JSONL evaluation set.

    Text files contain one prompt per non-empty line. JSONL files contain one
    object per line and read the string field selected by ``prompt_key``.

    Args:
        evaluation_set: Path to a TXT or JSONL evaluation set.
        prompt_key: JSONL field containing the prompt.
        max_prompts: Maximum number of prompts to load; zero keeps all prompts.

    Returns:
        Non-empty prompt strings in file order.

    Raises:
        FileNotFoundError: If the evaluation set does not exist.
        ValueError: If the limit or evaluation-set contents are invalid.
    """
    if max_prompts < 0:
        raise ValueError(f"max_prompts must be non-negative, got {max_prompts}.")
    if not os.path.isfile(evaluation_set):
        raise FileNotFoundError(f"Evaluation set not found: {evaluation_set}")

    prompts: List[str] = []
    with open(evaluation_set, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if evaluation_set.lower().endswith(".jsonl"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {evaluation_set}:{line_number}: {exc.msg}"
                    ) from exc
                prompt = record.get(prompt_key) if isinstance(record, dict) else None
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"Expected a non-empty string field '{prompt_key}' at "
                        f"{evaluation_set}:{line_number}."
                    )
                prompts.append(prompt.strip())
            else:
                prompts.append(line)
            if max_prompts and len(prompts) >= max_prompts:
                break

    if not prompts:
        raise ValueError(f"Evaluation set contains no prompts: {evaluation_set}")
    return prompts


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a user-facing dtype name."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Options: {sorted(dtype_map)}")
    return dtype_map[dtype_name]


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve an explicit or automatically detected inference device.

    Args:
        device: Optional torch device string.

    Returns:
        Explicit device unchanged, otherwise ``npu``, ``cuda``, or ``cpu`` in
        availability order.
    """
    if device is None:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    device_type = device.split(":")[0]
    if device_type not in ("npu", "cuda", "cpu"):
        raise ValueError(f"Unsupported inference device '{device}'. Options: npu, cuda, cpu.")
    if device_type == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        raise RuntimeError("NPU inference requested, but torch.npu is not available.")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested, but torch.cuda is not available.")
    return device


def _get_device_count(device_type: str) -> int:
    """Return the number of locally available accelerator devices."""
    if device_type == "npu":
        return torch.npu.device_count()
    if device_type == "cuda":
        return torch.cuda.device_count()
    return 1


def load_base_pipeline(
    base_model: str,
    dtype_str: str,
    device: Optional[str] = None,
) -> StableDiffusion3Pipeline:
    """Load an SD3.5 pipeline with the Flow-Factory ODE scheduler.

    Args:
        base_model: Hugging Face model identifier or local model path.
        dtype_str: One of ``bfloat16``, ``float16``, or ``float32``.
        device: Torch device receiving the pipeline. Auto-detected when omitted.

    Returns:
        Loaded SD3.5 pipeline in evaluation mode.
    """
    dtype = _resolve_dtype(dtype_str)
    resolved_device = resolve_device(device)
    with _PIPELINE_LOAD_LOCK:
        pipeline = StableDiffusion3Pipeline.from_pretrained(base_model, torch_dtype=dtype)
        scheduler = FlowMatchEulerDiscreteSDEScheduler.from_config(
            pipeline.scheduler.config,
            dynamics_type="ODE",
        )
        scheduler.eval()
        pipeline.scheduler = scheduler
        pipeline = pipeline.to(resolved_device)
    return pipeline


def apply_lora(
    pipeline: StableDiffusion3Pipeline,
    checkpoint_path: str,
    dtype: torch.dtype,
) -> None:
    """Load one LoRA checkpoint onto an SD3.5 pipeline.

    Args:
        pipeline: Base pipeline to modify.
        checkpoint_path: PEFT checkpoint directory.
        dtype: Dtype used to load the adapter weights.
    """
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        checkpoint_path,
        torch_dtype=dtype,
    )


def unload_lora(pipeline: StableDiffusion3Pipeline) -> None:
    """Unload the current LoRA adapter and restore the base transformer.

    Args:
        pipeline: Pipeline whose transformer currently carries a PEFT adapter.
    """
    pipeline.transformer = pipeline.transformer.unload()


def _expected_outputs(
    output_dir: str,
    step: int,
    prompts: List[str],
    num_samples: int,
    base_seed: int,
) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    """Return expected relative paths and missing generation slots."""
    paths: List[str] = []
    missing: List[Tuple[int, int, int]] = []
    sample_index = 0
    for prompt_index in range(len(prompts)):
        for sample_index_within_prompt in range(num_samples):
            relative_path = f"checkpoint_{step}/p{prompt_index}_s{sample_index_within_prompt}.png"
            paths.append(relative_path)
            if not os.path.isfile(os.path.join(output_dir, relative_path)):
                missing.append((prompt_index, sample_index_within_prompt, base_seed + sample_index))
            sample_index += 1
    return paths, missing


def _generate_batches(
    runner: "EvaluationRunner",
    prompts: List[str],
    output_dir: str,
    step: int,
    generation_kwargs: Dict[str, Any],
    batch_size: int,
    missing: List[Tuple[int, int, int]],
) -> None:
    """Generate missing images in batches and save them to disk."""
    device_type = runner.device.split(":")[0]
    use_autocast = device_type in ("cuda", "npu")
    pipeline_kwargs = {key: value for key, value in generation_kwargs.items() if key != "seed"}
    pipeline_kwargs["output_type"] = "pil"

    for batch_start in range(0, len(missing), batch_size):
        batch = missing[batch_start : batch_start + batch_size]
        batch_prompts = [prompts[prompt_index] for prompt_index, _, _ in batch]
        generators = [
            torch.Generator(device=runner.device).manual_seed(seed) for _, _, seed in batch
        ]
        pipeline_kwargs["generator"] = generators
        if use_autocast:
            with torch.autocast(device_type=device_type, dtype=runner.dtype):
                result = runner.pipeline(batch_prompts, **pipeline_kwargs)
        else:
            result = runner.pipeline(batch_prompts, **pipeline_kwargs)

        if len(result.images) != len(batch):
            raise RuntimeError(
                "Pipeline output count does not match the requested batch: "
                f"requested({len(batch)}), returned({len(result.images)})."
            )
        for (prompt_index, sample_index, _), image in zip(batch, result.images):
            path = os.path.join(
                output_dir,
                f"checkpoint_{step}/p{prompt_index}_s{sample_index}.png",
            )
            image.save(path, "PNG")


class EvaluationRunner:
    """Generate SD3.5 images from a base model and LoRA checkpoints."""

    def __init__(
        self,
        base_model: str,
        dtype_str: str,
        device: Optional[str] = None,
    ) -> None:
        """Initialize a lazy-loading inference runner.

        Args:
            base_model: Hugging Face model identifier or local model path.
            dtype_str: One of ``bfloat16``, ``float16``, or ``float32``.
            device: Torch inference device. Auto-detects NPU, then CUDA, then CPU.
        """
        self.device = resolve_device(device)
        self.dtype_str = dtype_str
        self.dtype = _resolve_dtype(dtype_str)
        self.base_model = base_model
        self._pipeline: Optional[StableDiffusion3Pipeline] = None

    @property
    def pipeline(self) -> StableDiffusion3Pipeline:
        """Return the lazily loaded base pipeline."""
        if self._pipeline is None:
            self._pipeline = load_base_pipeline(
                self.base_model,
                self.dtype_str,
                self.device,
            )
        return self._pipeline

    @property
    def pipe(self) -> StableDiffusion3Pipeline:
        """Return the pipeline using the legacy property name."""
        return self.pipeline

    @torch.no_grad()
    def generate_for_checkpoint(
        self,
        checkpoint_path: str,
        prompts: List[str],
        output_dir: str,
        step: Optional[int] = None,
        num_samples: int = 4,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
        base_seed: int = 42,
        **legacy_kwargs: Any,
    ) -> List[str]:
        """Generate all missing evaluation images for one checkpoint.

        Args:
            checkpoint_path: LoRA checkpoint directory.
            prompts: Evaluation prompts.
            output_dir: Root directory for generated images.
            step: Checkpoint step used in the output directory name. The legacy
                ``epoch`` keyword is also accepted.
            num_samples: Number of images generated for each prompt.
            generation_kwargs: Keyword arguments forwarded to the pipeline.
            batch_size: Maximum images generated by one pipeline call.
            base_seed: Seed assigned to the first output slot.
            **legacy_kwargs: Compatibility aliases ``epoch``, ``gen_kwargs``, and
                ``gen_batch_size`` used by the previous analysis-local module.

        Returns:
            Relative paths for all expected images, including resumed outputs.

        Raises:
            ValueError: If generation sizes are invalid or legacy aliases conflict.
        """
        if "epoch" in legacy_kwargs:
            legacy_step = legacy_kwargs.pop("epoch")
            if step is not None and step != legacy_step:
                raise ValueError(f"Conflicting step({step}) and epoch({legacy_step}).")
            step = legacy_step
        if "gen_kwargs" in legacy_kwargs:
            if generation_kwargs is not None:
                raise ValueError("Specify only one of generation_kwargs and gen_kwargs.")
            generation_kwargs = legacy_kwargs.pop("gen_kwargs")
        if "gen_batch_size" in legacy_kwargs:
            if batch_size != 16:
                raise ValueError("Specify only one of batch_size and gen_batch_size.")
            batch_size = legacy_kwargs.pop("gen_batch_size")
        if legacy_kwargs:
            raise TypeError(f"Unexpected generation arguments: {sorted(legacy_kwargs)}")
        if step is None:
            raise TypeError("Missing required checkpoint step. Specify step or epoch.")
        if not prompts:
            raise ValueError("prompts must contain at least one evaluation prompt.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        generation_kwargs = generation_kwargs or {}
        os.makedirs(os.path.join(output_dir, f"checkpoint_{step}"), exist_ok=True)
        paths, missing = _expected_outputs(
            output_dir,
            step,
            prompts,
            num_samples,
            base_seed,
        )
        if not missing:
            print(f"      All {len(paths)} images already generated - skipping.")
            return paths

        print(
            f"      Generating {len(missing)}/{len(paths)} images " f"(batch size={batch_size}) ..."
        )
        apply_lora(self.pipeline, checkpoint_path, self.dtype)
        try:
            _generate_batches(
                self,
                prompts,
                output_dir,
                step,
                generation_kwargs,
                batch_size,
                missing,
            )
        finally:
            self.unload_lora()
        return paths

    def unload_lora(self) -> None:
        """Unload the active LoRA adapter while retaining the base pipeline."""
        if self._pipeline is None:
            return
        unload_lora(self._pipeline)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device.startswith("npu") and hasattr(torch, "npu"):
            torch.npu.empty_cache()

    def close(self) -> None:
        """Release the loaded pipeline and cached accelerator memory."""
        self._pipeline = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device.startswith("npu") and hasattr(torch, "npu"):
            torch.npu.empty_cache()


class ParallelEvaluationRunner:
    """Generate checkpoint images concurrently across multiple accelerators."""

    def __init__(
        self,
        base_model: str,
        dtype_str: str,
        num_processes: int,
        device: Optional[str] = None,
    ) -> None:
        """Initialize one lazy runner per accelerator device.

        Args:
            base_model: Hugging Face model identifier or local model path.
            dtype_str: One of ``bfloat16``, ``float16``, or ``float32``.
            num_processes: Number of accelerator workers used concurrently.
            device: Accelerator type, currently ``cuda`` or ``npu``. Auto-detected
                when omitted.

        Raises:
            ValueError: If the device count or device type is invalid.
        """
        if num_processes < 2:
            raise ValueError(f"num_processes must be at least 2, got {num_processes}.")
        device_type = resolve_device(device).split(":")[0]
        if device_type not in ("cuda", "npu"):
            raise ValueError(
                f"Multi-device inference requires device 'cuda' or 'npu', got '{device}'."
            )
        device_count = _get_device_count(device_type)
        if num_processes > device_count:
            raise ValueError(
                f"num_processes({num_processes}) exceeds available {device_type} "
                f"devices({device_count})."
            )
        self._num_processes = num_processes
        self._runners = [
            EvaluationRunner(base_model, dtype_str, device=f"{device_type}:{index}")
            for index in range(num_processes)
        ]

    def generate_for_checkpoint(
        self,
        checkpoint_path: str,
        prompts: List[str],
        output_dir: str,
        step: Optional[int] = None,
        num_samples: int = 4,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
        base_seed: int = 42,
        **legacy_kwargs: Any,
    ) -> List[str]:
        """Generate one checkpoint's evaluation images across all devices.

        Args:
            checkpoint_path: LoRA checkpoint directory.
            prompts: Evaluation prompts.
            output_dir: Root directory for generated images.
            step: Checkpoint step used in the output directory name. The legacy
                ``epoch`` keyword is also accepted.
            num_samples: Number of images generated for each prompt.
            generation_kwargs: Keyword arguments forwarded to the pipeline.
            batch_size: Maximum images generated per pipeline call.
            base_seed: Seed assigned to the first output slot.
            **legacy_kwargs: Compatibility aliases ``epoch``, ``gen_kwargs``, and
                ``gen_batch_size``.

        Returns:
            Relative paths for all expected images.
        """
        if "epoch" in legacy_kwargs:
            legacy_step = legacy_kwargs.pop("epoch")
            if step is not None and step != legacy_step:
                raise ValueError(f"Conflicting step({step}) and epoch({legacy_step}).")
            step = legacy_step
        if "gen_kwargs" in legacy_kwargs:
            if generation_kwargs is not None:
                raise ValueError("Specify only one of generation_kwargs and gen_kwargs.")
            generation_kwargs = legacy_kwargs.pop("gen_kwargs")
        if "gen_batch_size" in legacy_kwargs:
            if batch_size != 16:
                raise ValueError("Specify only one of batch_size and gen_batch_size.")
            batch_size = legacy_kwargs.pop("gen_batch_size")
        if legacy_kwargs:
            raise TypeError(f"Unexpected generation arguments: {sorted(legacy_kwargs)}")
        if step is None:
            raise TypeError("Missing required checkpoint step. Specify step or epoch.")

        generation_kwargs = generation_kwargs or {}
        os.makedirs(os.path.join(output_dir, f"checkpoint_{step}"), exist_ok=True)
        paths, missing = _expected_outputs(
            output_dir,
            step,
            prompts,
            num_samples,
            base_seed,
        )
        if not missing:
            print(f"      All {len(paths)} images already generated - skipping.")
            return paths

        chunk_size = (len(missing) + self._num_processes - 1) // self._num_processes
        chunks = [
            (
                process_index,
                missing[process_index * chunk_size : (process_index + 1) * chunk_size],
            )
            for process_index in range(self._num_processes)
        ]
        chunks = [(process_index, chunk) for process_index, chunk in chunks if chunk]
        print(
            f"      Generating {len(missing)}/{len(paths)} images "
            f"({self._num_processes} processes, batch size={batch_size}) ..."
        )

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(
                    self._generate_chunk,
                    process_index,
                    self._runners[process_index],
                    checkpoint_path,
                    prompts,
                    output_dir,
                    step,
                    generation_kwargs,
                    batch_size,
                    chunk,
                ): process_index
                for process_index, chunk in chunks
            }
            for future in as_completed(futures):
                future.result()
        return paths

    @staticmethod
    def _generate_chunk(
        process_index: int,
        runner: EvaluationRunner,
        checkpoint_path: str,
        prompts: List[str],
        output_dir: str,
        step: int,
        generation_kwargs: Dict[str, Any],
        batch_size: int,
        chunk: List[Tuple[int, int, int]],
    ) -> None:
        """Generate one device's assigned image slots."""
        if runner.device.startswith("npu"):
            torch.npu.set_device(process_index)
        elif runner.device.startswith("cuda"):
            torch.cuda.set_device(process_index)

        apply_lora(runner.pipeline, checkpoint_path, runner.dtype)
        try:
            _generate_batches(
                runner,
                prompts,
                output_dir,
                step,
                generation_kwargs,
                batch_size,
                chunk,
            )
        finally:
            runner.unload_lora()

    def close(self) -> None:
        """Release all loaded pipelines and cached accelerator memory."""
        for runner in self._runners:
            runner.close()


def _build_manifest_rows(
    checkpoints: List[Checkpoint],
    prompts: List[str],
    num_samples: int,
    base_seed: int,
) -> List[ManifestRow]:
    """Build deterministic output metadata for every inference slot."""
    rows: List[ManifestRow] = []
    for step, checkpoint_path in checkpoints:
        output_index = 0
        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(num_samples):
                rows.append(
                    {
                        "checkpoint_step": step,
                        "checkpoint_path": checkpoint_path,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "seed": base_seed + output_index,
                        "prompt": prompt,
                        "image_path": (f"checkpoint_{step}/p{prompt_index}_s{sample_index}.png"),
                    }
                )
                output_index += 1
    return rows


def run_evaluation_set(
    runner: Union[EvaluationRunner, ParallelEvaluationRunner],
    checkpoints: List[Checkpoint],
    prompts: List[str],
    output_dir: str,
    num_samples: int = 4,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 16,
    base_seed: int = 42,
) -> Dict[int, List[str]]:
    """Run all checkpoints over an evaluation set and write a manifest.

    Args:
        runner: Single-process or parallel inference runner.
        checkpoints: ``(step, checkpoint_path)`` tuples to evaluate.
        prompts: Evaluation prompts in deterministic file order.
        output_dir: Root directory for images and ``manifest.jsonl``.
        num_samples: Number of images generated for each prompt and checkpoint.
        generation_kwargs: Keyword arguments forwarded to the SD3.5 pipeline.
        batch_size: Maximum images generated per pipeline call and device.
        base_seed: Seed assigned to the first image slot of each checkpoint.

    Returns:
        Mapping from checkpoint step to relative generated image paths.

    Raises:
        ValueError: If checkpoints are empty or contain duplicate steps.
    """
    if not checkpoints:
        raise ValueError("checkpoints must contain at least one checkpoint.")
    if not prompts:
        raise ValueError("prompts must contain at least one evaluation prompt.")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    steps = [step for step, _ in checkpoints]
    if len(set(steps)) != len(steps):
        raise ValueError(f"Checkpoint steps must be unique, got {steps}.")

    os.makedirs(output_dir, exist_ok=True)
    generated: Dict[int, List[str]] = {}
    for step, checkpoint_path in checkpoints:
        print(f"[Inference] checkpoint_step={step}, checkpoint_path={checkpoint_path}")
        generated[step] = runner.generate_for_checkpoint(
            checkpoint_path=checkpoint_path,
            prompts=prompts,
            output_dir=output_dir,
            step=step,
            num_samples=num_samples,
            generation_kwargs=generation_kwargs,
            batch_size=batch_size,
            base_seed=base_seed,
        )

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for row in _build_manifest_rows(checkpoints, prompts, num_samples, base_seed):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return generated
