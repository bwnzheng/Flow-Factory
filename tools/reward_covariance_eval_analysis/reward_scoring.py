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

"""Resumable actual reward-model inference for generated evaluation images."""

from __future__ import annotations

import inspect
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from flow_factory.hparams import RewardArguments
from flow_factory.rewards.registry import get_reward_model_class


@dataclass(frozen=True)
class _AcceleratorView:
    """Provide the accelerator fields consumed by offline reward models."""

    device: torch.device
    local_process_index: int


def score_reward(
    reward_config: Dict[str, Any],
    manifest_rows: List[Dict[str, Any]],
    image_root: Path,
    prompt_records: List[Any],
    output_path: Path,
    device: str,
    dtype: str,
    num_processes: int,
    batch_size: int,
) -> Dict[str, float]:
    """Score every manifest image with one actual Flow-Factory reward model.

    Args:
        reward_config: Reward name, registry identifier, and optional model settings.
        manifest_rows: Generated-image manifest rows.
        image_root: Directory against which manifest image paths are resolved.
        prompt_records: Prompt records carrying optional JSON metadata.
        output_path: Resumable JSONL reward-cache path.
        device: CUDA, NPU, or CPU device string.
        dtype: Reward-model dtype name.
        num_processes: Number of accelerator worker processes.
        batch_size: Per-worker reward forward batch size.

    Returns:
        Mapping from deterministic sample key to finite reward value.
    """
    cached = _load_cached_scores(output_path)
    missing = [row for row in manifest_rows if _sample_key(row) not in cached]
    if missing:
        chunks = _partition(missing, num_processes)
        results: Dict[str, float] = {}
        if len(chunks) == 1:
            results.update(
                _score_chunk(
                    reward_config,
                    chunks[0],
                    image_root,
                    prompt_records,
                    _worker_device(device, 0, num_processes),
                    dtype,
                    batch_size,
                )
            )
        else:
            with ProcessPoolExecutor(
                max_workers=len(chunks), mp_context=get_context("spawn")
            ) as executor:
                futures = {
                    executor.submit(
                        _score_chunk,
                        reward_config,
                        chunk,
                        image_root,
                        prompt_records,
                        _worker_device(device, worker_index, num_processes),
                        dtype,
                        batch_size,
                    ): worker_index
                    for worker_index, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    results.update(future.result())
        cached.update(results)
        _write_scores(output_path, cached)
    expected = {_sample_key(row) for row in manifest_rows}
    if set(cached) != expected:
        raise ValueError(
            f"Reward cache keys do not match the inference manifest for {output_path}: "
            f"expected({len(expected)}), cached({len(cached)})."
        )
    return cached


def _score_chunk(
    reward_config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    image_root: Path,
    prompt_records: List[Any],
    device: str,
    dtype: str,
    batch_size: int,
) -> Dict[str, float]:
    device_object = torch.device(device)
    device_index = device_object.index or 0
    if device_object.type == "cuda":
        torch.cuda.set_device(device_index)
    elif device_object.type == "npu":
        torch.npu.set_device(device_index)
    config = RewardArguments.from_dict(
        {
            **reward_config,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        }
    )
    model_class = get_reward_model_class(str(config.reward_model))
    model = model_class(
        config=config,
        accelerator=_AcceleratorView(device=device_object, local_process_index=device_index),
    )
    results: Dict[str, float] = {}
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            images = [Image.open(image_root / row["image_path"]).convert("RGB") for row in batch]
            prompts = [str(row["prompt"]) for row in batch]
            metadata = [prompt_records[int(row["prompt_index"])].metadata for row in batch]
            call_parameters = inspect.signature(model.__call__).parameters
            call_kwargs: Dict[str, Any] = {"prompt": prompts, "image": images}
            if "metadata" in call_parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in call_parameters.values()
            ):
                call_kwargs["metadata"] = metadata
            output = model(**call_kwargs)
            raw_values = output.rewards
            if isinstance(raw_values, torch.Tensor):
                raw_values = raw_values.detach().float().cpu().numpy()
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            if values.shape != (len(batch),) or not np.isfinite(values).all():
                raise ValueError(f"Reward {config.name!r} returned invalid shape/values: {values}.")
            results.update({_sample_key(row): float(value) for row, value in zip(batch, values)})
            for image in images:
                image.close()
    finally:
        del model
        if device_object.type == "cuda":
            torch.cuda.empty_cache()
        elif device_object.type == "npu":
            torch.npu.empty_cache()
    return results


def _partition(rows: List[Dict[str, Any]], num_processes: int) -> List[List[Dict[str, Any]]]:
    """Partition complete prompt groups without splitting them across workers."""
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}.")
    by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(int(row["prompt_index"]), []).append(row)
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(min(num_processes, len(by_prompt)))]
    for group_index, prompt_index in enumerate(sorted(by_prompt)):
        chunks[group_index % len(chunks)].extend(by_prompt[prompt_index])
    return [chunk for chunk in chunks if chunk]


def _worker_device(device: str, worker_index: int, num_processes: int) -> str:
    if num_processes == 1:
        return device
    device_type = device.split(":", maxsplit=1)[0]
    return f"{device_type}:{worker_index}"


def _sample_key(row: Dict[str, Any]) -> str:
    return f"p{int(row['prompt_index'])}_s{int(row['sample_index'])}"


def _load_cached_scores(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {}
    values: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = str(row["sample_key"])
        if key in values:
            raise ValueError(f"Duplicate reward cache key {key!r} in {path}.")
        values[key] = float(row["value"])
    return values


def _write_scores(path: Path, values: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(values):
            handle.write(
                json.dumps(
                    {"sample_key": key, "value": values[key]},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
