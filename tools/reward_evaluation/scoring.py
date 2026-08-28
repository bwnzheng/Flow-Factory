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

"""Resumable registry-backed scoring for generated image manifests."""

from __future__ import annotations

import gc
import inspect
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from flow_factory.hparams import RewardArguments
from flow_factory.rewards.abc import GroupwiseRewardModel
from flow_factory.rewards.registry import get_reward_model_class
from flow_factory.rewards.vision_reward import (
    VisionRewardModel,
    compose_vision_reward_score,
    load_vision_reward_score_head,
)

_VISION_REWARD_STAGE_KEY = "_evaluation_stage"
_VISION_REWARD_ALIGNMENT_STAGE = "alignment"
_VISION_REWARD_VQA_STAGE = "vqa_features"


def _progress(message: str) -> None:
    """Emit evaluator progress immediately, including from spawned workers."""
    print(message, flush=True)


@dataclass(frozen=True)
class _AcceleratorView:
    """Provide the accelerator fields consumed by offline reward models."""

    device: torch.device
    local_process_index: int

    def wait_for_everyone(self) -> None:
        """Provide a no-op barrier for an independent offline worker."""
        return None


def score_reward(
    reward_config: Mapping[str, Any],
    manifest_rows: List[Dict[str, Any]],
    image_root: Path,
    prompt_records: List[Any],
    output_path: Path,
    device: str,
    dtype: str,
    num_processes: int,
    batch_size: int,
) -> Dict[str, float]:
    """Score a manifest with any registry reward that accepts image inputs.

    Pointwise rewards are batched normally.  ``GroupwiseRewardModel`` rewards
    are called once per prompt group, preserving the group boundary required by
    rank-based rewards such as ``pickscore_rank``.  Results are cached in JSONL
    so a failed or interrupted evaluation can be resumed safely.

    Rewards whose contract requires audio, video, or condition images are not
    silently adapted to image input; the original model error is wrapped with a
    useful configuration hint.
    """
    if not manifest_rows:
        raise ValueError("manifest_rows must contain at least one generated sample.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}.")

    reward_model_spec = str(reward_config.get("reward_model"))
    is_vision_reward_spec = reward_model_spec.lower() == "vision_reward" or reward_model_spec == (
        "flow_factory.rewards.vision_reward.VisionRewardModel"
    )
    if is_vision_reward_spec and _staged_vision_reward_enabled(reward_config):
        if get_reward_model_class(reward_model_spec) is not VisionRewardModel:
            raise TypeError(
                f"VisionReward spec {reward_model_spec!r} did not resolve to " "VisionRewardModel."
            )
        return _score_vision_reward_staged(
            reward_config=reward_config,
            manifest_rows=manifest_rows,
            image_root=image_root,
            output_path=output_path,
            device=device,
            dtype=dtype,
            num_processes=num_processes,
            batch_size=batch_size,
        )

    cached = _load_cached_scores(output_path)
    expected = {_sample_key(row) for row in manifest_rows}
    cache_scope = _sample_scope(next(iter(expected)))
    unknown = {key for key in cached if _sample_scope(key) == cache_scope and key not in expected}
    if unknown:
        raise ValueError(
            f"Reward cache {output_path} contains {len(unknown)} samples not present "
            "in the current inference manifest. Remove the stale cache to resume."
        )

    missing = [row for row in manifest_rows if _sample_key(row) not in cached]
    reward_name = str(reward_config.get("name", reward_config.get("reward_model", "reward")))
    if not missing:
        _progress(
            f"[Reward] cache complete name={reward_name} samples={len(expected)} "
            f"output={output_path}"
        )
    if missing:
        _validate_reward_config(reward_config, device, dtype, batch_size)
        chunks = _partition_by_prompt(missing, num_processes)
        results: Dict[str, float] = {}
        _progress(
            f"[Reward] start name={reward_name} samples={len(manifest_rows)} "
            f"missing={len(missing)} workers={len(chunks)} batch_size={batch_size} "
            f"output={output_path}"
        )
        if len(chunks) == 1:
            worker_device = _worker_device(device, 0, num_processes)
            chunk_results = _score_chunk(
                dict(reward_config),
                chunks[0],
                manifest_rows,
                image_root,
                prompt_records,
                worker_device,
                dtype,
                batch_size,
            )
            results.update(chunk_results)
            cached.update(chunk_results)
            _write_scores(output_path, cached)
            _progress(
                f"[Reward] worker complete name={reward_name} device={worker_device} "
                f"samples={len(chunk_results)} cached={len(cached)}/{len(expected)}"
            )
        else:
            with ProcessPoolExecutor(
                max_workers=len(chunks), mp_context=get_context("spawn")
            ) as executor:
                futures = {
                    executor.submit(
                        _score_chunk,
                        dict(reward_config),
                        chunk,
                        manifest_rows,
                        image_root,
                        prompt_records,
                        _worker_device(device, index, num_processes),
                        dtype,
                        batch_size,
                    ): index
                    for index, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    worker_index = futures[future]
                    chunk_results = future.result()
                    results.update(chunk_results)
                    cached.update(chunk_results)
                    _write_scores(output_path, cached)
                    _progress(
                        f"[Reward] worker complete name={reward_name} "
                        f"worker={worker_index} samples={len(chunk_results)} "
                        f"cached={len(cached)}/{len(expected)}"
                    )
        cached.update(results)
        _write_scores(output_path, cached)
        _progress(
            f"[Reward] complete name={reward_name} samples={len(expected)} " f"output={output_path}"
        )

    if not expected.issubset(cached):
        raise ValueError(
            f"Reward cache keys do not match the inference manifest for {output_path}: "
            f"expected({len(expected)}), cached({len(cached)})."
        )
    return {key: cached[key] for key in sorted(expected)}


def _staged_vision_reward_enabled(reward_config: Mapping[str, Any]) -> bool:
    """Return whether the standalone evaluator should split VisionReward."""
    value = reward_config.get("staged_evaluation")
    extra_kwargs = reward_config.get("extra_kwargs")
    if value is None and isinstance(extra_kwargs, Mapping):
        value = extra_kwargs.get("staged_evaluation")
    if value is None:
        return True
    if not isinstance(value, bool):
        raise TypeError("VisionReward `staged_evaluation` must be a boolean, " f"got {value!r}.")
    return value


def _score_vision_reward_staged(
    reward_config: Mapping[str, Any],
    manifest_rows: List[Dict[str, Any]],
    image_root: Path,
    output_path: Path,
    device: str,
    dtype: str,
    num_processes: int,
    batch_size: int,
) -> Dict[str, float]:
    """Run VisionReward components in disjoint worker lifetimes and combine on CPU."""
    expected = {_sample_key(row) for row in manifest_rows}
    cached = _load_cached_scores(output_path)
    _validate_cache_scope(output_path, cached, expected)
    missing_final = [row for row in manifest_rows if _sample_key(row) not in cached]
    reward_name = str(reward_config.get("name", "vision_reward"))
    if not missing_final:
        _progress(
            f"[Reward] cache complete name={reward_name} samples={len(expected)} "
            f"output={output_path}"
        )
        return {key: cached[key] for key in sorted(expected)}

    _validate_reward_config(reward_config, device, dtype, batch_size)
    coefficients, intercept = _vision_reward_score_head(reward_config, device, dtype, batch_size)
    feature_count = int(coefficients.size - 1)
    if feature_count <= 0:
        raise ValueError(
            f"VisionReward score head must contain alignment and VQA weights, got "
            f"{coefficients.size} coefficients."
        )

    alignment_path = _stage_cache_path(output_path, _VISION_REWARD_ALIGNMENT_STAGE)
    vqa_path = _stage_cache_path(output_path, _VISION_REWARD_VQA_STAGE)
    alignments = _load_cached_scores(alignment_path)
    vqa_features = _load_feature_vectors(vqa_path, feature_count)
    _validate_cache_scope(alignment_path, alignments, expected)
    _validate_cache_scope(vqa_path, vqa_features, expected)

    _progress(
        f"[Reward] staged start name={reward_name} samples={len(manifest_rows)} "
        f"missing={len(missing_final)} workers={num_processes} batch_size={batch_size}"
    )
    alignment_rows = [row for row in missing_final if _sample_key(row) not in alignments]
    _run_vision_reward_stage(
        stage=_VISION_REWARD_ALIGNMENT_STAGE,
        reward_config=reward_config,
        rows=alignment_rows,
        image_root=image_root,
        output_path=alignment_path,
        cached=alignments,
        device=device,
        dtype=dtype,
        num_processes=num_processes,
        batch_size=batch_size,
    )

    # _run_vision_reward_stage owns a complete process pool lifetime. Reaching
    # this point means all alignment workers have run cleanup and exited before
    # any SAT VisionReward worker is created.
    vqa_rows = [row for row in missing_final if _sample_key(row) not in vqa_features]
    _run_vision_reward_stage(
        stage=_VISION_REWARD_VQA_STAGE,
        reward_config=reward_config,
        rows=vqa_rows,
        image_root=image_root,
        output_path=vqa_path,
        cached=vqa_features,
        device=device,
        dtype=dtype,
        num_processes=num_processes,
        batch_size=batch_size,
    )

    missing_keys = {_sample_key(row) for row in missing_final}
    if not missing_keys.issubset(alignments) or not missing_keys.issubset(vqa_features):
        raise ValueError(
            "VisionReward staged caches are incomplete after both passes: "
            f"alignment={len(missing_keys.intersection(alignments))}/{len(missing_keys)}, "
            f"vqa_features={len(missing_keys.intersection(vqa_features))}/{len(missing_keys)}."
        )

    for key in sorted(missing_keys):
        cached[key] = compose_vision_reward_score(
            alignment=alignments[key],
            vqa_features=vqa_features[key],
            coefficients=coefficients,
            intercept=intercept,
        )
    _write_scores(output_path, cached)
    _progress(
        f"[Reward] staged complete name={reward_name} samples={len(expected)} "
        f"output={output_path}"
    )
    return {key: cached[key] for key in sorted(expected)}


def _vision_reward_score_head(
    reward_config: Mapping[str, Any], device: str, dtype: str, batch_size: int
) -> tuple[np.ndarray, float]:
    """Resolve VisionReward's linear head without constructing either model."""
    config = RewardArguments.from_dict(
        {
            **dict(reward_config),
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        }
    )
    return load_vision_reward_score_head(config.extra_kwargs or {})


def _run_vision_reward_stage(
    stage: str,
    reward_config: Mapping[str, Any],
    rows: List[Dict[str, Any]],
    image_root: Path,
    output_path: Path,
    cached: Dict[str, Any],
    device: str,
    dtype: str,
    num_processes: int,
    batch_size: int,
) -> None:
    """Run and persist one VisionReward component in its own process pool."""
    reward_name = str(reward_config.get("name", "vision_reward"))
    if not rows:
        _progress(
            f"[Reward pass] cache complete name={reward_name} stage={stage} "
            f"cached={len(cached)} output={output_path}"
        )
        return

    chunks = _partition_pointwise_rows(rows, num_processes)
    _progress(
        f"[Reward pass] start name={reward_name} stage={stage} samples={len(rows)} "
        f"workers={len(chunks)} output={output_path}"
    )

    def persist(chunk_results: Dict[str, Any], worker_index: int) -> None:
        cached.update(chunk_results)
        if stage == _VISION_REWARD_ALIGNMENT_STAGE:
            _write_scores(output_path, cached)
        else:
            _write_feature_vectors(output_path, cached)
        _progress(
            f"[Reward pass] worker complete name={reward_name} stage={stage} "
            f"worker={worker_index} samples={len(chunk_results)} cached={len(cached)}"
        )

    # Keep the direct path for lightweight CPU tests. Accelerator stages always
    # use a spawned process, even with one chunk, so process exit is the memory
    # reclamation boundary between the two heavyweight components.
    if len(chunks) == 1 and torch.device(device).type == "cpu":
        worker_device = _worker_device(device, 0, num_processes)
        persist(
            _score_vision_reward_stage_chunk(
                stage,
                dict(reward_config),
                chunks[0],
                image_root,
                worker_device,
                dtype,
                batch_size,
            ),
            0,
        )
    else:
        with ProcessPoolExecutor(
            max_workers=len(chunks), mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(
                    _score_vision_reward_stage_chunk,
                    stage,
                    dict(reward_config),
                    chunk,
                    image_root,
                    _worker_device(device, index, num_processes),
                    dtype,
                    batch_size,
                ): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                persist(future.result(), futures[future])

    _progress(f"[Reward pass] complete name={reward_name} stage={stage} " f"output={output_path}")


def _validate_reward_config(
    reward_config: Mapping[str, Any], device: str, dtype: str, batch_size: int
) -> None:
    """Run an optional lightweight validator before spawning reward workers.

    Heavy reward models are intentionally still constructed inside workers, but
    path/configuration errors should be reported once in the parent process
    instead of being repeated by every device worker.  Existing rewards do not
    need to implement this hook.
    """
    config = RewardArguments.from_dict(
        {
            **dict(reward_config),
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        }
    )
    model_class = get_reward_model_class(str(config.reward_model))
    validator = getattr(model_class, "validate_config", None)
    if validator is not None:
        validator(config)


def _score_vision_reward_stage_chunk(
    stage: str,
    reward_config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    image_root: Path,
    device: str,
    dtype: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Score one VisionReward component on one assigned device."""
    reward_name = str(reward_config.get("name", "vision_reward"))
    _progress(
        f"[Reward worker] start name={reward_name} stage={stage} "
        f"device={device} samples={len(rows)}"
    )
    device_object = torch.device(device)
    device_index = device_object.index or 0
    if device_object.type == "cuda":
        torch.cuda.set_device(device_index)
    elif device_object.type == "npu":
        torch.npu.set_device(device_index)

    stage_extras = dict(reward_config.get("extra_kwargs") or {})
    stage_extras[_VISION_REWARD_STAGE_KEY] = stage
    config = RewardArguments.from_dict(
        {
            **reward_config,
            "extra_kwargs": stage_extras,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        }
    )
    model_class = get_reward_model_class(str(config.reward_model))
    if model_class is not VisionRewardModel:
        raise TypeError(
            "The staged VisionReward worker requires VisionRewardModel, got "
            f"{model_class.__name__}."
        )
    model = model_class(
        config=config,
        accelerator=_AcceleratorView(device=device_object, local_process_index=device_index),
    )
    _progress(f"[Reward worker] model ready name={reward_name} stage={stage} device={device}")
    results: Dict[str, Any] = {}
    processed = 0
    progress_interval = max(1, (len(rows) + 9) // 10)
    last_reported = 0

    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            batch_index = start // batch_size + 1
            batch_count = (len(rows) + batch_size - 1) // batch_size
            started = time.perf_counter()
            _progress(
                f"[Reward worker] batch start name={reward_name} stage={stage} "
                f"device={device} batch={batch_index}/{batch_count} items={len(batch)}"
            )
            values = _call_vision_reward_stage(model, stage, batch, image_root)
            _progress(
                f"[Reward worker] batch done name={reward_name} stage={stage} "
                f"device={device} batch={batch_index}/{batch_count} items={len(batch)} "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
            results.update({_sample_key(row): value for row, value in zip(batch, values)})
            processed += len(batch)
            if processed == len(rows) or processed - last_reported >= progress_interval:
                _progress(
                    f"[Reward worker] progress name={reward_name} stage={stage} "
                    f"device={device} processed={processed}/{len(rows)}"
                )
                last_reported = processed
    finally:
        del model
        gc.collect()
        if device_object.type == "cuda":
            torch.cuda.empty_cache()
        elif device_object.type == "npu":
            torch.npu.empty_cache()
        _progress(f"[Reward worker] done name={reward_name} stage={stage} device={device}")
    return results


def _call_vision_reward_stage(
    model: Any,
    stage: str,
    rows: List[Dict[str, Any]],
    image_root: Path,
) -> List[Any]:
    """Load images and extract one staged VisionReward feature family."""
    images: List[Image.Image] = []
    try:
        images = [Image.open(image_root / str(row["image_path"])).convert("RGB") for row in rows]
        prompts = [str(row["prompt"]) for row in rows]
        if stage == _VISION_REWARD_ALIGNMENT_STAGE:
            raw_values = model.score_alignment_features(prompts, images)
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            if values.shape != (len(rows),) or not np.isfinite(values).all():
                raise ValueError(
                    f"VisionReward alignment stage returned invalid values {values}; "
                    f"expected {len(rows)} finite scalars."
                )
            return [float(value) for value in values]

        if stage != _VISION_REWARD_VQA_STAGE:
            raise ValueError(f"Unsupported VisionReward evaluation stage {stage!r}.")
        raw_features = model.score_vqa_features(images)
        features = np.asarray(raw_features, dtype=np.float64)
        expected_features = model.vqa_feature_count
        if features.shape != (len(rows), expected_features) or not np.isfinite(features).all():
            raise ValueError(
                "VisionReward VQA stage returned invalid feature shape/values: "
                f"shape={features.shape}, expected={(len(rows), expected_features)}."
            )
        return [[float(value) for value in row] for row in features]
    finally:
        for image in images:
            image.close()


def _score_chunk(
    reward_config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    all_manifest_rows: List[Dict[str, Any]],
    image_root: Path,
    prompt_records: List[Any],
    device: str,
    dtype: str,
    batch_size: int,
) -> Dict[str, float]:
    reward_name = str(reward_config.get("name", reward_config.get("reward_model", "reward")))
    _progress(f"[Reward worker] start name={reward_name} device={device} samples={len(rows)}")
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
    is_groupwise = issubclass(model_class, GroupwiseRewardModel)
    model = model_class(
        config=config,
        accelerator=_AcceleratorView(device=device_object, local_process_index=device_index),
    )
    _progress(f"[Reward worker] model ready name={reward_name} device={device}")
    results: Dict[str, float] = {}
    progress_interval = max(1, (len(rows) + 9) // 10)
    processed = 0
    last_reported = 0

    def report_progress() -> None:
        nonlocal last_reported
        if processed == len(rows) or processed - last_reported >= progress_interval:
            _progress(
                f"[Reward worker] progress name={reward_name} device={device} "
                f"processed={processed}/{len(rows)}"
            )
            last_reported = processed

    try:
        if is_groupwise:
            # A cache may contain some samples from a group. Re-score the full
            # group so ranks are always computed against the same candidate set.
            groups = _groups_for_rows(rows, all_manifest_rows)
            for group_index, group_rows in enumerate(groups, start=1):
                started = time.perf_counter()
                _progress(
                    f"[Reward worker] batch start name={reward_name} device={device} "
                    f"group={group_index}/{len(groups)} items={len(group_rows)}"
                )
                values = _call_model(model, group_rows, image_root, prompt_records)
                _progress(
                    f"[Reward worker] batch done name={reward_name} device={device} "
                    f"group={group_index}/{len(groups)} items={len(group_rows)} "
                    f"elapsed={time.perf_counter() - started:.1f}s"
                )
                results.update({_sample_key(row): value for row, value in zip(group_rows, values)})
                processed += len(group_rows)
                report_progress()
        else:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                batch_index = start // batch_size + 1
                batch_count = (len(rows) + batch_size - 1) // batch_size
                started = time.perf_counter()
                _progress(
                    f"[Reward worker] batch start name={reward_name} device={device} "
                    f"batch={batch_index}/{batch_count} items={len(batch)}"
                )
                values = _call_model(model, batch, image_root, prompt_records)
                _progress(
                    f"[Reward worker] batch done name={reward_name} device={device} "
                    f"batch={batch_index}/{batch_count} items={len(batch)} "
                    f"elapsed={time.perf_counter() - started:.1f}s"
                )
                results.update({_sample_key(row): value for row, value in zip(batch, values)})
                processed += len(batch)
                report_progress()
    finally:
        del model
        if device_object.type == "cuda":
            torch.cuda.empty_cache()
        elif device_object.type == "npu":
            torch.npu.empty_cache()
        _progress(f"[Reward worker] done name={reward_name} device={device}")
    return results


def _call_model(
    model: Any,
    rows: List[Dict[str, Any]],
    image_root: Path,
    prompt_records: List[Any],
) -> List[float]:
    images = []
    try:
        images = [Image.open(image_root / str(row["image_path"])).convert("RGB") for row in rows]
        prompts = [str(row["prompt"]) for row in rows]
        metadata = [prompt_records[int(row["prompt_index"])].metadata for row in rows]
        parameters = inspect.signature(model.__call__).parameters
        kwargs: Dict[str, Any] = {"prompt": prompts, "image": images}
        if "metadata" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            kwargs["metadata"] = metadata
        try:
            output = model(**kwargs)
        except Exception as exc:
            name = getattr(getattr(model, "config", None), "name", type(model).__name__)
            raise RuntimeError(
                f"Reward {name!r} could not score image inputs. If this reward requires "
                "video/audio/condition images, provide a modality-specific evaluator "
                "or remove it from this image evaluation config."
            ) from exc
        raw_values = output.rewards
        if isinstance(raw_values, torch.Tensor):
            raw_values = raw_values.detach().float().cpu().numpy()
        values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        if values.shape != (len(rows),) or not np.isfinite(values).all():
            name = getattr(getattr(model, "config", None), "name", type(model).__name__)
            raise ValueError(
                f"Reward {name!r} returned invalid shape/values: {values}. "
                f"Expected {len(rows)} finite values."
            )
        return [float(value) for value in values]
    finally:
        for image in images:
            image.close()


def _groups_for_rows(
    rows: Iterable[Dict[str, Any]], all_rows: Iterable[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    selected = {int(row["prompt_index"]) for row in rows}
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in all_rows:
        prompt_index = int(row["prompt_index"])
        if prompt_index in selected:
            grouped.setdefault(prompt_index, []).append(row)
    return [
        sorted(group, key=lambda row: int(row["sample_index"]))
        for _, group in sorted(grouped.items())
    ]


def _partition_by_prompt(
    rows: List[Dict[str, Any]], num_processes: int
) -> List[List[Dict[str, Any]]]:
    by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(int(row["prompt_index"]), []).append(row)
    if not by_prompt:
        return []
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(min(num_processes, len(by_prompt)))]
    for index, prompt_index in enumerate(sorted(by_prompt)):
        chunks[index % len(chunks)].extend(by_prompt[prompt_index])
    return [chunk for chunk in chunks if chunk]


def _partition_pointwise_rows(
    rows: List[Dict[str, Any]], num_processes: int
) -> List[List[Dict[str, Any]]]:
    """Balance independent samples across every available worker."""
    if not rows:
        return []
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(min(num_processes, len(rows)))]
    for index, row in enumerate(rows):
        chunks[index % len(chunks)].append(row)
    return chunks


def _worker_device(device: str, worker_index: int, num_processes: int) -> str:
    if num_processes == 1:
        return device
    return f"{device.split(':', maxsplit=1)[0]}:{worker_index}"


def _sample_key(row: Mapping[str, Any]) -> str:
    step = row.get("checkpoint_step")
    checkpoint_prefix = f"c{int(step)}_" if step is not None else ""
    return f"{checkpoint_prefix}p{int(row['prompt_index'])}_s{int(row['sample_index'])}"


def _sample_scope(key: str) -> str:
    """Return the checkpoint prefix used to isolate one cache scope."""
    return key.partition("p")[0]


def _validate_cache_scope(path: Path, cached: Mapping[str, Any], expected: set[str]) -> None:
    """Reject stale samples from the active checkpoint while retaining older scopes."""
    cache_scope = _sample_scope(next(iter(expected)))
    unknown = {key for key in cached if _sample_scope(key) == cache_scope and key not in expected}
    if unknown:
        raise ValueError(
            f"Reward cache {path} contains {len(unknown)} samples not present "
            "in the current inference manifest. Remove the stale cache to resume."
        )


def _stage_cache_path(output_path: Path, stage: str) -> Path:
    """Return a sibling cache path for a VisionReward feature pass."""
    return output_path.with_name(f"{output_path.stem}.{stage}{output_path.suffix}")


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


def _load_feature_vectors(path: Path, feature_count: int) -> Dict[str, List[float]]:
    """Load and validate a resumable VisionReward VQA-feature cache."""
    if not path.is_file():
        return {}
    values: Dict[str, List[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = str(row["sample_key"])
        if key in values:
            raise ValueError(f"Duplicate reward cache key {key!r} in {path}.")
        features = np.asarray(row["features"], dtype=np.float64).reshape(-1)
        if features.shape != (feature_count,) or not np.isfinite(features).all():
            raise ValueError(
                f"Invalid VisionReward VQA features for {key!r} in {path}: "
                f"expected {feature_count} finite values, got shape={features.shape}."
            )
        values[key] = [float(value) for value in features]
    return values


def _write_scores(path: Path, values: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for key in sorted(values):
            handle.write(
                json.dumps(
                    {"sample_key": key, "value": values[key]},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary_path, path)


def _write_feature_vectors(path: Path, values: Mapping[str, Sequence[float]]) -> None:
    """Atomically persist staged VisionReward VQA features."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for key in sorted(values):
            handle.write(
                json.dumps(
                    {"sample_key": key, "features": list(values[key])},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary_path, path)
