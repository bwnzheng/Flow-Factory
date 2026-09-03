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

"""YAML configuration loading for standalone model inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import yaml

from .model_types import validate_model_type


@dataclass(frozen=True)
class ModelInferenceConfig:
    """Store validated standalone inference configuration."""

    base_model: str
    checkpoint_dir: Optional[str]
    checkpoint_path: Optional[str]
    evaluation_set: str
    output_dir: str
    model_type: str = "sd3-5"
    dtype: str = "bfloat16"
    device: Optional[str] = None
    num_processes: int = 1
    checkpoint_step: Optional[int] = None
    prompt_key: str = "prompt"
    max_prompts: int = 0
    num_samples: int = 4
    batch_size: int = 16
    seed: int = 42
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


def _require_mapping(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a required mapping section."""
    section = raw.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return section


def _reject_unknown_keys(section_name: str, section: Dict[str, Any], allowed: Set[str]) -> None:
    """Reject misspelled structural configuration fields."""
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Unknown fields in '{section_name}': {unknown}")


def _require_string(section_name: str, section: Dict[str, Any], key: str) -> str:
    """Return a required non-empty string field."""
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{section_name}.{key}' must be a non-empty string.")
    return value.strip()


def _optional_string(section_name: str, section: Dict[str, Any], key: str) -> Optional[str]:
    """Return an optional string field, normalizing empty strings to ``None``."""
    value = section.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config field '{section_name}.{key}' must be a string or null.")
    return value


def _integer_field(
    section_name: str,
    section: Dict[str, Any],
    key: str,
    default: int,
) -> int:
    """Return an integer field without accepting booleans."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config field '{section_name}.{key}' must be an integer.")
    return value


def load_inference_config(path: str) -> ModelInferenceConfig:
    """Load and validate standalone model inference YAML.

    Args:
        path: YAML configuration path.

    Returns:
        Validated model inference configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML schema or values are invalid.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Model inference config must contain a top-level mapping.")

    allowed_top_level = {"model", "checkpoint", "evaluation", "generation", "output"}
    _reject_unknown_keys("root", raw, allowed_top_level)
    model = _require_mapping(raw, "model")
    checkpoint = _require_mapping(raw, "checkpoint")
    evaluation = _require_mapping(raw, "evaluation")
    generation = _require_mapping(raw, "generation")
    output = _require_mapping(raw, "output")

    _reject_unknown_keys(
        "model",
        model,
        {"model_type", "base_model", "dtype", "device", "num_processes"},
    )
    _reject_unknown_keys("checkpoint", checkpoint, {"dir", "path", "step"})
    _reject_unknown_keys(
        "evaluation",
        evaluation,
        {"dataset", "prompt_key", "max_prompts", "num_samples", "batch_size", "seed"},
    )
    _reject_unknown_keys("output", output, {"dir"})

    base_model = _require_string("model", model, "base_model")
    model_type = validate_model_type(model.get("model_type", "sd3-5"))
    dtype = model.get("dtype", "bfloat16")
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Config field 'model.dtype' must be one of: bfloat16, float16, float32.")
    device = _optional_string("model", model, "device")
    num_processes = _integer_field("model", model, "num_processes", 1)
    if num_processes <= 0:
        raise ValueError(f"model.num_processes must be positive, got {num_processes}.")

    checkpoint_dir = _optional_string("checkpoint", checkpoint, "dir")
    checkpoint_path = _optional_string("checkpoint", checkpoint, "path")
    if bool(checkpoint_dir) == bool(checkpoint_path):
        raise ValueError("Specify exactly one of checkpoint.dir or checkpoint.path.")
    checkpoint_step = checkpoint.get("step")
    if checkpoint_step is not None and (
        isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int)
    ):
        raise ValueError("Config field 'checkpoint.step' must be an integer or null.")
    if checkpoint_step is not None and checkpoint_step < 0:
        raise ValueError(f"checkpoint.step must be non-negative, got {checkpoint_step}.")
    if checkpoint_dir and checkpoint_step is not None:
        raise ValueError("checkpoint.step is only valid with checkpoint.path.")

    evaluation_set = _require_string("evaluation", evaluation, "dataset")
    prompt_key = evaluation.get("prompt_key", "prompt")
    if not isinstance(prompt_key, str) or not prompt_key.strip():
        raise ValueError("Config field 'evaluation.prompt_key' must be a non-empty string.")
    max_prompts = _integer_field("evaluation", evaluation, "max_prompts", 0)
    num_samples = _integer_field("evaluation", evaluation, "num_samples", 4)
    batch_size = _integer_field("evaluation", evaluation, "batch_size", 16)
    seed = _integer_field("evaluation", evaluation, "seed", 42)
    if max_prompts < 0:
        raise ValueError(f"evaluation.max_prompts must be non-negative, got {max_prompts}.")
    if num_samples <= 0:
        raise ValueError(f"evaluation.num_samples must be positive, got {num_samples}.")
    if batch_size <= 0:
        raise ValueError(f"evaluation.batch_size must be positive, got {batch_size}.")

    output_dir = _require_string("output", output, "dir")
    return ModelInferenceConfig(
        base_model=base_model,
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        evaluation_set=evaluation_set,
        prompt_key=prompt_key.strip(),
        max_prompts=max_prompts,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        generation_kwargs=dict(generation),
        output_dir=output_dir,
        dtype=dtype,
        device=device,
        num_processes=num_processes,
    )
