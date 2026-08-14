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

"""Strict reader for saved prompt-group reward pickles."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_KEYS = {
    "step",
    "prompts",
    "src_groups",
    "nondom_sizes_in_group",
    "schema_version",
}


@dataclass(frozen=True)
class RewardGroup:
    """One immutable prompt-local reward matrix from a saved training step."""

    step: int
    group_id: int
    prompt: str
    reward_names: tuple[str, ...]
    rewards: np.ndarray
    probabilities: np.ndarray | None


@dataclass(frozen=True)
class SavedRewardWeightContext:
    """Resolved training reward weights preserved in a media run context."""

    weights_by_source: dict[str, dict[str, float]]


def load_train_reward_groups(rewards_dir: str | Path) -> dict[int, list[RewardGroup]]:
    """Read ``train_step_*.pkl`` files without flattening prompt groups.

    All active dimensions in a group must be either entirely finite or entirely
    NaN. A partially missing reward would make sample-level comparisons
    ambiguous, so it is rejected instead of filtering individual samples.
    """
    directory = Path(rewards_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Saved rewards directory does not exist: {directory}")

    grouped_by_step: dict[int, list[RewardGroup]] = {}
    paths = sorted(directory.glob("train_step_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No train_step_*.pkl files found in {directory}")

    for path in paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        groups = _parse_train_payload(payload, path)
        if not groups:
            raise ValueError(f"No multi-reward prompt groups found in {path}.")
        step = groups[0].step
        if step in grouped_by_step:
            raise ValueError(f"Duplicate reward-log step {step} in {directory}.")
        grouped_by_step[step] = groups
    return dict(sorted(grouped_by_step.items()))


def load_saved_reward_weight_context(run_dir: str | Path) -> SavedRewardWeightContext | None:
    """Load resolved training reward weights from ``logs/media.jsonl`` when present.

    Current local-media manifests start with a ``run_context`` record whose
    serialized configuration contains the fully resolved per-dataset reward
    weights. Older manifests and runs without locally saved media do not have
    this record; callers should then use an explicit analysis-YAML fallback.
    """
    path = Path(run_dir) / "logs" / "media.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}.") from error
            if record.get("record_type") != "run_context":
                continue
            configuration = record.get("configuration")
            if not isinstance(configuration, dict):
                raise ValueError(f"run_context.configuration must be a mapping in {path}.")
            return _parse_saved_reward_weight_context(configuration, path)
    return None


def _parse_train_payload(payload: Any, path: Path) -> list[RewardGroup]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary in {path}, got {type(payload).__name__}.")
    if "step" not in payload or "prompts" not in payload:
        raise ValueError(f"Saved train rewards require 'step' and 'prompts': {path}")

    step = int(payload["step"])
    prompts_raw = payload["prompts"]
    if not isinstance(prompts_raw, (list, tuple)) or not prompts_raw:
        raise ValueError(f"'prompts' must be a non-empty sequence in {path}.")
    prompts = [str(prompt) for prompt in prompts_raw]
    n_groups = len(prompts)
    reward_names = _reward_names(payload, n_groups, path)
    src_metadata = _src_metadata_by_group(payload.get("src_groups"), n_groups, path)

    result: list[RewardGroup] = []
    for group_id, prompt in enumerate(prompts):
        arrays: dict[str, np.ndarray] = {}
        available: list[str] = []
        sample_counts: set[int] = set()
        for reward_name in reward_names:
            values = np.asarray(payload[reward_name][group_id], dtype=np.float64).reshape(-1)
            if values.size == 0:
                raise ValueError(
                    f"Empty reward group at step {step}, group {group_id}, "
                    f"reward {reward_name!r} ({path})."
                )
            if np.isinf(values).any():
                raise ValueError(
                    f"Infinite reward at step {step}, group {group_id}, "
                    f"reward {reward_name!r} ({path})."
                )
            arrays[reward_name] = values
            sample_counts.add(values.size)
            if np.isfinite(values).all():
                available.append(reward_name)
            elif not np.isnan(values).all():
                raise ValueError(
                    f"Partially missing reward at step {step}, group {group_id}, "
                    f"reward {reward_name!r} ({path})."
                )

        if len(sample_counts) != 1:
            raise ValueError(
                f"Reward sample counts differ at step {step}, group {group_id}: "
                f"{sorted(sample_counts)} ({path})."
            )
        if len(available) < 2:
            raise ValueError(
                f"Reward-disagreement analysis requires at least two finite rewards; "
                f"step {step}, group {group_id} has {available} ({path})."
            )

        metadata = src_metadata.get(group_id)
        probabilities = None if metadata is None else metadata["probabilities"]
        n_samples = next(iter(sample_counts))
        if probabilities is not None and probabilities.shape != (n_samples,):
            raise ValueError(
                f"SRC probabilities at step {step}, group {group_id} have shape "
                f"{probabilities.shape}, expected ({n_samples},) ({path})."
            )
        result.append(
            RewardGroup(
                step=step,
                group_id=group_id,
                prompt=prompt,
                reward_names=tuple(available),
                rewards=np.column_stack([arrays[name] for name in available]),
                probabilities=probabilities,
            )
        )
    return result


def _reward_names(payload: dict[str, Any], n_groups: int, path: Path) -> list[str]:
    names: list[str] = []
    for name, values in payload.items():
        if name in _METADATA_KEYS:
            continue
        if not isinstance(values, (list, tuple)) or len(values) != n_groups:
            raise ValueError(
                f"Unexpected non-reward field {name!r} in {path}; expected a sequence of "
                f"{n_groups} prompt-group score arrays."
            )
        names.append(str(name))
    if len(names) < 2:
        raise ValueError(f"Expected at least two reward fields in {path}, got {names}.")
    return sorted(names)


def _src_metadata_by_group(
    source: Any,
    n_groups: int,
    path: Path,
) -> dict[int, dict[str, Any]]:
    if source is None:
        return {}
    if not isinstance(source, (list, tuple)) or len(source) != n_groups:
        raise ValueError(f"'src_groups' must be a sequence of {n_groups} entries in {path}.")

    result: dict[int, dict[str, Any]] = {}
    for entry in source:
        if not isinstance(entry, dict) or "group_id" not in entry or "probabilities" not in entry:
            raise ValueError(
                f"Each src_groups entry must include group_id and probabilities ({path})."
            )
        group_id = int(entry["group_id"])
        if group_id in result or group_id < 0 or group_id >= n_groups:
            raise ValueError(f"Invalid or duplicate SRC group_id {group_id} in {path}.")
        probabilities = np.asarray(entry["probabilities"], dtype=np.float64).reshape(-1)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
            raise ValueError(f"Invalid SRC probabilities for group {group_id} in {path}.")
        total = float(probabilities.sum())
        if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"SRC probabilities for group {group_id} sum to {total:.9g}, expected 1 ({path})."
            )

        result[group_id] = {"probabilities": probabilities / total}
    if set(result) != set(range(n_groups)):
        raise ValueError(f"SRC group IDs must cover 0..{n_groups - 1} in {path}.")
    return result


def _parse_saved_reward_weight_context(
    configuration: dict[str, Any],
    path: Path,
) -> SavedRewardWeightContext:
    rewards_raw = configuration.get("reward")
    if isinstance(rewards_raw, dict):
        reward_entries = list(rewards_raw.values())
    elif isinstance(rewards_raw, list):
        reward_entries = rewards_raw
    else:
        raise ValueError(f"run_context.configuration.reward must be a mapping or list in {path}.")

    weights_by_source: dict[str, dict[str, float]] = {}
    for index, entry in enumerate(reward_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"configuration.reward[{index}] must be a mapping in {path}.")
        reward_name = str(entry.get("name", "")).strip()
        weights = entry.get("weight")
        if not reward_name or not isinstance(weights, dict):
            raise ValueError(
                "Saved run context must contain each reward name and resolved per-source weight "
                f"mapping; malformed reward entry {index} in {path}."
            )
        for source, value in weights.items():
            weight = float(value)
            if not np.isfinite(weight):
                raise ValueError(
                    f"Saved reward weight for {reward_name!r}, source {source!r} is not finite in {path}."
                )
            source_weights = weights_by_source.setdefault(str(source), {})
            if reward_name in source_weights:
                raise ValueError(
                    f"Duplicate saved reward {reward_name!r} for source {source!r} in {path}."
                )
            source_weights[reward_name] = weight
    if not weights_by_source:
        raise ValueError(f"Saved run context has no training reward weights in {path}.")
    return SavedRewardWeightContext(weights_by_source=weights_by_source)
