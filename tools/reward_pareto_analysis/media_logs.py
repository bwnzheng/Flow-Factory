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

"""JSONL log reader — replaces EventAccumulator for the new log format.

Reads ``logs/media.jsonl`` from a training run directory.  Each line is a JSON
object with keys: ``step``, ``key``, ``path``, ``prompt``, ``reward``.

The ``path`` field is relative to ``{log_dir}`` (e.g. ``images/step_0.png``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def load_media_samples(
    log_dir: str,
    datasets: Optional[List[str]] = None,
    max_per_step: int = 0,
) -> Dict[int, List[Dict[str, Any]]]:
    """Read rollout images from ``media.jsonl``, grouped by step.

    Filters entries whose ``key`` starts with ``train_samples/`` (dataset "train")
    or ``eval/`` (dataset "eval").

    Args:
        log_dir: Path to the ``logs/`` directory inside a run folder.
        datasets: If non-empty, only keep images from these datasets
            (e.g. ``["train"]``).
        max_per_step: If > 0, cap images per step.

    Returns:
        ``{step: [{"image_path": str, "prompt": str, "tag_idx": int, "dataset": str}, ...]}``
        where ``image_path`` is an absolute path to a PNG file.
    """
    media_path = os.path.join(log_dir, "media.jsonl")
    if not os.path.isfile(media_path):
        raise FileNotFoundError(f"media.jsonl not found: {media_path}")

    dataset_set = set(datasets) if datasets else None

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    step_counts: Dict[int, int] = {}

    with open(media_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            key = entry.get("key", "")
            step = entry.get("step", 0)
            prompt = entry.get("prompt", "")
            rel_path = entry.get("path", "")

            if not key or not rel_path:
                continue

            # Determine dataset
            if key.startswith("train_samples/"):
                dataset = "train"
            elif key.startswith("eval/"):
                dataset = "eval"
            else:
                continue  # skip non-image entries

            if dataset_set is not None and dataset not in dataset_set:
                continue

            # Parse tag index from key (e.g. "train_samples/15" → 15)
            try:
                tag_idx = int(key.rsplit("/", 1)[-1])
            except (ValueError, IndexError):
                tag_idx = 0

            if max_per_step > 0 and step_counts.get(step, 0) >= max_per_step:
                continue

            # Resolve image path (rel_path is relative to log_dir, not images_dir)
            if os.path.isabs(rel_path):
                img_path = rel_path
            else:
                img_path = os.path.join(log_dir, rel_path)

            if step not in grouped:
                grouped[step] = []
            grouped[step].append(
                {
                    "image_path": img_path,
                    "prompt": prompt,
                    "tag_idx": tag_idx,
                    "dataset": dataset,
                }
            )
            step_counts[step] = step_counts.get(step, 0) + 1

    # Sort within each step by tag_idx
    for step in grouped:
        grouped[step].sort(key=lambda x: x["tag_idx"])

    return grouped
