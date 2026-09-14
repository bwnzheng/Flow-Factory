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

"""Prompt records shared by standalone reward-evaluation tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass(frozen=True)
class PromptRecord:
    """Store one prompt and its JSON-encoded reward metadata."""

    prompt: str
    metadata: str


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
