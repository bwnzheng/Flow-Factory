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

"""Regression tests for fresh checkpoint reward covariance analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.reward_covariance_eval_analysis.analyze import (
    AnalysisConfig,
    EvaluationConfig,
    ModelConfig,
    PromptRecord,
    RunConfig,
    SourceConfig,
    _generate_images,
    _write_analysis_artifacts,
    load_config,
)
from tools.reward_covariance_eval_analysis.reward_scoring import _partition, _worker_device


def test_default_config_is_weight_free_and_uses_fresh_rollouts() -> None:
    root = Path(__file__).parents[2]
    config = load_config(root / "tools/reward_covariance_eval_analysis/default.yaml")
    assert config.evaluation.num_samples_per_prompt == 16
    assert config.model.num_processes == 1
    assert config.model.device is None
    assert [source.name for source in config.sources] == ["pickscore", "ocr"]


def test_multi_process_config_rejects_indexed_device(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        """
model: {base_model: model, dtype: bfloat16, device: 'cuda:2', num_processes: 2}
evaluation: {num_samples_per_prompt: 2, generation_batch_size: 1, reward_batch_size: 1, seed: 1}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: run, checkpoint: saves/run/checkpoint-1}
output: {dir: output}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="accelerator type"):
        load_config(config_path)


def test_multi_process_config_accepts_npu_device_type(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        """
model: {base_model: model, dtype: bfloat16, device: npu, num_processes: 2}
evaluation: {num_samples_per_prompt: 2, generation_batch_size: 1, reward_batch_size: 1, seed: 1}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: run, checkpoint: saves/run/checkpoint-1}
output: {dir: output}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.model.device == "npu"
    assert config.model.num_processes == 2


def test_partition_keeps_prompt_groups_on_one_worker() -> None:
    rows = [
        {"prompt_index": prompt, "sample_index": sample}
        for prompt in range(5)
        for sample in range(3)
    ]
    chunks = _partition(rows, num_processes=3)
    owners = {}
    for worker, chunk in enumerate(chunks):
        for row in chunk:
            owners.setdefault(row["prompt_index"], set()).add(worker)
    assert all(len(worker_ids) == 1 for worker_ids in owners.values())
    assert _worker_device("cuda", 1, 2) == "cuda:1"
    assert _worker_device("npu", 1, 2) == "npu:1"


def test_generate_images_uses_configured_parallel_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class FakeParallelRunner:
        def __init__(self, base_model, dtype, num_processes, device):
            captured.update(
                base_model=base_model,
                dtype=dtype,
                num_processes=num_processes,
                device=device,
            )

        def close(self):
            captured["closed"] = True

    def fake_run_evaluation_set(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        row = {
            "prompt_index": 0,
            "sample_index": 0,
            "seed": 42,
            "prompt": "prompt",
            "image_path": "checkpoint_7/p0_s0.png",
        }
        (output_dir / "manifest.jsonl").write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(
        "tools.reward_covariance_eval_analysis.analyze.ParallelEvaluationRunner",
        FakeParallelRunner,
    )
    monkeypatch.setattr(
        "tools.reward_covariance_eval_analysis.analyze.run_evaluation_set",
        fake_run_evaluation_set,
    )
    config = AnalysisConfig(
        model=ModelConfig("model", "bfloat16", "cuda", 2),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=[],
        output_dir=str(tmp_path),
    )
    rows = _generate_images(
        config,
        RunConfig("run", "Run", "checkpoint-7"),
        7,
        SourceConfig("source", "prompts.txt", "prompt", 0, []),
        [PromptRecord("prompt", "{}")],
        tmp_path / "images",
    )
    assert captured == {
        "base_model": "model",
        "dtype": "bfloat16",
        "num_processes": 2,
        "device": "cuda",
        "closed": True,
    }
    assert rows[0]["prompt"] == "prompt"


def test_artifacts_preserve_samples_and_prompt_local_matrices(tmp_path: Path) -> None:
    config = AnalysisConfig(
        model=ModelConfig("model", "bfloat16", "cpu", 1),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=[],
        output_dir=str(tmp_path),
    )
    run = RunConfig("run", "Run", "checkpoint-7")
    source = SourceConfig("source", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])
    prompts = [PromptRecord("prompt zero", "{}"), PromptRecord("prompt one", "{}")]
    manifest = []
    values_a = {}
    values_b = {}
    for prompt_index in range(2):
        for sample_index in range(2):
            key = f"p{prompt_index}_s{sample_index}"
            manifest.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "seed": 42 + prompt_index * 2 + sample_index,
                    "prompt": prompts[prompt_index].prompt,
                    "image_path": f"checkpoint_7/{key}.png",
                }
            )
            values_a[key] = float(prompt_index + sample_index)
            values_b[key] = float(prompt_index + sample_index * 2)
    summary = _write_analysis_artifacts(
        config,
        run,
        source,
        7,
        prompts,
        manifest,
        {"a": values_a, "b": values_b},
        tmp_path,
    )
    sample_rows = [
        json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()
    ]
    metric_rows = [
        json.loads(line) for line in (tmp_path / "prompt_metrics.jsonl").read_text().splitlines()
    ]
    assert len(sample_rows) == 4
    assert len(metric_rows) == 2
    np.testing.assert_allclose(metric_rows[0]["covariance"], [[0.5, 1.0], [1.0, 2.0]])
    assert summary["n_prompts"] == 2
