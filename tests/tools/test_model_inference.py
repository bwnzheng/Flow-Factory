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

"""Tests for reusable evaluation-set model inference utilities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import yaml
from PIL import Image

from tools.model_inference import (
    EvaluationRunner,
    ParallelEvaluationRunner,
    cli,
    discover_checkpoints,
    load_evaluation_prompts,
    load_inference_config,
    resolve_checkpoints,
    resolve_device,
    run_evaluation_set,
)
from tools.model_inference.runner import _expected_outputs


class _FakeRunner:
    """Record inference requests without loading a model."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate_for_checkpoint(self, **kwargs: Any) -> List[str]:
        """Record one checkpoint request and return its expected paths."""
        self.calls.append(kwargs)
        step = kwargs["step"]
        prompts = kwargs["prompts"]
        num_samples = kwargs["num_samples"]
        return [
            f"checkpoint_{step}/p{prompt_index}_s{sample_index}.png"
            for prompt_index in range(len(prompts))
            for sample_index in range(num_samples)
        ]


def test_discover_checkpoints_ignores_noncanonical_directories(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-20").mkdir()
    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "checkpoint-4-extra").mkdir()
    (tmp_path / "checkpoint-1").write_text("not a directory", encoding="utf-8")

    checkpoints = discover_checkpoints(str(tmp_path))

    assert checkpoints == [
        (3, str(tmp_path / "checkpoint-3")),
        (20, str(tmp_path / "checkpoint-20")),
    ]


def test_resolve_single_checkpoint_infers_or_accepts_step(tmp_path: Path) -> None:
    canonical = tmp_path / "checkpoint-17"
    canonical.mkdir()
    custom = tmp_path / "best"
    custom.mkdir()

    assert resolve_checkpoints(checkpoint_path=str(canonical)) == [(17, str(canonical))]
    assert resolve_checkpoints(checkpoint_path=str(custom), checkpoint_step=9) == [(9, str(custom))]
    with pytest.raises(ValueError, match="Cannot infer checkpoint step"):
        resolve_checkpoints(checkpoint_path=str(custom))


def test_load_evaluation_prompts_supports_text_and_jsonl(tmp_path: Path) -> None:
    text_path = tmp_path / "test.txt"
    text_path.write_text("first prompt\n\nsecond prompt\nthird prompt\n", encoding="utf-8")
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "edit one"}),
                json.dumps({"instruction": "edit two"}),
            ]
        ),
        encoding="utf-8",
    )

    assert load_evaluation_prompts(str(text_path), max_prompts=2) == [
        "first prompt",
        "second prompt",
    ]
    assert load_evaluation_prompts(str(jsonl_path), prompt_key="instruction") == [
        "edit one",
        "edit two",
    ]


def test_load_evaluation_prompts_reports_invalid_jsonl_field(tmp_path: Path) -> None:
    evaluation_set = tmp_path / "test.jsonl"
    evaluation_set.write_text(json.dumps({"text": "missing prompt"}), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string field 'prompt'"):
        load_evaluation_prompts(str(evaluation_set))


def test_run_evaluation_set_calls_runner_and_writes_manifest(tmp_path: Path) -> None:
    runner = _FakeRunner()
    checkpoints = [(2, "/checkpoints/checkpoint-2"), (5, "/checkpoints/checkpoint-5")]
    prompts = ["alpha", "beta"]

    generated = run_evaluation_set(
        runner=runner,
        checkpoints=checkpoints,
        prompts=prompts,
        output_dir=str(tmp_path),
        num_samples=2,
        generation_kwargs={"num_inference_steps": 10},
        batch_size=3,
        base_seed=100,
    )

    assert sorted(generated) == [2, 5]
    assert [call["step"] for call in runner.calls] == [2, 5]
    assert all(call["batch_size"] == 3 for call in runner.calls)
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert rows[0] == {
        "checkpoint_step": 2,
        "checkpoint_path": "/checkpoints/checkpoint-2",
        "prompt_index": 0,
        "sample_index": 0,
        "seed": 100,
        "prompt": "alpha",
        "image_path": "checkpoint_2/p0_s0.png",
    }
    assert rows[4]["checkpoint_step"] == 5
    assert rows[4]["seed"] == 100


def test_legacy_evaluation_runner_keywords_remain_supported(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    checkpoint_output = output_dir / "checkpoint_4"
    checkpoint_output.mkdir(parents=True)
    Image.new("RGB", (1, 1), color="black").save(checkpoint_output / "p0_s0.png", "PNG")
    runner = EvaluationRunner("unused-base-model", "float32", device="cpu")

    paths = runner.generate_for_checkpoint(
        checkpoint_path=str(tmp_path / "unused-checkpoint"),
        prompts=["cached prompt"],
        output_dir=str(output_dir),
        epoch=4,
        num_samples=1,
        gen_kwargs={},
        gen_batch_size=1,
    )

    assert paths == ["checkpoint_4/p0_s0.png"]


def test_expected_outputs_marks_truncated_png_as_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    image_dir = output_dir / "checkpoint_4"
    image_dir.mkdir(parents=True)
    (image_dir / "p0_s0.png").write_bytes(b"")

    paths, missing = _expected_outputs(str(output_dir), 4, ["prompt"], 1, 42)

    assert paths == ["checkpoint_4/p0_s0.png"]
    assert missing == [(0, 0, 42)]


def test_standalone_cli_resolves_evaluation_set_and_generation_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    evaluation_set = tmp_path / "test.txt"
    evaluation_set.write_text("one\ntwo\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "base_model": "base-model",
                    "dtype": "bfloat16",
                    "device": "npu",
                    "num_processes": 2,
                },
                "checkpoint": {"dir": None, "path": str(checkpoint), "step": None},
                "evaluation": {
                    "dataset": str(evaluation_set),
                    "prompt_key": "prompt",
                    "max_prompts": 0,
                    "num_samples": 2,
                    "batch_size": 16,
                    "seed": 42,
                },
                "generation": {"num_inference_steps": 8, "max_sequence_length": 256},
                "output": {"dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )
    calls: Dict[str, Any] = {}

    class _FakeCLIRunner:
        """Record runner construction and cleanup."""

        def __init__(
            self,
            base_model: str,
            dtype_str: str,
            num_processes: int,
            device: str,
        ) -> None:
            calls["runner_init"] = (base_model, dtype_str, num_processes, device)

        def close(self) -> None:
            """Record explicit CLI cleanup."""
            calls["closed"] = True

    def _fake_run_evaluation_set(**kwargs: Any) -> Dict[int, List[str]]:
        calls["run_kwargs"] = kwargs
        return {12: []}

    monkeypatch.setattr(cli, "ParallelEvaluationRunner", _FakeCLIRunner)
    monkeypatch.setattr(cli, "run_evaluation_set", _fake_run_evaluation_set)

    exit_code = cli.main(["-c", str(config_path)])

    assert exit_code == 0
    assert calls["runner_init"] == ("base-model", "bfloat16", 2, "npu")
    assert calls["closed"] is True
    run_kwargs = calls["run_kwargs"]
    assert run_kwargs["checkpoints"] == [(12, str(checkpoint))]
    assert run_kwargs["prompts"] == ["one", "two"]
    assert run_kwargs["num_samples"] == 2
    assert run_kwargs["generation_kwargs"]["num_inference_steps"] == 8
    assert run_kwargs["generation_kwargs"]["max_sequence_length"] == 256


def test_default_yaml_uses_the_public_config_schema() -> None:
    config_path = Path(__file__).parents[2] / "tools/model_inference/default.yaml"

    config = load_inference_config(str(config_path))

    assert config.base_model == "stabilityai/stable-diffusion-3.5-medium"
    assert config.num_processes == 1
    assert config.generation_kwargs["num_inference_steps"] == 50


def test_standalone_parser_only_accepts_the_config_path() -> None:
    option_destinations = {
        action.dest
        for action in cli.build_parser()._actions
        if action.option_strings and action.dest != "help"
    }

    assert option_destinations == {"config"}


def test_inference_yaml_rejects_unknown_structural_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"base_model": "base", "num_gpus": 2},
                "checkpoint": {"dir": "checkpoints", "path": None, "step": None},
                "evaluation": {"dataset": "test.txt"},
                "generation": {},
                "output": {"dir": "output"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown fields in 'model'.*num_gpus"):
        load_inference_config(str(config_path))


def test_npu_is_auto_detected_for_single_and_parallel_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    assert resolve_device() == "npu"
    single_runner = EvaluationRunner("unused", "bfloat16")
    parallel_runner = ParallelEvaluationRunner("unused", "bfloat16", num_processes=2)

    assert single_runner.device == "npu"
    assert [runner.device for runner in parallel_runner._runners] == ["npu:0", "npu:1"]
    with pytest.raises(ValueError, match=r"num_processes\(3\).*npu devices\(2\)"):
        ParallelEvaluationRunner("unused", "bfloat16", num_processes=3)
