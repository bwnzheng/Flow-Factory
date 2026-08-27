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

"""Tests for the standalone registry-backed reward evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from flow_factory.rewards.unireward import _single_device_map
from flow_factory.rewards.vision_reward import (
    VisionRewardModel,
    _load_image_score_head,
    _resolve_model_path,
)
from tools.reward_covariance_eval_analysis.analyze import PromptRecord
from tools.reward_evaluation import scoring as scoring_module
from tools.reward_evaluation.evaluate import (
    EvaluationConfig,
    EvaluationSuiteConfig,
    ModelConfig,
    RunConfig,
    SourceConfig,
    _resolve_run_checkpoints,
    _write_artifacts,
    load_config,
)
from tools.reward_evaluation.scoring import (
    _AcceleratorView,
    _groups_for_rows,
    _partition_by_prompt,
)


def test_default_config_contains_ascend_reward_suite() -> None:
    root = Path(__file__).parents[2]
    config = load_config(root / "tools/reward_evaluation/default.yaml")
    assert [(source.name, source.dataset_dir, source.split) for source in config.sources] == [
        ("pickscore", "dataset/pickscore", "test"),
        ("geneval", "dataset/geneval", "test"),
        ("ocr", "dataset/ocr", "test"),
    ]
    assert [reward["reward_model"] for reward in config.sources[0].rewards] == [
        "pickscore",
        "hpsv2",
        "clip",
        "unireward",
        "vision_reward",
    ]
    assert [reward["reward_model"] for reward in config.sources[1].rewards] == [
        "pickscore",
        "hpsv2",
        "clip",
        "unireward",
        "vision_reward",
        "geneval_ascend",
    ]
    assert [reward["reward_model"] for reward in config.sources[2].rewards] == [
        "pickscore",
        "hpsv2",
        "clip",
        "unireward",
        "vision_reward",
        "ocr",
    ]


def test_reward_options_are_preserved_for_external_models(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        """
model: {base_model: model, dtype: bfloat16, device: cpu, num_processes: 1}
evaluation: {num_samples_per_prompt: 2, generation_batch_size: 1, reward_batch_size: 1, seed: 1}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - name: uni
        reward_model: unireward
        model_path: /models/uni
        batch_size: 1
      - name: vision
        reward_model: vision_reward
        repo_path: /models/vision
runs:
  - {name: run, checkpoint: saves/run/checkpoint-1}
output: {dir: output}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.sources[0].rewards[0]["model_path"] == "/models/uni"
    assert config.sources[0].rewards[1]["repo_path"] == "/models/vision"


def test_checkpoint_dir_run_resolves_all_checkpoints(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "checkpoint-20").mkdir(parents=True)
    (checkpoint_dir / "checkpoint-5").mkdir()
    (checkpoint_dir / "not-a-checkpoint").mkdir()
    run = RunConfig("run", "Run", None, str(checkpoint_dir))
    assert _resolve_run_checkpoints(run) == [
        (5, str(checkpoint_dir / "checkpoint-5")),
        (20, str(checkpoint_dir / "checkpoint-20")),
    ]


def test_groupwise_helpers_keep_complete_prompt_groups() -> None:
    rows = [
        {"prompt_index": prompt, "sample_index": sample}
        for prompt in range(4)
        for sample in range(3)
    ]
    chunks = _partition_by_prompt(rows, num_processes=2)
    owners = {}
    for worker, chunk in enumerate(chunks):
        for row in chunk:
            owners.setdefault(row["prompt_index"], set()).add(worker)
    assert all(len(worker_ids) == 1 for worker_ids in owners.values())
    groups = _groups_for_rows(rows[:6], rows)
    assert [[row["prompt_index"] for row in group] for group in groups] == [[0] * 3, [1] * 3]


def test_offline_accelerator_view_provides_noop_barrier() -> None:
    view = _AcceleratorView(device=torch.device("cpu"), local_process_index=0)
    assert view.wait_for_everyone() is None


def test_vision_reward_uses_extracted_checkpoint_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "visionreward"
    checkpoint.mkdir()
    (checkpoint / "model_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "latest").write_text("1\n", encoding="utf-8")
    (checkpoint / "1").mkdir()
    (checkpoint / "1" / "mp_rank_00_model_states.pt").write_bytes(b"placeholder")

    assert _resolve_model_path({"model_path": str(checkpoint)}) == checkpoint.resolve()


def test_vision_reward_rejects_huggingface_id_before_worker_start() -> None:
    with pytest.raises(FileNotFoundError, match="Hugging Face repository ID"):
        _resolve_model_path({"model_path": "THUDM/VisionReward-Image-bf16"})


def test_vision_reward_model_environment_overrides_template_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "visionreward"
    checkpoint.mkdir()
    (checkpoint / "model_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "latest").write_text("1\n", encoding="utf-8")
    (checkpoint / "1").mkdir()
    (checkpoint / "1" / "mp_rank_00_model_states.pt").write_bytes(b"placeholder")
    monkeypatch.setenv("VISIONREWARD_MODEL", str(checkpoint))

    assert (
        _resolve_model_path({"model_path": "THUDM/VisionReward-Image-bf16"}) == checkpoint.resolve()
    )


def test_vision_reward_selected_head_matches_upstream_files() -> None:
    root = Path(__file__).parents[2] / "VisionReward" / "VisionReward_Image"
    questions, coefficients, intercept = _load_image_score_head(
        root / "VisionReward_image_qa_select.txt", root / "weight_select.json"
    )
    assert len(questions) == 30
    assert coefficients.shape == (28,)
    assert intercept == 0.0


def test_vision_reward_score_uses_official_feature_order() -> None:
    class Alignment:
        def __call__(self, *, images, texts):
            assert len(images) == len(texts) == 1
            return torch.tensor([[0.25]])

    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return "yes<|end_of_text|>", [], None

    model = object.__new__(VisionRewardModel)
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model._questions = [f"question-{index}" for index in range(30)]
    model._score_coefficients = torch.ones(28).numpy()
    model._score_intercept = 0.0
    model._vqa_scorer = Alignment()
    model._chat_fn = chat
    model._model = object()
    model._text_processor = type("TextProcessor", (), {"invalid_slices": []})()
    model._image_processor = object()
    model._max_length = 32
    model._top_p = 0.4
    model._top_k = 1
    model._temperature = 0.8
    model._chat_args = type("ChatArgs", (), {})()

    result = model._score_single("a prompt", Image.new("RGB", (2, 2)))

    assert result == {"overall": 27.25, "alignment": 0.25}
    assert len(calls) == 30
    assert all("img_processor" in call and "image_processor" not in call for call in calls)


def test_unireward_device_map_pins_model_to_worker_device() -> None:
    assert _single_device_map(torch.device("cuda:3")) == {"": 3}
    assert _single_device_map(torch.device("cpu")) == {"": "cpu"}


def test_reward_cache_keeps_previous_checkpoint_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_score_chunk(
        reward_config, rows, all_rows, image_root, prompt_records, device, dtype, batch_size
    ):
        return {scoring_module._sample_key(row): float(row["checkpoint_step"]) for row in rows}

    monkeypatch.setattr(scoring_module, "_score_chunk", fake_score_chunk)
    monkeypatch.setattr(scoring_module, "get_reward_model_class", lambda _: object)
    rows_step_1 = [
        {
            "checkpoint_step": 1,
            "prompt_index": 0,
            "sample_index": 0,
            "prompt": "prompt",
            "image_path": "checkpoint_1/p0_s0.png",
        }
    ]
    rows_step_2 = [
        {
            "checkpoint_step": 2,
            "prompt_index": 0,
            "sample_index": 0,
            "prompt": "prompt",
            "image_path": "checkpoint_2/p0_s0.png",
        }
    ]
    cache_path = tmp_path / "reward.jsonl"
    first = scoring_module.score_reward(
        {"name": "fake", "reward_model": "fake"},
        rows_step_1,
        tmp_path,
        [],
        cache_path,
        "cpu",
        "float32",
        1,
        1,
    )
    second = scoring_module.score_reward(
        {"name": "fake", "reward_model": "fake"},
        rows_step_2,
        tmp_path,
        [],
        cache_path,
        "cpu",
        "float32",
        1,
        1,
    )
    assert first == {"c1_p0_s0": 1.0}
    assert second == {"c2_p0_s0": 2.0}
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2
    output = capsys.readouterr().out
    assert "[Reward] start name=fake" in output
    assert "[Reward] complete name=fake" in output


def test_reward_validator_runs_before_worker_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class ValidatedReward:
        @classmethod
        def validate_config(cls, config):
            calls.append((str(config.device), config.batch_size))

    def fake_score_chunk(
        reward_config, rows, all_rows, image_root, prompt_records, device, dtype, batch_size
    ):
        return {scoring_module._sample_key(row): 1.0 for row in rows}

    monkeypatch.setattr(scoring_module, "get_reward_model_class", lambda _: ValidatedReward)
    monkeypatch.setattr(scoring_module, "_score_chunk", fake_score_chunk)
    rows = [
        {
            "prompt_index": 0,
            "sample_index": 0,
            "prompt": "prompt",
            "image_path": "p0_s0.png",
        }
    ]

    result = scoring_module.score_reward(
        {"name": "validated", "reward_model": "validated"},
        rows,
        tmp_path,
        [],
        tmp_path / "reward.jsonl",
        "cpu",
        "float32",
        1,
        3,
    )

    assert result == {"p0_s0": 1.0}
    assert calls == [("cpu", 3)]


def test_write_artifacts_reports_all_reward_statistics(tmp_path: Path) -> None:
    config = EvaluationSuiteConfig(
        model=ModelConfig("model", "bfloat16", "cpu", 1),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=[],
        output_dir=str(tmp_path),
    )
    run = RunConfig("run", "Run", "checkpoint-7", None)
    source = SourceConfig(
        name="source",
        prompts_file="prompts.txt",
        dataset_dir=None,
        split="test",
        prompt_key="prompt",
        max_prompts=0,
        rewards=[
            {"name": "uni", "reward_model": "unireward"},
            {"name": "vision", "reward_model": "vision_reward"},
        ],
    )
    prompts = [PromptRecord("a prompt", "{}")]
    manifest = [
        {
            "prompt_index": 0,
            "sample_index": index,
            "seed": 42 + index,
            "prompt": "a prompt",
            "image_path": f"checkpoint_7/p0_s{index}.png",
        }
        for index in range(2)
    ]
    summary = _write_artifacts(
        config,
        run,
        source,
        7,
        "checkpoint-7",
        prompts,
        manifest,
        {"uni": {"p0_s0": 1.0, "p0_s1": 3.0}, "vision": {"p0_s0": 2.0, "p0_s1": 4.0}},
        tmp_path,
    )
    assert summary["n_samples"] == 2
    assert summary["rewards"]["uni"]["mean"] == 2.0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert rows[0]["rewards"] == {"uni": 1.0, "vision": 2.0}


def test_multi_process_config_rejects_indexed_device(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        """
model: {base_model: model, dtype: bfloat16, device: 'npu:2', num_processes: 2}
evaluation: {num_samples_per_prompt: 2, generation_batch_size: 1, reward_batch_size: 1, seed: 1}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards: [{name: uni, reward_model: unireward}]
runs: [{name: run, checkpoint: saves/run/checkpoint-1}]
output: {dir: output}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="accelerator type"):
        load_config(path)
