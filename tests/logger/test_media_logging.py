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

import json
from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import LogArguments
from flow_factory.logger import LogImage, prepare_sample_for_media
from flow_factory.logger.abc import LocalFileLogger, Logger
from flow_factory.samples import T2ISample
from flow_factory.trainers.abc import BaseTrainer


class _RecordingBackendLogger(Logger):
    def _init_platform(self):
        self.platform = None
        self.records = []

    def _convert_to_platform(self, value, height=None, width=None):
        return value

    def _log_impl(self, data, step):
        self.records.append((step, data))


def _config(tmp_path, *, save_media_locally, process_index=0, num_processes=1):
    config = SimpleNamespace(
        process_index=process_index,
        num_processes=num_processes,
        log_args=SimpleNamespace(
            save_dir=str(tmp_path),
            run_name="media-run",
            save_media_locally=save_media_locally,
            media_save_freq=20,
            image_save_format="jpg",
            image_save_quality=87,
            log_metrics_jsonl=False,
        ),
    )
    config.to_dict = lambda: {
        "log": {
            "run_name": "media-run",
            "media_save_freq": 20,
        }
    }
    return config


def _media_sample():
    sample = T2ISample(
        image=torch.zeros(3, 8, 8),
        timesteps=torch.tensor([1.0, 0.5, 0.0]),
        log_probs=torch.tensor([-0.2, -0.1]),
        prompt="a red cube",
        source="unit-test",
        source_id=3,
        applicable_rewards={"quality", "alignment"},
        _unique_id=42,
        extra_kwargs={
            "rewards": {"quality": 0.8, "alignment": 0.6},
            "advantage": 1.25,
            "sample_weight": 0.75,
            "masked_reward": float("nan"),
        },
    )
    return prepare_sample_for_media(
        sample,
        {
            "run": {
                "run_name": "media-run",
                "step": 20,
                "epoch": 2,
                "rank": 1,
                "world_size": 2,
            },
            "media": {"category": "training", "context": "final"},
            "sampling": {"num_inference_steps": 28, "guidance_scale": 4.5},
            "configuration": {
                "log": {
                    "run_name": "media-run",
                    "media_save_freq": 20,
                }
            },
        },
    )


class _MediaTrainerHarness:
    should_log_media = BaseTrainer.should_log_media
    _build_media_context = BaseTrainer._build_media_context
    log_media_samples = BaseTrainer.log_media_samples

    def __init__(self):
        self.step = 20
        self.epoch = 2
        self.log_args = SimpleNamespace(
            run_name="media-run",
            media_save_freq=20,
            max_log_samples=None,
            save_media_locally=True,
            logging_backend=None,
        )
        self.accelerator = SimpleNamespace(
            process_index=0,
            num_processes=2,
            is_main_process=True,
        )
        self.config = SimpleNamespace(to_dict=lambda: {"log": {"run_name": "media-run"}})
        self.records = []

    def log_data(self, data, step):
        self.records.append((step, data))


def test_media_samples_are_gathered_and_reindexed_without_rank_keys(monkeypatch):
    trainer = _MediaTrainerHarness()
    local_sample = T2ISample(
        image=torch.zeros(3, 8, 8),
        prompt="local sample",
        _unique_id=42,
    )
    remote_sample = T2ISample(
        image=torch.zeros(3, 8, 8),
        prompt="remote sample",
        _unique_id=42,
    )
    remote_media_sample = prepare_sample_for_media(
        remote_sample,
        {
            "run": {"rank": 1, "step": 20, "epoch": 2},
            "media": {
                "category": "training",
                "context": "final",
                "local_index": 0,
                "group_index": 0,
            },
            "configuration": {},
        },
    )

    monkeypatch.setattr(
        "flow_factory.trainers.abc.gather_object",
        lambda local_media: local_media + [remote_media_sample],
    )

    trainer.log_media_samples(
        [local_sample],
        category="training",
        context_name="final",
    )

    assert len(trainer.records) == 1
    step, payload = trainer.records[0]
    assert step == 20
    assert list(payload) == [
        "media/training/final/group_42/sample_000000",
        "media/training/final/group_42/sample_000001",
    ]
    contexts = [sample.extra_kwargs["_media_metadata"]["context"] for sample in payload.values()]
    assert [context["run"]["rank"] for context in contexts] == [0, 1]
    assert [context["media"]["group_index"] for context in contexts] == [0, 1]
    assert [context["media"]["global_index"] for context in contexts] == [0, 1]


def test_log_arguments_validate_media_settings(tmp_path):
    args = LogArguments(save_dir=str(tmp_path))
    assert args.media_save_freq == 20
    assert args.max_log_samples is None

    with pytest.raises(ValueError, match="media_save_freq"):
        LogArguments(save_dir=str(tmp_path), media_save_freq=-1)
    with pytest.raises(ValueError, match="max_log_samples"):
        LogArguments(save_dir=str(tmp_path), max_log_samples=0)
    with pytest.raises(ValueError, match="image_save_format"):
        LogArguments(save_dir=str(tmp_path), image_save_format="webp")


def test_local_media_interval_path_and_json_sidecar(tmp_path):
    logger = LocalFileLogger(
        _config(
            tmp_path,
            save_media_locally=True,
            process_index=1,
            num_processes=2,
        )
    )
    key = "media/training/final/group_42/sample_000000"
    sample = _media_sample()

    logger.log_data({key: sample}, step=19)
    image_path = (
        tmp_path
        / "media-run"
        / "logs"
        / "images"
        / "training"
        / "step_000020"
        / "group_42"
        / "final"
        / "sample_000000.jpg"
    )
    assert not image_path.exists()

    logger.log_data({key: sample}, step=20)
    sidecar_path = image_path.with_suffix(".json")
    assert image_path.exists()
    assert sidecar_path.exists()
    assert (image_path.parents[5] / "media.jsonl").exists()

    metadata = json.loads(sidecar_path.read_text())
    assert metadata["schema_version"] == 1
    assert metadata["sample"]["unique_id"] == 42
    assert metadata["sample"]["extra_kwargs"]["advantage"] == 1.25
    assert metadata["sample"]["extra_kwargs"]["sample_weight"] == 0.75
    assert metadata["sample"]["extra_kwargs"]["masked_reward"] == {
        "type": "nonfinite_float",
        "value": "nan",
    }
    assert metadata["sample"]["fields"]["timesteps"] == [1.0, 0.5, 0.0]
    assert metadata["sample"]["fields"]["log_probs"] == pytest.approx([-0.2, -0.1])
    assert metadata["context"]["sampling"] == {
        "num_inference_steps": 28,
        "guidance_scale": 4.5,
    }
    assert metadata["context"]["run"] == {"epoch": 2, "rank": 1}
    assert "configuration" not in metadata["context"]
    assert metadata["reward"] == {"quality": 0.8, "alignment": 0.6}
    manifest_records = [
        json.loads(line)
        for line in (image_path.parents[5] / "media.jsonl").read_text().splitlines()
    ]
    assert manifest_records[0] == {
        "record_type": "run_context",
        "schema_version": 2,
        "run": {"run_name": "media-run", "world_size": 2},
        "configuration": {
            "log": {
                "run_name": "media-run",
                "media_save_freq": 20,
            }
        },
    }
    manifest = manifest_records[1]
    assert manifest == {
        "step": 20,
        "key": key,
        "path": "images/training/step_000020/group_42/final/sample_000000.jpg",
        "prompt": "a red cube",
        "reward": {"quality": 0.8, "alignment": 0.6},
        "metadata_path": "images/training/step_000020/group_42/final/sample_000000.json",
    }


def test_media_manifest_writes_run_context_once(tmp_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))

    logger.log_data(
        {"media/training/initial/group_42/sample_000000": _media_sample()},
        step=20,
    )
    logger.log_data(
        {"media/training/final/group_42/sample_000000": _media_sample()},
        step=40,
    )
    resumed_logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))
    resumed_logger.log_data(
        {"media/evaluation/benchmark/group_42/sample_000000": _media_sample()},
        step=60,
    )

    manifest_path = tmp_path / "media-run" / "logs" / "media.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert [record.get("record_type") for record in records].count("run_context") == 1
    assert len(records) == 4


@pytest.mark.parametrize(
    ("key", "relative_path"),
    [
        (
            "media/evaluation/benchmark/group_42/sample_000000",
            "images/evaluation/step_000020/group_42/benchmark/sample_000000.jpg",
        ),
        (
            "media/ga/gen2/group_42/candidate_000004",
            "images/training/step_000020/group_42/gen2/candidate_000004.jpg",
        ),
    ],
)
def test_local_media_uses_step_group_context_layout(tmp_path, key, relative_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))

    logger.log_data({key: _media_sample()}, step=20)

    assert (tmp_path / "media-run" / "logs" / relative_path).exists()


def test_media_sample_filters_inapplicable_rewards_without_mutating_training_sample(tmp_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))
    sample = T2ISample(
        image=torch.zeros(3, 8, 8),
        prompt="a red cube",
        source="quality-dataset",
        applicable_rewards={"quality"},
        _unique_id=42,
        extra_kwargs={
            "rewards": {
                "quality": torch.tensor(0.8),
                "alignment": torch.tensor(float("nan")),
            }
        },
    )

    media_sample = prepare_sample_for_media(sample)
    key = "media/training/final/group_42/sample_000000"
    logger.log_data({key: media_sample}, step=20)

    assert set(sample.extra_kwargs["rewards"]) == {"quality", "alignment"}
    assert set(media_sample.extra_kwargs["rewards"]) == {"quality"}
    sidecar_path = (
        tmp_path
        / "media-run"
        / "logs"
        / "images"
        / "training"
        / "step_000020"
        / "group_42"
        / "final"
        / "sample_000000.json"
    )
    metadata = json.loads(sidecar_path.read_text())
    assert metadata["sample"]["applicable_rewards"] == ["quality"]
    assert metadata["reward"] == {"quality": pytest.approx(0.8)}


def test_media_sample_keeps_all_rewards_without_applicability_bookkeeping():
    sample = T2ISample(
        image=torch.zeros(3, 8, 8),
        prompt="legacy sample",
        _unique_id=42,
        extra_kwargs={"rewards": {"quality": 0.8, "alignment": 0.6}},
    )

    media_sample = prepare_sample_for_media(sample)

    assert media_sample.extra_kwargs["rewards"] == {
        "quality": 0.8,
        "alignment": 0.6,
    }


def test_local_media_sidecar_serializes_nonfinite_tensor_rewards(tmp_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))
    key = "media/training/final/group_42/sample_000000"
    sample = _media_sample()
    sample.extra_kwargs["rewards"] = {
        "quality": torch.tensor(float("nan")),
        "alignment": torch.tensor(float("inf")),
    }

    logger.log_data({key: sample}, step=20)

    sidecar_path = (
        tmp_path
        / "media-run"
        / "logs"
        / "images"
        / "training"
        / "step_000020"
        / "group_42"
        / "final"
        / "sample_000000.json"
    )
    metadata = json.loads(sidecar_path.read_text())
    assert metadata["reward"] == {
        "quality": {"type": "nonfinite_float", "value": "nan"},
        "alignment": {"type": "nonfinite_float", "value": "inf"},
    }


def test_local_media_sidecar_normalizes_direct_logimage_metadata(tmp_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))
    key = "media/training/final/group_42/direct"
    image = LogImage(
        torch.zeros(3, 8, 8),
        metadata={"diagnostic": float("-inf")},
    )

    logger.log_data({key: image}, step=20)

    sidecar_path = (
        tmp_path
        / "media-run"
        / "logs"
        / "images"
        / "training"
        / "step_000020"
        / "group_42"
        / "final"
        / "direct.json"
    )
    metadata = json.loads(sidecar_path.read_text())
    assert metadata["diagnostic"] == {
        "type": "nonfinite_float",
        "value": "-inf",
    }


def test_backend_media_uses_the_same_interval(tmp_path):
    logger = _RecordingBackendLogger(_config(tmp_path, save_media_locally=False))
    key = "media/evaluation/benchmark/group_42/sample_000000"
    sample = _media_sample()

    logger.log_data({key: sample}, step=19)
    assert logger.records == []

    logger.log_data({key: sample}, step=20)
    assert len(logger.records) == 1
    step, payload = logger.records[0]
    assert step == 20
    assert list(payload) == [key]
    assert isinstance(payload[key], LogImage)


def test_backend_media_normalizes_nonfinite_tensor_rewards(tmp_path):
    logger = _RecordingBackendLogger(_config(tmp_path, save_media_locally=False))
    key = "media/training/final/group_42/sample_000000"
    sample = _media_sample()
    sample.extra_kwargs["rewards"] = {
        "quality": torch.tensor(float("nan")),
        "alignment": torch.tensor(float("-inf")),
    }

    logger.log_data({key: sample}, step=20)

    media = logger.records[0][1][key]
    assert media.metadata["reward"] == {
        "quality": {"type": "nonfinite_float", "value": "nan"},
        "alignment": {"type": "nonfinite_float", "value": "-inf"},
    }
    assert media.metadata["context"]["run"]["run_name"] == "media-run"
    assert media.metadata["context"]["run"]["world_size"] == 2
    assert media.metadata["context"]["configuration"]["log"]["run_name"] == "media-run"
