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
from flow_factory.samples import I2ISample, T2AVSample, T2ISample
from flow_factory.trainers.abc import BaseTrainer


class _RecordingBackendLogger(Logger):
    def _init_platform(self):
        self.platform = None
        self.records = []

    def _convert_to_platform(self, value, height=None, width=None):
        return value

    def _log_impl(self, data, step):
        self.records.append((step, data))


class _RecordingLocalMediaLogger:
    def __init__(self):
        self.rank = None
        self.saved = []
        self.manifests = []
        self.backend_manifests = []

    def set_media_rank(self, rank):
        self.rank = rank

    def save_media_locally(self, data, step):
        self.saved.append((self.rank, data, step))
        return [
            {
                "step": step,
                "key": key,
                "path": f"rank_{self.rank}/{key}",
                "metadata_path": f"rank_{self.rank}/{key}.json",
            }
            for key in data
        ]

    def write_media_manifest(self, entries):
        self.manifests.append(entries)

    def log_media_files(self, entries, step):
        self.backend_manifests.append((entries, step))


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

    def __init__(self, save_media_locally=True):
        self.step = 20
        self.epoch = 2
        self.log_args = SimpleNamespace(
            run_name="media-run",
            media_save_freq=20,
            max_log_samples=None,
            save_media_locally=save_media_locally,
            logging_backend=None if save_media_locally else "wandb",
        )
        self.accelerator = SimpleNamespace(
            process_index=0,
            num_processes=2,
            is_main_process=True,
        )
        self.config = SimpleNamespace(to_dict=lambda: {"log": {"run_name": "media-run"}})
        self.records = []
        self.logger = _RecordingLocalMediaLogger() if save_media_locally else None

    def log_data(self, data, step):
        self.records.append((step, data))


def test_local_media_gathers_only_manifest_records(monkeypatch):
    trainer = _MediaTrainerHarness()
    local_sample = T2ISample(
        image=torch.zeros(3, 8, 8),
        prompt="local sample",
        _unique_id=42,
    )
    gathered_inputs = []
    remote_manifest = {
        "step": 20,
        "key": "media/training/final/group_42/sample_000001",
        "path": "rank_1/images/training/step_000020/group_42/final/sample_000001.jpg",
        "metadata_path": "rank_1/images/training/step_000020/group_42/final/sample_000001.json",
    }

    def record_gather(local_manifest):
        gathered_inputs.append(local_manifest)
        return local_manifest + [remote_manifest]

    monkeypatch.setattr("flow_factory.trainers.abc.gather_object", record_gather)

    trainer.log_media_samples(
        [local_sample],
        category="training",
        context_name="final",
    )

    assert len(gathered_inputs) == 1
    assert len(gathered_inputs[0]) == 1
    assert "sample" not in gathered_inputs[0][0]
    assert len(trainer.logger.saved) == 1
    saved_rank, saved_payload, saved_step = trainer.logger.saved[0]
    assert saved_rank == 0
    assert saved_step == 20
    assert list(saved_payload) == ["media/training/final/group_42/sample_000000"]
    assert len(trainer.logger.manifests) == 1
    assert trainer.logger.manifests[0] == [gathered_inputs[0][0], remote_manifest]


def test_non_main_rank_saves_local_media_without_writing_manifest(monkeypatch):
    trainer = _MediaTrainerHarness()
    trainer.accelerator.process_index = 1
    trainer.accelerator.is_main_process = False
    monkeypatch.setattr(
        "flow_factory.trainers.abc.gather_object",
        lambda local_manifest: local_manifest,
    )

    trainer.log_media_samples(
        [T2ISample(image=torch.zeros(3, 8, 8), prompt="rank one", _unique_id=43)],
        category="training",
        context_name="final",
    )

    assert len(trainer.logger.saved) == 1
    assert trainer.logger.saved[0][0] == 1
    assert trainer.logger.manifests == []
    assert trainer.records == []


def test_local_media_logs_manifest_files_to_backend_on_main(monkeypatch):
    trainer = _MediaTrainerHarness()
    trainer.log_args.logging_backend = "tensorboard"
    monkeypatch.setattr(
        "flow_factory.trainers.abc.gather_object",
        lambda local_manifest: local_manifest,
    )

    trainer.log_media_samples(
        [T2ISample(image=torch.zeros(3, 8, 8), prompt="backend sample", _unique_id=44)],
        category="training",
        context_name="final",
    )

    assert len(trainer.logger.manifests) == 1
    assert trainer.logger.backend_manifests == [(trainer.logger.manifests[0], 20)]


def test_backend_media_skips_replay_metadata_before_gather(monkeypatch):
    trainer = _MediaTrainerHarness(save_media_locally=False)
    sample = T2ISample(
        image=torch.zeros(3, 8, 8),
        prompt="backend sample",
        _unique_id=42,
        extra_kwargs={"rewards": {"quality": 0.8}},
    )
    gathered = []

    def record_gather(media_records):
        gathered.extend(media_records)
        return media_records

    monkeypatch.setattr("flow_factory.trainers.abc.gather_object", record_gather)

    trainer.log_media_samples(
        [sample],
        category="training",
        context_name="final",
    )

    assert len(gathered) == 1
    gathered_sample = gathered[0]["sample"]
    assert gathered[0]["group_id"] == 42
    assert "_media_metadata" not in gathered_sample.extra_kwargs
    assert gathered_sample.extra_kwargs == {"rewards": {"quality": 0.8}}
    assert gathered_sample.prompt == "backend sample"
    assert gathered_sample.source is None
    assert gathered_sample._unique_id is None
    payload = trainer.records[0][1]
    assert list(payload) == ["media/training/final/group_42/sample_000000"]


def test_backend_media_keeps_only_fields_required_by_multimodal_formatters():
    image_sample = I2ISample(
        image=torch.zeros(3, 8, 8),
        condition_images=[torch.ones(3, 8, 8)],
        prompt="edit image",
        source="edit-dataset",
        _unique_id=42,
    )
    audio_video_sample = T2AVSample(
        video=torch.zeros(2, 3, 8, 8),
        audio=torch.zeros(160),
        audio_sample_rate=16000,
        prompt="sound and motion",
        source="video-dataset",
        _unique_id=43,
    )

    image_media = prepare_sample_for_media(image_sample, include_metadata=False)
    audio_video_media = prepare_sample_for_media(
        audio_video_sample,
        include_metadata=False,
    )

    assert image_media.condition_images is not None
    assert image_media.source is None
    assert audio_video_media.video is not None
    assert audio_video_media.audio is not None
    assert audio_video_media.audio_sample_rate == 16000
    assert audio_video_media.source is None


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
        "caption": "quality: 0.80, alignment: 0.60 | a red cube",
        "prompt": "a red cube",
        "reward": {"quality": 0.8, "alignment": 0.6},
        "metadata_path": "images/training/step_000020/group_42/final/sample_000000.json",
    }


def test_local_media_can_save_rank_shard_before_manifest_append(tmp_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))
    logger.set_media_rank(1)
    key = "media/training/final/group_42/sample_000000"

    entries = logger.save_media_locally({key: _media_sample()}, step=20)

    image_path = (
        tmp_path
        / "media-run"
        / "logs"
        / "images"
        / "training"
        / "step_000020"
        / "rank_1"
        / "group_42"
        / "final"
        / "sample_000000.jpg"
    )
    manifest_path = tmp_path / "media-run" / "logs" / "media.jsonl"
    assert image_path.exists()
    assert not manifest_path.exists()
    assert entries[0]["path"] == (
        "images/training/step_000020/rank_1/group_42/final/sample_000000.jpg"
    )

    logger.write_media_manifest(entries)

    assert manifest_path.exists()
    manifest_records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert manifest_records[-1]["metadata_path"] == (
        "images/training/step_000020/rank_1/group_42/final/sample_000000.json"
    )


def test_backend_can_log_rank_local_media_from_manifest(tmp_path):
    config = _config(tmp_path, save_media_locally=True)
    local_logger = LocalFileLogger(config)
    local_logger.set_media_rank(1)
    key = "media/training/final/group_42/sample_000000"
    entries = local_logger.save_media_locally({key: _media_sample()}, step=20)

    backend_logger = _RecordingBackendLogger(config)
    backend_logger.log_media_files(entries, step=20)

    assert len(backend_logger.records) == 1
    step, payload = backend_logger.records[0]
    assert step == 20
    assert list(payload) == [key]
    assert isinstance(payload[key], LogImage)
    assert payload[key].value == str(tmp_path / "media-run" / "logs" / entries[0]["path"])


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
            "images/evaluation/step_000020/benchmark/group_42_sample_000000.jpg",
        ),
        (
            "media/ga/gen2/group_42/candidate_000004",
            "images/training/step_000020/group_42/gen2/candidate_000004.jpg",
        ),
    ],
)
def test_local_media_uses_category_specific_layout(tmp_path, key, relative_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))

    logger.log_data({key: _media_sample()}, step=20)

    assert (tmp_path / "media-run" / "logs" / relative_path).exists()


def test_evaluation_media_flattens_group_and_table_items_within_each_step(tmp_path):
    logger = LocalFileLogger(_config(tmp_path, save_media_locally=True))
    key = "media/evaluation/benchmark/group_42/sample_000000/generated_video/0"

    logger.log_data({key: LogImage(torch.zeros(3, 8, 8))}, step=20)
    logger.log_data({key: LogImage(torch.zeros(3, 8, 8))}, step=40)

    eval_dir = tmp_path / "media-run" / "logs" / "images" / "evaluation"
    assert sorted(path.name for path in eval_dir.iterdir()) == [
        "step_000020",
        "step_000040",
    ]
    expected_files = [
        "group_42_sample_000000_generated_video_0.jpg",
        "group_42_sample_000000_generated_video_0.json",
    ]
    step_20_dir = eval_dir / "step_000020" / "benchmark"
    step_40_dir = eval_dir / "step_000040" / "benchmark"
    assert sorted(path.name for path in step_20_dir.iterdir()) == expected_files
    assert sorted(path.name for path in step_40_dir.iterdir()) == expected_files


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
    sample = prepare_sample_for_media(_media_sample(), include_metadata=False)
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
    assert "context" not in media.metadata
