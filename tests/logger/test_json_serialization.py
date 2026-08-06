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
import pickle
from types import SimpleNamespace

import numpy as np

from flow_factory.logger.abc import LocalFileLogger, Logger


class _TestMainLogger(Logger):
    def _init_platform(self):
        self.platform = None

    def _convert_to_platform(self, value, height=None, width=None):
        return value

    def _log_impl(self, data, step):
        pass


def test_metrics_jsonl_keeps_scalar_summaries_and_omits_nested_raw_values(tmp_path):
    logger = LocalFileLogger(
        SimpleNamespace(
            log_args=SimpleNamespace(
                save_dir=str(tmp_path),
                run_name="json-safe",
                save_media_locally=False,
                log_metrics_jsonl=True,
            )
        )
    )
    diagnostics = {
        "np_bool_metric": np.bool_(True),
        "np_float_metric": np.float32(0.25),
        "nested/diagnostics": [{"scores": [0.1, 0.2]}],
        "raw/vector": np.array([1.0, 2.0], dtype=np.float32),
    }

    logger._write_metrics_jsonl(diagnostics, step=7)

    metrics_path = tmp_path / "json-safe" / "logs" / "metrics.jsonl"
    record = json.loads(metrics_path.read_text().strip())
    assert record == {
        "step": 7,
        "np_bool_metric": True,
        "np_float_metric": 0.25,
        "raw/vector": 1.5,
    }


def test_ga_raw_selections_are_saved_to_pkl_and_omitted_from_metrics_jsonl(tmp_path):
    logger = _TestMainLogger(
        SimpleNamespace(
            log_args=SimpleNamespace(
                save_dir=str(tmp_path),
                run_name="ga-raw",
                save_media_locally=False,
                log_metrics_jsonl=True,
            )
        )
    )
    event = {
        "rank": 0,
        "gid": 17,
        "gen": 1,
        "candidate_rewards": {"quality": np.array([0.1, 0.9], dtype=np.float32)},
        "selected_ids": np.array([1], dtype=np.int64),
    }

    logger.log_data(
        {
            "ga/n_groups": 2,
            "ga/gen0/src/frozen_score": 0.25,
            "ga/raw_selections": [event],
        },
        step=7,
    )

    metrics_path = tmp_path / "ga-raw" / "logs" / "metrics.jsonl"
    metrics = json.loads(metrics_path.read_text().strip())
    assert metrics == {
        "step": 7,
        "ga/n_groups": 2,
        "ga/gen0/src/frozen_score": 0.25,
    }

    ga_path = tmp_path / "ga-raw" / "logs" / "ga" / "train_step_000007.pkl"
    with ga_path.open("rb") as handle:
        raw = pickle.load(handle)
    assert raw["schema_version"] == 1
    assert raw["step"] == 7
    assert len(raw["selections"]) == 1
    np.testing.assert_array_equal(raw["selections"][0]["selected_ids"], [1])
    np.testing.assert_allclose(raw["selections"][0]["candidate_rewards"]["quality"], [0.1, 0.9])


def test_train_prompts_and_rewards_are_kept_in_pkl_only(tmp_path):
    logger = _TestMainLogger(
        SimpleNamespace(
            log_args=SimpleNamespace(
                save_dir=str(tmp_path),
                run_name="train-raw",
                save_media_locally=False,
                log_metrics_jsonl=True,
            )
        )
    )
    logger.log_data(
        {
            "train/reward_quality_mean": 0.5,
            "train/prompts": ["first prompt"],
            "train/rewards": [{"prompt": "first prompt", "rewards": {"quality": [0.2, 0.8]}}],
            "train/src_groups": [
                {
                    "group_id": 0,
                    "scores": [0.1, 0.9],
                    "probabilities": [0.25, 0.75],
                }
            ],
            "train/nondom_sizes_in_group": {"values": [2]},
        },
        step=3,
    )

    metrics_path = tmp_path / "train-raw" / "logs" / "metrics.jsonl"
    assert json.loads(metrics_path.read_text().strip()) == {
        "step": 3,
        "train/reward_quality_mean": 0.5,
    }

    rewards_path = tmp_path / "train-raw" / "logs" / "rewards" / "train_step_000003.pkl"
    with rewards_path.open("rb") as handle:
        raw = pickle.load(handle)
    assert raw["prompts"] == ["first prompt"]
    np.testing.assert_allclose(raw["quality"][0], [0.2, 0.8])
    assert raw["src_groups"] == [
        {
            "group_id": 0,
            "scores": [0.1, 0.9],
            "probabilities": [0.25, 0.75],
        }
    ]
    assert raw["nondom_sizes_in_group"] == [2]
