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

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import cycle_reward as cycle_reward_module
from flow_factory.rewards import get_reward_model_class
from flow_factory.rewards.cycle_reward import CycleRewardModel


class _FakeCycleReward:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def score(self, images, prompts):
        self.calls.append((images, prompts))
        return torch.arange(len(prompts), dtype=torch.float32).reshape(-1, 1) + 0.5


def test_registry_resolves_cycle_reward_case_insensitively():
    assert get_reward_model_class("Cycle_Reward") is CycleRewardModel
    assert get_reward_model_class("CycleReward") is CycleRewardModel


def test_cycle_reward_batches_and_uses_upstream_preprocess(monkeypatch):
    fake_model = _FakeCycleReward()

    def fake_factory(**kwargs):
        assert kwargs == {
            "device": torch.device("cpu"),
            "model_type": "CycleReward-T2I",
            "cache_dir": "/tmp/cycle",
        }
        return fake_model, lambda image: torch.full((3, 2, 2), float(image.getpixel((0, 0))[0]))

    monkeypatch.setattr(cycle_reward_module, "_cyclereward_factory", fake_factory)
    config = RewardArguments.from_dict(
        {
            "reward_model": "cycle_reward",
            "device": "cpu",
            "batch_size": 2,
            "model_type": "CycleReward-T2I",
            "cache_dir": "/tmp/cycle",
        }
    )
    model = CycleRewardModel(config, accelerator=None)
    images = [Image.new("RGB", (2, 2), color=(index, 0, 0)) for index in range(3)]

    result = model(["first", "second", "third"], images)

    assert torch.equal(result.rewards, torch.tensor([0.5, 1.5, 0.5]))
    assert [call[1] for call in fake_model.calls] == [["first", "second"], ["third"]]
    assert all(call[0].shape == (len(call[1]), 3, 2, 2) for call in fake_model.calls)


def test_cycle_reward_supports_video_first_frame(monkeypatch):
    fake_model = _FakeCycleReward()
    monkeypatch.setattr(
        cycle_reward_module,
        "_cyclereward_factory",
        lambda **_: (fake_model, lambda image: torch.zeros(3, 2, 2)),
    )
    model = CycleRewardModel(RewardArguments(device="cpu"), accelerator=None)
    videos = [[Image.new("RGB", (2, 2))], [Image.new("RGB", (2, 2))]]

    result = model(["first", "second"], video=videos)

    assert result.rewards.shape == (2,)


def test_cycle_reward_rejects_missing_dependency(monkeypatch):
    monkeypatch.setattr(cycle_reward_module, "_cyclereward_factory", None)

    with pytest.raises(ImportError, match="cyclereward"):
        CycleRewardModel(RewardArguments(device="cpu"), accelerator=None)


def test_cycle_reward_rejects_invalid_model_type(monkeypatch):
    monkeypatch.setattr(
        cycle_reward_module,
        "_cyclereward_factory",
        SimpleNamespace(),
    )
    config = RewardArguments.from_dict(
        {"reward_model": "cycle_reward", "device": "cpu", "model_type": "unknown"}
    )

    with pytest.raises(ValueError, match="Unsupported CycleReward model_type"):
        CycleRewardModel(config, accelerator=None)
