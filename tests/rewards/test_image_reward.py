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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import get_reward_model_class
from flow_factory.rewards import image_reward as image_reward_module
from flow_factory.rewards.image_reward import ImageRewardModel


class _FakeImageReward:
    def __init__(self) -> None:
        self.calls = []

    def eval(self):
        return self

    def score(self, prompt, image):
        self.calls.append((prompt, image))
        return 0.25 if prompt == "first" else -1.0


def test_registry_resolves_imagereward_case_insensitively() -> None:
    assert get_reward_model_class("ImageReward") is ImageRewardModel


def test_image_reward_delegates_to_upstream_scorer(monkeypatch) -> None:
    fake_model = _FakeImageReward()

    def fake_load(name, **kwargs):
        assert name == "local-checkpoint.pt"
        assert kwargs == {"device": torch.device("cpu"), "download_root": "/tmp/cache"}
        return fake_model

    monkeypatch.setattr(
        image_reward_module,
        "_image_reward",
        SimpleNamespace(load=fake_load),
    )
    config = RewardArguments.from_dict(
        {
            "reward_model": "imagereward",
            "device": "cpu",
            "model_path": "local-checkpoint.pt",
            "download_root": "/tmp/cache",
        }
    )
    model = ImageRewardModel(config, accelerator=None)
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]

    result = model(["first", "second"], images)

    assert torch.equal(result.rewards, torch.tensor([0.25, -1.0]))
    assert [prompt for prompt, _ in fake_model.calls] == ["first", "second"]


def test_image_reward_rejects_mismatched_inputs(monkeypatch) -> None:
    fake_model = _FakeImageReward()
    monkeypatch.setattr(
        image_reward_module,
        "_image_reward",
        SimpleNamespace(load=lambda *args, **kwargs: fake_model),
    )
    model = ImageRewardModel(RewardArguments(device="cpu"), accelerator=None)

    with pytest.raises(ValueError, match="prompts but 1 images"):
        model(["first", "second"], [Image.new("RGB", (4, 4))])


def test_image_reward_requires_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr(image_reward_module, "_image_reward", None)

    with pytest.raises(ImportError, match="image-reward"):
        ImageRewardModel(RewardArguments(device="cpu"), accelerator=None)
