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
from transformers.modeling_outputs import BaseModelOutputWithPooling

from flow_factory.rewards.aesthetic_score import AestheticScoreRewardModel


class _FakeProcessor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}


class _FakeCLIPModel:
    def __init__(self, output):
        self.output = output

    def get_image_features(self, *, pixel_values):
        assert pixel_values.shape[0] == 2
        return self.output


class _RecordingMLP:
    def __init__(self):
        self.inputs = []

    def __call__(self, features):
        self.inputs.append(features.clone())
        return features.sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("wrap_in_model_output", [False, True])
def test_aesthetic_score_l2_normalizes_clip_features(wrap_in_model_output):
    features = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    output = (
        BaseModelOutputWithPooling(pooler_output=features)
        if wrap_in_model_output
        else features
    )
    model = AestheticScoreRewardModel.__new__(AestheticScoreRewardModel)
    model.config = SimpleNamespace(batch_size=2)
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.processor = _FakeProcessor()
    model.clip_model = _FakeCLIPModel(output)
    model.mlp = _RecordingMLP()

    result = model(
        prompt=["unused", "unused"],
        image=[Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
    )

    normalized_features = model.mlp.inputs[0]
    assert torch.allclose(normalized_features.norm(p=2, dim=-1), torch.ones(2))
    assert torch.allclose(normalized_features, torch.tensor([[0.6, 0.8], [0.0, 1.0]]))
    assert torch.allclose(result.rewards, torch.tensor([1.4, 1.0]))
