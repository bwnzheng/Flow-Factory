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

import pytest
import torch

from flow_factory.trainers.nft import _compute_nft_policy_objective


def test_nft_src_weight_multiplies_complete_per_sample_objective():
    positive_loss = torch.tensor([1.0, 4.0])
    negative_loss = torch.tensor([3.0, 2.0])
    advantage = torch.tensor([2.0, -1.0])
    sample_weight = torch.tensor([1.5, 0.5])

    policy_loss, per_sample_loss, probability, clipping_ratio = _compute_nft_policy_objective(
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        advantage=advantage,
        adv_clip_range=(-5.0, 5.0),
        nft_beta=1.0,
        sample_weight=sample_weight,
    )

    expected_probability = torch.tensor([0.7, 0.4])
    expected_per_sample = (
        expected_probability * positive_loss + (1.0 - expected_probability) * negative_loss
    )
    expected_loss = (sample_weight * expected_per_sample * 5.0).mean()

    torch.testing.assert_close(probability, expected_probability)
    torch.testing.assert_close(per_sample_loss, expected_per_sample)
    torch.testing.assert_close(policy_loss, expected_loss)
    assert clipping_ratio.item() == pytest.approx(0.0)


def test_nft_objective_without_src_matches_baseline_formula():
    positive_loss = torch.tensor([1.0, 4.0])
    negative_loss = torch.tensor([3.0, 2.0])
    advantage = torch.tensor([5.0, -5.0])

    policy_loss, per_sample_loss, probability, clipping_ratio = _compute_nft_policy_objective(
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        advantage=advantage,
        adv_clip_range=(-5.0, 5.0),
        nft_beta=1.0,
    )

    torch.testing.assert_close(probability, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(per_sample_loss, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(policy_loss, torch.tensor(7.5))
    assert clipping_ratio.item() == pytest.approx(1.0)
