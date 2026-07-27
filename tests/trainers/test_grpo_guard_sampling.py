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

from flow_factory.trainers import grpo as grpo_module
from flow_factory.trainers.grpo import GRPOGuardTrainer


def test_grpo_guard_sample_uses_standard_generation_pipeline(monkeypatch):
    trainer = GRPOGuardTrainer.__new__(GRPOGuardTrainer)
    trainer.adapter = SimpleNamespace(scheduler=SimpleNamespace(train_timesteps=["train-timestep"]))
    trainer.training_args = SimpleNamespace(num_inference_steps=7)
    trainer.reward_buffer = object()

    trajectory_indices = [1, 3, 5]
    monkeypatch.setattr(
        grpo_module,
        "compute_trajectory_indices",
        lambda **kwargs: trajectory_indices,
    )

    captured = {}
    expected_samples = [object()]

    def generate_samples(**kwargs):
        captured.update(kwargs)
        return expected_samples

    trainer.generate_samples = generate_samples

    assert trainer.sample() is expected_samples
    assert captured == {
        "reward_buffer": trainer.reward_buffer,
        "compute_log_prob": True,
        "trajectory_indices": trajectory_indices,
        "extra_call_back_kwargs": ["next_latents_mean"],
    }
