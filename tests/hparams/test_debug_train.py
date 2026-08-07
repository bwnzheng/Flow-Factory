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

from flow_factory.hparams import Arguments


def _config(
    *,
    trainer_type: str = "grpo",
    debug_train: bool = False,
    gradient_accumulation_steps="auto",
):
    return {
        "data": {
            "datasets": [
                {
                    "name": "default",
                    "dataset_dir": "data",
                    "train": {},
                }
            ]
        },
        "train": {
            "trainer_type": trainer_type,
            "debug_train": debug_train,
            "per_device_batch_size": 1,
            "group_size": 4,
            "unique_sample_num_per_epoch": 48,
            "gradient_step_per_epoch": 2,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        "rewards": [
            {
                "name": "quality",
                "reward_model": "CLIP",
                "weight": 1.0,
            }
        ],
    }


@pytest.mark.parametrize("trainer_type", ["grpo", "nft"])
def test_debug_train_uses_minimum_auto_accumulation_geometry(trainer_type):
    args = Arguments.from_dict(_config(trainer_type=trainer_type, debug_train=True))

    assert args.training_args.debug_train is True
    assert args.training_args.unique_sample_num_per_epoch == 2
    assert args.training_args.num_batches_per_epoch == 8


def test_debug_train_false_preserves_configured_epoch_size():
    args = Arguments.from_dict(_config())

    assert args.training_args.debug_train is False
    assert args.training_args.unique_sample_num_per_epoch == 48


def test_debug_train_is_first_exported_training_field():
    args = Arguments.from_dict(_config())

    assert next(iter(args.training_args.to_dict())) == "debug_train"


def test_debug_train_respects_manual_gradient_accumulation_semantics():
    args = Arguments.from_dict(
        _config(debug_train=True, gradient_accumulation_steps=3),
    )

    assert args.training_args.unique_sample_num_per_epoch == 1
    assert args.training_args.num_batches_per_epoch == 4
    assert args.training_args.gradient_accumulation_steps == 3


def test_debug_train_rejects_nonpositive_configured_sample_count():
    config = _config(debug_train=True)
    config["train"]["unique_sample_num_per_epoch"] = 0

    with pytest.raises(ValueError, match="must be positive"):
        Arguments.from_dict(config)


def test_debug_train_keeps_every_weighted_training_source():
    config = _config(debug_train=True)
    config["data"]["datasets"] = [
        {
            "name": "source_a",
            "dataset_dir": "data/a",
            "train": {"weight": 1},
        },
        {
            "name": "source_b",
            "dataset_dir": "data/b",
            "train": {"weight": 2},
        },
    ]

    args = Arguments.from_dict(config)
    training_datasets = args.data_args.training_datasets

    assert args.training_args.unique_sample_num_per_epoch == 6
    assert training_datasets[0].train.unique_sample_num_per_epoch == 2
    assert training_datasets[1].train.unique_sample_num_per_epoch == 4
