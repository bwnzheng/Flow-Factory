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

import numpy as np
import pytest
import torch

from flow_factory.logger.abc import LocalFileLogger


def test_metrics_jsonl_normalizes_nested_numpy_and_tensor_values(tmp_path):
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
        "ga/samples": [
            {
                "selection_diagnostics": {
                    "degenerate_scalar_contrast": np.bool_(False),
                    "elite_id": np.int64(3),
                    "frozen_score": np.float32(0.25),
                    "contributions": np.array([[0.1, 0.2]], dtype=np.float32),
                    "tensor_scalar": torch.tensor(2.0),
                    "tensor_vector": torch.tensor([1, 2]),
                }
            }
        ],
    }

    logger._write_metrics_jsonl(diagnostics, step=7)

    metrics_path = tmp_path / "json-safe" / "logs" / "metrics.jsonl"
    record = json.loads(metrics_path.read_text().strip())
    assert record["np_bool_metric"] is True
    selection = record["ga/samples"][0]["selection_diagnostics"]
    assert selection == {
        "degenerate_scalar_contrast": False,
        "elite_id": 3,
        "frozen_score": 0.25,
        "contributions": [[pytest.approx(0.1), pytest.approx(0.2)]],
        "tensor_scalar": 2.0,
        "tensor_vector": [1, 2],
    }
