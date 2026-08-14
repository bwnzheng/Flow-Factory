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

"""Regression tests for offline prompt-local reward-disagreement analysis."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.reward_disagreement_analysis.analyze import (
    AnalysisConfig,
    RunSpec,
    _parse_config,
    run_analysis,
)
from tools.reward_disagreement_analysis.metrics import (
    aggregate_group_metrics,
    compute_reward_disagreement_metrics,
)
from tools.reward_disagreement_analysis.plots import plot_per_reward_disagreement_trajectories
from tools.reward_disagreement_analysis.reward_logs import load_train_reward_groups


def test_group_metrics_keep_uniform_centering_when_effective_mass_is_supplied() -> None:
    rewards = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )

    metrics = compute_reward_disagreement_metrics(
        rewards,
        reward_weights=np.asarray([1.0, 0.25]),
        sample_weights=np.asarray([0.1, 0.2, 0.7]),
        reward_eps=0.0,
        scalar_eps=0.0,
    )

    np.testing.assert_allclose(metrics["disagreement_rate_per_reward"], [0.0, 1.0])
    assert metrics["fully_concordant_ratio"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        metrics["effective_disagreement_rate_per_reward"],
        [0.0, 1.0],
    )
    assert metrics["effective_fully_concordant_ratio"] == pytest.approx(0.0)
    assert metrics["scalar_advantage_identity_max_abs_error"] < 1e-12


def test_neutral_samples_are_excluded_from_rates_and_fully_concordant_ratio() -> None:
    rewards = np.asarray(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ]
    )

    metrics = compute_reward_disagreement_metrics(
        rewards,
        reward_weights=np.asarray([1.0, 1.0]),
        reward_eps=0.1,
        scalar_eps=0.1,
    )

    assert metrics["fully_valid_ratio"] == pytest.approx(0.0)
    assert np.isnan(metrics["fully_concordant_ratio"])
    assert np.isnan(metrics["mean_disagreement_count"])
    assert np.isnan(metrics["disagreement_count_histogram"]).all()


def test_positive_and_negative_fully_concordant_ratios_use_their_own_denominators() -> None:
    metrics = compute_reward_disagreement_metrics(
        np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
        reward_eps=0.0,
        scalar_eps=0.0,
    )

    assert metrics["fully_concordant_positive_ratio"] == pytest.approx(1.0)
    assert metrics["fully_concordant_negative_ratio"] == pytest.approx(1.0)


def test_aggregate_group_metrics_macro_averages_prompt_groups() -> None:
    first = compute_reward_disagreement_metrics(
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        reward_weights=np.asarray([0.25, 1.0]),
        reward_eps=0.0,
        scalar_eps=0.0,
    )
    second = compute_reward_disagreement_metrics(
        np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
        reward_weights=np.asarray([1.0, 0.25]),
        reward_eps=0.0,
        scalar_eps=0.0,
    )

    aggregate = aggregate_group_metrics([first, second])

    assert aggregate["n_groups"] == 2
    np.testing.assert_allclose(aggregate["disagreement_rate_per_reward"], [0.5, 0.5])


def _write_train_pickle(path: Path, step: int) -> None:
    payload = {
        "step": step,
        "prompts": ["first", "second"],
        "pick_score": [np.asarray([0.0, 1.0]), np.asarray([0.0, 2.0])],
        "clip_score": [np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])],
        "src_groups": [
            {
                "group_id": 0,
                "probabilities": [0.25, 0.75],
            },
            {
                "group_id": 1,
                "probabilities": [0.75, 0.25],
            },
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_saved_reward_reader_and_analysis_need_explicit_weights(tmp_path: Path) -> None:
    rewards_dir = tmp_path / "saves" / "run" / "logs" / "rewards"
    rewards_dir.mkdir(parents=True)
    _write_train_pickle(rewards_dir / "train_step_000007.pkl", step=7)

    groups = load_train_reward_groups(rewards_dir)

    assert list(groups) == [7]
    assert [group.reward_names for group in groups[7]] == [
        ("clip_score", "pick_score"),
        ("clip_score", "pick_score"),
    ]
    config = AnalysisConfig(
        save_dir=str(tmp_path / "saves"),
        runs=[
            RunSpec(name="run", label="SRC", reward_weights={"pick_score": 1.0, "clip_score": 1.0})
        ],
        reward_epsilon=0.0,
        scalar_epsilon=0.0,
        output_dir=str(tmp_path / "output"),
    )

    rows, metadata = run_analysis(config)

    assert {row["metric"] for row in rows} >= {
        "natural_disagreement_rate",
        "effective_disagreement_rate",
        "natural_fully_concordant_ratio",
        "effective_fully_concordant_ratio",
    }
    assert metadata["runs"][0]["n_effective_groups"] == 2


def test_analysis_recovers_weights_from_saved_media_run_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "saves" / "run"
    rewards_dir = run_dir / "logs" / "rewards"
    rewards_dir.mkdir(parents=True)
    _write_train_pickle(rewards_dir / "train_step_000007.pkl", step=7)
    context = {
        "record_type": "run_context",
        "configuration": {
            "reward": {
                "reward_0": {"name": "pick_score", "weight": {"source": 1.0}},
                "reward_1": {"name": "clip_score", "weight": {"source": 1.0}},
            }
        },
    }
    (run_dir / "logs" / "media.jsonl").write_text(json.dumps(context) + "\n", encoding="utf-8")
    config = AnalysisConfig(
        save_dir=str(tmp_path / "saves"),
        runs=[RunSpec(name="run", label="SRC", reward_weights={})],
        reward_epsilon=0.0,
        scalar_epsilon=0.0,
        output_dir=str(tmp_path / "output"),
    )

    _, metadata = run_analysis(config)

    assert metadata["runs"][0]["reward_weight_sources"] == {
        "clip_score__pick_score": "saved_media_run_context:source"
    }


def test_analysis_rejects_missing_historical_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        "runs:\n  - name: run\nreward_weights:\n  pick_score: 1.0\n",
        encoding="utf-8",
    )

    config = _parse_config(config_path)
    rewards_dir = tmp_path / "saves" / "run" / "logs" / "rewards"
    rewards_dir.mkdir(parents=True)
    _write_train_pickle(rewards_dir / "train_step_000007.pkl", step=7)
    config = AnalysisConfig(
        save_dir=str(tmp_path / "saves"),
        runs=config.runs,
        output_dir=str(tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="Cannot recover scalarization weights"):
        run_analysis(config)


def test_per_reward_disagreement_plots_are_written_separately(tmp_path: Path) -> None:
    rows = [
        {
            "run_label": "SRC-NFT",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": reward,
            "metric": metric,
            "value": value,
        }
        for reward, values in {
            "clip_score": ((0.2, 0.3), (0.1, 0.2)),
            "pick_score": ((0.4, 0.5), (0.3, 0.4)),
        }.items()
        for metric, metric_values in zip(
            ("natural_disagreement_rate", "effective_disagreement_rate"), values
        )
        for step, value in enumerate(metric_values)
    ]

    plot_per_reward_disagreement_trajectories(rows, tmp_path)

    output_dir = tmp_path / "clip_score__pick_score" / "per_reward_disagreement"
    assert (output_dir / "clip_score.png").stat().st_size > 0
    assert (output_dir / "pick_score.png").stat().st_size > 0
