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

"""Regression tests for offline prompt-local reward-concordance analysis."""

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
    compute_reward_concordance_metrics,
)
from tools.reward_disagreement_analysis.plots import (
    plot_per_reward_conflict_score_trajectories,
    plot_reward_concordance_lower_bound_trajectories,
)
from tools.reward_disagreement_analysis.reward_logs import load_train_reward_groups


def test_group_metrics_report_raw_conflict_scores_and_lower_bound() -> None:
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
        reward_weights=np.asarray([1.0, 0.25]),
    )

    np.testing.assert_allclose(metrics["per_reward_conflict_score"], [0.5, -0.125])
    assert metrics["reward_concordance_lower_bound"] == pytest.approx(-0.125)


def test_lower_bound_uses_the_weakest_score_for_each_sample() -> None:
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )

    np.testing.assert_allclose(metrics["per_reward_conflict_score"], [0.0, 0.0])
    assert metrics["reward_concordance_lower_bound"] == pytest.approx(0.0)


def test_aggregate_group_metrics_macro_averages_prompt_groups() -> None:
    first = compute_reward_concordance_metrics(
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        reward_weights=np.asarray([0.25, 1.0]),
    )
    second = compute_reward_concordance_metrics(
        np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
        reward_weights=np.asarray([1.0, 0.25]),
    )

    aggregate = aggregate_group_metrics([first, second])

    assert aggregate["n_groups"] == 2
    np.testing.assert_allclose(
        aggregate["per_reward_conflict_score"],
        [0.2265625, 0.03125],
    )
    assert aggregate["reward_concordance_lower_bound"] == pytest.approx(-0.0859375)


def _write_train_pickle(path: Path, step: int) -> None:
    payload = {
        "step": step,
        "prompts": ["first", "second"],
        "pick_score": [np.asarray([0.0, 1.0]), np.asarray([0.0, 2.0])],
        "clip_score": [np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])],
        "src_groups": [
            {"group_id": 0, "probabilities": [0.25, 0.75]},
            {"group_id": 1, "probabilities": [0.75, 0.25]},
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_analysis_uses_only_saved_rewards_not_saved_src_probabilities(tmp_path: Path) -> None:
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
        output_dir=str(tmp_path / "output"),
    )

    rows, metadata = run_analysis(config)

    assert {row["metric"] for row in rows} == {
        "per_reward_conflict_score",
        "reward_concordance_lower_bound",
    }
    assert "n_effective_groups" not in metadata["runs"][0]


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


def test_analysis_rejects_removed_neutral_threshold_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        "runs:\n  - name: run\nanalysis:\n  reward_epsilon: 1.0e-8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="analysis is no longer supported"):
        _parse_config(config_path)


def test_lower_bound_and_per_reward_conflict_score_plots_are_written(tmp_path: Path) -> None:
    rows = [
        {
            "run_label": "SRC-NFT",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": reward,
            "metric": "per_reward_conflict_score",
            "value": value,
        }
        for reward, values in {"clip_score": (0.2, 0.3), "pick_score": (-0.4, -0.5)}.items()
        for step, value in enumerate(values)
    ]
    rows.extend(
        {
            "run_label": "SRC-NFT",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": "",
            "metric": "reward_concordance_lower_bound",
            "value": value,
        }
        for step, value in enumerate((-0.3, -0.4))
    )

    plot_per_reward_conflict_score_trajectories(rows, tmp_path)
    plot_reward_concordance_lower_bound_trajectories(rows, tmp_path)

    output_dir = tmp_path / "clip_score__pick_score"
    assert (output_dir / "per_reward_conflict_score" / "clip_score.png").stat().st_size > 0
    assert (output_dir / "per_reward_conflict_score" / "pick_score.png").stat().st_size > 0
    assert (output_dir / "reward_concordance_lower_bound.png").stat().st_size > 0
