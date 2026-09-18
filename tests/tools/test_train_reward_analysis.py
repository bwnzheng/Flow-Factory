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

"""Regression tests for offline train-reward analysis."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest
from matplotlib.lines import Line2D

from tools.train_reward_analysis import analyze
from tools.train_reward_analysis.analyze import (
    CACHE_MODES,
    AnalysisConfig,
    RunSpec,
    _apply_overrides,
    _build_parser,
    _parse_config,
    _render_figures,
    run_analysis,
)
from tools.train_reward_analysis.metrics import (
    aggregate_group_metrics,
    compute_reward_concordance_metrics,
)
from tools.train_reward_analysis.plots import (
    _percent_of_own_range,
    _percent_of_sample_share,
    _smoothed_series,
    plot_agreement_count_distribution_trajectories,
    plot_agreement_count_expectation_trajectories,
    plot_per_reward_conflict_score_trajectories,
    plot_per_reward_disagreement_trajectories,
    plot_concordance_rate_trajectories,
    plot_reward_concordance_lower_bound_trajectories,
    plot_run_training_progress_trajectories,
    plot_standardized_reward_covariance_trajectories,
)
from tools.train_reward_analysis.reward_logs import load_train_reward_groups


def test_group_metrics_report_raw_conflict_scores_and_lower_bound() -> None:
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
        reward_weights=np.asarray([1.0, 0.25]),
    )

    np.testing.assert_allclose(metrics["per_reward_conflict_score"], [1.0, -0.25])
    np.testing.assert_allclose(metrics["per_reward_disagreement"], [0.0, 2.0 / 3.0])
    np.testing.assert_allclose(
        metrics["standardized_reward_covariance"], [[1.0, -1.0], [-1.0, 1.0]]
    )
    assert metrics["reward_concordance_lower_bound"] == pytest.approx(-0.25)


def test_lower_bound_uses_the_weakest_score_for_each_sample() -> None:
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )

    np.testing.assert_allclose(metrics["per_reward_conflict_score"], [0.0, 0.0])
    np.testing.assert_allclose(metrics["per_reward_disagreement"], [0.0, 0.0])
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
        [0.375, 0.375],
    )
    assert aggregate["reward_concordance_lower_bound"] == pytest.approx(-0.25)


def test_agreement_count_distribution_exposes_joint_conflict_structure() -> None:
    """Equal per-reward disagreement can still hide different joint conflict."""
    diffuse = np.asarray(
        [
            [2.0, 0.0, 2.0],
            [1.0, 4.0, 0.0],
            [0.0, 3.0, 3.0],
            [4.0, 2.0, 1.0],
            [1.0, 1.0, 3.0],
            [2.0, 1.0, 1.0],
        ]
    )
    polarized = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 4.0],
            [2.0, 4.0, 3.0],
            [4.0, 0.0, 2.0],
            [4.0, 3.0, 1.0],
        ]
    )
    weights = np.ones(3)

    diffuse_metrics = compute_reward_concordance_metrics(diffuse, weights)
    polarized_metrics = compute_reward_concordance_metrics(polarized, weights)

    # The marginal disagreement rates, and therefore the redundant mean count,
    # are identical for both groups.
    np.testing.assert_allclose(
        diffuse_metrics["per_reward_disagreement"], polarized_metrics["per_reward_disagreement"]
    )
    assert diffuse_metrics["mean_agreement_count"] == pytest.approx(
        polarized_metrics["mean_agreement_count"]
    )

    # The count distribution separates them: every diffuse sample keeps two of
    # the three rewards, while the other group spreads over one to three.
    np.testing.assert_allclose(
        diffuse_metrics["sample_agreement_count_distribution"], [0.0, 1.0 / 6.0, 5.0 / 6.0, 0.0]
    )
    np.testing.assert_allclose(
        polarized_metrics["sample_agreement_count_distribution"], [0.0, 1.0 / 3.0, 0.5, 1.0 / 6.0]
    )
    assert diffuse_metrics["fully_concordant_sample_rate"] == 0.0
    assert polarized_metrics["fully_concordant_sample_rate"] == pytest.approx(1.0 / 6.0)


def test_agreement_count_mean_is_the_complement_of_disagreement() -> None:
    """The mean count is derived, and total conflict is structurally unreachable."""
    rewards = np.asarray(
        [
            [0.0, 5.0, 1.0],
            [3.0, 0.5, 4.0],
            [1.0, 3.0, 2.0],
            [4.0, 1.5, 0.0],
            [2.0, 2.5, 3.0],
        ]
    )
    metrics = compute_reward_concordance_metrics(rewards, np.asarray([1.0, 0.5, 2.0]))

    # A sample below the group mean on every reward is also below the mean of
    # the weighted scalar, so no sample agrees with the scalar on zero rewards.
    assert metrics["sample_agreement_count_distribution"][0] == 0.0
    assert metrics["mean_agreement_count"] == pytest.approx(
        rewards.shape[1] - metrics["per_reward_disagreement"].sum()
    )


def test_aggregate_group_metrics_macro_averages_agreement_count_distribution() -> None:
    first = compute_reward_concordance_metrics(
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        reward_weights=np.asarray([0.25, 1.0]),
    )
    second = compute_reward_concordance_metrics(
        np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
        reward_weights=np.asarray([1.0, 0.25]),
    )

    aggregate = aggregate_group_metrics([first, second])

    first_distribution = first["sample_agreement_count_distribution"]
    second_distribution = second["sample_agreement_count_distribution"]

    np.testing.assert_allclose(
        aggregate["sample_agreement_count_distribution"],
        (first_distribution + second_distribution) / 2.0,
    )
    assert aggregate["mean_agreement_count"] == pytest.approx(
        (first["mean_agreement_count"] + second["mean_agreement_count"]) / 2.0
    )
    assert aggregate["fully_concordant_sample_rate"] == pytest.approx(
        (first["fully_concordant_sample_rate"] + second["fully_concordant_sample_rate"]) / 2.0
    )


def test_signed_full_concordance_splits_full_concordance_by_direction() -> None:
    """Positive and negative full agreement are the two halves of c = n_rewards."""
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )

    # Both rewards rise together, so the two lowest samples sit below the group
    # mean on both and the two highest sit above it on both.
    assert metrics["positive_fully_concordant_sample_rate"] == pytest.approx(0.5)
    assert metrics["negative_fully_concordant_sample_rate"] == pytest.approx(0.5)
    assert metrics["fully_concordant_sample_rate"] == pytest.approx(1.0)


def test_signed_full_concordance_excludes_degenerate_rewards_from_both_halves() -> None:
    """A reward with no group variance is neither positive nor negative.

    Such a reward standardizes to an exact zero, which still counts as agreeing
    with the scalar, so it can leave full concordance above the two signed
    halves put together.
    """
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )

    assert metrics["positive_fully_concordant_sample_rate"] == pytest.approx(0.0)
    assert metrics["negative_fully_concordant_sample_rate"] == pytest.approx(0.0)
    assert metrics["fully_concordant_sample_rate"] == pytest.approx(1.0)


def test_per_reward_mean_reward_is_the_raw_group_mean() -> None:
    """Progress curves need the raw level, not the standardized one."""
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 10.0], [2.0, 20.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )

    np.testing.assert_allclose(metrics["per_reward_mean_reward"], [1.0, 15.0])


def test_aggregate_group_metrics_macro_averages_signed_concordance_and_raw_level() -> None:
    """Raw levels average over prompt groups, so group size never reweights them."""
    first = compute_reward_concordance_metrics(
        np.asarray([[0.0, 10.0], [2.0, 20.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )
    second = compute_reward_concordance_metrics(
        np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )

    aggregate = aggregate_group_metrics([first, second])

    np.testing.assert_allclose(aggregate["per_reward_mean_reward"], [1.0, 7.5])
    assert aggregate["positive_fully_concordant_sample_rate"] == pytest.approx(
        (
            first["positive_fully_concordant_sample_rate"]
            + second["positive_fully_concordant_sample_rate"]
        )
        / 2.0
    )
    assert aggregate["negative_fully_concordant_sample_rate"] == pytest.approx(
        (
            first["negative_fully_concordant_sample_rate"]
            + second["negative_fully_concordant_sample_rate"]
        )
        / 2.0
    )


def test_percent_of_own_range_stretches_each_series_between_its_own_extremes() -> None:
    """Rewards on different scales become comparable in shape, not in level."""
    np.testing.assert_allclose(
        _percent_of_own_range(np.asarray([0.25, 0.30, 0.35])),
        [0.0, 50.0, 100.0],
    )
    # A reward that never moves has no range to express progress against.
    np.testing.assert_allclose(_percent_of_own_range(np.asarray([0.7, 0.7, 0.7])), [0.0, 0.0, 0.0])


def test_percent_of_sample_share_never_rescales_an_agreement_rate() -> None:
    """Agreement curves are already shares of samples and are plotted as they are."""
    np.testing.assert_allclose(
        _percent_of_sample_share(np.asarray([0.0, 0.125, 1.0])),
        [0.0, 12.5, 100.0],
    )


def test_aggregate_group_metrics_rejects_mismatched_agreement_distribution() -> None:
    metrics = compute_reward_concordance_metrics(
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        reward_weights=np.asarray([1.0, 1.0]),
    )
    truncated = {**metrics, "sample_agreement_count_distribution": np.asarray([0.0, 1.0])}

    with pytest.raises(ValueError, match="agreement-count distribution"):
        aggregate_group_metrics([truncated])


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
        "per_reward_disagreement",
        "per_reward_bottleneck_rate",
        "standardized_reward_covariance",
        "reward_concordance_lower_bound",
        "agreement_count_c0",
        "agreement_count_c1",
        "agreement_count_c2",
        "mean_agreement_count",
        "fully_concordant_sample_rate",
        "positive_fully_concordant_sample_rate",
        "negative_fully_concordant_sample_rate",
        "per_reward_mean_reward",
        "adv_positive",
        "adv_negative",
        "adv_zero",
        "sample_count_adv_positive",
        "sample_count_adv_negative",
        "sample_count_adv_zero",
    }
    assert not any("weight_" in row["metric"] for row in rows)
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


def test_cache_mode_flag_defaults_to_leaving_the_config_alone() -> None:
    """An omitted override must not replace the config value with a default."""
    parser = _build_parser()

    assert parser.parse_args(["-c", "analysis.yaml"]).cache_mode is None
    for mode in CACHE_MODES:
        assert parser.parse_args(["-c", "analysis.yaml", "--cache-mode", mode]).cache_mode == mode

    with pytest.raises(SystemExit):
        parser.parse_args(["-c", "analysis.yaml", "--cache-mode", "refresh"])


def test_cache_mode_flag_overrides_the_config_value() -> None:
    config = AnalysisConfig(cache_mode="regenerate")

    overridden = _apply_overrides(config, argparse.Namespace(cache_mode="reuse"))
    assert overridden.cache_mode == "reuse"
    # The rest of the configuration survives the override untouched.
    assert overridden.output_dir == config.output_dir
    assert overridden.smoothing_window == config.smoothing_window

    assert _apply_overrides(config, argparse.Namespace(cache_mode=None)).cache_mode == "regenerate"


def test_cache_mode_flag_is_honoured_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag reaches the run loop, not just the parser.

    The config asks to regenerate, so a reuse that never happens would mean the
    override was dropped somewhere between parsing and use. Asserting on the
    failure that reuse must produce when no cache exists is what proves the
    command line won.
    """
    rewards_dir = tmp_path / "saves" / "run" / "logs" / "rewards"
    rewards_dir.mkdir(parents=True)
    _write_train_pickle(rewards_dir / "train_step_000007.pkl", step=7)
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        "save_dir: {save_dir}\n"
        "runs:\n  - name: run\n    reward_weights:\n      pick_score: 1.0\n      clip_score: 1.0\n"
        "output:\n  dir: {output_dir}\n  cache_mode: regenerate\n".format(
            save_dir=tmp_path / "saves", output_dir=tmp_path / "output"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze", "-c", str(config_path), "--cache-mode", "reuse"],
    )

    with pytest.raises(FileNotFoundError, match="Plot-data cache not found"):
        analyze.main()


def test_plot_smoothing_window_is_configurable_and_requires_an_odd_integer(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        "runs:\n  - name: run\nplot:\n  smoothing_window: 3\n",
        encoding="utf-8",
    )
    assert _parse_config(config_path).smoothing_window == 3

    config_path.write_text(
        "runs:\n  - name: run\nplot:\n  smoothing_window: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive odd integer"):
        _parse_config(config_path)


def test_plot_format_accepts_pdf_and_rejects_unknown_value(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        "runs:\n  - name: run\noutput:\n  plot_format: pdf\n",
        encoding="utf-8",
    )
    assert _parse_config(config_path).plot_format == "pdf"

    config_path.write_text(
        "runs:\n  - name: run\noutput:\n  plot_format: svg\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="png.*pdf"):
        _parse_config(config_path)


def test_centered_smoothing_uses_available_edge_points() -> None:
    rows = [{"step": step, "value": value} for step, value in enumerate((0.0, 3.0, 6.0, 9.0, 12.0))]

    steps, values = _smoothed_series(rows, smoothing_window=3)

    np.testing.assert_array_equal(steps, np.arange(5))
    np.testing.assert_allclose(values, [1.5, 3.0, 6.0, 9.0, 10.5])


def test_lower_bound_and_per_reward_conflict_score_plots_are_written(tmp_path: Path) -> None:
    rows = [
        {
            "run_label": "SRC-NFT",
            "dataset": "pickscore",
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
            "dataset": "pickscore",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": "",
            "metric": "reward_concordance_lower_bound",
            "value": value,
        }
        for step, value in enumerate((-0.3, -0.4))
    )
    rows.extend(
        {
            "run_label": "SRC-NFT",
            "dataset": "pickscore",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": reward,
            "metric": "per_reward_disagreement",
            "value": value,
        }
        for reward, values in {"clip_score": (0.2, 0.3), "pick_score": (0.7, 0.6)}.items()
        for step, value in enumerate(values)
    )

    plot_per_reward_conflict_score_trajectories(rows, tmp_path)
    plot_per_reward_disagreement_trajectories(rows, tmp_path)
    rows.extend(
        {
            "run_label": "SRC-NFT",
            "dataset": "pickscore",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": "",
            "reward_pair": "clip_score__pick_score",
            "metric": "standardized_reward_covariance",
            "value": value,
        }
        for step, value in enumerate((0.2, 0.3))
    )
    plot_standardized_reward_covariance_trajectories(rows, tmp_path)
    plot_reward_concordance_lower_bound_trajectories(rows, tmp_path)

    output_dir = tmp_path / "pickscore"
    assert (output_dir / "per_reward_conflict_score" / "clip_score.png").stat().st_size > 0
    assert (output_dir / "per_reward_conflict_score" / "pick_score.png").stat().st_size > 0
    assert (output_dir / "per_reward_disagreement" / "clip_score.png").stat().st_size > 0
    assert (output_dir / "per_reward_disagreement" / "pick_score.png").stat().st_size > 0
    covariance_dir = tmp_path / "pickscore" / "standardized_reward_covariance"
    assert (covariance_dir / "clip_score__pick_score.png").stat().st_size > 0
    assert (output_dir / "reward_concordance_lower_bound.png").stat().st_size > 0


def _write_stage_marker(rows, output_dir, smoothing_window, plot_format) -> None:
    """Picklable figure-stage stand-in for the rendering-plumbing test."""
    Path(output_dir, f"stage_{smoothing_window}_{plot_format}.txt").write_text(
        str(len(rows)), encoding="utf-8"
    )


def test_render_figures_forwards_arguments_to_worker_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stages receive rows, output directory, smoothing and format in workers."""
    rows = [{"value": 1.0}, {"value": 2.0}]
    config = AnalysisConfig(smoothing_window=3, plot_format="pdf")
    monkeypatch.setattr(analyze, "_plot_worker_count", lambda functions: 2)

    _render_figures(config, rows, tmp_path, (_write_stage_marker,))

    assert (tmp_path / "stage_3_pdf.txt").read_text(encoding="utf-8") == "2"


def test_render_figures_renders_in_process_when_only_one_worker_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"value": 1.0}]
    config = AnalysisConfig(smoothing_window=5, plot_format="png")
    monkeypatch.setattr(analyze, "_plot_worker_count", lambda functions: 1)

    _render_figures(config, rows, tmp_path, (_write_stage_marker,))

    assert (tmp_path / "stage_5_png.txt").read_text(encoding="utf-8") == "1"


def _agreement_count_rows(steps: int = 24) -> list[dict]:
    """Rows long enough that sparse markers apply, with a structurally empty c = 0 bin."""
    rows = [
        {
            "run_label": label,
            "dataset": "pickscore",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": "",
            "reward_pair": f"count_{agreeing_count}",
            "metric": f"agreement_count_c{agreeing_count}",
            "value": 0.0 if agreeing_count == 0 else value + 0.001 * step,
        }
        for label in ("SRC-NFT", "NFT (uniform)")
        for agreeing_count, value in ((0, 0.0), (1, 0.4), (2, 0.6))
        for step in range(steps)
    ]
    rows.extend(
        {
            "run_label": label,
            "dataset": "pickscore",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": "",
            "reward_pair": "",
            "metric": "mean_agreement_count",
            "value": 1.6 + 0.001 * step,
        }
        for label in ("SRC-NFT", "NFT (uniform)")
        for step in range(steps)
    )
    return rows


def test_agreement_count_distribution_plot_draws_each_bin_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each run/bin pair is drawn exactly once, and the legend keeps the runs apart.

    Series sharing one agreeing count also share a color, so the dash pattern is
    the only cue separating the runs: the plotted series must keep sparse markers,
    and the legend handles must carry no marker at all, since the centre marker
    would cover the single dash gap that fits inside a short handle.
    """
    drawn: list[tuple[str, int]] = []
    handles: list[Line2D] = []
    original_plot = matplotlib.axes.Axes.plot
    original_legend = matplotlib.axes.Axes.legend

    def counting_plot(self, *args, **kwargs):
        drawn.append((str(kwargs.get("linestyle")), int(kwargs.get("markevery"))))
        return original_plot(self, *args, **kwargs)

    def capturing_legend(self, *args, **kwargs):
        handles.extend(kwargs.get("handles") or [])
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", counting_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", capturing_legend)

    plot_agreement_count_distribution_trajectories(_agreement_count_rows(), tmp_path)

    assert len(drawn) == 4, "each run/bin pair is drawn exactly once"
    assert all(markevery > 1 for _, markevery in drawn)

    labels = [handle.get_label() for handle in handles]
    assert sorted(labels) == sorted(
        f"{label} | c={agreeing_count}"
        for label in ("SRC-NFT", "NFT (uniform)")
        for agreeing_count in (1, 2)
    )
    for agreeing_count in (1, 2):
        styles = {
            str(handle.get_linestyle())
            for handle in handles
            if handle.get_label().endswith(f"| c={agreeing_count}")
        }
        assert len(styles) == 2, f"runs share one dash pattern for c={agreeing_count}: {styles}"
    assert all(handle.get_marker() in ("", "None", None) for handle in handles)


def test_agreement_count_plots_are_written_per_dataset(tmp_path: Path) -> None:
    rows = _agreement_count_rows()

    plot_agreement_count_distribution_trajectories(rows, tmp_path)
    plot_agreement_count_expectation_trajectories(rows, tmp_path)

    assert (tmp_path / "pickscore" / "agreement_count.png").stat().st_size > 0
    assert (tmp_path / "pickscore" / "agreement_count_expectation.png").stat().st_size > 0


def _training_progress_rows(dataset: str = "pickscore") -> list[dict]:
    rows = []
    for label in ("SRC-NFT", "NFT (uniform)"):
        for step in range(10):
            for reward in ("clip_score", "pick_score"):
                rows.append(
                    {
                        "run_label": label,
                        "dataset": dataset,
                        "step": step,
                        "reward_combination": "clip_score__pick_score",
                        "reward": reward,
                        "reward_pair": "",
                        "metric": "per_reward_mean_reward",
                        "value": 0.25 + 0.01 * step,
                    }
                )
            for metric, value in (
                ("positive_fully_concordant_sample_rate", 0.15),
                ("negative_fully_concordant_sample_rate", 0.24),
            ):
                rows.append(
                    {
                        "run_label": label,
                        "dataset": dataset,
                        "step": step,
                        "reward_combination": "clip_score__pick_score",
                        "reward": "",
                        "reward_pair": "",
                        "metric": metric,
                        "value": value,
                    }
                )
    return rows


def test_training_progress_figures_are_written_into_their_own_folder(tmp_path: Path) -> None:
    """One reward-progress figure per run, plus one all-run agreement figure."""
    rows = _training_progress_rows()

    plot_run_training_progress_trajectories(rows, tmp_path)
    plot_concordance_rate_trajectories(rows, tmp_path)

    folder = tmp_path / "pickscore" / "training_progress"
    assert (folder / "SRC-NFT.png").stat().st_size > 0
    assert (folder / "NFT__uniform_.png").stat().st_size > 0
    assert (folder / "concordance.png").stat().st_size > 0


def test_run_progress_figure_aggregates_every_reward_into_one_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rewards collapse to a single curve, so the figure cannot grow with them.

    Each reward is normalized to its own range before averaging, so the
    aggregate is the mean of per-reward percentages rather than of raw levels.
    """
    drawn: list[tuple[float, ...]] = []
    original_plot = matplotlib.axes.Axes.plot

    def counting_plot(self, *args, **kwargs):
        drawn.append(tuple(np.asarray(args[1], dtype=float)))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", counting_plot)

    rows = [
        row
        for row in _training_progress_rows()
        if row["run_label"] == "SRC-NFT" and row["metric"] != "per_reward_mean_reward"
    ]
    # Two rewards on wildly different scales, so averaging raw levels and
    # averaging per-reward percentages would disagree.
    rows.extend(
        {
            "run_label": "SRC-NFT",
            "dataset": "pickscore",
            "step": step,
            "reward_combination": "clip_score__pick_score",
            "reward": reward,
            "reward_pair": "",
            "metric": "per_reward_mean_reward",
            "value": low + (high - low) * step / 9.0,
        }
        for reward, low, high in (("clip_score", 0.0, 1.0), ("pick_score", 100.0, 200.0))
        for step in range(10)
    )

    # Smoothing is off here so the drawn series is the aggregate itself.
    plot_run_training_progress_trajectories(rows, tmp_path, smoothing_window=1)

    # Three series: the aggregate, positive, and negative, each drawn twice as a
    # faint raw trace and then a foreground.
    assert len(drawn) == 6, f"expected 3 series drawn as raw + smoothed, got {len(drawn)}"
    aggregate, foreground = drawn[0], drawn[1]
    # Both rewards span their own range equally, so every step averages to
    # 100 * step / 9 — which averaging raw levels could not produce, since
    # clip_score and pick_score live on scales 100x apart.
    np.testing.assert_allclose(aggregate, 100.0 * np.arange(10) / 9.0)
    np.testing.assert_allclose(foreground, aggregate)


def test_concordance_figure_gives_runs_the_colour_and_direction_the_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs own colour and marker; positive/negative own solid/dashed.

    Direction has only two values while runs do not, so the strongest channel
    goes to the runs and the dash pattern — the one that survives greyscale —
    carries direction.
    """
    drawn: list[tuple[str, str, str]] = []
    handles: list[Line2D] = []
    original_plot = matplotlib.axes.Axes.plot
    original_legend = matplotlib.axes.Axes.legend

    def recording_plot(self, *args, **kwargs):
        drawn.append(
            (
                str(kwargs.get("color")),
                str(kwargs.get("linestyle")),
                str(kwargs.get("marker")),
            )
        )
        return original_plot(self, *args, **kwargs)

    def capturing_legend(self, *args, **kwargs):
        handles.extend(kwargs.get("handles") or [])
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", capturing_legend)

    plot_concordance_rate_trajectories(_training_progress_rows(), tmp_path)

    # Drawn run -> direction -> (raw, smoothed), so the odd entries are the
    # four foregrounded series in that order.
    assert len(drawn) == 8, f"expected 4 series drawn twice, got {len(drawn)}"
    positive_run_a, negative_run_a, positive_run_b, negative_run_b = drawn[1::2]

    assert [series[1] for series in drawn[1::2]] == ["-", "--", "-", "--"]
    # Direction never moves the colour or the marker off its run.
    for positive, negative in ((positive_run_a, negative_run_a), (positive_run_b, negative_run_b)):
        assert positive[0] == negative[0], "one colour per run"
        assert positive[2] == negative[2], "one marker per run"
    # And the two runs share neither.
    assert positive_run_a[0] != positive_run_b[0], "runs must not share a colour"
    assert positive_run_a[2] != positive_run_b[2], "runs must not share a marker"

    labels = [handle.get_label() for handle in handles]
    assert sorted(labels) == sorted(
        f"{label} | {legend_name}"
        for label in ("SRC-NFT", "NFT (uniform)")
        for _, legend_name in (
            ("", "positive concordant rate"),
            ("", "negative concordant rate"),
        )
    )
    assert {str(handle.get_linestyle()) for handle in handles} == {"-", "--"}


def _record_axis_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[float, float]], list[object]]:
    """Record every y-limit written and every twin axis created.

    Matplotlib's autoscale writes limits through this same call, so the
    recordings hold both explicit pins and autoscaled ranges.
    """
    written: list[tuple[float, float]] = []
    twins: list[object] = []
    original_set_ylim = matplotlib.axes.Axes.set_ylim
    original_twinx = matplotlib.axes.Axes.twinx

    def recording_set_ylim(self, *args, **kwargs):
        # Autoscale passes one sequence; explicit pins pass two bounds.
        bounds = (
            args[0] if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)) else args
        )
        written.append(tuple(float(bound) for bound in bounds))
        return original_set_ylim(self, *args, **kwargs)

    def recording_twinx(self, *args, **kwargs):
        twin = original_twinx(self, *args, **kwargs)
        twins.append(twin)
        return twin

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", recording_set_ylim)
    monkeypatch.setattr(matplotlib.axes.Axes, "twinx", recording_twinx)
    return written, twins


def test_run_progress_figure_puts_rates_on_a_second_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rates get their own floating axis beside the fixed progress axis.

    Progress is anchored at both ends by its own min-max normalization, so it
    keeps 0-100%. Rates sit in a narrow band whose bounds are arbitrary, so
    pinning them to the same range would flatten the differences the figure
    exists to show.
    """
    written, twins = _record_axis_limits(monkeypatch)

    plot_run_training_progress_trajectories(_training_progress_rows(), tmp_path)

    # One twin rate axis per run, and only the progress axes are pinned.
    assert len(twins) == 2, f"expected one rate axis per run, got {len(twins)}"
    pinned = [limit for limit in written if limit == (0.0, 100.0)]
    assert len(pinned) == 2, f"only the two progress axes are pinned, got {written}"
    for twin in twins:
        assert twin.get_ylim() != (0.0, 100.0), "the rate axis must follow its own data"


def test_concordance_figure_autoscales_to_its_own_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The all-run figure drops the fixed 0-100% range to magnify the rates."""
    written, twins = _record_axis_limits(monkeypatch)

    plot_concordance_rate_trajectories(_training_progress_rows(), tmp_path)

    assert not twins, "the all-run figure has only rates, so it needs no second axis"
    assert (0.0, 100.0) not in written, f"the rate axis must not be pinned, got {written}"
    low, high = (float(bound) for bound in written[-1])
    assert 0.0 < low and high < 100.0, f"axis should track the rate band, got {written[-1]}"


def test_agreement_count_plots_reject_a_dataset_with_two_reward_sets(tmp_path: Path) -> None:
    """A dataset fixes its reward set; mixing two would mix agreeing-count scales."""
    rows = _agreement_count_rows()
    second_reward_set = [
        {**row, "reward_combination": "clip_score__ocr_reward__pick_score"} for row in rows
    ]
    rows.extend(second_reward_set)

    with pytest.raises(ValueError, match="more than one reward combination"):
        plot_agreement_count_distribution_trajectories(rows, tmp_path)

    with pytest.raises(ValueError, match="more than one reward combination"):
        plot_agreement_count_expectation_trajectories(rows, tmp_path)
