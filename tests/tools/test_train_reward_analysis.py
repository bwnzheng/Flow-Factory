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

from tools.train_reward_analysis import analyze, plots
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
from tools.train_reward_analysis.figure_spec import (
    SPEC_VERSION,
    FigureAxis,
    FigureFontSizes,
    FigureLegend,
    FigureSeries,
    FigureSpec,
    LegendEntry,
    read_spec,
)
from tools.train_reward_analysis.metrics import (
    aggregate_group_metrics,
    compute_reward_concordance_metrics,
)
from tools.train_reward_analysis.plots import (
    _CATEGORICAL_COLORS,
    _moving_average,
    _percent_of_own_range,
    _percent_points,
    build_agreement_count_distribution_figures,
    build_agreement_count_expectation_figures,
    build_concordance_rate_figures,
    build_figures,
    build_run_training_progress_figures,
    render_figure,
    write_figure_data,
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


def test_percent_points_never_rescale_an_agreement_rate() -> None:
    """Agreement curves are already shares of samples and are plotted as they are."""
    rows = [{"step": step, "value": value} for step, value in enumerate([0.0, 0.125, 1.0])]

    assert _percent_points(rows) == [[0.0, 0.0], [1.0, 12.5], [2.0, 100.0]]
    assert _percent_points([]) == []


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

    with pytest.raises(FileNotFoundError, match="No metadata.json"):
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
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(_moving_average(values, 3), [1.5, 2.0, 3.0, 3.5])
    np.testing.assert_allclose(_moving_average(values, 1), values)
    with pytest.raises(ValueError, match="positive odd integer"):
        _moving_average(values, 4)


def test_every_figure_carries_the_configured_smoothing_window() -> None:
    """A spec must never fall back to its own default and ignore the config."""
    outputs = build_figures(_training_progress_rows(), smoothing_window=3)

    assert outputs
    assert {spec.smoothing_window for _, spec in outputs} == {3}


def test_figure_data_round_trips_through_its_own_file(tmp_path: Path) -> None:
    stem, spec = build_figures(_training_progress_rows())[0]

    write_figure_data([(stem, spec)], tmp_path)

    assert read_spec(tmp_path / f"{stem}.json") == spec
    assert (tmp_path / f"{stem}.json").is_file()


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_older_figure_data_remains_readable(tmp_path: Path, version: int) -> None:
    stem, spec = build_figures(_training_progress_rows())[0]
    write_figure_data([(stem, spec)], tmp_path)
    path = tmp_path / f"{stem}.json"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'"spec_version": {SPEC_VERSION}', f'"spec_version": {version}'
        ),
        encoding="utf-8",
    )

    assert read_spec(path) == spec


def test_font_sizes_round_trip_and_apply_to_continuous_dual_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    font_sizes = FigureFontSizes(
        title=16.0,
        x_label=13.0,
        left_y_label=14.0,
        right_y_label=15.0,
        x_tick=9.0,
        left_y_tick=10.0,
        right_y_tick=11.0,
        legend=12.0,
    )
    spec = FigureSpec(
        title="Font sizes",
        x_label="Training step",
        left=FigureAxis(label="Left"),
        right=FigureAxis(label="Right"),
        series=[
            FigureSeries(label="left", points=[[0.0, 0.1]], color="C0"),
            FigureSeries(
                label="right",
                points=[[0.0, 0.9]],
                color="C1",
                axis="right",
            ),
        ],
        legend=FigureLegend(
            entries=[LegendEntry(label="left", color="C0")],
            ncol=2,
        ),
        smoothing_window=1,
        border_width=2.0,
        top_margin=0.04,
        font_sizes=font_sizes,
    )
    write_figure_data([("fonts", spec)], tmp_path)
    loaded = read_spec(tmp_path / "fonts.json")
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: None)

    render_figure(loaded, tmp_path, "fonts", "png")
    figure = plots.plt.gcf()
    left_axis, right_axis = figure.axes

    assert loaded == spec
    assert json.loads((tmp_path / "fonts.json").read_text(encoding="utf-8"))["font_sizes"] == {
        "title": 16.0,
        "x_label": 13.0,
        "left_y_label": 14.0,
        "right_y_label": 15.0,
        "x_tick": 9.0,
        "left_y_tick": 10.0,
        "right_y_tick": 11.0,
        "legend": 12.0,
    }
    assert left_axis.title.get_fontsize() == pytest.approx(16.0)
    assert left_axis.xaxis.label.get_fontsize() == pytest.approx(13.0)
    assert left_axis.yaxis.label.get_fontsize() == pytest.approx(14.0)
    assert right_axis.yaxis.label.get_fontsize() == pytest.approx(15.0)
    assert {tick.get_fontsize() for tick in left_axis.get_xticklabels()} == {9.0}
    assert {tick.get_fontsize() for tick in left_axis.get_yticklabels()} == {10.0}
    assert {tick.get_fontsize() for tick in right_axis.get_yticklabels()} == {11.0}
    assert {text.get_fontsize() for text in left_axis.get_legend().get_texts()} == {12.0}
    assert left_axis.get_legend()._ncols == 2
    assert all(
        spine.get_linewidth() == pytest.approx(2.0)
        for axis in figure.axes
        for spine in axis.spines.values()
    )
    assert figure.subplotpars.top == pytest.approx(0.96)
    original_close(figure)


def test_broken_axis_round_trips_and_renders_two_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = FigureSpec(
        title="Broken reward axis",
        x_label="Training step",
        left=FigureAxis(
            label="Reward",
            segments=[[0.0, 0.2], [0.8, 1.0]],
            segment_height_ratios=[2.0, 1.0],
        ),
        series=[
            FigureSeries(
                label="run",
                points=[[0.0, 0.1], [1.0, 0.9]],
                color="C0",
            )
        ],
        smoothing_window=1,
        break_gap=0.08,
        break_mark_size=0.015,
        border_width=2.0,
        top_margin=0.04,
        font_sizes=FigureFontSizes(
            title=16.0,
            x_label=13.0,
            left_y_label=14.0,
            x_tick=9.0,
            left_y_tick=10.0,
        ),
    )
    write_figure_data([("broken", spec)], tmp_path)
    loaded = read_spec(tmp_path / "broken.json")
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: None)

    path = render_figure(loaded, tmp_path, "broken", "png")
    figure = plots.plt.gcf()

    assert loaded == spec
    assert path.stat().st_size > 0
    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylim() == pytest.approx((0.8, 1.0))
    assert figure.axes[1].get_ylim() == pytest.approx((0.0, 0.2))
    assert figure.axes[0].spines["bottom"].get_visible() is False
    assert figure.axes[1].spines["top"].get_visible() is False
    assert figure.axes[0].title.get_fontsize() == pytest.approx(16.0)
    assert figure.axes[1].xaxis.label.get_fontsize() == pytest.approx(13.0)
    assert {tick.get_fontsize() for tick in figure.axes[1].get_xticklabels()} == {9.0}
    assert all(
        {tick.get_fontsize() for tick in axis.get_yticklabels()} == {10.0} for axis in figure.axes
    )
    assert next(text for text in figure.texts if text.get_text() == "Reward").get_fontsize() == (
        pytest.approx(14.0)
    )
    assert all(
        spine.get_linewidth() == pytest.approx(2.0)
        for axis in figure.axes
        for spine in axis.spines.values()
    )
    break_marks = [line for axis in figure.axes for line in axis.lines if not line.get_clip_on()]
    assert len(break_marks) == 4
    assert {line.get_linewidth() for line in break_marks} == {2.0}
    assert figure.subplotpars.top == pytest.approx(0.96)
    original_close(figure)


def test_matching_dual_y_breaks_render_both_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segments = [[0.0, 0.2], [0.8, 1.0]]
    spec = FigureSpec(
        title="Dual broken axes",
        x_label="Training step",
        left=FigureAxis(label="Left", segments=segments),
        right=FigureAxis(label="Right", segments=[[0.0, 20.0], [80.0, 100.0]]),
        series=[
            FigureSeries(label="left", points=[[0.0, 0.1]], color="C0"),
            FigureSeries(
                label="right",
                points=[[0.0, 90.0]],
                color="C1",
                axis="right",
            ),
        ],
        smoothing_window=1,
        font_sizes=FigureFontSizes(right_y_label=15.0, right_y_tick=11.0),
    )
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: None)

    path = render_figure(spec, tmp_path, "dual-broken", "png")
    figure = plots.plt.gcf()

    assert path.stat().st_size > 0
    right_axes = figure.axes[2:]
    assert all(
        {tick.get_fontsize() for tick in axis.get_yticklabels()} == {11.0} for axis in right_axes
    )
    assert next(text for text in figure.texts if text.get_text() == "Right").get_fontsize() == (
        pytest.approx(15.0)
    )
    original_close(figure)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_font_sizes_fail_fast(tmp_path: Path, value: float) -> None:
    spec = FigureSpec(
        title="invalid",
        x_label="step",
        left=FigureAxis(label="value"),
        font_sizes=FigureFontSizes(title=value),
    )

    with pytest.raises(ValueError, match="font_sizes.title"):
        write_figure_data([("invalid", spec)], tmp_path)


def test_unknown_font_size_field_fails_fast(tmp_path: Path) -> None:
    stem, spec = build_figures(_training_progress_rows())[0]
    write_figure_data([(stem, spec)], tmp_path)
    path = tmp_path / f"{stem}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["font_sizes"] = {"x_ticks": 11.0}
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields.*x_ticks"):
        read_spec(path)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_border_width_fails_fast(tmp_path: Path, value: float) -> None:
    spec = FigureSpec(
        title="invalid",
        x_label="step",
        left=FigureAxis(label="value"),
        border_width=value,
    )

    with pytest.raises(ValueError, match="border_width"):
        write_figure_data([("invalid", spec)], tmp_path)


@pytest.mark.parametrize("value", [-1.0, 1.0, float("nan"), float("inf")])
def test_invalid_top_margin_fails_fast(tmp_path: Path, value: float) -> None:
    spec = FigureSpec(
        title="invalid",
        x_label="step",
        left=FigureAxis(label="value"),
        top_margin=value,
    )

    with pytest.raises(ValueError, match="top_margin"):
        write_figure_data([("invalid", spec)], tmp_path)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_legend_column_count_fails_fast(tmp_path: Path, value: object) -> None:
    spec = FigureSpec(
        title="invalid",
        x_label="step",
        left=FigureAxis(label="value"),
        legend=FigureLegend(entries=[], ncol=value),
    )

    with pytest.raises(ValueError, match="legend.ncol"):
        write_figure_data([("invalid", spec)], tmp_path)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            FigureSpec(
                title="invalid",
                x_label="step",
                left=FigureAxis(
                    label="value",
                    limits=[0.0, 1.0],
                    segments=[[0.0, 0.2], [0.8, 1.0]],
                ),
            ),
            "both limits and segments",
        ),
        (
            FigureSpec(
                title="invalid",
                x_label="step",
                left=FigureAxis(label="value", segments=[[0.0, 0.6], [0.5, 1.0]]),
            ),
            "ascending and separated",
        ),
        (
            FigureSpec(
                title="invalid",
                x_label="step",
                left=FigureAxis(
                    label="value",
                    segments=[[0.0, 0.2], [0.8, 1.0]],
                    segment_height_ratios=[1.0],
                ),
            ),
            "one value per segment",
        ),
        (
            FigureSpec(
                title="invalid",
                x_label="step",
                left=FigureAxis(label="left", segments=[[0.0, 0.2], [0.8, 1.0]]),
                right=FigureAxis(label="right"),
            ),
            "same number of segments",
        ),
    ],
)
def test_invalid_broken_axis_layouts_fail_fast(
    tmp_path: Path, spec: FigureSpec, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        write_figure_data([("invalid", spec)], tmp_path)


def test_figure_data_leads_with_each_line_and_its_points(tmp_path: Path) -> None:
    """The on-disk structure should read like the figure, not analysis rows."""
    stem, spec = build_figures(_training_progress_rows())[0]

    write_figure_data([(stem, spec)], tmp_path)
    raw = json.loads((tmp_path / f"{stem}.json").read_text(encoding="utf-8"))

    assert list(raw)[:3] == ["spec_version", "title", "x_label"]
    assert raw["series"]
    assert list(raw["series"][0])[:3] == ["label", "points", "color"]
    assert all(len(point) == 2 for series in raw["series"] for point in series["points"])
    assert "rows" not in raw
    assert "metric" not in raw["series"][0]


def test_regeneration_cleanup_removes_only_superseded_aggregate_data(tmp_path: Path) -> None:
    for name in analyze.LEGACY_DATA_FILES:
        (tmp_path / name).write_text("legacy", encoding="utf-8")
    unrelated = tmp_path / "keep.json"
    unrelated.write_text("keep", encoding="utf-8")

    removed = analyze._remove_legacy_data_files(tmp_path)

    assert removed == list(analyze.LEGACY_DATA_FILES)
    assert all(not (tmp_path / name).exists() for name in analyze.LEGACY_DATA_FILES)
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_every_image_is_recoverable_from_its_data_alone(tmp_path: Path) -> None:
    """Redrawing from the specs alone must reproduce the images byte for byte.

    This is what the data files are for: an image and the description of it
    travel together, so nothing upstream of them has to be re-derived.
    """
    outputs = build_figures(_training_progress_rows())
    write_figure_data(outputs, tmp_path)
    for stem, spec in outputs:
        render_figure(spec, tmp_path, stem, "png")
    before = {stem: (tmp_path / f"{stem}.png").read_bytes() for stem, _ in outputs}

    for path in tmp_path.rglob("*.png"):
        path.unlink()
    for stem, _ in outputs:
        render_figure(read_spec(tmp_path / f"{stem}.json"), tmp_path, stem, "png")

    assert {stem: (tmp_path / f"{stem}.png").read_bytes() for stem, _ in outputs} == before


def test_stale_figure_data_is_rejected_rather_than_redrawn(tmp_path: Path) -> None:
    """A spec from another version would redraw a figure that no longer matches."""
    stem, spec = build_figures(_training_progress_rows())[0]
    write_figure_data([(stem, spec)], tmp_path)
    path = tmp_path / f"{stem}.json"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'"spec_version": {SPEC_VERSION}', '"spec_version": 0'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="spec_version"):
        read_spec(path)


def test_render_figures_writes_one_image_per_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyze.os, "cpu_count", lambda: 1)
    config = AnalysisConfig(output_dir=str(tmp_path))
    outputs = build_figures(_training_progress_rows())

    analyze._render_figures(config, outputs, tmp_path)

    for stem, _ in outputs:
        assert (tmp_path / f"{stem}.png").stat().st_size > 0


def test_render_figures_spreads_specs_over_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyze.os, "cpu_count", lambda: 2)
    config = AnalysisConfig(output_dir=str(tmp_path))
    outputs = build_figures(_training_progress_rows())

    analyze._render_figures(config, outputs, tmp_path)

    for stem, _ in outputs:
        assert (tmp_path / f"{stem}.png").stat().st_size > 0


def test_figure_index_refuses_to_guess_what_to_redraw(tmp_path: Path) -> None:
    """Reuse reads the recorded index, so a deleted spec is reported."""
    with pytest.raises(ValueError, match="lists no figures"):
        analyze._figure_stems({}, tmp_path)

    with pytest.raises(FileNotFoundError, match="Figure data missing"):
        analyze._figure_stems({"figures": ["ocr/missing"]}, tmp_path)


def test_figure_data_rejects_duplicate_output_names(tmp_path: Path) -> None:
    stem, spec = build_figures(_training_progress_rows())[0]

    with pytest.raises(ValueError, match="duplicate output stems"):
        write_figure_data([(stem, spec), (stem, spec)], tmp_path)


def _agreement_count_rows(steps: int = 24) -> list[dict]:
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


def test_agreement_count_figure_maps_every_bin_and_keeps_handles_marker_free() -> None:
    """Bins that no run populates are dropped, and legend handles stay bare.

    Series sharing one agreeing count also share a color, so the dash pattern is
    the only cue separating the runs: the drawn series keep sparse markers, and
    the handles carry none, since a handle's centre marker would cover the single
    dash gap that fits inside a short handle.
    """
    ((stem, spec),) = build_agreement_count_distribution_figures(_agreement_count_rows())

    assert stem == "pickscore/agreement_count"
    drawn = {series.label: series for series in spec.series}
    assert set(drawn) == {
        f"{label} | c={agreeing_count}"
        for label in ("SRC-NFT", "NFT (uniform)")
        for agreeing_count in (1, 2)
    }
    assert all(series.markevery > 1 for series in spec.series)
    assert all(series.faint_raw_trace is False for series in spec.series)

    handles = {entry.label: entry for entry in spec.legend.entries}
    assert set(handles) == set(drawn)
    assert all(entry.marker == "" for entry in spec.legend.entries)
    assert spec.legend.handlelength == 2.8
    for agreeing_count in (1, 2):
        styles = {
            entry.linestyle
            for entry in spec.legend.entries
            if entry.label.endswith(f"| c={agreeing_count}")
        }
        assert len(styles) == 2, f"runs share one dash pattern for c={agreeing_count}: {styles}"


def test_agreement_count_figures_reject_a_dataset_with_two_reward_sets() -> None:
    """A dataset fixes its reward set; mixing two would mix agreeing-count scales."""
    rows = _agreement_count_rows()
    rows.extend(
        {**row, "reward_combination": "clip_score__ocr_reward__pick_score"} for row in list(rows)
    )

    for builder in (
        build_agreement_count_distribution_figures,
        build_agreement_count_expectation_figures,
    ):
        with pytest.raises(ValueError, match="more than one reward combination"):
            builder(rows)


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


def test_training_progress_is_one_figure_per_run_beside_one_all_run_figure() -> None:
    progress = build_run_training_progress_figures(_training_progress_rows())
    concordance = build_concordance_rate_figures(_training_progress_rows())

    assert sorted(stem for stem, _ in progress) == [
        "pickscore/training_progress/NFT__uniform_",
        "pickscore/training_progress/SRC-NFT",
    ]
    assert [stem for stem, _ in concordance] == ["pickscore/training_progress/concordance"]


def test_run_progress_figure_aggregates_every_reward_into_one_curve() -> None:
    """Rewards collapse to a single curve, so the figure cannot grow with them.

    Each reward is normalized to its own range before averaging, so the aggregate
    is the mean of per-reward percentages rather than of raw levels.
    """
    rows = [
        row
        for row in _training_progress_rows()
        if row["run_label"] == "SRC-NFT" and row["metric"] != "per_reward_mean_reward"
    ]
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
        # Two rewards on scales 100x apart: averaging raw levels and averaging
        # per-reward percentages cannot both produce the same curve.
        for reward, low, high in (("clip_score", 0.0, 1.0), ("pick_score", 100.0, 200.0))
        for step in range(10)
    )

    spec = dict(build_run_training_progress_figures(rows))["pickscore/training_progress/SRC-NFT"]

    progress = spec.series[0]
    assert progress.label == "mean reward progress (left)"
    np.testing.assert_allclose([value for _, value in progress.points], 100.0 * np.arange(10) / 9.0)


def test_run_progress_figure_puts_rates_on_a_second_axis() -> None:
    """Progress keeps the anchored 0-100%; the rates float to their own band."""
    spec = dict(build_run_training_progress_figures(_training_progress_rows()))[
        "pickscore/training_progress/SRC-NFT"
    ]

    assert spec.left.limits == [0.0, 100.0]
    assert spec.left.grid is True
    assert spec.right is not None
    assert spec.right.limits is None, "the rate axis follows its own data"
    assert spec.right.grid is False, "only the anchored axis offers gridlines"
    assert {series.axis for series in spec.series} == {"left", "right"}
    # Every curve is named with the side it belongs to, so none is read against
    # the wrong scale.
    assert all(
        series.label.endswith("(left)") or series.label.endswith("(right)")
        for series in spec.series
    )


def test_concordance_figure_gives_runs_the_colour_and_direction_the_dash() -> None:
    """Runs own colour and marker; positive/negative own solid/dashed.

    Direction has only two values while runs do not, so the strongest channel
    goes to the runs and the dash pattern — the one that survives greyscale —
    carries direction.
    """
    ((stem, spec),) = build_concordance_rate_figures(_training_progress_rows())

    assert stem == "pickscore/training_progress/concordance"
    assert spec.left.limits is None, "the all-run figure autoscales to its own band"

    run_styles: dict[str, set[tuple[str, str]]] = {}
    for series in spec.series:
        run, _, direction = series.label.partition(" | ")
        assert direction, f"every legend label names its run and direction: {series.label}"
        assert series.linestyle == ("-" if direction.startswith("positive") else "--")
        run_styles.setdefault(run, set()).add((series.color, series.marker))
    assert set(run_styles) == {"SRC-NFT", "NFT (uniform)"}
    for styles in run_styles.values():
        assert len(styles) == 1, "a run keeps one colour and marker across both directions"
    assert (
        len({next(iter(styles)) for styles in run_styles.values()}) == 2
    ), "the two runs must differ in colour and marker"


def test_concordance_figure_rejects_more_runs_than_the_palette_holds() -> None:
    """Cycling colours would give two runs the same hue, which defeats the point."""
    rows = _training_progress_rows()
    rows.extend(
        {**row, "run_label": f"run_{index}"}
        for index in range(len(_CATEGORICAL_COLORS))
        for row in _training_progress_rows()
    )

    with pytest.raises(ValueError, match="validated palette"):
        build_concordance_rate_figures(rows)
