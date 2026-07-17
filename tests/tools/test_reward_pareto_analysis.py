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

"""Regression tests for group-aware Pareto convexity analysis."""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize

from tools.reward_pareto_analysis.analyze import (
    AnalysisConfig,
    _parse_config,
    _validate_config,
)
from tools.reward_pareto_analysis.plots import (
    _adaptive_metric_ylim,
    _compute_convexity_metrics,
    _convexified_hypervolume,
    _hypervolume,
    _is_convex_supported,
    _metric_summaries,
    _metric_summary,
    _resolve_worker_count,
    plot_distribution_1d,
    plot_pareto_convexity_metrics,
    plot_reward_percentiles,
)
from tools.reward_pareto_analysis.reward_logs import load_train_rewards


def _write_train_pickle(path: Path, step: int, partial_ocr: bool = False) -> None:
    ocr_missing = np.array([np.nan, 0.4]) if partial_ocr else np.array([np.nan, np.nan])
    data = {
        "step": step,
        "prompts": ["ocr prompt", "general prompt"],
        "clip_score": [np.array([0.2, 0.8]), np.array([0.3, 0.7])],
        "ocr_reward": [np.array([0.1, 0.9]), ocr_missing],
        "pick_score": [np.array([0.7, 0.4]), np.array([0.2, 0.9])],
    }
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def test_train_reader_partitions_complete_group_reward_combinations(tmp_path: Path) -> None:
    _write_train_pickle(tmp_path / "train_step_000000.pkl", step=0)
    _write_train_pickle(tmp_path / "train_step_000001.pkl", step=1)

    combinations, reward_names = load_train_rewards(str(tmp_path))

    assert reward_names == ["clip_score", "ocr_reward", "pick_score"]
    assert set(combinations) == {
        ("clip_score", "ocr_reward", "pick_score"),
        ("clip_score", "pick_score"),
    }
    for step_data in combinations.values():
        assert sorted(step_data) == [0, 1]
        assert all(record["n_groups"] == 1 for record in step_data.values())
        assert all(record["points"].shape[0] == 2 for record in step_data.values())


def test_train_reader_rejects_partial_group_reward_missingness(tmp_path: Path) -> None:
    _write_train_pickle(tmp_path / "train_step_000000.pkl", step=0, partial_ocr=True)

    with pytest.raises(ValueError, match="Partially missing reward"):
        load_train_rewards(str(tmp_path))


def test_exact_convexified_hypervolume_uses_all_box_vertices() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ref = np.zeros(3)

    assert _hypervolume(points, ref) == pytest.approx(0.0)
    assert _convexified_hypervolume(points, ref) == pytest.approx(1.0 / 6.0)


def _inclusion_exclusion_hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    volume = 0.0
    for size in range(1, len(points) + 1):
        sign = 1.0 if size % 2 else -1.0
        for subset in itertools.combinations(points, size):
            upper = np.min(np.asarray(subset), axis=0)
            volume += sign * float(np.prod(np.maximum(upper - ref, 0.0)))
    return volume


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_hso_matches_inclusion_exclusion_oracle(dimension: int) -> None:
    rng = np.random.default_rng(1200 + dimension)
    points = rng.uniform(0.05, 1.0, size=(6, dimension))
    ref = np.zeros(dimension)

    expected = _inclusion_exclusion_hypervolume(points, ref)

    assert _hypervolume(points, ref) == pytest.approx(expected, abs=1e-12)


def test_hypervolume_is_unchanged_by_dominated_and_duplicate_points() -> None:
    pareto = np.array([[0.3, 0.9, 0.5], [0.8, 0.4, 0.7]])
    augmented = np.vstack([pareto, pareto[0], [0.2, 0.3, 0.4]])
    ref = np.zeros(3)

    assert _hypervolume(augmented, ref) == pytest.approx(_hypervolume(pareto, ref))


def test_convexity_metrics_detect_convexly_dominated_pareto_point() -> None:
    # All three points are discretely Pareto-optimal, but the middle point is
    # dominated by a convex combination of the two endpoints.
    points = np.array([[0.0, 1.0], [0.5, 0.4], [1.0, 0.0]])

    metrics = _compute_convexity_metrics(points, np.zeros(2))

    assert metrics["pareto_size"] == 3
    assert metrics["supported_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["hv_gap_abs"] >= 0.0
    assert 0.0 <= metrics["hv_gap_ratio"] <= 1.0


def test_convex_supported_ratio_is_stable_under_large_reward_translation() -> None:
    points = np.array([[0.0, 1.0], [0.5, 0.4], [1.0, 0.0]]) + 1e10

    metrics = _compute_convexity_metrics(points, np.zeros(2))

    assert metrics["supported_ratio"] == pytest.approx(2.0 / 3.0)


def test_convex_supported_ratio_handles_tightly_clustered_4d_points() -> None:
    points = np.random.default_rng(109).random((16, 4)) * 1e-6

    metrics = _compute_convexity_metrics(points, np.zeros(4))

    assert 0.0 <= metrics["supported_ratio"] <= 1.0


def test_convex_supported_ratio_handles_reported_scipy_115_failure() -> None:
    points = np.array(
        [
            [0.78208486, 0.62266379, 0.42857143, 0.55956669],
            [0.69408067, 0.62844588, 0.92857140, 0.53425255],
            [0.57255107, 0.52255244, 0.07142857, 0.53118092],
            [0.68988999, 0.59253365, 0.42857143, 0.62559438],
            [0.59350445, 0.55917003, 0.00000000, 0.44955906],
            [0.71503405, 0.62522773, 0.64285713, 0.57689646],
            [0.76951283, 0.61847272, 0.21428572, 0.56023250],
            [0.74855946, 0.66028372, 0.50000000, 0.60026096],
            [0.70246202, 0.63647396, 0.85714287, 0.58783220],
            [0.72341540, 0.63554302, 0.92857140, 0.51289711],
            [0.71922472, 0.63960493, 0.71428573, 0.56751478],
            [0.76532216, 0.62416492, 1.00000000, 0.58798347],
            [0.78208486, 0.63053714, 0.00000000, 0.58225677],
            [0.74436878, 0.62446060, 0.78571427, 0.51523338],
            [0.79465689, 0.62963261, 0.00000000, 0.56674216],
            [0.77370351, 0.64595874, 0.00000000, 0.57413814],
        ]
    )

    metrics = _compute_convexity_metrics(points, np.zeros(4))

    assert metrics["pareto_size"] == 14
    assert 0.0 <= metrics["supported_ratio"] <= 1.0


def test_convex_support_retries_unknown_highs_status(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.array([[0.0, 1.0], [1.0, 0.0]])
    real_linprog = scipy.optimize.linprog
    methods: list[str] = []

    def flaky_linprog(*args: object, method: str, **kwargs: object) -> object:
        methods.append(method)
        if method == "highs-ds":
            return scipy.optimize.OptimizeResult(
                success=False,
                status=4,
                message="HiGHS model status is unknown",
            )
        return real_linprog(*args, method=method, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", flaky_linprog)

    assert _is_convex_supported(points[0], points)
    assert methods == ["highs-ds", "highs-ipm"]


@pytest.mark.parametrize("metric", ["hv_gap_ratio", "supported_ratio"])
def test_ratio_plot_limits_adapt_to_drawn_statistics(metric: str) -> None:
    rows = [
        {"step": step, metric: value}
        for step, values in enumerate(([0.002, 0.004, 0.006], [0.005, 0.007, 0.009]))
        for value in values
    ]

    summary = _metric_summary(rows, metric)
    lower, upper = _adaptive_metric_ylim(summary, metric)

    _, means, medians, q25s, q75s = summary
    drawn = np.concatenate([means, medians, q25s, q75s])
    assert 0.0 <= lower < drawn.min()
    assert drawn.max() < upper <= 1.0
    assert upper - lower < 0.1


def test_ratio_plot_limits_reject_out_of_domain_statistics() -> None:
    rows = [{"step": 0, "hv_gap_ratio": 1.01}]

    with pytest.raises(ValueError, match="fall outside"):
        _adaptive_metric_ylim(_metric_summary(rows, "hv_gap_ratio"), "hv_gap_ratio")


def test_ratio_plot_limits_add_headroom_at_upper_boundary() -> None:
    rows = [{"step": 0, "supported_ratio": 1.0} for _ in range(4)]

    lower, upper = _adaptive_metric_ylim(
        _metric_summary(rows, "supported_ratio"), "supported_ratio"
    )

    assert lower < 1.0
    assert upper == pytest.approx(1.02)


def test_metric_summaries_group_rows_once_for_multiple_metrics() -> None:
    class CountingRows(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    rows = CountingRows(
        [
            {"step": 0, "first": 1.0, "second": 10.0},
            {"step": 0, "first": 3.0, "second": 20.0},
            {"step": 1, "first": 5.0, "second": 30.0},
        ]
    )

    summaries = _metric_summaries(rows, ["first", "second"])

    assert rows.iterations == 1
    np.testing.assert_array_equal(summaries["first"][0], [0, 1])
    np.testing.assert_allclose(summaries["first"][1], [2.0, 5.0])
    np.testing.assert_allclose(summaries["second"][1], [15.0, 30.0])


def test_worker_count_supports_auto_and_explicit_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.reward_pareto_analysis.plots.os.cpu_count", lambda: 40)

    assert _resolve_worker_count(800, 0) == 16
    assert _resolve_worker_count(8, 0) == 8
    assert _resolve_worker_count(800, 6) == 6
    assert _resolve_worker_count(3, 6) == 3
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_worker_count(10, -1)


def test_config_parses_pareto_compute_worker_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text("compute:\n  max_workers: 12\n")

    assert _parse_config(str(config_path)).max_workers == 12

    config_path.write_text("compute:\n  max_workers: -1\n")
    with pytest.raises(ValueError, match="compute.max_workers"):
        _parse_config(str(config_path))


def test_minimal_rewards_only_config_needs_no_reward_models(tmp_path: Path) -> None:
    config_path = tmp_path / "rewards_only.yaml"
    config_path.write_text("""run_name: "test_run"
save_dir: "saves"
rewards_analysis:
  enabled: true
""")

    config = _parse_config(str(config_path))

    assert config.rewards_analysis_enabled is True
    assert config.images_analysis_enabled is False
    assert config.evaluation_enabled is False
    assert config.rewards == []
    _validate_config(config)


@pytest.mark.parametrize("source", ["images_analysis_enabled", "evaluation_enabled"])
def test_image_based_sources_require_reward_models(source: str) -> None:
    config = AnalysisConfig()
    setattr(config, source, True)

    with pytest.raises(ValueError, match="rewards must be configured"):
        _validate_config(config)

    config.rewards = [{"name": "clip_score", "reward_model": "CLIP"}]
    _validate_config(config)


def test_default_config_explicitly_enables_only_saved_rewards() -> None:
    project_root = Path(__file__).parents[2]
    config = _parse_config(str(project_root / "tools" / "reward_pareto_analysis" / "default.yaml"))

    assert config.rewards_analysis_enabled is True
    assert config.images_analysis_enabled is False
    assert config.evaluation_enabled is False


def test_distribution_and_percentile_plots_are_generated(tmp_path: Path) -> None:
    all_steps = {
        step: {"points": np.asarray([[0.2 + step], [0.4 + step], [0.8 + step]])}
        for step in range(2)
    }

    distribution_path = tmp_path / "distribution_1d.png"
    percentiles_path = tmp_path / "reward_percentiles.png"
    plot_distribution_1d(all_steps, "reward", str(distribution_path))
    plot_reward_percentiles(all_steps, ["reward"], str(percentiles_path))

    assert distribution_path.stat().st_size > 0
    assert percentiles_path.stat().st_size > 0


def test_plot_writes_overview_vector_pdfs_csv_and_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    all_steps = {
        0: {
            "points": np.array([[0.0, 1.0], [0.5, 0.4], [1.0, 0.0]]),
            "prompt_idx": np.zeros(3, dtype=int),
            "step": 0,
            "n_groups": 1,
        },
        1: {
            "points": np.array([[0.0, 1.0], [0.5, 0.6], [1.0, 0.0]]),
            "prompt_idx": np.zeros(3, dtype=int),
            "step": 1,
            "n_groups": 1,
        },
    }

    output_dir = tmp_path / "clip_score__pick_score"
    for dirname, suffix in (("pdf", "pdf"), ("svg", "svg")):
        stale_dir = output_dir / "figures" / dirname
        stale_dir.mkdir(parents=True)
        (stale_dir / f"convexity_depth.{suffix}").write_text("stale")
    plot_pareto_convexity_metrics(
        all_steps,
        ["clip_score", "pick_score"],
        str(output_dir),
        title="Test Convexity",
    )

    assert (output_dir / "overview.png").stat().st_size > 0
    expected_stems = {
        "pareto_front_size",
        "convexification_hv_gap_absolute",
        "convexification_hv_gap_relative",
        "convex_supported_ratio",
    }
    pdf_dir = output_dir / "figures" / "pdf"
    assert {path.stem for path in pdf_dir.glob("*.pdf")} == expected_stems
    assert all(path.read_bytes().startswith(b"%PDF") for path in pdf_dir.glob("*.pdf"))
    svg_dir = output_dir / "figures" / "svg"
    assert {path.stem for path in svg_dir.glob("*.svg")} == expected_stems
    assert all(b"<svg" in path.read_bytes() for path in svg_dir.glob("*.svg"))
    assert all(b"<text" in path.read_bytes() for path in svg_dir.glob("*.svg"))
    metrics_csv = output_dir / "per_group_metrics.csv"
    assert metrics_csv.stat().st_size > 0
    assert "convexity_depth" not in metrics_csv.read_text().splitlines()[0]
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["metric_version"] == 3
    assert metadata["reward_names"] == ["clip_score", "pick_score"]
    assert metadata["reference_point"] == [0.0, 0.0]
    assert metadata["n_steps"] == 2
    assert metadata["compute_workers"] == 1
    assert set(metadata["interpretation"]) == {
        "pareto_size",
        "hv_gap_abs",
        "hv_gap_ratio",
        "supported_ratio",
    }
    captured = capsys.readouterr()
    assert "[Pareto] clip_score + pick_score" in captured.out
    assert "Steps: 2" in captured.out
    assert "Groups: 2" in captured.out
    assert "Samples/group: 3" in captured.out
    assert "Workers: 1 (compute.max_workers(0, auto))" in captured.out
    assert "Geometry computation:" in captured.out
    assert "Metric aggregation/export:" in captured.out
    assert "Figure generation:" in captured.out
    assert "Wrote: 1 PNG, 4 PDFs, 4 SVGs, 1 CSV, 1 JSON" in captured.out
    assert "Pareto geometry (2D)" in captured.err


def test_plot_adds_step_and_reward_context_to_geometry_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    all_steps = {
        7: {
            "points": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "prompt_idx": np.zeros(2, dtype=int),
        }
    }

    def fail_geometry(*args, **kwargs):
        raise ValueError("synthetic geometry failure")

    monkeypatch.setattr(
        "tools.reward_pareto_analysis.plots._compute_step_convexity_rows",
        fail_geometry,
    )

    with pytest.raises(
        RuntimeError,
        match=r"step\(7\).*reward_combination\(clip_score \+ pick_score\)",
    ):
        plot_pareto_convexity_metrics(
            all_steps,
            ["clip_score", "pick_score"],
            str(tmp_path / "failure"),
        )


def test_plot_skips_metrics_without_prompt_group_indices(tmp_path: Path) -> None:
    all_steps = {
        0: {
            "points": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "step": 0,
        }
    }
    output_dir = tmp_path / "ungrouped"
    for dirname in ("pdf", "svg"):
        vector_dir = output_dir / "figures" / dirname
        vector_dir.mkdir(parents=True)
        (vector_dir / "stale.txt").write_text("stale")

    plot_pareto_convexity_metrics(
        all_steps,
        ["clip_score", "pick_score"],
        str(output_dir),
    )

    assert not (output_dir / "overview.png").exists()
    assert not (output_dir / "per_group_metrics.csv").exists()
    assert not (output_dir / "figures").exists()
