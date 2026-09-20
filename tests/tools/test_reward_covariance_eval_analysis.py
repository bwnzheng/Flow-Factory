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

"""Regression tests for fresh checkpoint reward covariance analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest
import torch

from tools.eval_reward_analysis import analyze
from tools.eval_reward_analysis.analyze import (
    AnalysisConfig,
    EvaluationConfig,
    ModelConfig,
    PromptRecord,
    RunConfig,
    SourceConfig,
    _generate_images,
    _resolve_reward_weights,
    _write_analysis_artifacts,
    load_config,
    load_prompt_records,
    run_analysis,
)
from tools.eval_reward_analysis.plots import (
    build_agreement_count_figure,
    build_covariance_figure,
    build_jsr_figure,
)
from tools.eval_reward_analysis.reward_scoring import (
    _AcceleratorView,
    _partition,
    _worker_device,
)
from tools.figures import (
    FIGURE_INDEX_NAME,
    FigureSpec,
    MatrixFigureSpec,
    read_spec,
    render_figure,
)


def test_default_config_is_weight_free_and_uses_fresh_rollouts() -> None:
    root = Path(__file__).parents[2]
    config = load_config(root / "tools/eval_reward_analysis/default.yaml")
    assert config.evaluation.num_samples_per_prompt == 16
    assert config.model.num_processes == 1
    assert config.model.device is None
    assert config.plot_format == "png"
    assert [source.name for source in config.sources] == ["pickscore", "ocr"]
    assert [source.max_prompts for source in config.sources] == [100, 100]


def test_load_config_accepts_base_model_only_run(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        """
model: {base_model: model}
evaluation: {num_samples_per_prompt: 2}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: base, label: Base, base_model_only: true}
output: {dir: output}
""",
        encoding="utf-8",
    )
    run = load_config(config_path).runs[0]
    assert run.base_model_only is True
    assert run.checkpoint is None


def test_load_config_rejects_checkpoint_and_base_model_only_together(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        """
model: {base_model: model}
evaluation: {num_samples_per_prompt: 2}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: invalid, checkpoint: checkpoint-1, base_model_only: true}
output: {dir: output}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_config(config_path)


def test_load_prompt_records_respects_source_limit(tmp_path: Path) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("zero\none\ntwo\n", encoding="utf-8")

    records = load_prompt_records(prompts_path, max_prompts=2)

    assert [record.prompt for record in records] == ["zero", "one"]


def test_multi_process_config_rejects_indexed_device(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        """
model: {base_model: model, dtype: bfloat16, device: 'cuda:2', num_processes: 2}
evaluation: {num_samples_per_prompt: 2, generation_batch_size: 1, reward_batch_size: 1, seed: 1}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: run, checkpoint: saves/run/checkpoint-1}
output: {dir: output}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="accelerator type"):
        load_config(config_path)


def test_multi_process_config_accepts_npu_device_type(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        """
model: {base_model: model, dtype: bfloat16, device: npu, num_processes: 2}
evaluation: {num_samples_per_prompt: 2, generation_batch_size: 1, reward_batch_size: 1, seed: 1}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: run, checkpoint: saves/run/checkpoint-1}
output: {dir: output}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.model.device == "npu"
    assert config.model.num_processes == 2


def test_plot_format_accepts_pdf_and_rejects_unknown_value(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        """
model: {base_model: model}
evaluation: {num_samples_per_prompt: 2}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: run, checkpoint: saves/run/checkpoint-1}
output: {dir: output, plot_format: pdf}
""",
        encoding="utf-8",
    )
    assert load_config(config_path).plot_format == "pdf"
    config_path.write_text(config_path.read_text().replace("plot_format: pdf", "plot_format: svg"))
    with pytest.raises(ValueError, match="png.*pdf"):
        load_config(config_path)


def test_partition_keeps_prompt_groups_on_one_worker() -> None:
    rows = [
        {"prompt_index": prompt, "sample_index": sample}
        for prompt in range(5)
        for sample in range(3)
    ]
    chunks = _partition(rows, num_processes=3)
    owners = {}
    for worker, chunk in enumerate(chunks):
        for row in chunk:
            owners.setdefault(row["prompt_index"], set()).add(worker)
    assert all(len(worker_ids) == 1 for worker_ids in owners.values())
    assert _worker_device("cuda", 1, 2) == "cuda:1"
    assert _worker_device("npu", 1, 2) == "npu:1"


def test_offline_accelerator_view_provides_noop_barrier() -> None:
    view = _AcceleratorView(device=torch.device("cuda:1"), local_process_index=1)
    assert view.device == torch.device("cuda:1")
    assert view.local_process_index == 1
    assert view.wait_for_everyone() is None


def test_generate_images_uses_configured_parallel_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class FakeParallelRunner:
        def __init__(self, base_model, dtype, num_processes, device):
            captured.update(
                base_model=base_model,
                dtype=dtype,
                num_processes=num_processes,
                device=device,
            )

        def close(self):
            captured["closed"] = True

    def fake_run_evaluation_set(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        row = {
            "prompt_index": 0,
            "sample_index": 0,
            "seed": 42,
            "prompt": "prompt",
            "image_path": "checkpoint_7/p0_s0.png",
        }
        (output_dir / "manifest.jsonl").write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(
        "tools.eval_reward_analysis.analyze.ParallelEvaluationRunner",
        FakeParallelRunner,
    )
    monkeypatch.setattr(
        "tools.eval_reward_analysis.analyze.run_evaluation_set",
        fake_run_evaluation_set,
    )
    config = AnalysisConfig(
        model=ModelConfig("model", "bfloat16", "cuda", 2),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=[],
        output_dir=str(tmp_path),
    )
    rows = _generate_images(
        config,
        RunConfig("run", "Run", "checkpoint-7"),
        7,
        SourceConfig("source", "prompts.txt", "prompt", 0, []),
        [PromptRecord("prompt", "{}")],
        tmp_path / "images",
    )
    assert captured == {
        "base_model": "model",
        "dtype": "bfloat16",
        "num_processes": 2,
        "device": "cuda",
        "closed": True,
    }
    assert rows[0]["prompt"] == "prompt"


def test_generate_images_passes_none_checkpoint_for_base_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, base_model, dtype, device):
            pass

        def close(self):
            pass

    def fake_run_evaluation_set(**kwargs):
        captured["checkpoints"] = kwargs["checkpoints"]
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "manifest.jsonl").write_text(
            json.dumps(
                {
                    "prompt_index": 0,
                    "sample_index": 0,
                    "seed": 42,
                    "prompt": "prompt",
                    "image_path": "checkpoint_0/p0_s0.png",
                }
            )
            + "\n"
        )

    monkeypatch.setattr("tools.eval_reward_analysis.analyze.EvaluationRunner", FakeRunner)
    monkeypatch.setattr(
        "tools.eval_reward_analysis.analyze.run_evaluation_set",
        fake_run_evaluation_set,
    )
    config = AnalysisConfig(
        model=ModelConfig("model", "bfloat16", "cpu", 1),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=[],
        output_dir=str(tmp_path),
    )
    _generate_images(
        config,
        RunConfig("base", "Base", None, True),
        0,
        SourceConfig("source", "prompts.txt", "prompt", 0, []),
        [PromptRecord("prompt", "{}")],
        tmp_path / "images",
    )
    assert captured["checkpoints"] == [(0, None)]


def test_artifacts_preserve_samples_and_prompt_local_matrices(tmp_path: Path) -> None:
    config = AnalysisConfig(
        model=ModelConfig("model", "bfloat16", "cpu", 1),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=[],
        output_dir=str(tmp_path),
    )
    run = RunConfig("run", "Run", "checkpoint-7")
    source = SourceConfig("source", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])
    prompts = [PromptRecord("prompt zero", "{}"), PromptRecord("prompt one", "{}")]
    manifest = []
    values_a = {}
    values_b = {}
    for prompt_index in range(2):
        for sample_index in range(2):
            key = f"p{prompt_index}_s{sample_index}"
            manifest.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "seed": 42 + prompt_index * 2 + sample_index,
                    "prompt": prompts[prompt_index].prompt,
                    "image_path": f"checkpoint_7/{key}.png",
                }
            )
            values_a[key] = float(prompt_index + sample_index)
            values_b[key] = float(prompt_index + sample_index * 2)
    summary = _write_analysis_artifacts(
        config,
        run,
        source,
        7,
        prompts,
        manifest,
        {"a": values_a, "b": values_b},
        tmp_path,
        {"a": 1.0, "b": 1.0},
        "run_config",
    )
    sample_rows = [
        json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()
    ]
    metric_rows = [
        json.loads(line) for line in (tmp_path / "prompt_metrics.jsonl").read_text().splitlines()
    ]
    assert len(sample_rows) == 4
    assert len(metric_rows) == 2
    np.testing.assert_allclose(metric_rows[0]["covariance"], [[0.5, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(metric_rows[0]["standardized_covariance"], [[1.0, 1.0], [1.0, 1.0]])
    assert summary["n_prompts"] == 2
    assert (tmp_path / summary["covariance_plot"]).is_file()
    assert (tmp_path / summary["agreement_count_plot"]).is_file()
    assert summary["reward_weights"] == {"a": 1.0, "b": 1.0}
    assert summary["reward_weight_source"] == "run_config"
    # Two rewards, two samples: each prompt's c = 0 bin is structurally empty.
    assert len(summary["agreement_count_distribution"]) == 3
    assert summary["agreement_count_distribution"][0] == 0.0
    assert summary["mean_agreement_count"] == pytest.approx(
        sum(
            count * fraction
            for count, fraction in enumerate(summary["agreement_count_distribution"])
        )
    )
    assert metric_rows[0]["agreement_count_distribution"][0] == 0.0
    assert metric_rows[0]["mean_agreement_count"] == pytest.approx(summary["mean_agreement_count"])


def test_artifacts_skip_agreement_statistics_without_weights(tmp_path: Path) -> None:
    # Covariance geometry needs no scalarization weights, so a run that cannot
    # resolve them still gets its covariance artifacts, just no agreeing counts.
    config = AnalysisConfig(
        model=ModelConfig("model", "bfloat16", "cpu", 1),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=[],
        output_dir=str(tmp_path),
    )
    run = RunConfig("base", "Base", None, base_model_only=True)
    source = SourceConfig("source", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])
    prompts = [PromptRecord("prompt zero", "{}")]
    manifest = []
    values = {"a": {}, "b": {}}
    for sample_index in range(2):
        key = f"p0_s{sample_index}"
        manifest.append(
            {
                "prompt_index": 0,
                "sample_index": sample_index,
                "seed": 42 + sample_index,
                "prompt": prompts[0].prompt,
                "image_path": f"checkpoint_0/{key}.png",
            }
        )
        values["a"][key] = float(sample_index)
        values["b"][key] = float(sample_index * 2)

    summary = _write_analysis_artifacts(
        config,
        run,
        source,
        0,
        prompts,
        manifest,
        values,
        tmp_path,
        None,
        "disabled",
    )

    metric_row = json.loads((tmp_path / "prompt_metrics.jsonl").read_text().splitlines()[0])
    assert "mean_agreement_count" not in metric_row
    assert "agreement_count_distribution" not in metric_row
    assert "covariance" in metric_row
    assert "mean_agreement_count" not in summary
    assert summary["reward_weights"] is None
    assert summary["reward_weight_source"] == "disabled"
    assert summary["agreement_count_plot"] is None
    assert not (tmp_path / "plots" / "agreement_count.png").exists()
    assert (tmp_path / summary["covariance_plot"]).is_file()


def _base_model_analysis_config(
    tmp_path: Path,
    figure_mode: str = "regenerate",
    checkpoint_run: Path | None = None,
    jsr: str = "",
) -> Path:
    """Write a config whose first run is the base model and needs no weights.

    ``checkpoint_run`` adds a second run at that checkpoint, which a run-based
    JSR section needs as its comparison.
    """
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("prompt zero\nprompt one\n", encoding="utf-8")
    extra_run = (
        f"  - {{name: ckpt, label: Ckpt, checkpoint: {checkpoint_run}}}\n"
        if checkpoint_run is not None
        else ""
    )
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        f"""
model: {{base_model: model, device: cpu}}
evaluation: {{num_samples_per_prompt: 2}}
sources:
  - name: test
    prompts_file: {prompts_path}
    rewards:
      - {{name: a, reward_model: A}}
      - {{name: b, reward_model: B}}
runs:
  - {{name: base, label: Base, base_model_only: true}}
{extra_run}agreement_count: {{enabled: false}}
{jsr}output: {{dir: {tmp_path / "out"}, figure_mode: {figure_mode}}}
""",
        encoding="utf-8",
    )
    return config_path


def _run_based_jsr_section() -> str:
    """A run-based JSR section comparing the base model against the checkpoint run."""
    return (
        "jsr:\n"
        "  enabled: true\n"
        "  reference_run: base\n"
        "  comparison_runs: [ckpt]\n"
        "  q_grid: [0.0, 0.5, 0.9, 1.0]\n"
    )


def _install_fake_rollouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace generation and reward scoring with deterministic stand-ins."""

    def fake_generate_images(config, run, step, source, prompt_records, image_root):
        image_root.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "seed": 42 + prompt_index * 2 + sample_index,
                "prompt": record.prompt,
                "image_path": f"checkpoint_{step}/p{prompt_index}_s{sample_index}.png",
            }
            for prompt_index, record in enumerate(prompt_records)
            for sample_index in range(2)
        ]
        (image_root / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return rows

    def fake_score_reward(**kwargs):
        return {
            f"p{row['prompt_index']}_s{row['sample_index']}": float(
                row["prompt_index"] + row["sample_index"]
            )
            for row in kwargs["manifest_rows"]
        }

    monkeypatch.setattr("tools.eval_reward_analysis.analyze._generate_images", fake_generate_images)
    monkeypatch.setattr("tools.eval_reward_analysis.analyze.score_reward", fake_score_reward)


def test_run_analysis_needs_no_weights_for_a_base_model_without_agreement_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a base-model run used to demand an explicit reward_weights
    # override even when the configured analysis never consumed one.
    _install_fake_rollouts(monkeypatch)

    summary = run_analysis(load_config(_base_model_analysis_config(tmp_path)))["experiments"][0]

    assert summary["reward_weights"] is None
    assert summary["reward_weight_source"] == "disabled"
    assert summary["agreement_count_plot"] is None
    assert "mean_agreement_count" not in summary
    assert "covariance" in summary


def test_figures_are_written_beside_their_data_and_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_rollouts(monkeypatch)
    config = load_config(_base_model_analysis_config(tmp_path))

    summary = run_analysis(config)["experiments"][0]

    plots_dir = Path(config.output_dir) / "base" / "test" / "plots"
    assert (plots_dir / "covariance_matrix.png").is_file()
    assert isinstance(read_spec(plots_dir / "covariance_matrix.json"), MatrixFigureSpec)
    index = json.loads((plots_dir / FIGURE_INDEX_NAME).read_text(encoding="utf-8"))
    assert index["figures"] == ["covariance_matrix"]
    assert index["figure_mode"] == "regenerate"
    assert index["plot_format"] == "png"
    assert summary["covariance_plot"] == "plots/covariance_matrix.png"


def test_reuse_redraws_every_figure_without_touching_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of the reuse figure mode: regenerate writes the specs, and a
    # later run redraws from them without generating images or scoring rewards.
    _install_fake_rollouts(monkeypatch)
    config = load_config(_base_model_analysis_config(tmp_path, figure_mode="reuse"))
    regenerate = load_config(_base_model_analysis_config(tmp_path))
    run_analysis(regenerate)

    plots_dir = Path(config.output_dir) / "base" / "test" / "plots"
    data_before = (plots_dir / "covariance_matrix.json").read_text(encoding="utf-8")
    (plots_dir / "covariance_matrix.png").unlink()

    def fail(*args, **kwargs):
        raise AssertionError("reuse must not generate images or score rewards")

    monkeypatch.setattr("tools.eval_reward_analysis.analyze._generate_images", fail)
    monkeypatch.setattr("tools.eval_reward_analysis.analyze.score_reward", fail)

    redrawn = analyze._redraw_figures(config)

    assert redrawn == [str(plots_dir / "covariance_matrix.png")]
    assert (plots_dir / "covariance_matrix.png").stat().st_size > 0
    assert (plots_dir / "covariance_matrix.json").read_text(encoding="utf-8") == data_before


def _write_cached_jsr_records(path: Path, offset: float) -> None:
    """Write one comparison model's per-image reward records."""
    rows = [
        {
            "prompt_id": f"p{prompt}",
            "image_id": f"i{sample}",
            "rewards": {"a": offset + 0.1 * sample + 0.05 * prompt, "b": offset + 0.2 * sample},
        }
        for prompt in range(4)
        for sample in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cached_jsr_config(tmp_path: Path, figure_mode: str = "regenerate") -> Path:
    """Write a cached-record JSR config, which needs no runs, sources, or weights."""
    reference = tmp_path / "base.jsonl"
    model = tmp_path / "src.jsonl"
    _write_cached_jsr_records(reference, 0.0)
    _write_cached_jsr_records(model, 0.3)
    config_path = tmp_path / "jsr.yaml"
    config_path.write_text(
        f"""
model: {{base_model: unused}}
evaluation: {{num_samples_per_prompt: 2}}
sources: []
runs: []
jsr:
  reference: {reference}
  models: {{src: {model}}}
  rewards: [a, b]
  q_grid: [0.0, 0.5, 0.9, 1.0]
output: {{dir: {tmp_path / "out"}, figure_mode: {figure_mode}}}
""",
        encoding="utf-8",
    )
    return config_path


def test_cached_jsr_writes_figure_data_and_redraws_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cached-record entry writes its JSR figure straight into the output
    # directory, so reuse has to look for the index exactly there.
    config = load_config(_cached_jsr_config(tmp_path))
    result = run_analysis(config)

    assert result["jsr"]["plot"] == "jsr_curves.png"
    output_root = Path(config.output_dir)
    assert (output_root / "jsr_curves.png").is_file()
    assert isinstance(read_spec(output_root / "jsr_curves.json"), FigureSpec)
    index = json.loads((output_root / FIGURE_INDEX_NAME).read_text(encoding="utf-8"))
    assert index["figures"] == ["jsr_curves"]

    (output_root / "jsr_curves.png").unlink()

    def fail(*args, **kwargs):
        raise AssertionError("reuse must not re-analyze cached records")

    monkeypatch.setattr("tools.eval_reward_analysis.analyze.analyze_cached_results", fail)
    redrawn = analyze._redraw_figures(load_config(_cached_jsr_config(tmp_path, "reuse")))

    assert redrawn == [str(output_root / "jsr_curves.png")]
    assert (output_root / "jsr_curves.png").stat().st_size > 0


def test_run_based_jsr_panels_are_redrawn_from_their_own_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every JSR panel -- one directory per source, plus "overall" when configured
    # -- carries the same figure data as a per-run figure, so reuse redraws each
    # from its own index.
    _install_fake_rollouts(monkeypatch)
    checkpoint = _checkpoint_fixture(tmp_path, {"test": {"a": 1.0, "b": 1.0}})
    run_analysis(
        load_config(
            _base_model_analysis_config(
                tmp_path, checkpoint_run=checkpoint, jsr=_run_based_jsr_section()
            )
        )
    )

    jsr_root = Path(load_config(_base_model_analysis_config(tmp_path)).output_dir) / "jsr"
    panels = sorted(path.name for path in jsr_root.iterdir() if path.is_dir())
    assert panels == ["test"]
    spec = read_spec(jsr_root / "test" / "jsr_curves.json")
    assert isinstance(spec, FigureSpec)
    assert [series.label for series in spec.series] == ["Ckpt"]
    (jsr_root / "test" / "jsr_curves.png").unlink()

    reuse = load_config(
        _base_model_analysis_config(
            tmp_path, "reuse", checkpoint_run=checkpoint, jsr=_run_based_jsr_section()
        )
    )
    redrawn = analyze._redraw_figures(reuse)

    assert str(jsr_root / "test" / "jsr_curves.png") in redrawn
    assert (jsr_root / "test" / "jsr_curves.png").stat().st_size > 0


def test_reuse_ignores_a_disabled_jsr_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_rollouts(monkeypatch)
    run_analysis(load_config(_base_model_analysis_config(tmp_path)))

    config = load_config(
        _base_model_analysis_config(tmp_path, "reuse", jsr="jsr: {enabled: false}\n")
    )

    # Nothing was written under output/jsr, and reuse must not go looking for it.
    redrawn = analyze._redraw_figures(config)

    plots_dir = Path(config.output_dir) / "base" / "test" / "plots"
    assert redrawn == [str(plots_dir / "covariance_matrix.png")]


def test_reuse_reports_a_missing_figure_index(tmp_path: Path) -> None:
    config = load_config(_base_model_analysis_config(tmp_path, figure_mode="reuse"))

    with pytest.raises(FileNotFoundError, match=FIGURE_INDEX_NAME):
        analyze._redraw_figures(config)


def test_figure_mode_rejects_an_unknown_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="figure_mode must be one of"):
        load_config(_base_model_analysis_config(tmp_path, figure_mode="rebuild"))


def test_load_config_reads_the_agreement_count_toggle(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yaml"
    template = """
model: {{base_model: model}}
evaluation: {{num_samples_per_prompt: 2}}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {{name: a, reward_model: A}}
      - {{name: b, reward_model: B}}
runs:
  - {{name: run, base_model_only: true}}
output: {{dir: output}}
{extra}"""
    config_path.write_text(template.format(extra=""), encoding="utf-8")
    assert load_config(config_path).agreement_count == {"enabled": True}

    config_path.write_text(
        template.format(extra="agreement_count: {enabled: false}\n"), encoding="utf-8"
    )
    assert load_config(config_path).agreement_count == {"enabled": False}

    config_path.write_text(
        template.format(extra="agreement_count: {enabled: 'yes'}\n"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        load_config(config_path)


def _weights_context_payload(weights_by_source: dict[str, dict[str, float]]) -> dict:
    """Build a run_context whose reward entries carry one weight per source."""
    reward_entries = {}
    names = sorted({name for weights in weights_by_source.values() for name in weights})
    for index, name in enumerate(names):
        reward_entries[f"reward_{index}"] = {
            "name": name,
            "weight": {
                source: float(weights[name])
                for source, weights in weights_by_source.items()
                if name in weights
            },
        }
    return {"record_type": "run_context", "configuration": {"reward": reward_entries}}


def _checkpoint_fixture(
    tmp_path: Path, weights_by_source: dict[str, dict[str, float]], name: str = "run"
) -> Path:
    run_dir = tmp_path / "saves" / name
    checkpoint = run_dir / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "media.jsonl").write_text(
        json.dumps(_weights_context_payload(weights_by_source)) + "\n", encoding="utf-8"
    )
    return checkpoint


def _analysis_config(runs: list[RunConfig]) -> AnalysisConfig:
    return AnalysisConfig(
        model=ModelConfig("model", "bfloat16", "cpu", 1),
        evaluation=EvaluationConfig(2, 1, 2, 42, {}),
        sources=[],
        runs=runs,
        output_dir="output",
    )


def test_resolve_reward_weights_reads_the_training_run_context(tmp_path: Path) -> None:
    checkpoint = _checkpoint_fixture(tmp_path, {"ocr": {"a": 0.2, "b": 1.0}})
    run = RunConfig("run", "Run", str(checkpoint))
    source = SourceConfig("ocr", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])

    weights, origin = _resolve_reward_weights(_analysis_config([run]), run, source)

    assert weights == {"a": 0.2, "b": 1.0}
    assert origin == "saved_run_context:ocr"


def test_resolve_reward_weights_prefers_explicit_run_weights(tmp_path: Path) -> None:
    checkpoint = _checkpoint_fixture(tmp_path, {"ocr": {"a": 0.2, "b": 1.0}})
    run = RunConfig("run", "Run", str(checkpoint), reward_weights={"a": 1.0, "b": 1.0})
    source = SourceConfig("ocr", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])

    weights, origin = _resolve_reward_weights(_analysis_config([run]), run, source)

    assert weights == {"a": 1.0, "b": 1.0}
    assert origin == "run_config"


def test_resolve_reward_weights_rejects_incomplete_names(tmp_path: Path) -> None:
    checkpoint = _checkpoint_fixture(tmp_path, {"ocr": {"a": 0.2, "b": 1.0}})
    run = RunConfig("run", "Run", str(checkpoint), reward_weights={"a": 1.0})
    source = SourceConfig("ocr", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])

    with pytest.raises(ValueError, match="must cover exactly"):
        _resolve_reward_weights(_analysis_config([run]), run, source)


def test_resolve_reward_weights_requires_weights_for_a_base_model_run(tmp_path: Path) -> None:
    run = RunConfig("base", "Base", None, base_model_only=True)
    source = SourceConfig("ocr", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])

    with pytest.raises(ValueError, match="evaluates the base model"):
        _resolve_reward_weights(_analysis_config([run]), run, source)


def test_resolve_reward_weights_requires_the_source_in_the_run_context(tmp_path: Path) -> None:
    checkpoint = _checkpoint_fixture(tmp_path, {"pickscore": {"a": 1.0, "b": 1.0}})
    run = RunConfig("run", "Run", str(checkpoint))
    source = SourceConfig("ocr", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])

    with pytest.raises(ValueError, match="no saved weights for source"):
        _resolve_reward_weights(_analysis_config([run]), run, source)


def test_resolve_reward_weights_inherits_for_a_base_model_run(tmp_path: Path) -> None:
    # A base-model run borrows the weights its compared checkpoints trained with,
    # so its agreeing counts stay on the same axis without a hand-copied override.
    weights = {"ocr": {"a": 0.2, "b": 1.0}}
    first = RunConfig("first", "First", str(_checkpoint_fixture(tmp_path, weights, name="first")))
    second = RunConfig(
        "second", "Second", str(_checkpoint_fixture(tmp_path, weights, name="second"))
    )
    base = RunConfig("base", "Base", None, base_model_only=True)
    source = SourceConfig("ocr", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])

    resolved, origin = _resolve_reward_weights(
        _analysis_config([base, first, second]), base, source
    )

    assert resolved == {"a": 0.2, "b": 1.0}
    assert origin == "inherited_run_context:first,second"


def test_resolve_reward_weights_rejects_disagreeing_comparison_runs(tmp_path: Path) -> None:
    first = RunConfig(
        "first", "First", str(_checkpoint_fixture(tmp_path, {"ocr": {"a": 0.2, "b": 1.0}}, "first"))
    )
    second = RunConfig(
        "second",
        "Second",
        str(_checkpoint_fixture(tmp_path, {"ocr": {"a": 0.5, "b": 1.0}}, "second")),
    )
    base = RunConfig("base", "Base", None, base_model_only=True)
    source = SourceConfig("ocr", "prompts.txt", "prompt", 0, [{"name": "a"}, {"name": "b"}])

    with pytest.raises(ValueError, match="disagree on the scalarization weights"):
        _resolve_reward_weights(_analysis_config([base, first, second]), base, source)


def test_parse_run_reads_reward_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    template = """
model: {base_model: model}
evaluation: {num_samples_per_prompt: 2}
sources:
  - name: test
    prompts_file: prompts.txt
    rewards:
      - {name: a, reward_model: A}
      - {name: b, reward_model: B}
runs:
  - {name: run, base_model_only: true, reward_weights: {a: 1.0, b: 0.25}}
output: {dir: output}
"""
    config_path.write_text(template, encoding="utf-8")
    assert load_config(config_path).runs[0].reward_weights == {"a": 1.0, "b": 0.25}

    config_path.write_text(template.replace("{a: 1.0, b: 0.25}", "{a: -1.0}"), encoding="utf-8")
    with pytest.raises(ValueError, match="must be finite and positive"):
        load_config(config_path)


def test_agreement_count_plot_writes_one_run_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels: list[str] = []
    original = matplotlib.axes.Axes.text

    def capturing_text(self, x, y, text, *args, **kwargs):
        labels.append(str(text))
        return original(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", capturing_text)

    stem, spec = build_agreement_count_figure(
        [0.0, 0.183, 0.421, 0.396],
        title="Agreement count: SRC checkpoint-60 (ocr)",
    )
    render_figure(spec, tmp_path / "plots", stem, "png")

    assert (tmp_path / "plots" / "agreement_count.png").stat().st_size > 0
    # One value label per drawn bar; the structurally empty c = 0 bin has none.
    assert labels == ["0.183", "0.421", "0.396"]
    assert spec.x_tick_labels == ["c=1", "c=2", "c=3"]


def test_agreement_count_plot_rejects_a_negative_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_agreement_count_figure([0.0, -0.1, 0.6, 0.5], title="bad")
