# Reward Evaluation Analysis

This tool evaluates prompt-local reward geometry and Joint Success Rate (JSR).
It supports fresh generation as well as already sampled/evaluated JSONL
records via `analyze_cached_results`.
Prompt parsing is shared through `tools.utils` with the standalone reward
evaluator; importing either tool does not load the other tool's CLI entrypoint.

```bash
python -m tools.eval_reward_analysis.analyze \
  -c tools/eval_reward_analysis/nft_src_vs_uniform.yaml
```

`model.num_processes` controls accelerator workers. `model.device: null`
auto-detects NPU, then CUDA, then CPU. Use `1` with an indexed device such as
`cuda:2` or `npu:2`. For multiple workers, use an accelerator type such as
`cuda` or `npu` and set `num_processes` no larger than the available devices.
Each prompt and all of its repeated samples stay on one reward worker.
Reward workers use spawned OS processes so CUDA/NPU runtime state is not
inherited through `fork`.

Each source has an independent `max_prompts` limit. The shipped configs use
`max_prompts: 100`; set it to `0` only when the entire prompt file should be
evaluated. Prompts are truncated in file order before inference, so generation
and every reward model process the same bounded prompt set.

The run is resumable. Existing generated images are skipped by
`tools.model_inference`; each reward has a separate JSONL cache. The output for
each `(run, source)` contains:

```text
images/manifest.jsonl
images/checkpoint_<step>/*.png
reward_scores/<reward>.jsonl
samples.jsonl
prompt_metrics.jsonl
plots/covariance_matrix.<plot_format>
plots/covariance_matrix.json
plots/figures.json
summary.json
```

Every figure is written twice: as an image, and as the spec that produced it.
Nothing draws from the metrics directly, so `plots/<stem>.json` is a complete
description of its image, and `plots/figures.json` lists which figures the last
run wrote. That is what makes `output.figure_mode: reuse` possible -- it reads
each directory's index and redraws those specs, skipping generation, reward
scoring, and metric computation entirely. Override the config per invocation
with `--figures reuse`:

```bash
python -m tools.eval_reward_analysis.analyze \
  -c tools/eval_reward_analysis/jsr_runs.yaml --figures reuse
```

This mode governs the figure stage only. Generated images and reward scores are
always reused when they exist, whatever it is set to, and a missing figure index
is a hard error rather than a silent regeneration. The figure specs, renderer,
and this reuse workflow are shared with `tools/train_reward_analysis` through
`tools.figures`; the train tool calls the same stage `cache_mode`.

`plots/agreement_count.<plot_format>` and its spec additionally show that run's
agreeing-count distribution, unless `agreement_count.enabled: false`: the
fraction of its fresh samples whose count of rewards pointing the same way as
the weighted scalar equals each value from 1 to ``n_rewards``. The count is
computed exactly as the training-side reward-concordance tool computes it, so
fresh-sample numbers here and training-batch numbers there share one definition.
Figures stay inside each run's own directory; compare runs by reading their
`summary.json` (`mean_agreement_count`, `fully_concordant_sample_rate`) side by
side.

Runs may evaluate the base model directly by setting `base_model_only: true`
instead of `checkpoint`. Base-model samples use `checkpoint_0/` and are marked
as `base_model` in the manifest and JSON artifacts.

`samples.jsonl` records every image path, prompt, seed, sample index, and reward
vector. `prompt_metrics.jsonl` records the full reward matrix, unbiased sample
covariance, Pearson correlation, negative-pair ratio, and mean negative
correlation, and standardized covariance for every prompt. Standardized
covariance is computed after z-scoring each reward within each prompt, so it is
numerically equivalent to the prompt-local Pearson correlation matrix. The
`plots/covariance_matrix.<plot_format>` heatmap uses the prompt-macro-averaged
standardized covariance matrix; raw covariance remains available in the JSONL
artifacts. Set `output.plot_format` to `png` (default) or `pdf` to choose the
output format; it applies to every figure, and the JSR curves use it too.

Covariance, correlation, negative-pair, and JSR metrics use no scalarization
weights. Only the agreement-count statistics do, because they compare each
reward against the weighted scalar, which is why `plots/agreement_count.*` and
the three agreeing-count keys in the JSONL artifacts are the only outputs
missing when `agreement_count.enabled: false` is set.

When the agreement counts do run, weights resolve per run and per source in this
order: an explicit `runs[].reward_weights`, which must cover exactly that
source's rewards; the run's own `logs/media.jsonl` run_context (the same source
the training-side analysis reads), matched by source name; and, for a
`base_model_only` run, the run contexts of the configured checkpoint runs, which
must agree with each other. A base-model run has no training log of its own, so
inheriting the weights of the runs it is compared against is what keeps its
counts on the same axis; because the run context stores weights per source, this
works for a source whose reward list differs from its siblings'. Every
`summary.json` records `reward_weights` and `reward_weight_source` for auditing,
and records `null` with a `disabled` source when the counts are off.

Because fresh rollouts are not shaped by the training-time sample selector, the
agreement-count numbers here are the ones that show what fine-tuning moved; the
training-side tool's figures describe the selected batch instead.

Use separate `output.cache_dir` and `output.jsr_dir` values when running the
run-based workflow. `cache_dir` contains generated images, manifests, reward
caches, and per-run summaries; `jsr_dir` contains only JSR result tables and
plots. The legacy `output.dir` key remains an alias for `cache_dir`. Each JSR
directory holds one `jsr_curves` figure with its spec and index, so reuse
redraws those curves from the same kind of data file as the per-run figures.

Where those JSR directories are depends on which JSR entry the config selects.
The run-based entry writes one directory per source under `output.jsr_dir`, and
an `overall` directory when `jsr.overall` is set; without `jsr_dir` that root is
`<output.dir>/jsr`. The cached-record entry, which takes `jsr.reference` and
`jsr.models` instead, computes a single curve from existing JSONL records and
writes its figure directly into `output.jsr_dir`, or into `output.dir` when
`jsr_dir` is unset. Reuse looks in exactly the place the selected entry writes.

Cached JSR records contain `prompt_id` (or `prompt_index`), `image_id` (or
`sample_index`), and a `rewards` mapping. The reference file is shared by all
comparison models. Thresholds use a prompt-equally-weighted empirical
generalized inverse (`q=0` is `-inf`), and joint success uses inclusive `>=`.

```python
from tools.eval_reward_analysis.jsr import analyze_cached_results

result = analyze_cached_results(
    "eval/base/samples.jsonl",
    {"baseline": "eval/baseline/samples.jsonl", "src": "eval/src/samples.jsonl"},
    rewards=["pick_score", "clip_score"], q_grid=[0, .5, .8, .9, 1],
    bootstrap_replicates=2000, bootstrap_seed=123,
)
```

For normal use, prefer the run-based configuration in
`jsr_runs.yaml`: list the shared reference and comparison names in the
`jsr` section, and define their checkpoints in the existing `runs` section.
Note that this is a separate entry from the cached-record workflow above, and
the two take different `jsr` sections: a `jsr.reference` path plus a `jsr.models`
mapping feeds existing JSONL records and needs no `runs`, no generation, and no
reward weights, while `jsr.reference_run` plus `jsr.comparison_runs` generates
fresh rollouts from the configured `runs` and resolves weights for the
agreeing counts. The `--cached-reference`/`--cached-model` CLI flags are the same
cached-record path.
The tool stores each run under `output.dir/<run>/<source>/`; if its manifest,
reward caches, or `samples.jsonl` are absent/incomplete, generation and scoring
resume automatically. Set `covariance.enabled: false` to skip covariance
artifacts while retaining the same run caches.
