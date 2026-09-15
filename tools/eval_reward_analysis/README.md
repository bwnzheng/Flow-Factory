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
summary.json
```

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
output format. No scalarization weights are used.

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
`jsr_cached.yaml`: list the shared reference and comparison names in the
`jsr` section, and define their checkpoints in the existing `runs` section.
The tool stores each run under `output.dir/<run>/<source>/`; if its manifest,
reward caches, or `samples.jsonl` are absent/incomplete, generation and scoring
resume automatically. Set `covariance.enabled: false` to skip covariance
artifacts while retaining the same run caches.
