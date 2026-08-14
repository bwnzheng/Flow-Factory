# Fresh Reward Covariance Evaluation

This tool evaluates prompt-local reward geometry on fresh images generated
from saved LoRA checkpoints. It never reads training reward PKLs.

```bash
python -m tools.reward_covariance_eval_analysis.analyze \
  -c tools/reward_covariance_eval_analysis/nft_src_vs_uniform.yaml
```

`model.num_processes` controls accelerator workers. `model.device: null`
auto-detects NPU, then CUDA, then CPU. Use `1` with an indexed device such as
`cuda:2` or `npu:2`. For multiple workers, use an accelerator type such as
`cuda` or `npu` and set `num_processes` no larger than the available devices.
Each prompt and all of its repeated samples stay on one reward worker.
Reward workers use spawned OS processes so CUDA/NPU runtime state is not
inherited through `fork`.

The run is resumable. Existing generated images are skipped by
`tools.model_inference`; each reward has a separate JSONL cache. The output for
each `(run, source)` contains:

```text
images/manifest.jsonl
images/checkpoint_<step>/*.png
reward_scores/<reward>.jsonl
samples.jsonl
prompt_metrics.jsonl
summary.json
```

`samples.jsonl` records every image path, prompt, seed, sample index, and reward
vector. `prompt_metrics.jsonl` records the full reward matrix, unbiased sample
covariance, Pearson correlation, negative-pair ratio, and mean negative
correlation for every prompt. No scalarization weights are used.
