# Standalone Reward Evaluation

This tool generates fresh images from one or more saved checkpoints with the
shared `tools.model_inference` implementation, then evaluates each image with
the reward models named in the Flow-Factory reward registry. It is independent
of training logs and can be resumed after an interruption.

Run it with:

```bash
python -m tools.reward_evaluation \
  --config tools/reward_evaluation/default.yaml
```

The template mirrors the three evaluation sources used by the Ascend training
setup: `dataset/pickscore/test.txt`, `dataset/geneval/test.jsonl`, and
`dataset/ocr/test.txt`. PickScore, HPSv2, CLIP, UniReward, and VisionReward are
attached to every source; GenEval is attached only to the GenEval source and
OCR only to the OCR source, matching the training source boundaries. UniReward
and VisionReward are active entries in the template rather than commented
examples; set their external model/repository paths before running.

The registry lookup is case-insensitive and also accepts a fully qualified
Python class path, so a project-local reward can be evaluated without changing
this tool. The currently registered identifiers are `pickscore`,
`pickscore_rank`, `clip`, `clap`, `imagebind`, `ocr`, `ocr_v27`,
`vllm_evaluate`, `rational_rewards_t2i`, `rational_rewards_edit`, `geneval`,
`geneval_ascend`, `geneval2_soft_tifa`, `hpsv2`, `qwen_image_bench`,
`aesthetic_score`, `vision_reward`, and `unireward`.

## Configuration

The top-level sections are `model`, `evaluation`, `sources`, `runs`, and
`output`. The recommended source form mirrors the training configuration:
`dataset_dir: dataset/pickscore` plus `split: test` resolves
`dataset/pickscore/test.jsonl` or `dataset/pickscore/test.txt`, exactly like
the training eval loader. This keeps prompt ordering, metadata, and source
boundaries aligned with the training `data.datasets` entries. An explicit
`prompts_file` can still be used for a standalone prompt file, but it is
mutually exclusive with `dataset_dir`.

For JSONL, `prompt_key` selects the generation prompt and all other fields are
forwarded as the reward metadata. A reward entry requires only `name` and
`reward_model`; arbitrary additional fields are passed to `RewardArguments` and
therefore to the selected reward model.

`model.num_processes` is the number of local inference/reward workers. Use an
accelerator type (`npu` or `cuda`) when it is greater than one. The evaluator
uses spawned reward processes and the same deterministic seed/path convention
as `tools.model_inference`. The built-in UniReward backends pin their complete
model to the worker's assigned accelerator rather than using automatic
cross-device model sharding.

Each `runs` entry can select one checkpoint or an entire checkpoint directory:

```yaml
runs:
  - name: "nft_run"
    label: "NFT"
    checkpoint_dir: "saves/nft_run/checkpoints"
```

With `checkpoint_dir`, every sorted `checkpoint-N` subdirectory is evaluated.
Use `checkpoint` instead when only one checkpoint is needed. A checkpoint is
marked complete only after all configured rewards have been written to its
JSONL result file, so rerunning the same command resumes generated images and
reward caches without losing completed checkpoints.

`pickscore_rank` is also supported. It is a groupwise reward, so the evaluator
calls it once for the complete set of samples belonging to each prompt rather
than mixing samples from different prompts. Pointwise rewards are batched by
`evaluation.reward_batch_size`.

The evaluator supplies generated images and prompts. Rewards whose contract
requires audio, video, or edit-condition images cannot be meaningfully scored
from this image-only manifest; they fail with an explicit modality error rather
than silently receiving the wrong input. A modality-specific manifest can use
the same registry/scoring pattern if such evaluation is needed.

As with training, the selected reward's optional dependencies and external
weights must be installed on the evaluation host. For example, `clap` and
`imagebind` require their audio stack even though this tool only supplies the
image field; registry import failures are reported directly by the command.

## Outputs

For each `(run, source)` pair, the output directory contains:

```text
images/manifest.jsonl
images/checkpoint_<step>/*.png
reward_scores/<reward>.jsonl
checkpoint_results/checkpoint-<step>.jsonl
results.jsonl
summary.json
```

`reward_scores/<reward>.jsonl` is a resumable cache keyed by checkpoint step,
prompt index, and sample index. `checkpoint_results/checkpoint-<step>.jsonl`
is written atomically at the end of each completed checkpoint, and
`results.jsonl` is rebuilt from those completed files after every checkpoint.
Each result row joins the generated image with its checkpoint, prompt, seed,
metadata, and complete reward vector. `summary.json` reports the statistics for
each processed checkpoint.

The command prints flushed stage, reward, and worker progress to stdout. Reward
scores are written after each completed worker, so a long-running evaluation
leaves a resumable `reward_scores/<reward>.jsonl` file before the full reward
finishes. The final checkpoint and summary artifacts are still written only
after all configured rewards for that checkpoint complete.
