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

Model-specific base-model presets are also provided:

```bash
python -m tools.reward_evaluation --config tools/reward_evaluation/default_sdxl.yaml
python -m tools.reward_evaluation --config tools/reward_evaluation/default_sd3_5_large.yaml
python -m tools.reward_evaluation --config tools/reward_evaluation/default_flux1_dev.yaml
```

These three presets evaluate the pure base model with PickScore, HPSv2, and CLIP on the
PickScore test source. They use model-specific starting parameters and output directories. The
full multi-source reward suite remains available in `default.yaml`.

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

Set `model.model_type` to select the image-generation backend:

- `sdxl` uses `StableDiffusionXLPipeline` (for example, SD-XL base 1.0).
- `sd3-5` uses `StableDiffusion3Pipeline` and covers SD3.5 Medium, Large, and Large Turbo;
  choose the exact variant with `model.base_model`.
- `flux1` uses `FluxPipeline` for FLUX.1-dev.

The selected checkpoint must match the selected base model. PEFT checkpoints are attached to
`unet` for SDXL and `transformer` for SD3.5/FLUX.1; Diffusers-format LoRA checkpoints use the
pipeline's native LoRA loader.

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

To evaluate the unmodified base model, use an explicit base-model-only run. This skips LoRA
loading and writes `checkpoint_path: "base_model"` in the result provenance:

```yaml
runs:
  - name: "sdxl_base"
    label: "SDXL base"
    base_model_only: true
```

The base-model run uses the same `model.model_type`, `model.base_model`, prompts, seeds, image
manifest, reward caches, and summary format as checkpoint runs. Keep it under a distinct run name
when comparing it with adapter checkpoints.

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

### VisionReward

The standalone evaluator runs VisionReward in two passes by default to avoid
keeping both heavyweight feature extractors in NPU memory:

1. `alignment` starts one worker per assigned device and loads only
   `clip-flant5-xxl` (including its CLIP vision tower). It writes
   `reward_scores/<reward>.alignment.jsonl`.
2. The alignment process pool shuts down completely. `vqa_features` then
   starts a new worker pool and loads only the SAT VisionReward checkpoint,
   its Llama-3 tokenizer, and its image/text processors. It writes the 27
   selected features to
   `reward_scores/<reward>.vqa_features.jsonl`.
3. The parent process computes `[alignment, *vqa_features] @ coef + intercept`
   in NumPy float64 on CPU and writes the normal final cache at
   `reward_scores/<reward>.jsonl`.

On non-CUDA devices such as Ascend NPU, the adapter replaces SAT's
CUDA-only Triton rotary kernel with an equivalent device-native PyTorch rotary
implementation. CUDA keeps the upstream fast kernel.

Both intermediate files are atomic, resumable caches. If the second pass is
interrupted, the next run reuses the completed alignment pass and only fills
missing VQA rows. The final score file and all downstream result formats are
unchanged. Set `staged_evaluation: false` on a VisionReward entry only when the
legacy single-worker-lifetime behavior is explicitly required; that mode loads
both components together and needs enough memory for their combined footprint.

`model.num_processes: 8` creates up to eight workers for each pass and assigns
them to `npu:0` through `npu:7`. `evaluation.reward_batch_size` controls the
number of image records handed to each adapter call, but the upstream
VisionReward implementation still performs its selected QA generations
sequentially within that call.

The VisionReward model checkpoint is not loaded directly from its Hugging Face
repository ID. The upstream SAT loader accepts a local directory containing
`model_config.json`; the released checkpoint is split into archive parts. In
the VisionReward checkout,
download the model and extract it once:

```bash
huggingface-cli download zai-org/VisionReward-Image-bf16 \
  --local-dir /path/to/VisionReward-Image-bf16
cd /path/to/VisionReward-Image-bf16
cat ckpts/split_part_* > ckpts/visionreward_image.tar
tar -xf ckpts/visionreward_image.tar
```

Then set `extra_kwargs.model_path` to the extracted directory (the directory
that contains `model_config.json`), not to
`THUDM/VisionReward-Image-bf16`. Alternatively, set
`VISIONREWARD_MODEL=/path/to/VisionReward-Image-bf16`; that environment
variable overrides the template's placeholder ID. The evaluator performs this check before
spawning workers, so a missing/incomplete checkpoint is reported once rather
than independently by every device process.

The checkpoint does not contain the Llama-3 tokenizer used by the upstream
image inference script. VisionReward's language checkpoint expects the
Llama-3 vocabulary (128,256 entries); a tokenizer from Qwen, Mistral, Llama-2,
or another family is not a drop-in replacement even if `AutoTokenizer` can
load it.

If access to the gated Meta repository is unavailable, a public technical
alternative is
[`hfl/llama-3-chinese-8b-instruct-v3`](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v3).
It exposes the standard Llama-3 tokenizer files and the expected vocabulary
metadata. [`NousResearch/Meta-Llama-3-8B-Instruct`](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct)
is another technical mirror, but its repository is explicitly marked with the
Llama-3 license. The HFL card is marked Apache-2.0, while its discussion also
documents the upstream Meta-Llama-3 lineage and license notice. Check the
repository terms, the [Meta Llama 3 Community
License](https://www.llama.com/llama3/license/), and your organization's policy
before using or redistributing either artifact; a Hugging Face license tag is
not a legal clearance.

Only the tokenizer/config files are needed (not the alternative model's
16-GB language-model weights). Download them on a networked host and transfer
the resulting directory when the evaluator host is offline:

```bash
TOKENIZER_DIR=/data/models/llama3-compatible-tokenizer
mkdir -p "$TOKENIZER_DIR"
huggingface-cli download hfl/llama-3-chinese-8b-instruct-v3 \
  --include config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir "$TOKENIZER_DIR"
```

Point the reward at that directory (not at an individual `tokenizer.json`):

```yaml
extra_kwargs:
  repo_path: /path/to/Flow-Factory/VisionReward
  model_path: /data/models/VisionReward-Image-bf16
  tokenizer_path: /data/models/llama3-compatible-tokenizer
```

For the template configs,
`VISIONREWARD_TOKENIZER=/data/models/llama3-compatible-tokenizer` is
equivalent. The evaluator also accepts a public repository ID and downloads
only the tokenizer files during preflight:

```yaml
extra_kwargs:
  repo_path: /path/to/Flow-Factory/VisionReward
  model_path: /data/models/VisionReward-Image-bf16
  tokenizer_repo: hfl/llama-3-chinese-8b-instruct-v3
  # Use false on a networked host; set true when the repo is already cached.
  tokenizer_local_files_only: false
```

`tokenizer_path: hfl/llama-3-chinese-8b-instruct-v3` is accepted as a shorter
equivalent. Set `tokenizer_revision` to pin a revision and
`tokenizer_cache_dir` to select the HF cache root. On an offline host, either
copy a complete local directory as shown above or set
`tokenizer_local_files_only: true` and make sure that cache snapshot is
already available. The evaluator validates the directory and rejects a
metadata `vocab_size` other than 128,256 before loading the large checkpoint,
so a missing or incompatible tokenizer does not make every worker load the
weights and fail later at `AutoTokenizer.from_pretrained`.
`VISIONREWARD_TOKENIZER_REPO=hfl/llama-3-chinese-8b-instruct-v3` is the
environment-variable equivalent of `tokenizer_repo`.

The official VisionReward stack pins older PyTorch/Transformers versions than
Flow-Factory. If those packages cannot coexist, run VisionReward in its own
environment behind the HTTP reward-server contract described in
`guidance/rewards.md` and configure the evaluator with
`flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel`:

```yaml
- name: "vision_reward"
  reward_model: "flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel"
  server_url: "http://127.0.0.1:18001"
  timeout: 300
  retry_attempts: 3
```

The server must expose `/health` and `/compute`; `reward_server/example_server.py`
is a protocol template. Keep `model.num_processes: 1` for one server instance,
or run one server per device/port when parallel isolated workers are needed.

## Outputs

For each `(run, source)` pair, the output directory contains:

```text
images/manifest.jsonl
images/checkpoint_<step>/*.png
reward_scores/<reward>.jsonl
reward_scores/<vision_reward>.alignment.jsonl
reward_scores/<vision_reward>.vqa_features.jsonl
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
each processed checkpoint. VisionReward's alignment cache stores one scalar
`value` per sample; its VQA cache stores a 27-element `features` list per
sample. These two files are evaluator internals and the final scalar cache
remains identical to every other reward cache.

The command prints flushed stage, reward, and worker progress to stdout. Reward
scores are written after each completed worker, so a long-running evaluation
leaves a resumable `reward_scores/<reward>.jsonl` file before the full reward
finishes. The final checkpoint and summary artifacts are still written only
after all configured rewards for that checkpoint complete.
