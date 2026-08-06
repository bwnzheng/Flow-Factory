# Model Inference

This module owns reusable SD3.5 LoRA checkpoint inference for offline analysis tools. It
can be imported from Python or executed directly over a TXT/JSONL evaluation set. Outputs
are resumable: existing image slots are skipped, and `manifest.jsonl` records each prompt,
seed, checkpoint, and relative image path.

## Standalone usage

Copy and edit `default.yaml`. All inference settings live in YAML:

```yaml
model:
  base_model: "stabilityai/stable-diffusion-3.5-medium"
  dtype: "bfloat16"
  device: "npu"                   # null auto-detects NPU, CUDA, then CPU
  num_processes: 4

checkpoint:
  dir: "saves/<run>/checkpoints"  # Or set path and optional step
  path: null
  step: null

evaluation:
  dataset: "dataset/pickscore/test.txt"
  prompt_key: "prompt"
  max_prompts: 0
  num_samples: 4
  batch_size: 16
  seed: 42

generation:
  num_inference_steps: 50
  guidance_scale: 1.0
  height: 512
  width: 512

output:
  dir: "analysis_output/model_inference/<run>"
```

Run the configured inference job:

```bash
python -m tools.model_inference -c tools/model_inference/default.yaml
```

The command line only selects the YAML file. `generation` fields are forwarded directly to
the diffusers pipeline, so model-specific pipeline options can be added without changing the
CLI. For one checkpoint, set `checkpoint.path`; when its directory is not named
`checkpoint-N`, also set `checkpoint.step`. Each parallel worker loads one base pipeline and
receives a deterministic subset of missing image slots.

## Python API

```python
from tools.model_inference import (
    EvaluationRunner,
    discover_checkpoints,
    load_evaluation_prompts,
    run_evaluation_set,
)

checkpoints = discover_checkpoints("saves/<run>/checkpoints")
prompts = load_evaluation_prompts("dataset/pickscore/test.txt")
runner = EvaluationRunner(
    "stabilityai/stable-diffusion-3.5-medium",
    "bfloat16",
    device="cuda",
)
try:
    run_evaluation_set(
        runner,
        checkpoints,
        prompts,
        "analysis_output/model_inference/<run>",
        generation_kwargs={"num_inference_steps": 50, "guidance_scale": 1.0},
    )
finally:
    runner.close()
```
