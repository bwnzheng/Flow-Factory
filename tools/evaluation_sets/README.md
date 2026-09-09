# Eval-only prompt sets

## DrawBench

Materialize the canonical DrawBench prompt set from the Google Research
spreadsheet export:

```bash
python -m tools.evaluation_sets.download_drawbench
```

The command writes `dataset/drawbench/test.jsonl` containing 200 records with
`prompt`, `category`, and `drawbench_index`. The downloader validates the
canonical size (200 prompts across 11 categories) before writing. No training
split is created.

Use the same directory in standalone reward evaluation:

```yaml
sources:
  - name: drawbench
    dataset_dir: "dataset/drawbench"
    split: "test"
    prompt_key: "prompt"
    max_prompts: 0
    rewards:
      - {name: "pick_score", reward_model: "pickscore"}
      - {name: "hpsv2", reward_model: "hpsv2"}
      - {name: "clip_score", reward_model: "clip"}
```

Use it in training-time evaluation without adding it to the training sampler:

```yaml
data:
  datasets:
    - name: drawbench
      dataset_dir: "dataset/drawbench"
      eval: {}
```

For selective reward routing, set `applicable_datasets: ["drawbench"]` on the
reward configuration. DrawBench is a prompt benchmark, not a reference-image
dataset; compare generated images with fixed seeds and retain the category for
per-category reporting.

