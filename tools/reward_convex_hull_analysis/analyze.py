#!/usr/bin/env python3
"""Reward Convex Hull Analysis Tool.

Three independent data sources:
  1. **Images analysis** — reads rollout images from ``media.jsonl``, scores them
     with reward models (CLIP, PickScore, …), caches results, and plots per-step
     convex hulls.  Supports train / eval dataset split.
  2. **Rewards analysis** — reads pre-computed scores from ``logs/rewards/*.pkl``
     directly (no image loading, no reward model), and generates distribution,
     percentile, and Pareto front plots.  Also supports train / eval split.
  3. **Evaluation inference** — loads LoRA weights from each checkpoint, generates
     fresh images on test prompts, scores them, and plots per-checkpoint hulls.

Usage::

    python tools/reward_convex_hull_analysis/analyze.py \\
        -c tools/reward_convex_hull_analysis/default_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from tools.reward_convex_hull_analysis.convex_hull import (
    plot_convex_hulls_2d,
    plot_convex_hulls_faceted,
    plot_convex_hulls_windows,
    plot_convex_hulls_windows_cumulative,
    plot_distribution_1d,
    plot_hull_area_curve,
    plot_per_group_hypervolume_and_gap,
    plot_reward_percentiles,
)
from tools.reward_convex_hull_analysis.evaluation_runner import (
    EvaluationRunner,
    MultiGPUEvaluationRunner,
    discover_checkpoints,
)
from tools.reward_convex_hull_analysis.log_reader import (
    load_media_samples,
)
from tools.reward_convex_hull_analysis.reward_computer import (
    MultiGPUComputer,
    StandaloneRewardComputer,
)
from tools.reward_convex_hull_analysis.parallel import run_tasks
from tools.reward_convex_hull_analysis.rewards_reader import (
    load_eval_rewards,
    load_train_rewards,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    # Sensible defaults for pipeline kwargs so users don't have to specify
    # every field in gen_kwargs.  Merged with YAML values in _parse_config.
    GEN_DEFAULTS: ClassVar[Dict[str, Any]] = {
        "num_inference_steps": 50,
        "guidance_scale": 1.0,
        "height": 512,
        "width": 512,
    }

    run_name: str = ""
    save_dir: str = "saves"

    images_analysis_enabled: bool = True
    images_max_images_per_step: int = 0
    evaluation_enabled: bool = True
    rewards_analysis_enabled: bool = True
    eval_checkpoint_dir: str = ""
    prompts_file: str = ""
    prompts: List[str] = field(default_factory=list)
    max_prompts: int = 0
    num_samples: int = 4
    gen_batch_size: int = 16
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)

    base_model: str = "stabilityai/stable-diffusion-3.5-medium"
    dtype: str = "bfloat16"
    device: str = "cuda"
    num_gpus: int = 1

    rewards: List[Dict[str, Any]] = field(default_factory=list)

    output_dir: str = "analysis_output"


def _parse_config(path: str) -> AnalysisConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    img = raw.get("images_analysis", {})
    ev = raw.get("evaluation", {})
    ra = raw.get("rewards_analysis", {})
    model = raw.get("model", {})
    output = raw.get("output", {})

    return AnalysisConfig(
        run_name=raw.get("run_name", ""),
        save_dir=raw.get("save_dir", "saves"),
        images_analysis_enabled=img.get("enabled", True),
        images_max_images_per_step=img.get("max_images_per_step", 0),
        evaluation_enabled=ev.get("enabled", True),
        rewards_analysis_enabled=ra.get("enabled", True),
        eval_checkpoint_dir=ev.get("checkpoint_dir", ""),
        prompts_file=ev.get("prompts_file", ""),
        prompts=ev.get("prompts", []),
        max_prompts=ev.get("max_prompts", 0),
        num_samples=ev.get("num_samples", 4),
        gen_batch_size=ev.get("gen_batch_size", 16),
        gen_kwargs={**AnalysisConfig.GEN_DEFAULTS, **ev.get("gen_kwargs", {})},
        base_model=model.get("base_model", "stabilityai/stable-diffusion-3.5-medium"),
        dtype=model.get("dtype", "bfloat16"),
        device=model.get("device", "cuda"),
        num_gpus=model.get("num_gpus", 1),
        rewards=raw.get("rewards", []),
        output_dir=output.get("dir", "analysis_output"),
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_run_name(config: AnalysisConfig) -> str:
    """Determine run name from config, auto-resolving if empty."""
    if config.run_name:
        return config.run_name
    ckpt = config.eval_checkpoint_dir
    if ckpt:
        ckpt = ckpt.rstrip("/")
        parent = os.path.dirname(ckpt)
        if os.path.basename(ckpt) == "checkpoints":
            return os.path.basename(parent)
        return os.path.basename(ckpt)
    # Try scanning saves/ for run directories that have logs/
    for name in sorted(os.listdir(config.save_dir), reverse=True):
        if os.path.isdir(os.path.join(config.save_dir, name, "logs")):
            return name
    raise ValueError("Cannot resolve run_name. Set run_name in config or provide checkpoint_dir.")


def _resolve_checkpoint_dir(config: AnalysisConfig, run_name: str) -> str:
    if config.eval_checkpoint_dir:
        return config.eval_checkpoint_dir
    return os.path.join(config.save_dir, run_name, "checkpoints")


def _load_prompts(config: AnalysisConfig) -> List[str]:
    """Load prompts from file or inline list."""
    if config.prompts:
        return list(config.prompts)
    if config.prompts_file:
        path = config.prompts_file
        if not os.path.isabs(path):
            if not os.path.isfile(path):
                path = os.path.join(os.getcwd(), path)
        if os.path.isfile(path):
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        print(f"  [WARN] prompts_file not found: {config.prompts_file}")
    return []


# ---------------------------------------------------------------------------
# Reward scoring helpers
# ---------------------------------------------------------------------------


def _score_images(
    computer: Union[StandaloneRewardComputer, MultiGPUComputer],
    images: List,
    prompts: List[str],
    batch_size: int = 16,
) -> Dict[str, List[float]]:
    if not prompts:
        prompts = ["a photo"] * len(images)
    if len(prompts) != len(images):
        prompts = [prompts[i % len(prompts)] for i in range(len(images))]
    return computer.compute(images, prompts, batch_size=batch_size)


def _build_step_data(
    step: int,
    rewards_dict: Dict[str, List[float]],
    reward_names: List[str],
    prompt_per_img: Optional[List[str]] = None,
) -> Dict[str, Any]:
    n_samples = len(rewards_dict[reward_names[0]]) if reward_names else 0
    if n_samples == 0:
        return {"points": np.empty((0, len(reward_names))), "step": step}
    pts = np.column_stack([rewards_dict[name] for name in reward_names])
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    result: Dict[str, Any] = {
        "points": pts,
        "step": step,
        "n_total": n_samples,
        "n_valid": len(pts),
    }
    if prompt_per_img and len(prompt_per_img) == n_samples:
        prompt_per_img_arr = np.array(prompt_per_img)
        unique_prompts = list(dict.fromkeys(prompt_per_img_arr))
        prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
        result["prompt_idx"] = np.array([prompt_to_idx[p] for p in prompt_per_img_arr])[mask]
        result["prompt_labels"] = unique_prompts
    return result


# ---------------------------------------------------------------------------
# Shared plot dispatch — used by both images and rewards analysis paths
# ---------------------------------------------------------------------------



def _dispatch_plots(
    all_step_data: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    out_dir: str,
    title_prefix: str,
    label_name: str = "Step",
    window_size: int = 20,
) -> None:
    """Generate plots for a single dataset/source combination.

    Uses the same dispatch logic for both images→reward_model and rewards/
    pickle data paths: ≥2 reward dims → convex hull suite, else 1-D distribution.
    """
    n_models = len(reward_names)
    if n_models >= 2:
        # Phase 1: independent 2-D plots
        run_tasks(
            [
                (
                    "convex hull overlay",
                    plot_convex_hulls_2d,
                    (all_step_data, reward_names),
                    {
                        "output_path": os.path.join(out_dir, "convex_hulls_2d.png"),
                        "title": f"{title_prefix} Reward Convex Hulls",
                        "label_name": label_name,
                    },
                ),
                (
                    "faceted hull grid",
                    plot_convex_hulls_faceted,
                    (all_step_data, reward_names),
                    {
                        "output_path": os.path.join(out_dir, "convex_hulls_faceted.png"),
                        "title": f"{title_prefix} Convex Hulls — Per-Step Evolution",
                        "label_name": label_name,
                    },
                ),
                (
                    "hull area curve",
                    plot_hull_area_curve,
                    (all_step_data, reward_names),
                    {
                        "output_path": os.path.join(out_dir, "hull_area_curve.png"),
                        "title": f"{title_prefix} Convex Hull Area Over Steps",
                        "label_name": label_name,
                    },
                ),
            ]
        )

        # Phase 2: late-stage faceted (depends on mid, cheap to compute)
        all_steps_sorted = sorted(all_step_data.keys())
        if len(all_steps_sorted) >= 2:
            mid = all_steps_sorted[len(all_steps_sorted) // 2]
            print(f"  [Plot] Generating late-stage faceted hulls (step >= {mid}) ...")
            plot_convex_hulls_faceted(
                all_step_data,
                reward_names,
                os.path.join(out_dir, "convex_hulls_faceted_late.png"),
                title=f"{title_prefix} Convex Hulls — Late Stage ({label_name} {mid}+)",
                label_name=label_name,
                step_range=(mid, all_steps_sorted[-1]),
            )

        # Phase 3: window + percentile + Pareto in parallel
        phase3_tasks = [
            (
                "window-averaged hull",
                plot_convex_hulls_windows,
                (all_step_data, reward_names),
                {
                    "output_path": os.path.join(out_dir, "convex_hulls_windows.png"),
                    "title": f"{title_prefix} Convex Hull Trend (Window-Averaged)",
                    "label_name": label_name,
                    "window_size": window_size,
                },
            ),
            (
                "cumulative window frames",
                plot_convex_hulls_windows_cumulative,
                (all_step_data, reward_names),
                {
                    "output_dir": os.path.join(out_dir, "convex_hulls_windows_frames"),
                    "title": f"{title_prefix} Convex Hull",
                    "label_name": label_name,
                    "window_size": window_size,
                },
            ),
            (
                "reward percentiles",
                plot_reward_percentiles,
                (all_step_data, reward_names),
                {
                    "output_path": os.path.join(out_dir, "reward_percentiles.png"),
                    "title": f"{title_prefix} Reward Percentile Trends",
                    "label_name": label_name,
                    "window_size": window_size,
                },
            ),
            (
                "per-group hypervolume + hull gap",
                plot_per_group_hypervolume_and_gap,
                (all_step_data, reward_names),
                {
                    "output_path": os.path.join(out_dir, "per_group_hypervolume.png"),
                    "title": f"{title_prefix} Per-Group Hypervolume & Hull Gap",
                    "label_name": label_name,
                },
            ),
        ]
        run_tasks(phase3_tasks)

    else:
        # 1-D: distribution first, then percentiles + Pareto in parallel
        print(f"  [Plot] Generating 1-D reward distribution ...")
        plot_distribution_1d(
            all_step_data,
            reward_names[0],
            os.path.join(out_dir, "distribution_1d.png"),
            title=f"{title_prefix} Reward Distribution",
            label_name=label_name,
        )
        run_tasks(
            [
                (
                    "reward percentiles",
                    plot_reward_percentiles,
                    (all_step_data, reward_names),
                    {
                        "output_path": os.path.join(out_dir, "reward_percentiles.png"),
                        "title": f"{title_prefix} Reward Percentile Trends",
                        "label_name": label_name,
                        "window_size": window_size,
                    },
                ),
                (
                    "per-group hypervolume",
                    plot_per_group_hypervolume_and_gap,
                    (all_step_data, reward_names),
                    {
                        "output_path": os.path.join(out_dir, "per_group_hypervolume.png"),
                        "title": f"{title_prefix} Per-Group Hypervolume",
                        "label_name": label_name,
                    },
                ),
            ]
        )


# ---------------------------------------------------------------------------
# Workflow: Images → reward model → hulls
# ---------------------------------------------------------------------------


def _run_images_analysis(
    config: AnalysisConfig,
    run_name: str,
    computer: Optional[Union[StandaloneRewardComputer, MultiGPUComputer]],
    all_prompts: List[str],
    output_dir: str,
    reward_names: List[str],
    dataset_filter: List[str],
    output_subdir: str,
    preloaded_images: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Read rollout images from media.jsonl, score with reward models, plot.

    Args:
        dataset_filter: e.g. ``["train"]`` or ``["eval"]``.
        output_subdir: e.g. ``"train_images"`` or ``"eval_images"``.
        preloaded_images: If provided, skip ``load_media_samples`` and reuse
            this data (avoids duplicate I/O from ``_check_cache``).
    """
    label = output_subdir.replace("_", " ").title()
    log_dir = os.path.join(config.save_dir, run_name, "logs")
    if not os.path.isdir(log_dir):
        print(f"[{label}] Logs directory not found: {log_dir} — skipping")
        return None

    if preloaded_images is not None:
        images_by_step = preloaded_images
    else:
        print(
            f"[{label}] Loading samples from {log_dir}/media.jsonl "
            f"(datasets={dataset_filter}) ..."
        )
        try:
            images_by_step = load_media_samples(
                log_dir,
                datasets=dataset_filter,
                max_per_step=config.images_max_images_per_step,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [ERROR] {exc}")
            return None

    if not images_by_step:
        print(f"  No images found for dataset filter {dataset_filter}.")
        return None

    steps = sorted(images_by_step.keys())
    total_imgs = sum(len(v) for v in images_by_step.values())
    datasets = sorted(set(e["dataset"] for entries in images_by_step.values() for e in entries))
    print(f"  Found {len(steps)} steps with images across datasets: {datasets}")
    print(f"  Total images: {total_imgs}")

    out_dir = os.path.join(output_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    step_img_counts = {s: len(entries) for s, entries in images_by_step.items()}

    cached = _load_reward_cache(out_dir, steps, reward_names)
    cache_valid = cached is not None and set(steps) == set(cached.keys())
    if cache_valid:
        for step in steps:
            expected = step_img_counts.get(step, 0)
            actual = cached[step].get("n_total", 0)
            if expected != actual:
                print(f"  [Reward] Image count changed — re-scoring needed.")
                cache_valid = False
                break

    if cache_valid:
        print(f"  [Reward] All {len(steps)} steps cached — skipping reward scoring.")
        all_step_data = cached
    else:
        assert computer is not None, "Reward model needed but not loaded"
        print(f"  [Reward] Scoring images with {reward_names} ...")

        t0 = time.time()

        # --- Build flat entry list (paths + prompts, no file handles yet) ---
        Entry = Tuple[str, str, int]  # (path, prompt, step)
        all_entries: List[Entry] = []
        for step in tqdm(steps, desc="  Indexing", unit="step"):
            for e in images_by_step.get(step, []):
                img_path = e["image_path"]
                if os.path.isfile(img_path):
                    all_entries.append((img_path, e.get("prompt", "a photo"), step))

        total_images = len(all_entries)
        SCORING_CHUNK = 500  # bound file handles per batch

        step_reward_buf: Dict[int, Dict[str, List[float]]] = {}
        step_prompt_buf: Dict[int, List[str]] = {}

        for chunk_start in range(0, total_images, SCORING_CHUNK):
            chunk_entries = all_entries[chunk_start : chunk_start + SCORING_CHUNK]

            # Open only this chunk's images
            chunk_images = [Image.open(path) for path, _, _ in chunk_entries]
            chunk_prompts = [prompt for _, prompt, _ in chunk_entries]

            try:
                chunk_rewards = _score_images(computer, chunk_images, chunk_prompts)
            finally:
                for img in chunk_images:
                    img.close()

            # Scatter back to per-step buffers
            for i, (_, _, step) in enumerate(chunk_entries):
                if step not in step_reward_buf:
                    step_reward_buf[step] = {name: [] for name in reward_names}
                    step_prompt_buf[step] = []
                for name in reward_names:
                    step_reward_buf[step][name].append(chunk_rewards[name][i])
                step_prompt_buf[step].append(chunk_prompts[i])

        # --- Build final step data ---
        all_step_data = {}
        for step in sorted(step_reward_buf.keys()):
            all_step_data[step] = _build_step_data(
                step,
                step_reward_buf[step],
                reward_names,
                step_prompt_buf[step],
            )

        elapsed = time.time() - t0
        n_chunks = (total_images + SCORING_CHUNK - 1) // SCORING_CHUNK
        print(
            f"  [Reward] Scored {total_images} images in {n_chunks} chunk(s), "
            f"{elapsed:.1f}s ({elapsed / max(total_images, 1):.3f}s/image)"
        )
        _save_reward_cache(out_dir, all_step_data, reward_names)

    _eval_window = 1 if output_subdir in ("eval_images",) else 20

    _dispatch_plots(
        all_step_data,
        reward_names,
        out_dir,
        title_prefix=f"{label} Rollout",
        label_name="Step",
        window_size=_eval_window,
    )
    print(f"  {label} → {out_dir}/")
    return all_step_data


# ---------------------------------------------------------------------------
# Workflow: Evaluation inference → rewards → hulls
# ---------------------------------------------------------------------------


def _run_evaluation(
    config: AnalysisConfig,
    run_name: str,
    computer: Optional[Union[StandaloneRewardComputer, MultiGPUComputer]],
    prompts: List[str],
    output_dir: str,
    reward_names: List[str],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Discover checkpoints, generate images, score them, return epoch→data dict.

    Flow:
    1. Per-checkpoint reward cache check → skip already-cached checkpoints.
    2. Generate images for uncached checkpoints (batch inference, multi-GPU).
    3. Free the generation pipeline.
    4. Score ALL uncached checkpoints at once (single or multi-GPU).
    5. Merge results into cache.
    6. Dispatch plots.
    """
    ckpt_dir = _resolve_checkpoint_dir(config, run_name)
    try:
        checkpoints = discover_checkpoints(ckpt_dir)
    except FileNotFoundError as exc:
        print(f"[Evaluation] {exc} — skipping")
        return None

    if not checkpoints:
        print(f"[Evaluation] No checkpoint-* subdirectories in {ckpt_dir}")
        return None

    if not prompts:
        print("[Evaluation] No prompts configured — skipping evaluation analysis")
        return None

    if config.max_prompts > 0 and len(prompts) > config.max_prompts:
        prompts = prompts[: config.max_prompts]

    expected_per_ckpt = len(prompts) * config.num_samples
    all_epochs = [e for e, _ in checkpoints]

    print(f"[Evaluation] Found {len(checkpoints)} checkpoints: {all_epochs}")
    print(
        f"  Prompts: {len(prompts)}, samples per prompt: {config.num_samples}"
        f" → {expected_per_ckpt} images per checkpoint"
    )

    ev_out = os.path.join(output_dir, "evaluation")
    os.makedirs(ev_out, exist_ok=True)
    gen_dir = os.path.join(ev_out, "generated_images")

    # --- 1. Per-checkpoint cache check ---
    existing_cache = _load_reward_cache(ev_out, all_epochs, reward_names)

    # Validate n_total per cached step
    def _is_step_cached(step: int) -> bool:
        if existing_cache is None:
            return False
        data = existing_cache.get(step)
        if data is None:
            return False
        return data.get("n_total", 0) == expected_per_ckpt

    cached_epochs = set(e for e in all_epochs if _is_step_cached(e))
    uncached_epochs = [e for e in all_epochs if e not in cached_epochs]

    if cached_epochs:
        print(
            f"  Reward cache hit for epochs {sorted(cached_epochs)}"
            f" — skipping generation + scoring for these."
        )

    if not uncached_epochs:
        print(f"  All {len(all_epochs)} checkpoints fully cached.")
        all_epoch_data = existing_cache
    else:
        # --- 2. Generate images for uncached checkpoints ---
        if config.num_gpus > 1:
            gen_runner = MultiGPUEvaluationRunner(
                config.base_model,
                config.dtype,
                num_gpus=config.num_gpus,
                device=config.device,
            )
            print(
                f"  Using {config.num_gpus}-GPU generation" f" (batch size={config.gen_batch_size})"
            )
        else:
            gen_runner = EvaluationRunner(config.base_model, config.dtype, device=config.device)

        for epoch, ckpt_path in checkpoints:
            if epoch in cached_epochs:
                continue
            print(f"  [Gen] Epoch={epoch}: generating images ...")
            gen_runner.generate_for_checkpoint(
                ckpt_path,
                prompts,
                gen_dir,
                epoch,
                num_samples=config.num_samples,
                gen_kwargs=config.gen_kwargs,
                gen_batch_size=config.gen_batch_size,
                base_seed=config.gen_kwargs.get("seed", 42),
            )

        # --- 3. Free generation pipelines ---
        del gen_runner

        # --- 4. Score uncached checkpoints ---
        assert computer is not None, "Reward model needed but not loaded"
        all_epoch_data = dict(existing_cache) if existing_cache else {}

        # Collect all (image, prompt) pairs across uncached checkpoints
        all_images: List = []
        all_prompt_texts: List[str] = []
        ckpt_image_ranges: Dict[int, Tuple[int, int]] = {}  # epoch -> (start, end) in flat lists

        for epoch in uncached_epochs:
            start = len(all_images)
            for pi, p in enumerate(prompts):
                for si in range(config.num_samples):
                    full = os.path.join(gen_dir, f"checkpoint_{epoch}", f"p{pi}_s{si}.png")
                    all_images.append(Image.open(full))
                    all_prompt_texts.append(p)
            ckpt_image_ranges[epoch] = (start, len(all_images))

        total_images = len(all_images)
        print(f"  Scoring {total_images} images across {len(uncached_epochs)}" f" checkpoints ...")

        if all_images:
            try:
                all_rewards = _score_images(computer, all_images, all_prompt_texts)
                # Split flat results back per checkpoint
                for epoch in uncached_epochs:
                    start, end = ckpt_image_ranges[epoch]
                    epoch_rewards = {name: vals[start:end] for name, vals in all_rewards.items()}
                    epoch_prompts = all_prompt_texts[start:end]
                    all_epoch_data[epoch] = _build_step_data(
                        epoch, epoch_rewards, reward_names, epoch_prompts
                    )
            finally:
                for img in all_images:
                    img.close()

        # --- 5. Merge into cache ---
        _save_reward_cache(ev_out, all_epoch_data, reward_names)

    # --- 6. Plots ---
    _dispatch_plots(
        all_epoch_data,
        reward_names,
        ev_out,
        title_prefix="Evaluation (Test Prompts)",
        label_name="Epoch",
        window_size=1,
    )
    print(f"  Evaluation results saved to {ev_out}/")
    return all_epoch_data


# ---------------------------------------------------------------------------
# Workflow: Rewards/ pickles → distribution / percentile / Pareto
# ---------------------------------------------------------------------------


def _run_rewards_analysis(
    config: AnalysisConfig,
    run_name: str,
    output_dir: str,
) -> None:
    """Read pre-computed scores from ``logs/rewards/*.pkl`` and generate plots.

    This path does NOT load images or run reward models — it uses the scores
    that were saved during training directly.

    Generates separate plot sets for train data and each eval dataset:
    ``train_rewards/`` and ``eval_rewards/<dataset_name>/``.
    """
    rewards_dir = os.path.join(config.save_dir, run_name, "logs", "rewards")
    if not os.path.isdir(rewards_dir):
        print(f"[Rewards Analysis] Directory not found: {rewards_dir} — skipping")
        return

    print(f"[Rewards Analysis] Loading reward pickles from {rewards_dir} ...")

    # train_data: {step: {...}}, train_rnames: [str, ...]
    train_data, train_rnames = load_train_rewards(rewards_dir)
    # eval_data: {dataset_name: {step: {...}}}, eval_rnames: {dataset_name: [str, ...]}
    eval_data, eval_rnames = load_eval_rewards(rewards_dir)

    if not train_data and not eval_data:
        print("[Rewards Analysis] No reward pickle files found.")
        return

    # --- Train ---
    if train_data:
        tr_out = os.path.join(output_dir, "train_rewards")
        os.makedirs(tr_out, exist_ok=True)
        first_step = sorted(train_data.keys())[0]
        n_scores = train_data[first_step]["n_total"]
        print(
            f"  Train: {len(train_data)} steps, {n_scores} scores/step, "
            f"rewards={train_rnames}"
        )

        _dispatch_plots(
            train_data,
            train_rnames,
            tr_out,
            title_prefix="Train Rewards (from pickles)",
            label_name="Step",
            window_size=20,
        )
        print(f"  Train rewards → {tr_out}/")

    # --- Eval (per dataset) ---
    if eval_data:
        ev_base = os.path.join(output_dir, "eval_rewards")
        os.makedirs(ev_base, exist_ok=True)
        for ds_name, ds_step_data in eval_data.items():
            ds_out = os.path.join(ev_base, ds_name)
            os.makedirs(ds_out, exist_ok=True)
            ds_rnames = eval_rnames.get(ds_name, [])
            if not ds_rnames:
                print(f"  Eval/{ds_name}: no common reward keys — skipping")
                continue
            n_steps = len(ds_step_data)
            n_scores = ds_step_data[sorted(ds_step_data.keys())[0]]["n_total"]
            print(
                f"  Eval/{ds_name}: {n_steps} steps, {n_scores} scores/step, "
                f"rewards={ds_rnames}"
            )

            _dispatch_plots(
                ds_step_data,
                ds_rnames,
                ds_out,
                title_prefix=f"Eval/{ds_name} Rewards (from pickles)",
                label_name="Step",
                window_size=1,
            )
            print(f"  Eval/{ds_name} → {ds_out}/")


# ---------------------------------------------------------------------------
# Reward cache — skip reward model when scores already on disk
# ---------------------------------------------------------------------------


def _reward_cache_path(source_dir: str) -> str:
    return os.path.join(source_dir, "reward_cache.json")


def _load_reward_cache(
    source_dir: str, step_keys: List[int], reward_names: List[str]
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load cached reward scores, returning partial results on miss.

    Returns a dict keyed by the step_keys that are present and valid.
    Returns ``None`` only if the cache file is missing or *reward_names*
    have changed (full invalidation).  Callers are responsible for
    verifying ``n_total`` against the expected image count per step.
    """
    path = _reward_cache_path(source_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    cached_names = raw.get("reward_names", [])
    if cached_names != reward_names:
        print(f"  Reward models changed ({cached_names} → {reward_names}) — invalidating cache.")
        return None
    steps_data = raw.get("steps", {})
    result: Dict[int, Dict[str, Any]] = {}
    for step in step_keys:
        key = str(step)
        if key not in steps_data:
            continue
        rec = steps_data[key]
        pts_list = [rec.get(name, []) for name in reward_names]
        n_valid = min(len(lst) for lst in pts_list) if pts_list else 0
        if n_valid == 0:
            result[step] = {
                "points": np.empty((0, len(reward_names))),
                "step": step,
                "n_total": 0,
                "n_valid": 0,
            }
        else:
            pts = np.column_stack([pts_list[i][:n_valid] for i in range(len(reward_names))])
            result[step] = {
                "points": pts,
                "step": step,
                "n_total": rec.get("n_total", n_valid),
                "n_valid": n_valid,
            }
        prompt_idx = rec.get("prompt_idx")
        if prompt_idx is not None and len(prompt_idx) == result[step]["n_valid"]:
            result[step]["prompt_idx"] = np.array(prompt_idx)
        prompt_labels = rec.get("prompt_labels")
        if prompt_labels:
            result[step]["prompt_labels"] = prompt_labels
    return result if result else None


def _save_reward_cache(
    source_dir: str, all_data: Dict[int, Dict[str, Any]], reward_names: List[str]
):
    """Save reward scores, merging into existing cache so per-step granularity is preserved."""
    path = _reward_cache_path(source_dir)
    # Load existing cache so we don't clobber other steps
    steps_dict = {}
    if os.path.isfile(path):
        with open(path, "r") as f:
            existing = json.load(f)
        if existing.get("reward_names") == reward_names:
            steps_dict = existing.get("steps", {})
    for step, data in sorted(all_data.items()):
        pts = data.get("points")
        rec = {"n_total": data.get("n_total", 0), "n_valid": data.get("n_valid", 0)}
        if pts is not None and len(pts) > 0:
            for di, name in enumerate(reward_names):
                rec[name] = pts[:, di].tolist()
        else:
            for name in reward_names:
                rec[name] = []
        prompt_idx = data.get("prompt_idx")
        if prompt_idx is not None:
            rec["prompt_idx"] = prompt_idx.tolist()
        prompt_labels = data.get("prompt_labels")
        if prompt_labels:
            rec["prompt_labels"] = prompt_labels
        steps_dict[str(step)] = rec
    with open(path, "w") as f:
        json.dump({"reward_names": reward_names, "steps": steps_dict}, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path: str):
    print(f"Loading config: {config_path}")
    config = _parse_config(config_path)

    if not config.rewards:
        print("ERROR: No reward models configured.")
        sys.exit(1)

    run_name = _resolve_run_name(config)
    print(f"Run name: {run_name}")

    output_dir = os.path.join(config.output_dir, "reward_convex_hull_analysis", run_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump({k: v for k, v in config.__dict__.items() if not k.startswith("_")}, f)

    all_prompts = _load_prompts(config)
    reward_names = [r.get("name", r.get("reward_model", "?")) for r in config.rewards]

    # --- Check if images-based sources are fully cached ---
    train_cached = False
    eval_cached = False
    ev_cached = False

    # (cached, preloaded_images) — preloaded_images is reused to avoid a second
    # load_media_samples() call inside _run_images_analysis.
    def _check_cache(subdir: str, datasets: List[str]) -> Tuple[bool, Optional[Dict[int, Any]]]:
        out_dir = os.path.join(output_dir, subdir)
        log_dir = os.path.join(config.save_dir, run_name, "logs")
        if not os.path.isdir(log_dir):
            return False, None
        try:
            images_by_step = load_media_samples(
                log_dir,
                datasets=datasets,
                max_per_step=config.images_max_images_per_step,
            )
            steps = sorted(images_by_step.keys())
            if not steps:
                return False, None
            cached_all = _load_reward_cache(out_dir, steps, reward_names)
            if cached_all is None or set(steps) != set(cached_all.keys()):
                return False, images_by_step
            for step in steps:
                expected = len(images_by_step.get(step, []))
                actual = cached_all[step].get("n_total", 0)
                if expected != actual:
                    return False, images_by_step
            return True, None  # cache valid, no need to keep loaded data
        except (FileNotFoundError, ValueError):
            return False, None

    train_preloaded = None
    eval_preloaded = None
    if config.images_analysis_enabled:
        train_cached, train_preloaded = _check_cache("train_images", ["train"])
        eval_cached, eval_preloaded = _check_cache("eval_images", ["eval"])

    if config.evaluation_enabled:
        ev_out = os.path.join(output_dir, "evaluation")
        ckpt_dir = _resolve_checkpoint_dir(config, run_name)
        try:
            checkpoints = discover_checkpoints(ckpt_dir)
            epochs = [e for e, _ in checkpoints]
            if epochs:
                cached = _load_reward_cache(ev_out, epochs, reward_names)
                if cached is not None and set(epochs) == set(cached.keys()):
                    n_prompts = len(all_prompts)
                    if config.max_prompts > 0 and n_prompts > config.max_prompts:
                        n_prompts = config.max_prompts
                    expected = n_prompts * config.num_samples
                    if all(cached[e].get("n_total", 0) == expected for e in epochs):
                        ev_cached = True
        except FileNotFoundError:
            pass

    need_compute = (config.images_analysis_enabled and not (train_cached and eval_cached)) or (
        config.evaluation_enabled and not ev_cached
    )

    if need_compute:
        print(f"Loading reward models: {reward_names}")
        import torch as _torch

        device = config.device
        if device == "cuda" and _torch.cuda.is_available():
            free_mem = _torch.cuda.mem_get_info()[0] / (1024**3)
            print(f"  GPU free: {free_mem:.1f} GiB")
            if free_mem < 4.0:
                print(f'  [WARN] Low GPU memory — consider device: "cpu"')
        t0 = time.time()
        num_gpus = config.num_gpus
        if num_gpus > 1:
            print(f"  Creating {num_gpus}-GPU scorer ...")
            computer: Union[StandaloneRewardComputer, MultiGPUComputer] = MultiGPUComputer(
                config.rewards, num_gpus=num_gpus, device=config.device
            )
        else:
            computer = StandaloneRewardComputer(config.rewards, device=config.device)
        print(f"  Reward models loaded in {time.time() - t0:.1f}s: {reward_names}")
    else:
        print(f"All image caches valid — skipping model loading ({reward_names}).")
        computer = None

    # --- Images path: train ---
    if config.images_analysis_enabled:
        t0 = time.time()
        _run_images_analysis(
            config,
            run_name,
            computer,
            all_prompts,
            output_dir,
            reward_names,
            dataset_filter=["train"],
            output_subdir="train_images",
            preloaded_images=train_preloaded,
        )
        print(f"  [Timing] Train images analysis: {time.time() - t0:.1f}s")

    # --- Images path: eval ---
    if config.images_analysis_enabled:
        t0 = time.time()
        _run_images_analysis(
            config,
            run_name,
            computer,
            all_prompts,
            output_dir,
            reward_names,
            dataset_filter=["eval"],
            output_subdir="eval_images",
            preloaded_images=eval_preloaded,
        )
        print(f"  [Timing] Eval images analysis: {time.time() - t0:.1f}s")

    # --- Evaluation (checkpoint-based generation) ---
    if config.evaluation_enabled:
        t0 = time.time()
        _run_evaluation(config, run_name, computer, all_prompts, output_dir, reward_names)
        print(f"  [Timing] Evaluation analysis: {time.time() - t0:.1f}s")

    # --- Rewards path: train + eval from pickles ---
    if config.rewards_analysis_enabled:
        t0 = time.time()
        _run_rewards_analysis(config, run_name, output_dir)
        print(f"  [Timing] Rewards analysis: {time.time() - t0:.1f}s")

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Convex Hull Analysis")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "default_config.yaml"),
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    main(args.config)
