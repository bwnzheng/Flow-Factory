#!/usr/bin/env python3
"""Reward Convex Hull Analysis Tool.

Two independent data sources:
  1. **Training rollouts** — extracts logged rollout images from TensorBoard
     events across training steps, scores them with reward models, and plots
     per-step convex hulls.
  2. **Evaluation inference** — loads LoRA weights from each checkpoint,
     generates fresh images on test prompts, scores them, and plots
     per-checkpoint convex hulls.

When both sources are enabled, a combined overlay plot is produced as well.

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
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from tools.reward_convex_hull_analysis.convex_hull import (
    plot_combined_convex_hulls_2d,
    plot_convex_hulls_2d,
    plot_convex_hulls_faceted,
    plot_convex_hulls_windows,
    plot_distribution_1d,
    plot_hull_area_curve,
)
from tools.reward_convex_hull_analysis.evaluation_runner import (
    EvaluationRunner,
    discover_checkpoints,
)
from tools.reward_convex_hull_analysis.reward_computer import StandaloneRewardComputer
from tools.reward_convex_hull_analysis.tensorboard_extractor import (
    decode_images_to_disk,
    load_decoded_images,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    run_name: str = ""
    save_dir: str = "saves"

    training_enabled: bool = True
    tr_max_images_per_step: int = 0
    tr_datasets: List[str] = field(default_factory=list)
    tr_num_workers: int = 4
    evaluation_enabled: bool = True
    eval_checkpoint_dir: str = ""
    prompts_file: str = ""
    prompts: List[str] = field(default_factory=list)
    max_prompts: int = 0
    num_samples: int = 4
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)

    base_model: str = "stabilityai/stable-diffusion-3.5-medium"
    dtype: str = "bfloat16"
    device: str = "cuda"

    rewards: List[Dict[str, Any]] = field(default_factory=list)

    output_dir: str = "analysis_output"


def _parse_config(path: str) -> AnalysisConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    tr = raw.get("training", {})
    ev = raw.get("evaluation", {})
    model = raw.get("model", {})
    output = raw.get("output", {})

    return AnalysisConfig(
        run_name=raw.get("run_name", ""),
        save_dir=raw.get("save_dir", "saves"),
        training_enabled=tr.get("enabled", True),
        tr_max_images_per_step=tr.get("max_images_per_step", 0),
        tr_datasets=tr.get("datasets", []),
        tr_num_workers=tr.get("num_workers", 4),
        evaluation_enabled=ev.get("enabled", True),
        eval_checkpoint_dir=ev.get("checkpoint_dir", ""),
        prompts_file=ev.get("prompts_file", ""),
        prompts=ev.get("prompts", []),
        max_prompts=ev.get("max_prompts", 0),
        num_samples=ev.get("num_samples", 4),
        gen_kwargs=ev.get("gen_kwargs", {}),
        base_model=model.get("base_model", "stabilityai/stable-diffusion-3.5-medium"),
        dtype=model.get("dtype", "bfloat16"),
        device=model.get("device", "cuda"),
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
    tb_root = os.path.join(config.save_dir, "tensorboard")
    if os.path.isdir(tb_root):
        runs = sorted(os.listdir(tb_root))
        if runs:
            return runs[-1]
    raise ValueError(
        "Cannot resolve run_name. Set run_name in config or provide checkpoint_dir."
    )


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
# Prompt reconstruction from training sampler (GroupContiguousSampler)
# ---------------------------------------------------------------------------


def _reconstruct_training_prompts(
    prompts_file: str,
    steps: List[int],
    seed: int = 42,
    group_size: int = 16,
    unique_sample_num: int = 48,
    world_size: int = 4,
    per_device_batch_size: int = 1,
    dataset_size: Optional[int] = None,
) -> Dict[int, List[str]]:
    """Reconstruct training prompts using the exact ``GroupContiguousSampler`` logic."""
    path = prompts_file
    if not os.path.isabs(path):
        if not os.path.isfile(path):
            path = os.path.join(os.getcwd(), path)
    with open(path, "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    D = dataset_size if dataset_size else len(all_prompts)
    M = unique_sample_num
    K = group_size
    G = M // world_size

    result: Dict[int, List[str]] = {}
    for epoch in steps:
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        indices = torch.randperm(D, generator=g)[:M].tolist()
        group_perm = torch.randperm(M, generator=g).tolist()
        shuffled_groups = [indices[i] for i in group_perm]
        my_groups = shuffled_groups[:G]
        result[epoch] = [all_prompts[my_groups[0]], all_prompts[my_groups[1]]]

    return result


# ---------------------------------------------------------------------------
# Reward scoring helpers
# ---------------------------------------------------------------------------


def _score_images(
    computer: StandaloneRewardComputer,
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

    result: Dict[str, Any] = {"points": pts, "step": step, "n_total": n_samples, "n_valid": len(pts)}
    if prompt_per_img and len(prompt_per_img) == n_samples:
        prompt_per_img_arr = np.array(prompt_per_img)
        unique_prompts = list(dict.fromkeys(prompt_per_img_arr))
        prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
        result["prompt_idx"] = np.array([prompt_to_idx[p] for p in prompt_per_img_arr])[mask]
        result["prompt_labels"] = unique_prompts
    return result


# ---------------------------------------------------------------------------
# Workflow: Training rollouts → rewards → hulls
# ---------------------------------------------------------------------------


def _run_training(
    config: AnalysisConfig,
    run_name: str,
    computer: Optional[StandaloneRewardComputer],
    all_prompts: List[str],
    output_dir: str,
    reward_names: List[str],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Decode images from TensorBoard, score them, return step→data dict."""
    tb_dir = os.path.join(config.save_dir, "tensorboard", run_name)
    if not os.path.isdir(tb_dir):
        print(f"[Training] TensorBoard directory not found: {tb_dir} — skipping")
        return None

    datasets_filter = config.tr_datasets if config.tr_datasets else None

    decoded_dir = os.path.join(output_dir, "decoded_images")
    print(f"[Training] Decoding images to {decoded_dir} ...")
    try:
        image_index = decode_images_to_disk(
            tb_dir, decoded_dir,
            datasets=datasets_filter,
            max_per_step=config.tr_max_images_per_step,
            num_workers=config.tr_num_workers,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [ERROR] {exc}")
        return None

    if not image_index:
        print("  No rollout images found in TensorBoard events.")
        return None

    step_img_counts = Counter(e["step"] for e in image_index)
    steps = sorted(step_img_counts.keys())
    datasets = sorted(set(e["dataset"] for e in image_index))
    total_imgs = sum(step_img_counts.values())
    print(f"  Found {len(steps)} steps with images across datasets: {datasets}")
    print(f"  Total images: {total_imgs}")

    tr_out = os.path.join(output_dir, "training")
    os.makedirs(tr_out, exist_ok=True)

    cached = _load_reward_cache(tr_out, steps, reward_names)
    cache_valid = cached is not None
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
        print(f"[Training] Loading decoded images ({len(image_index)} total) ...")
        images_by_step = load_decoded_images(
            decoded_dir, image_index,
            max_per_step=config.tr_max_images_per_step,
        )
        print(f"  [Reward] Scoring images with {reward_names} ...")

        step_prompts: Dict[int, List[str]] = {}
        train_prompts_file = "dataset/pickscore/train.txt"
        if os.path.isfile(train_prompts_file):
            print(f"  [Reward] Reconstructing training prompts from {train_prompts_file} ...")
            step_prompts = _reconstruct_training_prompts(
                train_prompts_file, steps,
                seed=42, group_size=16, unique_sample_num=48,
                world_size=4, per_device_batch_size=1, dataset_size=1024,
            )

        t0 = time.time()
        all_step_data = {}
        for step in tqdm(steps, desc="  Scoring", unit="step"):
            entries = images_by_step.get(step, [])
            images = [e["image"] for e in entries]
            prompts = step_prompts.get(step, [])
            prompt_per_img = []
            for e in entries:
                ti = e.get("tag_idx", 0)
                if ti < 16 and len(prompts) > 0:
                    prompt_per_img.append(prompts[0])
                elif ti >= 16 and len(prompts) > 1:
                    prompt_per_img.append(prompts[1])
                elif ti < len(all_prompts):
                    prompt_per_img.append(all_prompts[ti])
                else:
                    prompt_per_img.append("a photo")
            rewards = _score_images(computer, images, prompt_per_img)
            all_step_data[step] = _build_step_data(step, rewards, reward_names,
                                                    prompt_per_img)
        elapsed = time.time() - t0
        print(f"  [Reward] Scored {len(steps)} steps in {elapsed:.1f}s "
              f"({elapsed / max(len(steps), 1):.1f}s/step)")
        _save_reward_cache(tr_out, all_step_data, reward_names)

    n_models = len(reward_names)
    if n_models >= 2:
        print(f"  [Plot] Generating convex hull overlay ...")
        plot_convex_hulls_2d(
            all_step_data, reward_names,
            os.path.join(tr_out, "convex_hulls_2d.png"),
            title="Training Rollout Reward Convex Hulls",
            label_name="Step",
        )
        print(f"  [Plot] Generating faceted hull grid ...")
        plot_convex_hulls_faceted(
            all_step_data, reward_names,
            os.path.join(tr_out, "convex_hulls_faceted.png"),
            title="Training Convex Hulls — Per-Step Evolution",
            label_name="Step",
        )
        print(f"  [Plot] Generating hull area curve ...")
        plot_hull_area_curve(
            all_step_data, reward_names,
            os.path.join(tr_out, "hull_area_curve.png"),
            title="Training Convex Hull Area Over Steps",
            label_name="Step",
        )

        all_steps_sorted = sorted(all_step_data.keys())
        mid = all_steps_sorted[len(all_steps_sorted) // 2]
        print(f"  [Plot] Generating late-stage faceted hulls (step >= {mid}) ...")
        plot_convex_hulls_faceted(
            all_step_data, reward_names,
            os.path.join(tr_out, "convex_hulls_faceted_late.png"),
            title=f"Training Convex Hulls — Late Stage (Step {mid}+)",
            label_name="Step",
            step_range=(mid, all_steps_sorted[-1]),
        )
        print(f"  [Plot] Generating window-averaged hull trend ...")
        plot_convex_hulls_windows(
            all_step_data, reward_names,
            os.path.join(tr_out, "convex_hulls_windows.png"),
            title="Training Convex Hull Trend (Window-Averaged)",
            label_name="Step",
        )
    else:
        plot_distribution_1d(
            all_step_data, reward_names[0],
            os.path.join(tr_out, "distribution_1d.png"),
            title="Training Rollout Reward Distribution",
            label_name="Step",
        )
    print(f"  Training → {tr_out}/")
    return all_step_data


# ---------------------------------------------------------------------------
# Workflow: Evaluation inference → rewards → hulls
# ---------------------------------------------------------------------------


def _run_evaluation(
    config: AnalysisConfig,
    run_name: str,
    computer: Optional[StandaloneRewardComputer],
    prompts: List[str],
    output_dir: str,
    reward_names: List[str],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Discover checkpoints, generate images, score them, return epoch→data dict."""
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
        prompts = prompts[:config.max_prompts]

    print(f"[Evaluation] Found {len(checkpoints)} checkpoints: "
          f"{[e for e, _ in checkpoints]}")
    print(f"  Prompts: {len(prompts)}, samples per prompt: {config.num_samples}")

    runner = EvaluationRunner(config.base_model, config.dtype, device=config.device)

    ev_out = os.path.join(output_dir, "evaluation")
    os.makedirs(ev_out, exist_ok=True)

    gen_dir = os.path.join(ev_out, "generated_images")

    epochs = [e for e, _ in checkpoints]
    cached = _load_reward_cache(ev_out, epochs, reward_names)
    if cached is not None:
        print(f"  All {len(epochs)} checkpoints cached — skipping reward scoring.")
        all_epoch_data = cached
    else:
        assert computer is not None, "Reward model needed but not loaded"
        all_epoch_data = {}
        for epoch, ckpt_path in checkpoints:
            print(f"  Evaluation epoch={epoch}: generating images ...")
            paths = runner.generate_for_checkpoint(
                ckpt_path, prompts, gen_dir, epoch,
                config.num_samples, config.gen_kwargs,
            )
            images = []
            prompt_per_img = []
            for pi, p in enumerate(prompts):
                for si in range(config.num_samples):
                    full = os.path.join(gen_dir, f"checkpoint_{epoch}", f"p{pi}_s{si}.png")
                    if os.path.isfile(full):
                        images.append(Image.open(full))
                        prompt_per_img.append(p)
            print(f"    Scoring {len(images)} images ...")
            if images:
                rewards = _score_images(computer, images, prompt_per_img)
                all_epoch_data[epoch] = _build_step_data(epoch, rewards, reward_names,
                                                          prompt_per_img)
        _save_reward_cache(ev_out, all_epoch_data, reward_names)

    del runner

    n_models = len(reward_names)
    if n_models >= 2:
        plot_convex_hulls_2d(
            all_epoch_data, reward_names,
            os.path.join(ev_out, "convex_hulls_2d.png"),
            title="Evaluation Reward Convex Hulls (Test Prompts)",
            label_name="Epoch",
        )
        plot_convex_hulls_faceted(
            all_epoch_data, reward_names,
            os.path.join(ev_out, "convex_hulls_faceted.png"),
            title="Evaluation Convex Hulls — Per-Epoch Evolution",
            label_name="Epoch",
        )
        plot_hull_area_curve(
            all_epoch_data, reward_names,
            os.path.join(ev_out, "hull_area_curve.png"),
            title="Evaluation Convex Hull Area Over Epochs",
            label_name="Epoch",
        )
    else:
        plot_distribution_1d(
            all_epoch_data, reward_names[0],
            os.path.join(ev_out, "distribution_1d.png"),
            title="Evaluation Reward Distribution (Test Prompts)",
            label_name="Epoch",
        )
    print(f"  Evaluation results saved to {ev_out}/")
    return all_epoch_data


# ---------------------------------------------------------------------------
# Reward cache — skip reward model when scores already on disk
# ---------------------------------------------------------------------------


def _load_manifest_quiet(decoded_dir: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(decoded_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        m = json.load(f)
    return m.get("images", None)


def _reward_cache_path(source_dir: str) -> str:
    return os.path.join(source_dir, "reward_cache.json")


def _load_reward_cache(
    source_dir: str, step_keys: List[int], reward_names: List[str]
) -> Optional[Dict[int, Dict[str, Any]]]:
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
            return None
        rec = steps_data[key]
        pts_list = [rec.get(name, []) for name in reward_names]
        n_valid = min(len(lst) for lst in pts_list) if pts_list else 0
        if n_valid == 0:
            result[step] = {"points": np.empty((0, len(reward_names))), "step": step,
                            "n_total": 0, "n_valid": 0}
        else:
            pts = np.column_stack([pts_list[i][:n_valid] for i in range(len(reward_names))])
            result[step] = {"points": pts, "step": step,
                            "n_total": rec.get("n_total", n_valid), "n_valid": n_valid}
        prompt_idx = rec.get("prompt_idx")
        if prompt_idx is not None and len(prompt_idx) == result[step]["n_valid"]:
            result[step]["prompt_idx"] = np.array(prompt_idx)
        prompt_labels = rec.get("prompt_labels")
        if prompt_labels:
            result[step]["prompt_labels"] = prompt_labels
    return result


def _save_reward_cache(source_dir: str, all_data: Dict[int, Dict[str, Any]],
                       reward_names: List[str]):
    steps_dict = {}
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
    with open(_reward_cache_path(source_dir), "w") as f:
        json.dump({"reward_names": reward_names, "steps": steps_dict}, f, indent=2)


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------


def _run_combined(
    tr_data: Dict[int, Dict[str, Any]],
    ev_data: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_dir: str,
) -> None:
    combined_out = os.path.join(output_dir, "combined")
    os.makedirs(combined_out, exist_ok=True)

    n_models = len(reward_names)
    if n_models >= 2:
        plot_combined_convex_hulls_2d(
            tr_data, ev_data, reward_names,
            os.path.join(combined_out, "convex_hulls_2d.png"),
            label_a="Training",
            label_b="Evaluation",
            title="Combined Reward Convex Hulls: Training vs Evaluation",
        )
        print(f"  Combined plot saved to {combined_out}/")
    else:
        print("  Skipping combined plot (need ≥2 reward models for convex hull)")


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

    # --- Check if all sources are fully cached ---
    tr_cached = False
    ev_cached = False

    if config.training_enabled:
        tr_out = os.path.join(output_dir, "training")
        decoded_dir = os.path.join(output_dir, "decoded_images")
        image_index = _load_manifest_quiet(decoded_dir)
        if image_index:
            if config.tr_datasets:
                image_index = [e for e in image_index
                               if e.get("dataset", "") in config.tr_datasets]
            steps = sorted(set(e["step"] for e in image_index))
            cached_all = _load_reward_cache(tr_out, steps, reward_names)
            if cached_all is not None:
                manifest_counts = Counter(e["step"] for e in image_index)
                tr_cached = True
                for step in steps:
                    expected = manifest_counts.get(step, 0)
                    actual = cached_all[step].get("n_total", 0)
                    if expected != actual:
                        tr_cached = False
                        break

    if config.evaluation_enabled:
        ev_out = os.path.join(output_dir, "evaluation")
        ckpt_dir = _resolve_checkpoint_dir(config, run_name)
        try:
            checkpoints = discover_checkpoints(ckpt_dir)
            epochs = [e for e, _ in checkpoints]
            if epochs and _load_reward_cache(ev_out, epochs, reward_names) is not None:
                ev_cached = True
        except FileNotFoundError:
            pass

    need_compute = (config.training_enabled and not tr_cached) or \
                   (config.evaluation_enabled and not ev_cached)

    if need_compute:
        print(f"Loading reward models: {reward_names}")
        import torch as _torch
        device = config.device
        if device == "cuda" and _torch.cuda.is_available():
            free_mem = _torch.cuda.mem_get_info()[0] / (1024**3)
            print(f"  GPU free: {free_mem:.1f} GiB")
            if free_mem < 4.0:
                print(f"  [WARN] Low GPU memory — consider device: \"cpu\"")
        t0 = time.time()
        computer = StandaloneRewardComputer(config.rewards, device=device)
        print(f"  Reward models loaded in {time.time() - t0:.1f}s: {reward_names}")
        reward_names = reward_names
    else:
        print(f"All data cached — skipping model loading ({reward_names}).")
        computer = None

    tr_data: Optional[Dict[int, Dict[str, Any]]] = None
    ev_data: Optional[Dict[int, Dict[str, Any]]] = None

    if config.training_enabled:
        t0 = time.time()
        tr_data = _run_training(config, run_name, computer, all_prompts, output_dir,
                                 reward_names)
        print(f"  [Timing] Training analysis: {time.time() - t0:.1f}s")

    if config.evaluation_enabled:
        t0 = time.time()
        ev_data = _run_evaluation(config, run_name, computer, all_prompts, output_dir,
                                   reward_names)
        print(f"  [Timing] Evaluation analysis: {time.time() - t0:.1f}s")

    if tr_data is not None and ev_data is not None and len(reward_names) >= 2:
        print(f"  [Plot] Generating combined overlay ...")
        _run_combined(tr_data, ev_data, reward_names, output_dir)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Convex Hull Analysis")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "default_config.yaml"),
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    main(args.config)
