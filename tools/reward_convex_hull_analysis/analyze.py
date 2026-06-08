#!/usr/bin/env python3
"""Reward Convex Hull Analysis Tool.

Two independent data sources:
  1. **TensorBoard events** — extracts logged eval rollout images across training
     steps, scores them with reward models, and plots per-step convex hulls.
  2. **Checkpoint inference** — loads LoRA weights from each checkpoint, generates
     fresh images on test prompts, scores them, and plots per-checkpoint convex hulls.

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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from tools.reward_convex_hull_analysis.convex_hull import (
    plot_combined_convex_hulls_2d,
    plot_convex_hulls_2d,
    plot_distribution_1d,
)
from tools.reward_convex_hull_analysis.checkpoint_runner import (
    CheckpointRunner,
    discover_checkpoints,
)
from tools.reward_convex_hull_analysis.reward_computer import StandaloneRewardComputer
from tools.reward_convex_hull_analysis.tensorboard_extractor import (
    extract_images_from_tensorboard,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    run_name: str = ""
    save_dir: str = "saves"

    tensorboard_enabled: bool = True
    checkpoint_enabled: bool = True
    checkpoint_dir: str = ""
    prompts_file: str = ""
    prompts: List[str] = field(default_factory=list)
    num_samples: int = 4
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)

    base_model: str = "stabilityai/stable-diffusion-3.5-medium"
    dtype: str = "bfloat16"
    device: str = "cuda"
    cache_dir: str = ""

    rewards: List[Dict[str, Any]] = field(default_factory=list)

    output_dir: str = "analysis_output"


def _parse_config(path: str) -> AnalysisConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    tb = raw.get("tensorboard", {})
    ckpt = raw.get("checkpoint", {})
    model = raw.get("model", {})
    output = raw.get("output", {})

    return AnalysisConfig(
        run_name=raw.get("run_name", ""),
        save_dir=raw.get("save_dir", "saves"),
        tensorboard_enabled=tb.get("enabled", True),
        checkpoint_enabled=ckpt.get("enabled", True),
        checkpoint_dir=ckpt.get("checkpoint_dir", ""),
        prompts_file=ckpt.get("prompts_file", ""),
        prompts=ckpt.get("prompts", []),
        num_samples=ckpt.get("num_samples", 4),
        gen_kwargs=ckpt.get("gen_kwargs", {}),
        base_model=model.get("base_model", "stabilityai/stable-diffusion-3.5-medium"),
        dtype=model.get("dtype", "bfloat16"),
        device=model.get("device", "cuda"),
        cache_dir=model.get("cache_dir", ""),
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
    # Try to infer from checkpoint_dir
    ckpt = config.checkpoint_dir
    if ckpt:
        # e.g. "saves/sd3-5_lora_nft_20260601_115058/checkpoints/" → run_name
        ckpt = ckpt.rstrip("/")
        parent = os.path.dirname(ckpt)
        if os.path.basename(ckpt) == "checkpoints":
            return os.path.basename(parent)
        return os.path.basename(ckpt)
    # Try scanning saves/ for tensorboard dirs
    tb_root = os.path.join(config.save_dir, "tensorboard")
    if os.path.isdir(tb_root):
        runs = sorted(os.listdir(tb_root))
        if runs:
            return runs[-1]  # most recent
    raise ValueError(
        "Cannot resolve run_name. Set run_name in config or provide checkpoint_dir."
    )


def _resolve_checkpoint_dir(config: AnalysisConfig, run_name: str) -> str:
    if config.checkpoint_dir:
        return config.checkpoint_dir
    return os.path.join(config.save_dir, run_name, "checkpoints")


def _load_prompts(config: AnalysisConfig) -> List[str]:
    """Load prompts from file or inline list."""
    if config.prompts:
        return list(config.prompts)
    if config.prompts_file:
        path = config.prompts_file
        if not os.path.isabs(path):
            # Try relative to cwd
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
    computer: StandaloneRewardComputer,
    images: List,
    prompts: List[str],
    batch_size: int = 16,
) -> Dict[str, List[float]]:
    """Score (image, prompt) pairs. Uses a dummy prompt if none provided."""
    if not prompts:
        prompts = ["a photo"] * len(images)
    if len(prompts) != len(images):
        # Repeat prompts cyclically if lengths mismatch
        prompts = [prompts[i % len(prompts)] for i in range(len(images))]
    return computer.compute(images, prompts, batch_size=batch_size)


def _build_step_data(
    step: int,
    rewards_dict: Dict[str, List[float]],
    reward_names: List[str],
) -> Dict[str, Any]:
    """Pack reward scores for one step into the format expected by plotting fns."""
    n_samples = len(rewards_dict[reward_names[0]]) if reward_names else 0
    if n_samples == 0:
        return {"points": np.empty((0, len(reward_names))), "step": step}
    pts = np.column_stack([rewards_dict[name] for name in reward_names])
    # Filter NaN/Inf
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    return {"points": pts, "step": step, "n_total": n_samples, "n_valid": len(pts)}


# ---------------------------------------------------------------------------
# Workflow: TensorBoard → rewards → hulls
# ---------------------------------------------------------------------------


def _run_tensorboard(
    config: AnalysisConfig,
    run_name: str,
    computer: StandaloneRewardComputer,
    all_prompts: List[str],
    output_dir: str,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Extract images from TensorBoard, score them, return step→data dict."""
    tb_dir = os.path.join(config.save_dir, "tensorboard", run_name)
    if not os.path.isdir(tb_dir):
        print(f"[TensorBoard] Directory not found: {tb_dir} — skipping")
        return None

    print(f"[TensorBoard] Extracting images from {tb_dir} ...")
    try:
        images_by_step, datasets = extract_images_from_tensorboard(tb_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [ERROR] {exc}")
        return None

    if not images_by_step:
        print("  No rollout images found in TensorBoard events.")
        return None

    print(f"  Found {len(images_by_step)} steps with images across datasets: {datasets}")
    total_imgs = sum(len(v) for v in images_by_step.values())
    print(f"  Total images: {total_imgs}")

    # Score images per step
    all_step_data: Dict[int, Dict[str, Any]] = {}
    for step in sorted(images_by_step):
        entries = images_by_step[step]
        images = [e["image"] for e in entries]
        # Map prompts by image index (cyclically)
        prompt_per_img = []
        for i in range(len(images)):
            if i < len(all_prompts):
                prompt_per_img.append(all_prompts[i])
            else:
                prompt_per_img.append("a photo")
        print(f"    Step {step}: scoring {len(images)} images ...")
        rewards = _score_images(computer, images, prompt_per_img)
        all_step_data[step] = _build_step_data(step, rewards, computer.reward_names)

    # Plot and save
    tb_out = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_out, exist_ok=True)

    _save_reward_data(all_step_data, computer.reward_names, tb_out, "tensorboard")
    n_models = len(computer.reward_names)
    if n_models >= 2:
        plot_convex_hulls_2d(
            all_step_data, computer.reward_names,
            os.path.join(tb_out, "convex_hulls_2d.png"),
            title="Training Rollout Reward Convex Hulls (TensorBoard)",
            label_name="Step",
        )
    else:
        plot_distribution_1d(
            all_step_data, computer.reward_names[0],
            os.path.join(tb_out, "distribution_1d.png"),
            title="Training Rollout Reward Distribution (TensorBoard)",
            label_name="Step",
        )
    print(f"  TensorBoard results saved to {tb_out}/")
    return all_step_data


# ---------------------------------------------------------------------------
# Workflow: Checkpoint inference → rewards → hulls
# ---------------------------------------------------------------------------


def _run_checkpoints(
    config: AnalysisConfig,
    run_name: str,
    computer: StandaloneRewardComputer,
    prompts: List[str],
    output_dir: str,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Discover checkpoints, generate images, score them, return epoch→data dict."""
    ckpt_dir = _resolve_checkpoint_dir(config, run_name)
    try:
        checkpoints = discover_checkpoints(ckpt_dir)
    except FileNotFoundError as exc:
        print(f"[Checkpoint] {exc} — skipping")
        return None

    if not checkpoints:
        print(f"[Checkpoint] No checkpoint-* subdirectories in {ckpt_dir}")
        return None

    if not prompts:
        print("[Checkpoint] No prompts configured — skipping checkpoint analysis")
        return None

    print(f"[Checkpoint] Found {len(checkpoints)} checkpoints: "
          f"{[e for e, _ in checkpoints]}")
    print(f"  Prompts: {len(prompts)}, samples per prompt: {config.num_samples}")

    runner = CheckpointRunner(config.base_model, config.dtype, device=config.device)

    all_epoch_data: Dict[int, Dict[str, Any]] = {}
    for epoch, ckpt_path in checkpoints:
        print(f"  Checkpoint epoch={epoch}: generating images ...")
        images = runner.generate_for_checkpoint(
            ckpt_path, prompts, config.num_samples, config.gen_kwargs,
        )
        # Build per-image prompt list: [p0]*N + [p1]*N + ...
        expanded_prompts = [p for p in prompts for _ in range(config.num_samples)]
        print(f"    Scoring {len(images)} images ...")
        rewards = _score_images(computer, images, expanded_prompts)
        all_epoch_data[epoch] = _build_step_data(epoch, rewards, computer.reward_names)

    # Cleanup
    del runner

    # Plot and save
    ckpt_out = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_out, exist_ok=True)

    _save_reward_data(all_epoch_data, computer.reward_names, ckpt_out, "checkpoints")
    n_models = len(computer.reward_names)
    if n_models >= 2:
        plot_convex_hulls_2d(
            all_epoch_data, computer.reward_names,
            os.path.join(ckpt_out, "convex_hulls_2d.png"),
            title="Checkpoint Reward Convex Hulls (Test Prompts)",
            label_name="Epoch",
        )
    else:
        plot_distribution_1d(
            all_epoch_data, computer.reward_names[0],
            os.path.join(ckpt_out, "distribution_1d.png"),
            title="Checkpoint Reward Distribution (Test Prompts)",
            label_name="Epoch",
        )
    print(f"  Checkpoint results saved to {ckpt_out}/")
    return all_epoch_data


# ---------------------------------------------------------------------------
# Save / load reward data
# ---------------------------------------------------------------------------


def _save_reward_data(
    all_data: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_dir: str,
    prefix: str,
) -> None:
    """Persist raw reward scores as JSON for reproducibility."""
    records = []
    for step, data in sorted(all_data.items()):
        pts = data.get("points")
        rec = {
            "step": step,
            "n_total": data.get("n_total", len(pts) if pts is not None else 0),
            "n_valid": data.get("n_valid", len(pts) if pts is not None else 0),
        }
        if pts is not None and len(pts) > 0:
            for di, name in enumerate(reward_names):
                rec[name] = pts[:, di].tolist()
        else:
            for name in reward_names:
                rec[name] = []
        records.append(rec)

    path = os.path.join(output_dir, f"{prefix}_reward_data.json")
    with open(path, "w") as f:
        json.dump({
            "reward_models": reward_names,
            "records": records,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------


def _run_combined(
    tb_data: Dict[int, Dict[str, Any]],
    ckpt_data: Dict[int, Dict[str, Any]],
    reward_names: List[str],
    output_dir: str,
) -> None:
    """Overlay TensorBoard and checkpoint convex hulls."""
    combined_out = os.path.join(output_dir, "combined")
    os.makedirs(combined_out, exist_ok=True)

    n_models = len(reward_names)
    if n_models >= 2:
        plot_combined_convex_hulls_2d(
            tb_data, ckpt_data, reward_names,
            os.path.join(combined_out, "convex_hulls_2d.png"),
            label_a="Training (TB)",
            label_b="Checkpoints",
            title="Combined Reward Convex Hulls: Training vs Checkpoints",
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

    # Resolve paths
    run_name = _resolve_run_name(config)
    print(f"Run name: {run_name}")

    output_dir = os.path.join(config.output_dir, "reward_convex_hull_analysis", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save effective config for reproducibility
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump({k: v for k, v in config.__dict__.items() if not k.startswith("_")}, f)

    # Prompts for TensorBoard (eval) vs Checkpoints (test)
    all_prompts = _load_prompts(config)

    # Load reward computer (shared between both sources)
    print(f"Loading reward models: {[r.get('name', r.get('reward_model', '?')) for r in config.rewards]}")
    # Set HF_HOME so from_pretrained finds cached models at the right path
    if config.cache_dir:
        os.environ["HF_HOME"] = os.path.expanduser(config.cache_dir)
        print(f"  HF_HOME set to: {os.environ['HF_HOME']}")
    # Check GPU memory if using cuda
    import torch as _torch
    device = config.device
    if device == "cuda" and _torch.cuda.is_available():
        free_mem = _torch.cuda.mem_get_info()[0] / (1024**3)
        if free_mem < 4.0:
            print(f"  [WARN] Only {free_mem:.1f} GiB free GPU memory; consider device: \"cpu\"")
    computer = StandaloneRewardComputer(config.rewards, device=device)
    print(f"  Reward dimensions: {computer.reward_names}")

    tb_data: Optional[Dict[int, Dict[str, Any]]] = None
    ckpt_data: Optional[Dict[int, Dict[str, Any]]] = None

    # --- TensorBoard analysis ---
    if config.tensorboard_enabled:
        tb_data = _run_tensorboard(config, run_name, computer, all_prompts, output_dir)

    # --- Checkpoint analysis ---
    if config.checkpoint_enabled:
        ckpt_data = _run_checkpoints(config, run_name, computer, all_prompts, output_dir)

    # --- Combined ---
    if tb_data is not None and ckpt_data is not None and len(computer.reward_names) >= 2:
        _run_combined(tb_data, ckpt_data, computer.reward_names, output_dir)

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
