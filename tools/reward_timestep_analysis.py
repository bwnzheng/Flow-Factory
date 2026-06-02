#!/usr/bin/env python3
"""
Reward Per-Timestep Analysis Tool for NFT Checkpoints.

Computes ||d(reward)/d(v_pred_t)|| — the gradient norm of each reward w.r.t.
the velocity prediction at each timestep — by backpropagating through the
full ODE denoising trajectory. This reveals which denoising stages each
reward is most sensitive to.

Usage:
    python tools/reward_timestep_analysis.py -c tools/reward_timestep_analysis.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── flow_factory imports ──────────────────────────────────────────────
from flow_factory.scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    set_scheduler_timesteps,
)

# ── matplotlib (optional) ─────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AnalysisConfig:
    base_model: str = "stabilityai/stable-diffusion-3.5-medium"
    checkpoint: str = ""
    dtype: str = "bfloat16"

    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    seed: int = 42

    num_analysis_timesteps: int = 20
    prompts: List[str] = field(default_factory=list)

    rewards: List[Dict[str, Any]] = field(default_factory=list)

    output_dir: str = "analysis_output"


def parse_config(path: str) -> AnalysisConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    model = raw.get("model", {})
    inference = raw.get("inference", {})
    analysis = raw.get("analysis", {})
    output = raw.get("output", {})

    return AnalysisConfig(
        base_model=model.get("base_model", "stabilityai/stable-diffusion-3.5-medium"),
        checkpoint=model.get("checkpoint", ""),
        dtype=model.get("dtype", "bfloat16"),
        num_inference_steps=inference.get("num_inference_steps", 50),
        guidance_scale=inference.get("guidance_scale", 1.0),
        height=inference.get("height", 512),
        width=inference.get("width", 512),
        seed=inference.get("seed", 42),
        num_analysis_timesteps=analysis.get("num_analysis_timesteps", 20),
        prompts=analysis.get("prompts", []),
        rewards=raw.get("rewards", []),
        output_dir=output.get("dir", "analysis_output"),
    )


def resolve_dtype(dtype_str: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]


# ============================================================================
# Pipeline + LoRA loading
# ============================================================================

def load_pipeline(config: AnalysisConfig) -> StableDiffusion3Pipeline:
    dtype = resolve_dtype(config.dtype)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    scheduler = FlowMatchEulerDiscreteSDEScheduler.from_config(
        pipe.scheduler.config,
        dynamics_type="ODE",
    )
    scheduler.eval()
    pipe.scheduler = scheduler
    pipe = pipe.to("cuda")
    return pipe


def load_lora(pipe: StableDiffusion3Pipeline, checkpoint_path: str, dtype: torch.dtype):
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        checkpoint_path,
        torch_dtype=dtype,
    )
    pipe.transformer = pipe.transformer.merge_and_unload()


# ============================================================================
# Differentiable reward functions
# ============================================================================

# CLIP normalization constants (OpenAI CLIP)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _preprocess_image_for_clip(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert VAE output tensor (in [-1, 1]) to CLIP-preprocessed tensor.
    All operations are differentiable.
    """
    image_01 = (image_tensor + 1.0) / 2.0
    image_01 = image_01.clamp(0.0, 1.0)
    image_224 = F.interpolate(image_01, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor(CLIP_MEAN, device=image_224.device, dtype=image_224.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=image_224.device, dtype=image_224.dtype).view(1, 3, 1, 1)
    return (image_224 - mean) / std


class DifferentiableCLIPReward:
    """CLIP cosine-similarity reward, fully differentiable."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name).eval().cuda()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> torch.Tensor:
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs).pooler_output
        return F.normalize(text_features, p=2, dim=-1)

    def __call__(self, image_tensor: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        pixel_values = _preprocess_image_for_clip(image_tensor)
        image_features = self.model.get_image_features(pixel_values=pixel_values).pooler_output
        image_features = F.normalize(image_features, p=2, dim=-1)
        return (image_features * text_features).sum(dim=-1).squeeze(0)


class DifferentiablePickScore:
    """PickScore reward (CLIP-ViT-H based), fully differentiable."""

    def __init__(self):
        self.model = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().cuda()
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.logit_scale = self.model.logit_scale.exp()

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> torch.Tensor:
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs).pooler_output
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    def __call__(self, image_tensor: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        pixel_values = _preprocess_image_for_clip(image_tensor)
        image_features = self.model.get_image_features(pixel_values=pixel_values).pooler_output
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        raw_score = self.logit_scale * (text_features * image_features).sum(dim=-1)
        return (raw_score / 26.0).squeeze(0)


def load_reward_fn(reward_cfg: Dict[str, Any]) -> Tuple[str, Any]:
    """Load a differentiable reward function by name."""
    rtype = reward_cfg.get("reward_model", reward_cfg.get("type", ""))
    rname = reward_cfg.get("name", rtype)
    if rtype in ("CLIP", "clip"):
        model_name = reward_cfg.get("model_name", "openai/clip-vit-large-patch14")
        return rname, DifferentiableCLIPReward(model_name=model_name)
    elif rtype in ("PickScore", "pickscore"):
        return rname, DifferentiablePickScore()
    else:
        raise ValueError(f"Unsupported reward type: {rtype}. Supported: CLIP, PickScore")


# ============================================================================
# Core gradient analysis
# ============================================================================

def _sample_analysis_indices(num_steps: int, num_samples: int) -> List[int]:
    """Sample evenly-spaced timestep indices from [0, num_steps-1]."""
    if num_samples >= num_steps:
        return list(range(num_steps))
    indices = np.linspace(0, num_steps - 1, num_samples).round().astype(int).tolist()
    return sorted(set(indices))


def _transformer_forward(transformer, hidden_states, timestep, encoder_hidden_states, pooled_projections):
    """Wrapper for checkpointing — must be a free function for pickle support."""
    return transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False,
    )[0]


class GradientAnalyzer:
    """Runs the ODE trajectory with gradient tracking and computes per-timestep reward sensitivities."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.dtype = resolve_dtype(config.dtype)
        self.device = torch.device("cuda")

        print("Loading SD3.5 pipeline...")
        self.pipe = load_pipeline(config)

        if config.checkpoint:
            print(f"Loading LoRA checkpoint: {config.checkpoint}")
            load_lora(self.pipe, config.checkpoint, self.dtype)

        self.pipe.transformer.eval()

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        do_cfg = self.config.guidance_scale > 1.0
        result = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=self.device,
            do_classifier_free_guidance=do_cfg,
        )
        if do_cfg:
            prompt_embeds, neg_embeds, pooled, neg_pooled = result
            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled,
                "negative_prompt_embeds": neg_embeds,
                "negative_pooled_prompt_embeds": neg_pooled,
                "do_cfg": True,
            }
        else:
            prompt_embeds, _, pooled, _ = result
            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled,
                "do_cfg": False,
            }

    def run_trajectory(self, prompt_embeds_info: Dict) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Run the ODE denoising trajectory with gradient tracking.
        Returns (v_preds, x_final, timesteps) where v_preds retains grad info.
        """
        pipe = self.pipe
        config = self.config
        do_cfg = prompt_embeds_info["do_cfg"]
        prompt_embeds = prompt_embeds_info["prompt_embeds"]
        pooled = prompt_embeds_info["pooled_prompt_embeds"]

        if do_cfg:
            neg_embeds = prompt_embeds_info["negative_prompt_embeds"]
            neg_pooled = prompt_embeds_info["negative_pooled_prompt_embeds"]

        # Prepare latents
        num_channels = pipe.transformer.config.in_channels
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        latents = pipe.prepare_latents(
            1, num_channels, config.height, config.width,
            self.dtype, self.device, generator,
        )

        # Set scheduler timesteps
        patch_size = pipe.transformer.config.patch_size
        image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
        timesteps = set_scheduler_timesteps(
            scheduler=pipe.scheduler,
            num_inference_steps=config.num_inference_steps,
            seq_len=image_seq_len,
            device=self.device,
        )
        # sigmas may be torch.Tensor or np.ndarray depending on diffusers version
        sigmas = pipe.scheduler.sigmas
        if isinstance(sigmas, np.ndarray):
            sigmas = torch.from_numpy(sigmas).to(self.device)
        else:
            sigmas = sigmas.to(self.device)

        v_preds = []
        x_t = latents

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_embed = t.expand(1).to(self.dtype)

            if do_cfg:
                x_input = torch.cat([x_t, x_t], dim=0)
                t_input = t_embed.repeat(2)
                pe_input = torch.cat([neg_embeds, prompt_embeds], dim=0)
                pp_input = torch.cat([neg_pooled, pooled], dim=0)
            else:
                x_input = x_t
                t_input = t_embed
                pe_input = prompt_embeds
                pp_input = pooled

            # Checkpointed transformer forward
            v_pred = torch.utils.checkpoint.checkpoint(
                _transformer_forward,
                self.pipe.transformer, x_input, t_input, pe_input, pp_input,
                use_reentrant=False,
            )

            if do_cfg:
                v_pred_uncond, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_uncond + config.guidance_scale * (v_pred_text - v_pred_uncond)

            v_preds.append(v_pred)

            # ODE step: x_{t+1} = x_t + v_pred * (sigma_next - sigma_t)
            sigma_t = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.tensor(0.0, device=self.device)
            dt = sigma_next - sigma_t
            x_t = x_t + v_pred * dt

        return v_preds, x_t, timesteps

    def compute_gradient_norms(
        self,
        prompt: str,
        prompt_embeds_info: Dict,
        reward_fn,
        reward_name: str,
    ) -> Tuple[Dict[int, float], float, torch.Tensor, torch.Tensor]:
        """
        Run trajectory, compute reward, backprop, and extract ||d(reward)/d(v_pred)||
        at analysis timesteps.

        Returns:
            grad_norms: dict mapping timestep index → gradient norm
            reward_value: scalar reward value
            timesteps: the scheduler timesteps (shifted, in [0, 1000])
            sigmas: the scheduler sigmas (in [0, 1])
        """
        config = self.config
        analysis_indices = _sample_analysis_indices(config.num_inference_steps, config.num_analysis_timesteps)
        print(f"  [{reward_name}] Analysis timestep indices: {analysis_indices}")

        # Run forward trajectory
        v_preds, x_final, timesteps = self.run_trajectory(prompt_embeds_info)

        # Register hooks on analysis v_preds
        grad_norms: Dict[int, float] = {}

        for idx in analysis_indices:
            def _hook(grad: torch.Tensor, _idx: int = idx) -> None:
                grad_norms[_idx] = grad.detach().norm().item()

            v_preds[idx].register_hook(_hook)

        # VAE decode (differentiable)
        latents = x_final / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        latents = latents.to(self.dtype)
        decoded = self.pipe.vae.decode(latents, return_dict=False)[0]

        # Encode text (non-differentiable, cached)
        text_features = reward_fn._encode_text(prompt)

        # Compute reward and backward
        reward_val = reward_fn(decoded, text_features)
        reward_val.backward()

        # Map analysis indices to sigma values (continuous time in [0, 1])
        sigmas = self.pipe.scheduler.sigmas
        if isinstance(sigmas, np.ndarray):
            sigmas = torch.from_numpy(sigmas).float()
        else:
            sigmas = sigmas.float()

        return dict(grad_norms), reward_val.item(), timesteps, sigmas


# ============================================================================
# Visualization
# ============================================================================

def save_results(results: List[Dict], output_dir: str):
    """Save raw results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    out = {
        "results": [
            {
                "prompt": r["prompt"],
                "reward": r["reward"],
                "reward_value": r["reward_value"],
                "timestep_indices": sorted(r["grad_norms"].keys()),
                "sigma_values": [float(r["sigmas"][i]) for i in sorted(r["grad_norms"].keys())],
                "timestep_values": [float(r["timesteps"][i]) for i in sorted(r["grad_norms"].keys())],
                "grad_norms": [r["grad_norms"][i] for i in sorted(r["grad_norms"].keys())],
            }
            for r in results
        ],
    }
    path = os.path.join(output_dir, "gradient_norms.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


def plot_gradient_norms(results: List[Dict], output_dir: str):
    """Plot gradient norms vs sigma (noise level) for each reward."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots.")
        return

    from collections import defaultdict

    reward_data: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        rname = r["reward"]
        indices = sorted(r["grad_norms"].keys())
        sigmas = r["sigmas"]
        for t_idx in indices:
            sigma_val = float(sigmas[t_idx])
            reward_data[rname][sigma_val].append(r["grad_norms"][t_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Raw gradient norms vs sigma (sigma=1 = pure noise, sigma=0 = clean)
    ax = axes[0]
    for rname, t_data in reward_data.items():
        sigma_vals = sorted(t_data.keys(), reverse=True)  # high sigma first (noisy → clean)
        means = [np.mean(t_data[s]) for s in sigma_vals]
        stds = [np.std(t_data[s]) for s in sigma_vals]
        ax.errorbar(sigma_vals, means, yerr=stds, marker="o", label=rname, capsize=3)
    ax.set_xlabel("Sigma (noise level: 1 = pure noise, 0 = clean)")
    ax.set_ylabel("||d(reward)/d(v_pred)||")
    ax.set_title("Per-timestep Reward Gradient Norm")
    ax.invert_xaxis()  # high sigma (noisy) on left, low sigma (clean) on right
    ax.legend()

    # Plot 2: Normalized
    ax = axes[1]
    for rname, t_data in reward_data.items():
        sigma_vals = sorted(t_data.keys(), reverse=True)
        means = np.array([np.mean(t_data[s]) for s in sigma_vals])
        means = (means - means.min()) / (means.max() - means.min() + 1e-8)
        ax.plot(sigma_vals, means, marker="o", label=rname)
    ax.set_xlabel("Sigma (noise level: 1 = pure noise, 0 = clean)")
    ax.set_ylabel("Normalized gradient norm")
    ax.set_title("Normalized Per-timestep Sensitivity")
    ax.invert_xaxis()
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "gradient_norms.png")
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main(config_path: str):
    config = parse_config(config_path)
    print(f"Output directory: {config.output_dir}")
    os.makedirs(config.output_dir, exist_ok=True)

    if not config.prompts:
        print("ERROR: No prompts specified in config.")
        sys.exit(1)
    if not config.rewards:
        print("ERROR: No rewards specified in config.")
        sys.exit(1)

    analyzer = GradientAnalyzer(config)

    # Load reward functions (each wraps a CLIP model)
    reward_fns: List[Tuple[str, Any]] = []
    for rcfg in config.rewards:
        name, fn = load_reward_fn(rcfg)
        reward_fns.append((name, fn))
        print(f"Loaded reward: {name}")

    all_results: List[Dict] = []

    for prompt_idx, prompt in enumerate(config.prompts):
        print(f"\n{'='*60}")
        print(f"Prompt [{prompt_idx + 1}/{len(config.prompts)}]: {prompt}")
        print(f"{'='*60}")

        # Encode prompt once
        prompt_embeds_info = analyzer._encode_prompt(prompt)

        for reward_name, reward_fn in reward_fns:
            print(f"  Computing gradients for {reward_name}...")
            grad_norms, reward_value, timesteps, sigmas = analyzer.compute_gradient_norms(
                prompt, prompt_embeds_info, reward_fn, reward_name,
            )
            print(f"    Reward value: {reward_value:.4f}")
            print(f"    Gradient norms range: [{min(grad_norms.values()):.6f}, {max(grad_norms.values()):.6f}]")

            all_results.append({
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "reward": reward_name,
                "reward_value": reward_value,
                "grad_norms": grad_norms,
                "timesteps": timesteps,
                "sigmas": sigmas,
            })

            # Clean up GPU memory
            torch.cuda.empty_cache()

    # Save and visualize
    save_results(all_results, config.output_dir)
    plot_gradient_norms(all_results, config.output_dir)

    print(f"\nDone. Results in {config.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Per-Timestep Analysis")
    parser.add_argument("-c", "--config", default="tools/reward_timestep_analysis.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    main(args.config)
