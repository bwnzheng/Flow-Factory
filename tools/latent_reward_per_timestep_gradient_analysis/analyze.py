#!/usr/bin/env python3
"""
Latent Reward Gradient Analysis Tool.

Computes ||d(reward)/d(v_pred_t)|| — the gradient norm of each reward w.r.t.
the velocity prediction at each timestep — by backpropagating through the
full ODE denoising trajectory. Supports batch analysis across multiple
checkpoints with multiprocessing (one GPU per process).

Usage:
    python tools/latent_reward_per_timestep_gradient_analysis/analyze.py -c tools/latent_reward_per_timestep_gradient_analysis/config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
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
from transformers import CLIPModel, CLIPProcessor

from flow_factory.scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    set_scheduler_timesteps,
)

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
    checkpoint_dir: str = ""
    dtype: str = "bfloat16"
    num_gpus: int = 1

    num_inference_steps: int = 50
    per_device_batch_size: int = 1   # >1 = batch multiple prompts per trajectory run
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    seed: int = 42

    num_analysis_timesteps: int = 20
    max_prompts: int = 0       # 0 = all, >0 = only first N prompts
    max_epochs: int = 0        # 0 = no limit
    prompts_file: str = ""
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
        checkpoint_dir=model.get("checkpoint_dir", ""),
        dtype=model.get("dtype", "bfloat16"),
        num_gpus=model.get("num_gpus", 1),
        num_inference_steps=inference.get("num_inference_steps", 50),
        per_device_batch_size=inference.get("per_device_batch_size", 1),
        guidance_scale=inference.get("guidance_scale", 1.0),
        height=inference.get("height", 512),
        width=inference.get("width", 512),
        seed=inference.get("seed", 42),
        num_analysis_timesteps=analysis.get("num_analysis_timesteps", 20),
        max_prompts=analysis.get("max_prompts", 0),
        max_epochs=analysis.get("max_epochs", 0),
        prompts_file=analysis.get("prompts_file", ""),
        prompts=analysis.get("prompts", []),
        rewards=raw.get("rewards", []),
        output_dir=output.get("dir", "analysis_output"),
    )


def resolve_dtype(dtype_str: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]


# ============================================================================
# Checkpoint discovery
# ============================================================================

def discover_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoint-N subdirectories, return sorted list of (epoch, path)."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    entries = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("checkpoint-"):
            try:
                epoch = int(name.split("checkpoint-")[1])
            except ValueError:
                continue
            path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(path):
                entries.append((epoch, path))
    return sorted(entries)


# ============================================================================
# Pipeline + LoRA loading
# ============================================================================

def load_pipeline(base_model: str, dtype_str: str) -> StableDiffusion3Pipeline:
    dtype = resolve_dtype(dtype_str)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    scheduler = FlowMatchEulerDiscreteSDEScheduler.from_config(
        pipe.scheduler.config, dynamics_type="ODE",
    )
    scheduler.eval()
    pipe.scheduler = scheduler
    pipe = pipe.to("cuda")
    return pipe


def load_lora(pipe: StableDiffusion3Pipeline, checkpoint_path: str, dtype: torch.dtype):
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer, checkpoint_path, torch_dtype=dtype,
    )
    pipe.transformer = pipe.transformer.merge_and_unload()


# ============================================================================
# Differentiable reward functions
# ============================================================================

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _preprocess_image_for_clip(image_tensor: torch.Tensor) -> torch.Tensor:
    image_01 = (image_tensor + 1.0) / 2.0
    image_01 = image_01.clamp(0.0, 1.0)
    image_224 = F.interpolate(image_01, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=image_224.device, dtype=image_224.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=image_224.device, dtype=image_224.dtype).view(1, 3, 1, 1)
    return (image_224 - mean) / std


class DifferentiableCLIPReward:
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
    if num_samples >= num_steps:
        return list(range(num_steps))
    indices = np.linspace(0, num_steps - 1, num_samples).round().astype(int).tolist()
    return sorted(set(indices))


def _transformer_forward(transformer, hidden_states, timestep, encoder_hidden_states, pooled_projections):
    return transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False,
    )[0]


class GradientAnalyzer:
    def __init__(self, config: AnalysisConfig, checkpoint_path: str = ""):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.dtype = resolve_dtype(config.dtype)
        self.device = torch.device("cuda")

        print(f"[GPU {torch.cuda.current_device()}] Loading SD3.5 pipeline...")
        self.pipe = load_pipeline(config.base_model, config.dtype)

        if checkpoint_path:
            print(f"[GPU {torch.cuda.current_device()}] Loading LoRA: {checkpoint_path}")
            load_lora(self.pipe, checkpoint_path, self.dtype)

        self.pipe.transformer.eval()

    @torch.no_grad()
    def pre_encode_all_prompts(self, prompts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Encode all prompts into SD3.5 text embeddings (batched), store on CPU.
        After encoding, frees T5 and CLIP text encoders (~10 GB saved).
        """
        chunk_size = 32
        all_embeds = []
        for start in range(0, len(prompts), chunk_size):
            chunk = prompts[start:start + chunk_size]
            do_cfg = self.config.guidance_scale > 1.0
            result = self.pipe.encode_prompt(
                prompt=chunk, prompt_2=chunk, prompt_3=chunk,
                device=self.device, do_classifier_free_guidance=do_cfg,
            )
            prompt_embeds, neg_embeds, pooled, neg_pooled = result
            for b in range(len(chunk)):
                info = {
                    "prompt_embeds": prompt_embeds[b:b + 1].cpu(),
                    "pooled_prompt_embeds": pooled[b:b + 1].cpu(),
                    "do_cfg": do_cfg,
                }
                if do_cfg and neg_embeds is not None and neg_pooled is not None:
                    info["negative_prompt_embeds"] = neg_embeds[b:b + 1].cpu()
                    info["negative_pooled_prompt_embeds"] = neg_pooled[b:b + 1].cpu()
                all_embeds.append(info)

        # Free text encoders — no longer needed, saves ~10 GB GPU memory
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            if hasattr(self.pipe, attr):
                setattr(self.pipe, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {torch.cuda.current_device()}] Text encoders freed after pre-encoding")
        return all_embeds

    def run_trajectory(
        self, batch_embeds: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ODE trajectory on a batch of pre-encoded prompt embeddings."""
        pipe = self.pipe
        config = self.config
        batch_size = len(batch_embeds)
        do_cfg = batch_embeds[0]["do_cfg"]

        # Concatenate per-prompt embeddings into batched tensors on GPU
        prompt_embeds = torch.cat([e["prompt_embeds"] for e in batch_embeds], dim=0).to(self.device)
        pooled = torch.cat([e["pooled_prompt_embeds"] for e in batch_embeds], dim=0).to(self.device)
        if do_cfg:
            neg_embeds = torch.cat([e["negative_prompt_embeds"] for e in batch_embeds], dim=0).to(self.device)
            neg_pooled = torch.cat([e["negative_pooled_prompt_embeds"] for e in batch_embeds], dim=0).to(self.device)

        num_channels = pipe.transformer.config.in_channels
        # Different noise per sample
        latents_list = []
        for b in range(batch_size):
            g = torch.Generator(device="cpu").manual_seed(config.seed + b)
            lat = pipe.prepare_latents(
                1, num_channels, config.height, config.width,
                self.dtype, self.device, g,
            )
            latents_list.append(lat)
        latents = torch.cat(latents_list, dim=0)
        latents.requires_grad_(True)

        patch_size = pipe.transformer.config.patch_size
        image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
        timesteps = set_scheduler_timesteps(
            scheduler=pipe.scheduler, num_inference_steps=config.num_inference_steps,
            seq_len=image_seq_len, device=self.device,
        )
        sigmas = pipe.scheduler.sigmas
        if isinstance(sigmas, np.ndarray):
            sigmas = torch.from_numpy(sigmas).to(self.device)
        else:
            sigmas = sigmas.to(self.device)

        v_preds = []
        x_t = latents
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_embed = t.expand(batch_size).to(self.dtype)
            if do_cfg:
                x_input = torch.cat([x_t, x_t], dim=0)
                t_input = t_embed.repeat(2)
                pe_input = torch.cat([neg_embeds, prompt_embeds], dim=0)
                pp_input = torch.cat([neg_pooled, pooled], dim=0)
            else:
                x_input = x_t; t_input = t_embed; pe_input = prompt_embeds; pp_input = pooled

            v_pred = torch.utils.checkpoint.checkpoint(
                _transformer_forward, self.pipe.transformer,
                x_input, t_input, pe_input, pp_input,
                use_reentrant=False,
            )
            if do_cfg:
                v_pred_uncond, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_uncond + config.guidance_scale * (v_pred_text - v_pred_uncond)
            v_preds.append(v_pred)

            sigma_t = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.tensor(0.0, device=self.device)
            x_t = x_t + v_pred * (sigma_next - sigma_t)

        return v_preds, x_t, timesteps

    def compute_gradient_norms_batched(
        self, prompts: List[str], batch_embeds: List[Dict[str, torch.Tensor]],
        reward_fn, reward_name: str, text_features_cache: Dict[str, torch.Tensor],
    ) -> Tuple[List[Tuple[Dict[int, float], Dict[int, torch.Tensor], float]], torch.Tensor, torch.Tensor]:
        """Run a batched trajectory using pre-encoded embeddings, compute per-prompt results.

        Returns:
            per_prompt: list of (grad_norms, grad_vectors, reward_value) per prompt
            timesteps: scheduler timesteps
            sigmas: scheduler sigmas
        """
        config = self.config
        batch_size = len(prompts)
        analysis_indices = _sample_analysis_indices(config.num_inference_steps, config.num_analysis_timesteps)

        v_preds, x_final, timesteps = self.run_trajectory(batch_embeds)

        # VAE decode (batched)
        latents = x_final / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        latents = latents.to(self.dtype)
        decoded = self.pipe.vae.decode(latents, return_dict=False)[0]  # (B, C, H, W)

        # Compute reward using pre-encoded text features
        rewards_list = []
        for b in range(batch_size):
            rv = reward_fn(decoded[b:b + 1], text_features_cache[prompts[b]])
            rewards_list.append(rv)

        # Per-sample gradient extraction via autograd.grad with grad_outputs
        target_v_preds = [v_preds[idx] for idx in analysis_indices]
        all_grad_norms: List[Dict[int, float]] = []
        all_grad_vectors: List[Dict[int, torch.Tensor]] = []
        all_reward_values: List[float] = []

        for b in range(batch_size):
            retain = (b < batch_size - 1)
            grads = torch.autograd.grad(
                rewards_list[b], target_v_preds,
                retain_graph=retain, allow_unused=False,
            )
            # Extract sample b's gradient from each batched v_pred
            grad_norms = {idx: g[b:b + 1].detach().norm().item() for idx, g in zip(analysis_indices, grads)}
            grad_vectors = {idx: g[b].detach().flatten().cpu() for idx, g in zip(analysis_indices, grads)}
            all_grad_norms.append(grad_norms)
            all_grad_vectors.append(grad_vectors)
            all_reward_values.append(rewards_list[b].item())

        del grads, target_v_preds, v_preds, x_final, decoded, latents, rewards_list
        gc.collect()
        torch.cuda.empty_cache()

        sigmas = self.pipe.scheduler.sigmas
        if isinstance(sigmas, np.ndarray):
            sigmas = torch.from_numpy(sigmas).float()
        else:
            sigmas = sigmas.float()

        return list(zip(all_grad_norms, all_grad_vectors, all_reward_values)), timesteps, sigmas


# ============================================================================
# Per-checkpoint worker (for multiprocessing)
# ============================================================================

def _analyze_one_checkpoint(worker_args: Tuple) -> Dict:
    """Run full analysis on a single checkpoint. Entry point for multiprocessing."""
    gpu_id, epoch, checkpoint_path, config_dict, prompts, reward_cfgs = worker_args

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Reconstruct config
    config = AnalysisConfig(**config_dict)

    print(f"\n[GPU {gpu_id}] Checkpoint epoch={epoch} | path={checkpoint_path}")

    # Load analyzer with this checkpoint
    analyzer = GradientAnalyzer(config, checkpoint_path)

    # Load reward functions
    reward_fns = []
    for rcfg in reward_cfgs:
        name, fn = load_reward_fn(rcfg)
        reward_fns.append((name, fn))

    results = []
    batch_size = config.per_device_batch_size

    # Pre-encode: SD3.5 text embeddings (T5 + CLIP) — once for all prompts
    print(f"[GPU {gpu_id}][ckpt {epoch}] Pre-encoding {len(prompts)} prompts...")
    all_sd3_embeds = analyzer.pre_encode_all_prompts(prompts)
    # Pre-encode: reward model text features
    prompt_embeds_cache = {}  # reward_name → {prompt_str: text_features}
    for reward_name, reward_fn in reward_fns:
        prompt_embeds_cache[reward_name] = {}
        for p in prompts:
            prompt_embeds_cache[reward_name][p] = reward_fn._encode_text(p)
    print(f"[GPU {gpu_id}][ckpt {epoch}] Pre-encoding done.")

    # Process prompts in batches using pre-encoded embeddings
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_prompts)))
        batch_embeds = all_sd3_embeds[batch_start:batch_start + batch_size]
        bs = len(batch_prompts)
        print(f"[GPU {gpu_id}][ckpt {epoch}] Batch [{batch_start + 1}-{batch_start + bs}/{len(prompts)}]")

        # Collect gradient vectors for all rewards across this batch
        batch_grads: Dict[str, List[Dict[int, torch.Tensor]]] = {}  # reward_name → [per-prompt {idx: vec}]
        batch_results: List[Dict] = []

        for reward_name, reward_fn in reward_fns:
            per_prompt_data, timesteps, sigmas = analyzer.compute_gradient_norms_batched(
                batch_prompts, batch_embeds, reward_fn, reward_name,
                prompt_embeds_cache[reward_name],
            )
            batch_grads[reward_name] = []
            for b, (grad_norms, grad_vectors, reward_value) in enumerate(per_prompt_data):
                batch_grads[reward_name].append(grad_vectors)
                batch_results.append({
                    "epoch": epoch,
                    "prompt": batch_prompts[b],
                    "prompt_idx": batch_indices[b],
                    "reward": reward_name,
                    "reward_value": reward_value,
                    "grad_norms": grad_norms,
                    "timesteps": timesteps.tolist(),
                    "sigmas": [float(sigmas[i]) for i in range(len(sigmas))],
                })
                print(f"[GPU {gpu_id}][ckpt {epoch}]   [{batch_indices[b]}] {reward_name}: "
                      f"value={reward_value:.4f}, "
                      f"grad_norms=[{min(grad_norms.values()):.6f}, {max(grad_norms.values()):.6f}]")

        # Compute pairwise cosine similarities for each prompt in the batch
        reward_names = [name for name, _ in reward_fns]
        analysis_indices = sorted(batch_grads[reward_names[0]][0].keys())
        sigmas_list = [float(sigmas[i]) for i in range(len(sigmas))]

        for b in range(bs):
            prompt_grads = {rn: batch_grads[rn][b] for rn in reward_names}
            cosine_sim: Dict[str, Dict[int, float]] = {}
            for i in range(len(reward_names)):
                for j in range(i + 1, len(reward_names)):
                    rA, rB = reward_names[i], reward_names[j]
                    key = f"{rA}_vs_{rB}"
                    cosine_sim[key] = {}
                    for idx in analysis_indices:
                        gA = prompt_grads[rA][idx]
                        gB = prompt_grads[rB][idx]
                        cos = (gA @ gB) / (gA.norm() * gB.norm() + 1e-8)
                        cosine_sim[key][idx] = cos.item()

            # Attach cosine sim to this prompt's entries in batch_results
            base_idx = b  # first entry for this prompt is at batch offset b
            for ri in range(len(reward_names)):
                entry_idx = base_idx + ri * bs
                batch_results[entry_idx]["cosine_sim"] = cosine_sim

        results.extend(batch_results)
        del batch_grads, batch_results
        gc.collect()
        torch.cuda.empty_cache()

    # Clean up to free GPU memory before next worker
    del analyzer
    gc.collect()
    torch.cuda.empty_cache()

    return {"epoch": epoch, "results": results}


# ============================================================================
# Visualization
# ============================================================================

def _expected_files(output_dir: str, epoch: int, n_rewards: int) -> List[str]:
    """Return list of expected output file paths for a checkpoint."""
    files = [
        os.path.join(output_dir, f"checkpoint-{epoch}_gradient_norms.json"),
        os.path.join(output_dir, f"checkpoint-{epoch}_gradient_norms.png"),
    ]
    if n_rewards >= 2:
        files.append(os.path.join(output_dir, f"checkpoint-{epoch}_cosine_sim.png"))
    return files


def _checkpoint_status(output_dir: str, epoch: int, n_rewards: int) -> str:
    """Check which output files exist. Returns 'complete', 'plots_missing', or 'missing'."""
    expected = _expected_files(output_dir, epoch, n_rewards)
    json_path = expected[0]
    png_paths = expected[1:]
    json_ok = os.path.exists(json_path)
    pngs_ok = all(os.path.exists(p) for p in png_paths)
    if json_ok and pngs_ok:
        return "complete"
    elif json_ok:
        return "plots_missing"
    else:
        return "missing"


def load_per_checkpoint_results(output_dir: str, epoch: int) -> Optional[Dict]:
    """Load previously saved checkpoint JSON, converting back to in-memory format."""
    path = os.path.join(output_dir, f"checkpoint-{epoch}_gradient_norms.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    # Reconstruct in-memory format: grad_norms as int→float dict
    results = []
    for r in data["results"]:
        indices = r["timestep_indices"]
        r["grad_norms"] = {idx: r["grad_norms"][i] for i, idx in enumerate(indices)}
        # Restore full sigmas/timesteps lists for downstream plot functions
        r["sigmas"] = r.get("sigmas", r.get("sigma_values", []))
        r["timesteps"] = r.get("timesteps", r.get("timestep_values", r.get("timestep_indices", [])))
        # Reconstruct cosine_sim if present
        if "cosine_sim" in r and r["cosine_sim"]:
            cs = {}
            for pair_key, cs_data in r["cosine_sim"].items():
                cs_indices = cs_data["timestep_indices"]
                cs[pair_key] = {idx: cs_data["cosine_sim"][i] for i, idx in enumerate(cs_indices)}
            r["cosine_sim"] = cs
        results.append(r)
    print(f"  Loaded existing results for checkpoint-{epoch} ({len(results)} records)")
    return {"epoch": data["epoch"], "results": results}


def save_per_checkpoint_results(results: List[Dict], output_dir: str, epoch: int):
    os.makedirs(output_dir, exist_ok=True)
    # Serialize: convert int-keyed dicts to lists for clean JSON
    serializable = []
    for r in results:
        # Extract full sigmas/timesteps as lists for JSON
        sigmas_full = [float(r["sigmas"][i]) for i in range(len(r["sigmas"]))] if not isinstance(r["sigmas"], list) else r["sigmas"]
        timesteps_full = r["timesteps"] if isinstance(r["timesteps"], list) else r["timesteps"].tolist() if hasattr(r["timesteps"], "tolist") else list(r["timesteps"])
        entry = {
            "epoch": r["epoch"],
            "prompt": r["prompt"],
            "prompt_idx": r["prompt_idx"],
            "reward": r["reward"],
            "reward_value": r["reward_value"],
            "timesteps": timesteps_full,
            "sigmas": sigmas_full,
            "timestep_indices": sorted(r["grad_norms"].keys()),
            "sigma_values": [float(r["sigmas"][i]) for i in sorted(r["grad_norms"].keys())],
            "grad_norms": [r["grad_norms"][i] for i in sorted(r["grad_norms"].keys())],
        }
        if "cosine_sim" in r and r["cosine_sim"]:
            cs_json = {}
            for pair_key, idx_dict in r["cosine_sim"].items():
                cs_json[pair_key] = {
                    "timestep_indices": sorted(idx_dict.keys()),
                    "sigma_values": [float(r["sigmas"][i]) for i in sorted(idx_dict.keys())],
                    "cosine_sim": [idx_dict[i] for i in sorted(idx_dict.keys())],
                }
            entry["cosine_sim"] = cs_json
        serializable.append(entry)

    path = os.path.join(output_dir, f"checkpoint-{epoch}_gradient_norms.json")
    with open(path, "w") as f:
        json.dump({"epoch": epoch, "results": serializable}, f, indent=2)
    print(f"  Saved {path}")


def plot_per_checkpoint(results: List[Dict], output_dir: str, epoch: int):
    """Single-checkpoint plot: individual per-prompt lines, one subplot per reward."""
    if not HAS_MPL:
        print("  [WARN] matplotlib not available, skipping gradient norm plot")
        return
    if not results:
        print("  [WARN] No results to plot for gradient norms")
        return
    from collections import defaultdict

    try:
        # Group by reward → prompt_idx → list of (sigma, grad_norm)
        reward_prompt_data = defaultdict(lambda: defaultdict(list))
        for r in results:
            rname = r["reward"]
            pidx = r["prompt_idx"]
            sigmas = r["sigmas"]
            for t_idx, gn in sorted(r["grad_norms"].items()):
                reward_prompt_data[rname][pidx].append((float(sigmas[t_idx]), gn))

        n_rewards = len(reward_prompt_data)
        fig, axes = plt.subplots(1, n_rewards, figsize=(7 * n_rewards, 5), squeeze=False)
        axes = axes[0]

        for ax, (rname, prompt_data) in zip(axes, sorted(reward_prompt_data.items())):
            for pidx, points in sorted(prompt_data.items()):
                svals = [p[0] for p in sorted(points, key=lambda x: x[0], reverse=True)]
                gvals = [p[1] for p in sorted(points, key=lambda x: x[0], reverse=True)]
                ax.plot(svals, gvals, marker=".", alpha=0.7, linewidth=0.8,
                        label=f"prompt {pidx}")
            ax.set_xlabel("Sigma (1 = noisy, 0 = clean)")
            ax.set_ylabel("||d(reward)/d(v_pred)||")
            ax.set_title(f"{rname} (epoch {epoch})")
            ax.invert_xaxis()
            if len(prompt_data) <= 10:
                ax.legend(fontsize=7)

        fig.tight_layout()
        path = os.path.join(output_dir, f"checkpoint-{epoch}_gradient_norms.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)
    except Exception as e:
        print(f"  [ERROR] Failed to plot gradient norms: {e}")


def plot_cosine_similarity(results: List[Dict], output_dir: str, epoch: int):
    """Per-checkpoint plot: cosine similarity per prompt, one line per prompt per pair."""
    if not HAS_MPL:
        print("  [WARN] matplotlib not available, skipping cosine similarity plot")
        return
    if not results:
        print("  [WARN] No results to plot for cosine similarity")
        return
    from collections import defaultdict

    try:
        # Group by pair_key → prompt_idx → list of (sigma, cos_sim)
        pair_prompt_data = defaultdict(lambda: defaultdict(list))
        for r in results:
            cs = r.get("cosine_sim", {})
            if not cs:
                continue
            sigmas = r["sigmas"]
            pidx = r["prompt_idx"]
            for pair_key, idx_dict in cs.items():
                for t_idx, cos_val in sorted(idx_dict.items()):
                    pair_prompt_data[pair_key][pidx].append((float(sigmas[t_idx]), cos_val))

        if not pair_prompt_data:
            print("  [WARN] No cosine similarity data to plot")
            return

        n_pairs = len(pair_prompt_data)
        fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5), squeeze=False)
        axes = axes[0]

        for ax, (pair_key, prompt_data) in zip(axes, sorted(pair_prompt_data.items())):
            for pidx, points in sorted(prompt_data.items()):
                svals = [p[0] for p in sorted(points, key=lambda x: x[0], reverse=True)]
                cvals = [p[1] for p in sorted(points, key=lambda x: x[0], reverse=True)]
                ax.plot(svals, cvals, marker=".", alpha=0.7, linewidth=0.8,
                        label=f"prompt {pidx}")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Sigma (1 = noisy, 0 = clean)")
            ax.set_ylabel("Cosine similarity")
            ax.set_title(f"Gradient cos-sim: {pair_key} (epoch {epoch})")
            ax.invert_xaxis()
            ax.set_ylim(-1.05, 1.05)
            if len(prompt_data) <= 10:
                ax.legend(fontsize=7)

        fig.tight_layout()
        path = os.path.join(output_dir, f"checkpoint-{epoch}_cosine_sim.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)
    except Exception as e:
        print(f"  [ERROR] Failed to plot cosine similarity: {e}")


def plot_aggregate(all_checkpoint_results: List[Dict], output_dir: str):
    """Aggregate plot: gradient norm heatmap across checkpoints × timesteps."""
    if not HAS_MPL or not all_checkpoint_results:
        return

    from collections import defaultdict
    from matplotlib.colors import Normalize

    # Build data: reward → epoch → sigma → mean_grad_norm
    reward_epoch_sigma: Dict[str, Dict[int, Dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    sigmas_global = set()

    for ckpt_data in all_checkpoint_results:
        epoch = ckpt_data["epoch"]
        for r in ckpt_data["results"]:
            rname = r["reward"]
            indices = sorted(r["grad_norms"].keys())
            sigmas = r["sigmas"]
            for t_idx in indices:
                s = float(sigmas[t_idx])
                sigmas_global.add(s)
                if s not in reward_epoch_sigma[rname][epoch]:
                    reward_epoch_sigma[rname][epoch][s] = []
                reward_epoch_sigma[rname][epoch][s].append(r["grad_norms"][t_idx])

    sigmas_sorted = sorted(sigmas_global, reverse=True)
    epochs_sorted = sorted(set(ckpt["epoch"] for ckpt in all_checkpoint_results))

    n_rewards = len(reward_epoch_sigma)
    fig, axes = plt.subplots(1, n_rewards, figsize=(7 * n_rewards, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, (rname, epoch_sigma_data) in enumerate(sorted(reward_epoch_sigma.items())):
        ax = axes[ax_idx]
        # Build matrix: rows=epochs, cols=sigmas
        matrix = np.zeros((len(epochs_sorted), len(sigmas_sorted)))
        for ei, epoch in enumerate(epochs_sorted):
            for si, s in enumerate(sigmas_sorted):
                vals = epoch_sigma_data.get(epoch, {}).get(s, [])
                matrix[ei, si] = np.mean(vals) if vals else np.nan

        # Normalize per reward for comparison
        vmin = np.nanmin(matrix)
        vmax = np.nanmax(matrix)
        im = ax.imshow(matrix, aspect="auto", origin="lower",
                       extent=[sigmas_sorted[-1], sigmas_sorted[0], epochs_sorted[0], epochs_sorted[-1]],
                       cmap="viridis", norm=Normalize(vmin=vmin, vmax=vmax))
        ax.set_xlabel("Sigma (1 = pure noise, 0 = clean)")
        ax.set_ylabel("Checkpoint epoch")
        ax.set_title(f"{rname} — ||d(reward)/d(v_pred)||")
        ax.invert_xaxis()
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    path = os.path.join(output_dir, "aggregate_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"Aggregate heatmap saved to {path}")
    plt.close(fig)

    # --- Aggregate cosine similarity heatmaps ---
    # Build data: reward_pair → epoch → sigma → mean_cos_sim
    pair_epoch_sigma: Dict[str, Dict[int, Dict[float, float]]] = defaultdict(lambda: defaultdict(dict))

    for ckpt_data in all_checkpoint_results:
        epoch = ckpt_data["epoch"]
        for r in ckpt_data["results"]:
            cs = r.get("cosine_sim", {})
            sigmas = r["sigmas"]
            for pair_key, idx_dict in cs.items():
                for t_idx, cos_val in idx_dict.items():
                    s = float(sigmas[t_idx])
                    if s not in pair_epoch_sigma[pair_key][epoch]:
                        pair_epoch_sigma[pair_key][epoch][s] = []
                    pair_epoch_sigma[pair_key][epoch][s].append(cos_val)

    if pair_epoch_sigma:
        n_pairs = len(pair_epoch_sigma)
        fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5), squeeze=False)
        axes = axes[0]
        for ax, (pair_key, epoch_sigma) in zip(axes, sorted(pair_epoch_sigma.items())):
            matrix = np.zeros((len(epochs_sorted), len(sigmas_sorted)))
            for ei, epoch in enumerate(epochs_sorted):
                for si, s in enumerate(sigmas_sorted):
                    vals = epoch_sigma.get(epoch, {}).get(s, [])
                    matrix[ei, si] = np.mean(vals) if vals else np.nan
            vmin = max(np.nanmin(matrix), -1.0)
            vmax = min(np.nanmax(matrix), 1.0)
            im = ax.imshow(matrix, aspect="auto", origin="lower",
                           extent=[sigmas_sorted[-1], sigmas_sorted[0], epochs_sorted[0], epochs_sorted[-1]],
                           cmap="RdBu_r", norm=Normalize(vmin=vmin, vmax=vmax))
            ax.set_xlabel("Sigma (1 = pure noise, 0 = clean)")
            ax.set_ylabel("Checkpoint epoch")
            ax.set_title(f"Gradient cos-sim: {pair_key}")
            ax.invert_xaxis()
            plt.colorbar(im, ax=ax)
        fig.tight_layout()
        path = os.path.join(output_dir, "aggregate_cosine_sim.png")
        fig.savefig(path, dpi=150)
        print(f"Aggregate cosine similarity saved to {path}")
        plt.close(fig)

    # Also create per-reward line plots comparing checkpoints at selected sigmas
    plot_checkpoint_comparison(reward_epoch_sigma, epochs_sorted, sigmas_sorted, output_dir)


def plot_checkpoint_comparison(reward_epoch_sigma, epochs_sorted, sigmas_sorted, output_dir):
    """Per-reward line plot: gradient norm vs sigma, one line per checkpoint."""
    if not HAS_MPL:
        return

    n_rewards = len(reward_epoch_sigma)
    fig, axes = plt.subplots(1, n_rewards, figsize=(7 * n_rewards, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, (rname, epoch_sigma_data) in enumerate(sorted(reward_epoch_sigma.items())):
        ax = axes[ax_idx]
        for epoch in epochs_sorted:
            svals = []; means = []
            for s in sigmas_sorted:
                vals = epoch_sigma_data.get(epoch, {}).get(s, [])
                if vals:
                    svals.append(s)
                    means.append(np.mean(vals))
            if svals:
                ax.plot(svals, means, marker=".", label=f"epoch {epoch}", alpha=0.7)
        ax.set_xlabel("Sigma (1 = pure noise, 0 = clean)")
        ax.set_ylabel("||d(reward)/d(v_pred)||")
        ax.set_title(f"{rname} — across checkpoints")
        ax.invert_xaxis(); ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "checkpoint_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Checkpoint comparison saved to {path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main(config_path: str):
    config = parse_config(config_path)
    # Create output subfolder named after the run (parent dir of checkpoints/)
    ckpt_dir = config.checkpoint_dir.rstrip("/")
    run_name = os.path.basename(os.path.dirname(ckpt_dir))
    output_dir = os.path.join(config.output_dir, "latent_reward_per_timestep_gradient_analysis", run_name)
    config.output_dir = output_dir  # update so workers use the correct path
    print(f"Checkpoint dir : {config.checkpoint_dir}")
    print(f"Output dir     : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts from file or use inline list
    if config.prompts_file:
        with open(config.prompts_file, "r") as f:
            config.prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(config.prompts)} prompts from {config.prompts_file}")
    if config.max_prompts > 0 and len(config.prompts) > config.max_prompts:
        config.prompts = config.prompts[:config.max_prompts]
        print(f"Truncated to first {config.max_prompts} prompts")
    if not config.prompts:
        print("ERROR: No prompts specified (prompts_file or prompts)."); sys.exit(1)
    if not config.rewards:
        print("ERROR: No rewards specified."); sys.exit(1)
    if not config.checkpoint_dir:
        print("ERROR: No checkpoint_dir specified."); sys.exit(1)

    # Discover checkpoints
    checkpoints = discover_checkpoints(config.checkpoint_dir)
    if config.max_epochs > 0:
        checkpoints = [(e, p) for e, p in checkpoints if e <= config.max_epochs]
    if not checkpoints:
        print(f"ERROR: No checkpoint-* subdirectories found in {config.checkpoint_dir}")
        sys.exit(1)
    print(f"Found {len(checkpoints)} checkpoints: {[e for e, _ in checkpoints]}")

    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs")

    # Prepare config dict for pickling (dataclass → dict)
    config_dict = {
        "base_model": config.base_model,
        "checkpoint_dir": config.checkpoint_dir,
        "dtype": config.dtype,
        "num_gpus": config.num_gpus,
        "num_inference_steps": config.num_inference_steps,
        "per_device_batch_size": config.per_device_batch_size,
        "guidance_scale": config.guidance_scale,
        "height": config.height, "width": config.width, "seed": config.seed,
        "num_analysis_timesteps": config.num_analysis_timesteps,
        "max_prompts": config.max_prompts,
        "prompts": config.prompts,
        "rewards": config.rewards,
        "output_dir": config.output_dir,
    }

    # Resume: check each checkpoint's files, load if possible, regenerate missing plots
    n_rewards = len(config.rewards)
    all_checkpoint_results = []
    pending_checkpoints = []
    for epoch, ckpt_path in checkpoints:
        status = _checkpoint_status(output_dir, epoch, n_rewards)
        if status == "complete":
            existing = load_per_checkpoint_results(output_dir, epoch)
            if existing is not None:
                all_checkpoint_results.append(existing)
        elif status == "plots_missing":
            print(f"  checkpoint-{epoch}: JSON exists, regenerating missing plots")
            existing = load_per_checkpoint_results(output_dir, epoch)
            if existing is not None:
                results = existing["results"]
                print(f"  Regenerating plots for checkpoint-{epoch}...")
                plot_per_checkpoint(results, output_dir, epoch)
                plot_cosine_similarity(results, output_dir, epoch)
                all_checkpoint_results.append(existing)
        else:
            pending_checkpoints.append((epoch, ckpt_path))

    if pending_checkpoints:
        print(f"Pending: {len(pending_checkpoints)} checkpoints to compute")
        # Build worker tasks for pending checkpoints
        tasks = []
        for i, (epoch, ckpt_path) in enumerate(pending_checkpoints):
            gpu_id = i % num_gpus
            tasks.append((gpu_id, epoch, ckpt_path, config_dict, config.prompts, config.rewards))

        if num_gpus > 1 and len(tasks) > 1:
            mp.set_start_method("spawn", force=True)
            with mp.Pool(processes=min(num_gpus, len(tasks))) as pool:
                for ckpt_result in pool.imap_unordered(_analyze_one_checkpoint, tasks):
                    epoch = ckpt_result["epoch"]
                    results = ckpt_result["results"]
                    print(f"\n=== Checkpoint epoch={epoch} complete: {len(results)} records ===")
                    save_per_checkpoint_results(results, output_dir, epoch)
                    print(f"  Generating plots for checkpoint-{epoch}...")
                    plot_per_checkpoint(results, output_dir, epoch)
                    plot_cosine_similarity(results, output_dir, epoch)
                    all_checkpoint_results.append(ckpt_result)
        else:
            for task in tasks:
                ckpt_result = _analyze_one_checkpoint(task)
                epoch = ckpt_result["epoch"]
                results = ckpt_result["results"]
                print(f"\n=== Checkpoint epoch={epoch} complete: {len(results)} records ===")
                save_per_checkpoint_results(results, output_dir, epoch)
                print(f"  Generating plots for checkpoint-{epoch}...")
                plot_per_checkpoint(results, output_dir, epoch)
                plot_cosine_similarity(results, output_dir, epoch)
                all_checkpoint_results.append(ckpt_result)
    else:
        print("All checkpoints already processed — skipping computation.")

    # Aggregate plots across all checkpoints
    all_checkpoint_results.sort(key=lambda x: x["epoch"])
    plot_aggregate(all_checkpoint_results, config.output_dir)

    # Save aggregate JSON
    agg_path = os.path.join(config.output_dir, "all_checkpoints_summary.json")
    summary = {
        "checkpoints": [{"epoch": c["epoch"], "num_results": len(c["results"])}
                        for c in all_checkpoint_results],
    }
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {config.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Per-Timestep Analysis")
    parser.add_argument("-c", "--config", default="tools/latent_reward_per_timestep_gradient_analysis/config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    main(args.config)
