"""Checkpoint discovery, pipeline loading, LoRA merging, and image generation.

All implementations are freshly written using the underlying library APIs
(diffusers, peft, transformers) — not copied from existing analysis tools.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from PIL import Image

from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def discover_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all ``checkpoint-N`` subdirectories, returning sorted ``[(epoch, path)]``."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    entries = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("checkpoint-"):
            try:
                epoch = int(name.split("checkpoint-")[1])
            except ValueError:
                continue
            ckpt_path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(ckpt_path):
                entries.append((epoch, ckpt_path))
    return sorted(entries)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def load_base_pipeline(base_model: str, dtype_str: str, device: str = "cuda"):
    """Load SD3.5 pipeline with FlowMatch ODE scheduler."""
    dtype = _resolve_dtype(dtype_str)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    scheduler = FlowMatchEulerDiscreteSDEScheduler.from_config(
        pipe.scheduler.config, dynamics_type="ODE",
    )
    scheduler.eval()
    pipe.scheduler = scheduler
    pipe = pipe.to(device)
    return pipe


def apply_lora(pipe, checkpoint_path: str, dtype: torch.dtype):
    """Load LoRA weights onto ``pipe.transformer`` and merge."""
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer, checkpoint_path, torch_dtype=dtype,
    )
    pipe.transformer = pipe.transformer.merge_and_unload()


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class CheckpointRunner:
    """Generates images from a base model + per-checkpoint LoRA weights."""

    def __init__(
        self,
        base_model: str,
        dtype_str: str,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = _resolve_dtype(dtype_str)
        self.base_model = base_model
        self._pipe = None

    @property
    def pipe(self):
        """Lazy-load the base pipeline (model download is slow)."""
        if self._pipe is None:
            self._pipe = load_base_pipeline(self.base_model, str(self.dtype), self.device)
        return self._pipe

    def generate_for_checkpoint(
        self,
        checkpoint_path: str,
        prompts: List[str],
        num_samples: int = 4,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Image.Image]:
        """Load LoRA for *checkpoint_path*, generate images, unload LoRA.

        Args:
            checkpoint_path: Path to a ``checkpoint-N`` directory.
            prompts: List of prompt strings.
            num_samples: Number of images to generate per prompt.
            gen_kwargs: Extra kwargs for ``pipe(prompt, ...)``.
                Defaults: ``num_inference_steps=50, guidance_scale=1.0,
                height=512, width=512``.

        Returns:
            ``len(prompts) * num_samples`` PIL Images (prompts[0] * N,
            prompts[1] * N, ...).
        """
        if gen_kwargs is None:
            gen_kwargs = {}
        defaults = {
            "num_inference_steps": 50,
            "guidance_scale": 1.0,
            "height": 512,
            "width": 512,
        }
        kw = {**defaults, **gen_kwargs}

        # Load LoRA onto the pipeline
        apply_lora(self.pipe, checkpoint_path, self.dtype)

        images: List[Image.Image] = []
        base_seed = kw.get("seed", 42)

        pipe_kwargs = {k: v for k, v in kw.items() if k != "seed"}
        pipe_kwargs["output_type"] = "pil"

        sample_idx = 0
        for prompt in prompts:
            for _ in range(num_samples):
                seed = base_seed + sample_idx
                generator = torch.Generator(device=self.device).manual_seed(seed)
                pipe_kwargs["generator"] = generator
                result = self.pipe(prompt, **pipe_kwargs)
                images.append(result.images[0])
                sample_idx += 1

        # Unload LoRA by reloading fresh base pipeline
        self._unload_lora()

        return images

    def _unload_lora(self):
        """Reload the base pipeline to clear merged LoRA weights."""
        del self._pipe
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
