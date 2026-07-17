# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint discovery, pipeline loading, LoRA adapter loading, and image generation.

All implementations are freshly written using the underlying library APIs
(diffusers, peft, transformers) — not copied from existing analysis tools.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from PIL import Image

from diffusers import StableDiffusion3Pipeline
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


_pipe_load_lock = threading.Lock()


def load_base_pipeline(
    base_model: str, dtype_str: str, device: str = "cuda"
) -> StableDiffusion3Pipeline:
    """Load SD3.5 pipeline with FlowMatch ODE scheduler.

    Serialised across threads via a module-level lock — parallel
    ``from_pretrained`` calls can trigger meta-tensor initialisation
    conflicts in transformers/diffusers.
    """
    dtype = _resolve_dtype(dtype_str)
    with _pipe_load_lock:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
        )
        scheduler = FlowMatchEulerDiscreteSDEScheduler.from_config(
            pipe.scheduler.config,
            dynamics_type="ODE",
        )
        scheduler.eval()
        pipe.scheduler = scheduler
        pipe = pipe.to(device)
    return pipe


def apply_lora(pipe: StableDiffusion3Pipeline, checkpoint_path: str, dtype: torch.dtype) -> None:
    """Load LoRA weights as a PEFT adapter (no merge)."""
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        checkpoint_path,
        torch_dtype=dtype,
    )


def unload_lora(pipe: StableDiffusion3Pipeline) -> None:
    """Unload the LoRA adapter, restoring the original base transformer."""
    pipe.transformer = pipe.transformer.unload()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan_missing_images(
    output_dir: str,
    epoch: int,
    prompts: List[str],
    num_samples: int,
    base_seed: int,
) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    """Scan disk for already-existing images, return (all_paths, missing).

    *all_paths* contains every expected relative path.
    *missing* is a list of ``(pi, si, seed)`` for images that need generation.
    """
    paths: List[str] = []
    missing: List[Tuple[int, int, int]] = []
    sample_idx = 0
    for pi in range(len(prompts)):
        for si in range(num_samples):
            rel = f"checkpoint_{epoch}/p{pi}_s{si}.png"
            full = os.path.join(output_dir, rel)
            paths.append(rel)
            if not os.path.isfile(full):
                seed = base_seed + sample_idx
                missing.append((pi, si, seed))
            sample_idx += 1
    return paths, missing


def _generate_batches(
    runner: "EvaluationRunner",
    prompts: List[str],
    output_dir: str,
    epoch: int,
    gen_kwargs: Dict[str, Any],
    gen_batch_size: int,
    missing: List[Tuple[int, int, int]],
) -> None:
    """Run batched pipeline inference and save images to disk."""
    dev_type = runner.device.split(":")[0] if ":" in runner.device else runner.device
    use_autocast = dev_type in ("cuda", "npu")

    # Drop "seed" from pipeline kwargs — per-image seeds are set explicitly
    # via torch.Generator, so a global "seed" key would be ignored or conflict.
    pipe_kwargs = {k: v for k, v in gen_kwargs.items() if k != "seed"}
    pipe_kwargs["output_type"] = "pil"

    total = len(missing)
    for batch_start in range(0, total, gen_batch_size):
        batch = missing[batch_start : batch_start + gen_batch_size]
        batch_prompts = [prompts[pi] for pi, _, _ in batch]
        batch_generators = [
            torch.Generator(device=runner.device).manual_seed(seed) for _, _, seed in batch
        ]

        pipe_kwargs["generator"] = batch_generators
        if use_autocast:
            with torch.autocast(device_type=dev_type, dtype=runner.dtype):
                result = runner.pipe(batch_prompts, **pipe_kwargs)
        else:
            result = runner.pipe(batch_prompts, **pipe_kwargs)

        for (pi, si, _seed), img in zip(batch, result.images):
            full = os.path.join(output_dir, f"checkpoint_{epoch}/p{pi}_s{si}.png")
            img.save(full, "PNG")


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class EvaluationRunner:
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
        self.dtype_str = dtype_str
        self.dtype = _resolve_dtype(dtype_str)
        self.base_model = base_model
        self._pipe = None

    @property
    def pipe(self) -> StableDiffusion3Pipeline:
        """Lazy-load the base pipeline (model download is slow)."""
        if self._pipe is None:
            self._pipe = load_base_pipeline(self.base_model, self.dtype_str, self.device)
        return self._pipe

    @torch.no_grad()
    def generate_for_checkpoint(
        self,
        checkpoint_path: str,
        prompts: List[str],
        output_dir: str,
        epoch: int,
        num_samples: int = 4,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        gen_batch_size: int = 16,
        base_seed: int = 42,
    ) -> List[str]:
        """Load LoRA, generate images with batched inference, save to disk.

        Images are saved to ``{output_dir}/checkpoint_{epoch}/p{pi}_s{si}.png``.
        Already-existing files are skipped (resume).  Missing images are
        generated in batches — one ``pipe()`` call produces up to
        *gen_batch_size* images in a single forward pass.

        Args:
            base_seed: Deterministic seed offset; the seed for image slot *k*
                is ``base_seed + k`` regardless of which images already exist
                on disk (ensures idempotent resume).

        Returns:
            List of relative file paths.
        """
        if gen_kwargs is None:
            gen_kwargs = {}

        ckpt_dir = os.path.join(output_dir, f"checkpoint_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # --- Scan disk for already-existing images (resume) ---
        paths, missing = _scan_missing_images(
            output_dir,
            epoch,
            prompts,
            num_samples,
            base_seed=base_seed,
        )

        if not missing:
            print(f"      All {len(paths)} images already generated — skipping.")
            return paths

        print(
            f"      Generating {len(missing)}/{len(paths)} images "
            f"(batch size={gen_batch_size}) ..."
        )

        # --- Load LoRA onto the pipeline → batch inference → unload ---
        apply_lora(self.pipe, checkpoint_path, self.dtype)
        try:
            _generate_batches(self, prompts, output_dir, epoch, gen_kwargs, gen_batch_size, missing)
        finally:
            self.unload_lora()

        return paths

    def unload_lora(self) -> None:
        """Unload LoRA adapter, keeping the base pipeline loaded."""
        if self._pipe is not None:
            unload_lora(self._pipe)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU evaluation runner
# ---------------------------------------------------------------------------


class MultiGPUEvaluationRunner:
    """Generates images in parallel across multiple GPUs.

    Creates one :class:`EvaluationRunner` per device.  Pipeline loading is
    lazy (on first :meth:`generate_for_checkpoint`), so memory is only
    consumed when work actually starts.

    Usage::

        runner = MultiGPUEvaluationRunner(
            "stabilityai/stable-diffusion-3.5-medium",
            dtype_str="bfloat16",
            num_gpus=4,
        )
        paths = runner.generate_for_checkpoint(
            checkpoint_path, prompts, output_dir, epoch,
            num_samples=16, gen_batch_size=16,
        )
    """

    def __init__(
        self,
        base_model: str,
        dtype_str: str,
        num_gpus: int,
        device: str = "cuda",
    ):
        if num_gpus < 2:
            raise ValueError(f"MultiGPUEvaluationRunner requires num_gpus >= 2, got {num_gpus}")

        self._num_gpus = num_gpus
        self._base_model = base_model
        self._dtype_str = dtype_str
        self._device_type = device.split(":")[0]  # "cuda", "npu", "cpu"

        # Create runners cheaply — pipelines are NOT loaded yet (lazy)
        self._runners: List[EvaluationRunner] = []
        for i in range(num_gpus):
            self._runners.append(
                EvaluationRunner(base_model, dtype_str, device=f"{self._device_type}:{i}")
            )

    def generate_for_checkpoint(
        self,
        checkpoint_path: str,
        prompts: List[str],
        output_dir: str,
        epoch: int,
        num_samples: int = 4,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        gen_batch_size: int = 16,
        base_seed: int = 42,
    ) -> List[str]:
        """Generate images, distributing work across GPUs.

        Missing images are split evenly across GPUs; results are merged in
        GPU-index order so output is deterministic.
        """
        if gen_kwargs is None:
            gen_kwargs = {}

        # --- Scan disk for missing images ---
        os.makedirs(os.path.join(output_dir, f"checkpoint_{epoch}"), exist_ok=True)

        paths, missing = _scan_missing_images(
            output_dir,
            epoch,
            prompts,
            num_samples,
            base_seed=base_seed,
        )

        if not missing:
            print(f"      All {len(paths)} images already generated — skipping.")
            return paths

        # --- Split missing items evenly across GPUs ---
        n = self._num_gpus
        chunk_size = (len(missing) + n - 1) // n
        chunks: List[Tuple[int, List[Tuple[int, int, int]]]] = []
        for gpu in range(n):
            chunk = missing[gpu * chunk_size : min((gpu + 1) * chunk_size, len(missing))]
            if chunk:
                chunks.append((gpu, chunk))

        total_missing = len(missing)
        print(
            f"      Generating {total_missing}/{len(paths)} images "
            f"({self._num_gpus} GPUs, batch size={gen_batch_size}) ..."
        )

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {}
            for gpu_idx, chunk in chunks:
                futures[
                    executor.submit(
                        self._generate_chunk,
                        gpu_idx,
                        self._runners[gpu_idx],
                        checkpoint_path,
                        prompts,
                        output_dir,
                        epoch,
                        gen_kwargs,
                        gen_batch_size,
                        chunk,
                    )
                ] = gpu_idx

            for future in as_completed(futures):
                gpu_idx = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"      [ERROR] GPU {gpu_idx}: {exc}")
                    raise

        print(f"      Generated {total_missing}/{len(paths)} images")
        return paths

    @staticmethod
    def _generate_chunk(
        gpu_idx: int,
        runner: EvaluationRunner,
        checkpoint_path: str,
        prompts: List[str],
        output_dir: str,
        epoch: int,
        gen_kwargs: Dict[str, Any],
        gen_batch_size: int,
        chunk: List[Tuple[int, int, int]],
    ) -> None:
        """Generate a subset of images on a specific GPU."""
        dev = runner.device
        if dev.startswith("npu"):
            torch.npu.set_device(gpu_idx)
        elif dev.startswith("cuda"):
            torch.cuda.set_device(gpu_idx)

        # Load LoRA → batch inference → unload
        apply_lora(runner.pipe, checkpoint_path, runner.dtype)
        try:
            _generate_batches(runner, prompts, output_dir, epoch, gen_kwargs, gen_batch_size, chunk)
        finally:
            runner.unload_lora()
