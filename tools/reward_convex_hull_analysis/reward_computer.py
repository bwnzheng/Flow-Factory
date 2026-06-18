"""Standalone reward model wrappers for offline PIL-image-based inference.

Loads CLIP and PickScore directly via HuggingFace APIs without depending on
the training-time reward infrastructure.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.utils.generic import ModelOutput


def _extract_feature_tensor(output: Any) -> torch.Tensor:
    """Extract tensor from get_*_features() output (compatible with transformers >=5)."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, ModelOutput):
        return output.pooler_output
    raise TypeError(f"Unexpected output type: {type(output).__name__}")


class StandaloneRewardComputer:
    """Lazy-loading CLIP and/or PickScore reward models for offline scoring.

    Models are NOT loaded at construction time — they are built on the first
    call to :meth:`compute`.  This makes it cheap to create many instances
    (one per GPU) and lets each load its models in parallel on its own device.

    Set the ``HF_HOME`` environment variable before instantiation to control
    where cached models are loaded from.
    """

    _SUPPORTED = {"CLIP", "PickScore"}

    def __init__(
        self,
        reward_configs: List[Dict[str, Any]],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif isinstance(dtype, str):
            # Resolve config-style string (e.g. "bfloat16") to torch.dtype
            dtype = getattr(torch, dtype)

        self.device = device
        self.dtype = dtype

        # Validate configs eagerly, defer model building
        self._configs: List[Dict[str, Any]] = []
        self._names: List[str] = []
        for cfg in reward_configs:
            rtype = cfg.get("reward_model", cfg.get("type", ""))
            rname = cfg.get("name", rtype)
            if rtype not in self._SUPPORTED:
                raise ValueError(
                    f"Unsupported reward model '{rtype}'. Supported: {self._SUPPORTED}"
                )
            self._configs.append(cfg)
            self._names.append(rname)

        self._models: Optional[Dict[str, callable]] = None

    @property
    def reward_names(self) -> List[str]:
        return self._names

    def _ensure_loaded(self) -> None:
        """Build and load models if not already done (called on first compute)."""
        if self._models is not None:
            return
        self._models = {}
        for cfg in self._configs:
            rtype = cfg.get("reward_model", cfg.get("type", ""))
            rname = cfg.get("name", rtype)
            self._models[rname] = self._build(rtype, cfg)

    def _build(self, rtype: str, cfg: Dict[str, Any]):
        if rtype == "CLIP":
            return _CLIPWrapper(
                model_name=cfg.get("model_name", "openai/clip-vit-large-patch14"),
                device=self.device,
                dtype=self.dtype,
            )
        if rtype == "PickScore":
            return _PickScoreWrapper(
                device=self.device,
                dtype=self.dtype,
                processor_name=cfg.get("processor_name", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
                model_name=cfg.get("model_name", "yuvalkirstain/PickScore_v1"),
            )
        raise ValueError(f"Unknown reward type: {rtype}")

    @torch.no_grad()
    def compute(
        self,
        images: List[Image.Image],
        prompts: List[str],
        batch_size: int = 16,
    ) -> Dict[str, List[float]]:
        """Score every (image, prompt) pair with every loaded reward model.

        Models are loaded on first call (lazy init).

        Args:
            images: List of PIL Images.
            prompts: List of prompt strings (same length as ``images``).
            batch_size: Max batch size passed to each model at once.

        Returns:
            ``{reward_name: [score_0, ...]}`` aligned with the input lists.
        """
        self._ensure_loaded()
        assert self._models is not None
        assert len(images) == len(
            prompts
        ), f"Mismatch: {len(images)} images vs {len(prompts)} prompts"
        results: Dict[str, List[float]] = {}
        for name in self._names:
            results[name] = self._models[name](images, prompts, batch_size)
        return results


class MultiGPUComputer:
    """Scores images in parallel across multiple accelerators (CUDA / NPU).

    Creates one :class:`StandaloneRewardComputer` per device — models are
    loaded lazily on first :meth:`compute` in each worker thread, so model
    loading is naturally parallel.

    Usage::

        computer = MultiGPUComputer(reward_configs, num_gpus=4, device="cuda")
        computer = MultiGPUComputer(reward_configs, num_gpus=8, device="npu")
        scores = computer.compute(images, prompts)
    """

    def __init__(
        self,
        reward_configs: List[Dict[str, Any]],
        num_gpus: int,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        if num_gpus < 2:
            raise ValueError(f"MultiGPUComputer requires num_gpus >= 2, got {num_gpus}")

        self._num_gpus = num_gpus
        self._device_type = device.split(":")[0]  # "cuda", "npu", "cpu"

        # Create instances cheaply — models are NOT loaded yet (lazy)
        self._computers: List[StandaloneRewardComputer] = []
        for i in range(num_gpus):
            self._computers.append(
                StandaloneRewardComputer(
                    reward_configs, device=f"{self._device_type}:{i}", dtype=dtype
                )
            )

    @property
    def reward_names(self) -> List[str]:
        return self._computers[0].reward_names

    @torch.no_grad()
    def compute(
        self,
        images: List[Image.Image],
        prompts: List[str],
        batch_size: int = 16,
    ) -> Dict[str, List[float]]:
        """Score (image, prompt) pairs, distributing work across GPUs."""
        # Split images evenly across GPUs
        n = self._num_gpus
        chunk_size = (len(images) + n - 1) // n

        chunks = []
        for i in range(n):
            start = i * chunk_size
            end = min(start + chunk_size, len(images))
            if start >= len(images):
                break
            chunks.append((i, images[start:end], prompts[start:end]))

        if not chunks:
            return {rname: [] for rname in self.reward_names}

        # Parallel scoring
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(
                    self._score_chunk,
                    gpu_idx,
                    self._computers[gpu_idx],
                    chunk_images,
                    chunk_prompts,
                    batch_size,
                ): gpu_idx
                for gpu_idx, chunk_images, chunk_prompts in chunks
            }
            results_by_gpu: Dict[int, Dict[str, List[float]]] = {}
            for future in futures:
                gpu_idx = futures[future]
                results_by_gpu[gpu_idx] = future.result()

        # Merge — concat per-reward-name lists in GPU order
        merged: Dict[str, List[float]] = {}
        first = results_by_gpu[0]
        for rname in first:
            merged[rname] = []
            for i, chunk_images, _ in chunks:
                merged[rname].extend(results_by_gpu[i][rname])
        return merged

    @staticmethod
    def _score_chunk(
        gpu_idx: int,
        computer: StandaloneRewardComputer,
        images: List[Image.Image],
        prompts: List[str],
        batch_size: int,
    ) -> Dict[str, List[float]]:
        dev = computer.device
        if dev.startswith("npu"):
            torch.npu.set_device(gpu_idx)
        elif dev.startswith("cuda"):
            torch.cuda.set_device(gpu_idx)
        # cpu — no-op
        return computer.compute(images, prompts, batch_size)


# ---------------------------------------------------------------------------
# CLIP wrapper
# ---------------------------------------------------------------------------


class _CLIPWrapper:
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self._autocast_kwargs = {
            "device_type": device.split(":")[0],
            "dtype": dtype,
        }

    @torch.no_grad()
    def __call__(
        self, images: List[Image.Image], prompts: List[str], batch_size: int
    ) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
            with torch.autocast(**self._autocast_kwargs):
                outputs = self.model(**inputs)
            img = F.normalize(outputs.image_embeds, p=2, dim=-1)
            txt = F.normalize(outputs.text_embeds, p=2, dim=-1)
            batch_scores = (img * txt).sum(dim=-1)
            scores.extend(batch_scores.float().cpu().tolist())
        return scores


# ---------------------------------------------------------------------------
# PickScore wrapper
# ---------------------------------------------------------------------------


class _PickScoreWrapper:
    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name: str = "yuvalkirstain/PickScore_v1",
    ):
        self.processor = CLIPProcessor.from_pretrained(processor_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self._autocast_kwargs = {
            "device_type": device.split(":")[0],
            "dtype": dtype,
        }

    @torch.no_grad()
    def __call__(
        self, images: List[Image.Image], prompts: List[str], batch_size: int
    ) -> List[float]:
        logit_scale = self.model.logit_scale.exp()
        scores: List[float] = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]

            img_inputs = self.processor(
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            img_inputs = {k: v.to(device=self.device) for k, v in img_inputs.items()}

            txt_inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            txt_inputs = {k: v.to(device=self.device) for k, v in txt_inputs.items()}

            with torch.autocast(**self._autocast_kwargs):
                img_emb = _extract_feature_tensor(self.model.get_image_features(**img_inputs))
                img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
                txt_emb = _extract_feature_tensor(self.model.get_text_features(**txt_inputs))
                txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

            batch_scores = logit_scale * (txt_emb * img_emb).sum(dim=-1)
            batch_scores = batch_scores / 26  # normalize to [0, 1]
            scores.extend(batch_scores.float().cpu().tolist())
        return scores
