"""Standalone reward model wrappers for offline PIL-image-based inference.

Loads CLIP and PickScore directly via HuggingFace APIs without depending on
the training-time reward infrastructure.
"""

from __future__ import annotations

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
    """Loads and runs CLIP and/or PickScore reward models for offline scoring.

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

        self.device = device
        self.dtype = dtype
        self._models: Dict[str, callable] = {}
        self._names: List[str] = []

        for cfg in reward_configs:
            rtype = cfg.get("reward_model", cfg.get("type", ""))
            rname = cfg.get("name", rtype)
            if rtype not in self._SUPPORTED:
                raise ValueError(
                    f"Unsupported reward model '{rtype}'. Supported: {self._SUPPORTED}"
                )
            self._models[rname] = self._build(rtype, cfg)
            self._names.append(rname)

    @property
    def reward_names(self) -> List[str]:
        return self._names

    def _build(self, rtype: str, cfg: Dict[str, Any]):
        if rtype == "CLIP":
            return _CLIPWrapper(
                model_name=cfg.get("model_name", "openai/clip-vit-large-patch14"),
                device=self.device,
                dtype=self.dtype,
            )
        if rtype == "PickScore":
            return _PickScoreWrapper(device=self.device, dtype=self.dtype)
        raise ValueError(f"Unknown reward type: {rtype}")

    @torch.no_grad()
    def compute(
        self,
        images: List[Image.Image],
        prompts: List[str],
        batch_size: int = 16,
    ) -> Dict[str, List[float]]:
        """Score every (image, prompt) pair with every loaded reward model.

        Args:
            images: List of PIL Images.
            prompts: List of prompt strings (same length as ``images``).
            batch_size: Max batch size passed to each model at once.

        Returns:
            ``{reward_name: [score_0, ...]}`` aligned with the input lists.
        """
        assert len(images) == len(prompts), (
            f"Mismatch: {len(images)} images vs {len(prompts)} prompts"
        )
        results: Dict[str, List[float]] = {}
        for name in self._names:
            results[name] = self._models[name](images, prompts, batch_size)
        return results


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

    @torch.no_grad()
    def __call__(
        self, images: List[Image.Image], prompts: List[str], batch_size: int
    ) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
    def __init__(self, device: str, dtype: torch.dtype):
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(
        self, images: List[Image.Image], prompts: List[str], batch_size: int
    ) -> List[float]:
        logit_scale = self.model.logit_scale.exp()
        scores: List[float] = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]

            img_inputs = self.processor(
                images=batch_images, padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            )
            img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}

            txt_inputs = self.processor(
                text=batch_prompts, padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            )
            txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}

            img_emb = _extract_feature_tensor(self.model.get_image_features(**img_inputs))
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

            txt_emb = _extract_feature_tensor(self.model.get_text_features(**txt_inputs))
            txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

            batch_scores = logit_scale * (txt_emb * img_emb).sum(dim=-1)
            batch_scores = batch_scores / 26  # normalize to [0, 1]
            scores.extend(batch_scores.float().cpu().tolist())
        return scores
