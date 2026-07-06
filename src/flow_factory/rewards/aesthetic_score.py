# Copyright 2026 Bowen-Zheng
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

# src/flow_factory/rewards/aesthetic_score.py
# LAION Improved Aesthetic Predictor — CLIP ViT-L/14 + 5-layer MLP head.
# Reference: https://github.com/christophschuhmann/improved-aesthetic-predictor

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__, rank_zero_only=True)

# ---------------------------------------------------------------------------
# MLP head architecture
# ---------------------------------------------------------------------------

_MLP_STATE_DICT_URL = "camenduru/improved-aesthetic-predictor"
_MLP_CHECKPOINT_FILE = "sac+logos+ava1-l14-linearMSE.pth"


class AestheticMLP(nn.Module):
    """5-layer MLP head matching the improved aesthetic predictor checkpoint.

    Architecture (from checkpoint keys):
        Linear(768→1024) → Dropout(0.2) → Linear(1024→128) → Dropout(0.2)
        → Linear(128→64) → Dropout(0.1) → Linear(64→16) → Linear(16→1)
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class AestheticScoreRewardModel(PointwiseRewardModel):
    """LAION Improved Aesthetic Predictor — image-only aesthetic quality scoring.

    Uses a frozen CLIP ViT-L/14 vision encoder followed by a 5-layer MLP head
    trained on human aesthetic ratings.  Output range is roughly 0–10 (higher
    is more aesthetic).

    Configuration (via RewardArguments extra_kwargs):
        clip_model: CLIP backbone (default ``"openai/clip-vit-large-patch14"``).
        mlp_repo: HuggingFace repo for the MLP checkpoint
            (default ``"camenduru/improved-aesthetic-predictor"``).
        mlp_file: checkpoint filename (default ``"sac+logos+ava1-l14-linearMSE.pth"``).
    """

    required_fields = ("image", "video")

    DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        extras = config.extra_kwargs or {}
        clip_model_id = extras.get("clip_model", self.DEFAULT_CLIP_MODEL)
        mlp_repo = extras.get("mlp_repo", _MLP_STATE_DICT_URL)
        mlp_file = extras.get("mlp_file", _MLP_CHECKPOINT_FILE)

        # --- CLIP backbone ---------------------------------------------------
        self.clip_model = (
            CLIPModel.from_pretrained(clip_model_id, torch_dtype=self.dtype).eval().to(self.device)
        )
        self.processor = CLIPProcessor.from_pretrained(clip_model_id)

        # --- MLP head --------------------------------------------------------
        # Use projection_dim (768 for ViT-L/14), which is the output dimension
        # of CLIPModel.get_image_features().  The aesthetic MLP checkpoint was
        # trained on 768-dim projected CLIP features.
        self._clip_input_dim: int = self.clip_model.projection_dim

        self.mlp = AestheticMLP(input_dim=self._clip_input_dim).to(
            device=self.device, dtype=self.dtype
        )

        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(mlp_repo, mlp_file)
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.mlp.eval()

        if accelerator is not None:
            accelerator.wait_for_everyone()

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute aesthetic scores for a batch of images.

        Args:
            prompt: Text prompts (unused by this reward – aesthetic is image-only).
            image: PIL Images.
            video: Optional videos (uses first frame of each).

        Returns:
            RewardModelOutput with aesthetic scores (roughly 0–10).
        """
        if image is None and video is not None:
            image = [v[0] for v in video]
        if image is None:
            raise ValueError("AestheticScore requires either 'image' or 'video' input.")

        batch_size = getattr(self.config, "batch_size", len(image))
        all_scores: List[torch.Tensor] = []

        for i in range(0, len(image), batch_size):
            chunk = image[i : i + batch_size]
            inputs = self.processor(
                images=chunk,
                return_tensors="pt",
            )
            inputs = {k: v.to(device=self.device) for k, v in inputs.items()}

            # CLIP projected image features (projection_dim, typically 768)
            pixel_values = inputs["pixel_values"]
            image_features = self.clip_model.get_image_features(pixel_values)
            image_features = image_features.pooler_output.to(dtype=self.dtype)

            scores = self.mlp(image_features).squeeze(-1)  # (batch,)
            all_scores.append(scores.float().cpu())

        rewards = torch.cat(all_scores, dim=0)
        return RewardModelOutput(rewards=rewards, extra_info={})


def download_model():
    """Pre-download helper."""
    AestheticScoreRewardModel(RewardArguments(device="cpu"), accelerator=None)


if __name__ == "__main__":
    download_model()
