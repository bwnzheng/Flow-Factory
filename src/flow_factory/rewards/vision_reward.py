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

# src/flow_factory/rewards/vision_reward.py
# VisionReward — fine-grained multi-dimensional human preference scoring.
# Reference: https://github.com/THUDM/VisionReward  /  arXiv:2412.21059
#
# Requires cloning the VisionReward repository and installing dependencies:
#
#     git clone https://github.com/THUDM/VisionReward.git /path/to/VisionReward
#     cd /path/to/VisionReward
#     pip install -r requirements.txt
#
# Then configure ``repo_path`` in the YAML extra_kwargs, or set the
# environment variable ``VISIONREWARD_REPO``.
#
# The model is eval-only and heavyweight (~20 GB VRAM for CogVLM2-19B).
# Recommendation: use ``VisionReward-Image-bf16`` for lower VRAM usage.

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__, rank_zero_only=True)

_DEFAULT_WEIGHT = {
    "Alignment": 1.0,
    "Composition": 1.0,
    "Quality": 1.0,
    "Fidelity": 1.0,
    "Safety_Emotion": 1.0,
}


# ---------------------------------------------------------------------------
# Soft import helpers
# ---------------------------------------------------------------------------


def _resolve_repo_path(extras: dict) -> Path:
    """Resolve VisionReward repo path from extra_kwargs or env var."""
    repo = extras.get("repo_path") or os.environ.get("VISIONREWARD_REPO")
    if repo is None:
        raise ImportError(
            "VisionReward requires the repo path. Either:\n"
            "  1. Set `repo_path` in YAML extra_kwargs, or\n"
            "  2. Set VISIONREWARD_REPO environment variable\n"
            "Clone: git clone https://github.com/THUDM/VisionReward.git"
        )
    repo = Path(repo)
    if not repo.is_dir():
        raise FileNotFoundError(f"VisionReward repo not found at {repo}")
    return repo


def _import_visionreward(extras: dict):
    """Import VisionReward inference utilities from the cloned repo."""
    repo = _resolve_repo_path(extras)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        from sat.model import AutoModel  # noqa: F811
        from sat.model.mixins import CachedAutoregressiveMixin
        from utils.utils import (
            chat,
            get_image_processor,
            llama2_text_processor_inference,
            llama3_tokenizer,
        )
        from VisionReward_Image.t2v_metrics.vqascore import VQAScore
    except ImportError as e:
        raise ImportError(
            f"Failed to import VisionReward modules from {repo}. "
            "Ensure dependencies are installed:\n"
            f"  cd {repo} && pip install -r requirements.txt\n"
            f"Original error: {e}"
        ) from e

    return (
        AutoModel,
        CachedAutoregressiveMixin,
        chat,
        get_image_processor,
        llama2_text_processor_inference,
        llama3_tokenizer,
        VQAScore,
    )


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class VisionRewardModel(PointwiseRewardModel):
    """VisionReward — CogVLM2-based multi-dimensional image quality evaluation.

    Scores each image across 5 dimensions: Alignment, Composition, Quality,
    Fidelity, and Safety & Emotion.  Each dimension is evaluated via a set
    of binary (yes/no) questions, and the final score is a weighted sum.

    .. attention::
        This model loads CogVLM2-19B (~20 GB VRAM).  Use
        ``model_path: "THUDM/VisionReward-Image-bf16"`` for reduced memory.
        This reward is intended for **evaluation only**.

    Configuration (via RewardArguments extra_kwargs):
        repo_path: Path to cloned VisionReward repo (or set VISIONREWARD_REPO).
        model_path: HF model ID (default ``"THUDM/VisionReward-Image-bf16"``).
        weight_json: Path to dimension weight file or dict of weights
            (default: all 1.0).
        max_new_tokens: Max tokens per question response (default 64).
        temperature: Generation temperature (default 0.0).
    """

    required_fields = ("prompt", "image", "video")

    DEFAULT_MODEL_PATH = "THUDM/VisionReward-Image-bf16"

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        extras = config.extra_kwargs or {}

        # Import VisionReward modules
        (
            AutoModel,
            CachedAutoregressiveMixin,
            self._chat_fn,
            get_image_processor,
            llama2_text_processor_inference,
            llama3_tokenizer,
            VQAScore,
        ) = _import_visionreward(extras)

        # Model
        model_path = extras.get("model_path", self.DEFAULT_MODEL_PATH)
        self._model = AutoModel.from_pretrained(
            model_path, device=str(self.device), inference=True, skip_init=True
        ).eval()
        self._model.add_mixin("auto-regressive", CachedAutoregressiveMixin())

        # Tokenizer & processors
        self._tokenizer = llama3_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")
        self._image_processor = get_image_processor()
        self._text_processor = llama2_text_processor_inference(
            self._tokenizer, 8192, self._model.image_length
        )

        # Dimension weights
        weight_cfg = extras.get("weight_json")
        if weight_cfg is None:
            self._weights = dict(_DEFAULT_WEIGHT)
        elif isinstance(weight_cfg, str):
            with open(weight_cfg) as f:
                self._weights = json.load(f)
        else:
            self._weights = dict(weight_cfg)

        # Scoring
        self._vqa_scorer = VQAScore()
        self._max_new_tokens = int(extras.get("max_new_tokens", 64))
        self._temperature = float(extras.get("temperature", 0.0))

        # Build dimension-question mapping from VQAScore
        self._dimension_questions: Dict[str, List[str]] = {}
        for dim, questions in self._vqa_scorer.questions.items():
            self._dimension_questions[dim] = list(questions.keys())

        if accelerator is not None:
            accelerator.wait_for_everyone()

        logger.info(
            f"VisionReward loaded: model={model_path}, device={self.device}, "
            f"dims={list(self._dimension_questions.keys())}"
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    @torch.no_grad()
    def _score_single(self, prompt: str, image: Image.Image) -> Dict[str, float]:
        """Score a single image across all dimensions.

        Saves the image to a temp file for VisionReward's file-based API,
        then scores each dimension via binary questions.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            dim_scores: Dict[str, float] = {}
            with self._autocast():
                for dim, questions in self._dimension_questions.items():
                    yes_count = 0
                    total = len(questions)
                    for question in questions:
                        answer = self._chat_fn(
                            image_path=tmp_path,
                            model=self._model,
                            text_processor=self._text_processor,
                            image_processor=self._image_processor,
                            query=question.format(prompt=prompt),
                            max_length=self._max_new_tokens,
                            top_p=1.0,
                            temperature=self._temperature,
                            top_k=1,
                        )
                        if isinstance(answer, str):
                            answer = answer.strip().lower()
                        yes_count += 1 if answer and answer[0] == "y" else 0
                    dim_scores[dim] = yes_count / total if total > 0 else 0.0

            # Weighted overall score
            overall = sum(
                dim_scores.get(dim, 0.0) * self._weights.get(dim, 1.0)
                for dim in self._dimension_questions
            )
            dim_scores["overall"] = overall

            return dim_scores
        finally:
            os.unlink(tmp_path)

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute VisionReward scores for (prompt, image) pairs.

        Args:
            prompt: Text prompts.
            image: Generated PIL images.
            video: Optional videos (uses first frame of each).

        Returns:
            RewardModelOutput where ``rewards`` is the weighted overall score
            (0–1 range).  ``extra_info`` contains per-dimension scores.
        """
        if image is None and video is not None:
            image = [v[0] for v in video]
        if image is None:
            raise ValueError("VisionReward requires either 'image' or 'video' input.")
        if len(prompt) != len(image):
            raise ValueError(f"prompt/image length mismatch: {len(prompt)} vs {len(image)}")

        all_overall: List[float] = []
        all_dims: Dict[str, List[float]] = {}

        for p, img in zip(prompt, image):
            dim_scores = self._score_single(p, img)
            all_overall.append(dim_scores.pop("overall"))
            for dim, score in dim_scores.items():
                all_dims.setdefault(dim, []).append(score)

        rewards = torch.tensor(all_overall, device=self.device, dtype=torch.float32)
        return RewardModelOutput(rewards=rewards, extra_info=all_dims)


def download_model():
    """Pre-download helper."""
    VisionRewardModel(
        RewardArguments(device="cpu", extra_kwargs={"model_path": "THUDM/VisionReward-Image-bf16"}),
        accelerator=None,
    )


if __name__ == "__main__":
    download_model()
