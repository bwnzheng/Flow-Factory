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

# src/flow_factory/rewards/ocr_v27.py
"""
OCR Reward Model — PaddleOCR 2.7.0.3 (legacy API), CPU-only.

Uses the older ``model.ocr(img)`` API instead of the PP-OCRv5 ``model.predict(img)``.
Compatible with ``paddleocr==2.7.0.3`` and ``paddlepaddle`` (CPU build).

Dependencies:
    pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install paddleocr==2.7.0.3
    pip install python-Levenshtein

Usage (YAML):
    - name: "ocr_reward"
      reward_model: "OCR_v27"
      device: "cpu"
      dtype: float32
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__)

# Suppress PaddleOCR / PaddlePaddle internal debug logging.
os.environ.setdefault("GLOG_v", "0")  # PaddlePaddle glog verbosity (0 = ERROR only)
logging.getLogger("paddleocr").setLevel(logging.WARNING)
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)

# Limit PaddlePaddle CPU threads to avoid multi-rank contention.
# Each rank runs PaddleOCR on CPU; if all ranks use all cores
# simultaneously, the cluster scheduler may kill the job.
_cpu_count = os.cpu_count() or 1
_ocr_threads = max(1, _cpu_count // 4)
os.environ.setdefault("OMP_NUM_THREADS", str(_ocr_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_ocr_threads))

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError(
        "paddleocr is required for OCR reward. Install with: pip install paddleocr==2.7.0.3"
    )

try:
    from Levenshtein import distance
except ImportError:
    raise ImportError(
        "python-Levenshtein is required for OCR reward. Install with: pip install python-Levenshtein"
    )


class OCRLegacyRewardModel(PointwiseRewardModel):
    """OCR reward using PaddleOCR 2.7.0.3 legacy API, CPU-only.

    Evaluates whether generated images contain specified target text (from
    prompts with double-quoted strings) using OCR recognition.

    Configuration (via RewardArguments extra_kwargs):
        batch_size: batch size for score computation (default: from config)
    """

    required_fields = ("prompt", "image", "video")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        device_type = str(self.device)
        if "cuda" in device_type or "npu" in device_type:
            logger.warning(
                f"PaddleOCR 2.7.0.3 is CPU-only. "
                f"Ignoring requested device '{device_type}' and using CPU."
            )

        logger.info("PaddleOCR 2.7.0.3 initializing on CPU ...")

        # PaddleOCR 2.7.0.3 uses boolean ``use_gpu`` flag rather than a device
        # string.  We always force CPU here.
        self.model = PaddleOCR(use_angle_cls=False, use_gpu=False)

    def _compute_scores_batch(
        self,
        prompt: List[str],
        image: List[Image.Image],
    ) -> List[float]:
        """Compute OCR reward for a batch of image-prompt pairs."""
        rewards = []
        for img, p in zip(image, prompt):
            if isinstance(img, Image.Image):
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img = np.array(img)

            # Extract quoted target text (e.g. 'a sign saying "Hello"' -> 'Hello')
            parts = p.split('"')
            target_text = parts[1] if len(parts) >= 2 else p

            try:
                # PaddleOCR 2.7.0.3 legacy API: returns list of pages, each
                # page a list of [bbox, (text, confidence)] tuples.
                result = self.model.ocr(img, cls=False)

                recognized_text = ""
                if result and result[0]:
                    for line in result[0]:
                        recognized_text += line[1][0]  # line = [[bbox], (text, score)]

                recognized_text = recognized_text.replace(" ", "").lower()
                target_text = target_text.replace(" ", "").lower()

                if target_text in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, target_text)

                # Cap penalty at len(target_text) when many unrelated chars
                if dist > len(target_text):
                    dist = len(target_text)

            except Exception as e:
                logger.error(f"OCR processing failed: {str(e)}")
                dist = len(target_text)  # maximum penalty

            reward = 1.0 - dist / max(len(target_text), 1)
            rewards.append(reward)

        return rewards

    def _compute_video_scores(
        self,
        prompt: List[str],
        video: List[List[Image.Image]],
        batch_size: int,
    ) -> List[float]:
        """Mean OCR reward across all frames for each video."""
        frame_counts = [len(clip) for clip in video]
        flat_images = [frame for clip in video for frame in clip]
        flat_prompts = [p for p, n in zip(prompt, frame_counts) for _ in range(n)]

        all_scores: List[float] = []
        for i in range(0, len(flat_images), batch_size):
            batch_scores = self._compute_scores_batch(
                flat_prompts[i : i + batch_size],
                flat_images[i : i + batch_size],
            )
            all_scores.extend(batch_scores)

        # Reconstruct: mean per video
        scores = []
        offset = 0
        for n in frame_counts:
            scores.append(sum(all_scores[offset : offset + n]) / n)
            offset += n
        return scores

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
    ) -> RewardModelOutput:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if image is not None and video is not None:
            raise ValueError("Only one of image or video can be provided.")

        batch_size = getattr(self.config, "batch_size", len(prompt))

        if video is not None:
            scores = self._compute_video_scores(prompt, video, batch_size)
        else:
            scores = self._compute_scores_batch(prompt, image)

        return RewardModelOutput(rewards=scores, extra_info={})


def download_model():
    """Pre-download PaddleOCR models for offline use."""
    ocr = PaddleOCR(use_angle_cls=False, use_gpu=False)
    logger.info("PaddleOCR 2.7.0.3 initialized successfully")


if __name__ == "__main__":
    download_model()
