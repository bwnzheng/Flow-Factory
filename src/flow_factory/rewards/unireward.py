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

# src/flow_factory/rewards/unireward.py
# UniReward — unified reward model for multimodal understanding and generation.
# Reference: https://github.com/CodeGoat24/UnifiedReward  /  arXiv:2503.05236
#
# This is a heavy VLM-based reward model (~7B parameters). It is primarily
# intended for *evaluation* rather than per-sample training rewards due to
# the computational cost.  Consider using smaller reward models for training
# and UniReward for held-out evaluation, as done in the MARBLE paper.
#
# Supported backends:
#   v2.0 (default) — Qwen2.5-VL-7B  (CodeGoat24/UnifiedReward-2.0-qwen-7b)
#   v1.5           — LLaVA-OneVision  (CodeGoat24/UnifiedReward-7b-v1.5)
#
# The v1.5 backend requires the ``llava`` package:
#     pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__, rank_zero_only=True)

# ---------------------------------------------------------------------------
# Default model paths
# ---------------------------------------------------------------------------

_UNIREWARD_V20 = "CodeGoat24/UnifiedReward-2.0-qwen-7b"
_UNIREWARD_V15 = "CodeGoat24/UnifiedReward-7b-v1.5"

# ---------------------------------------------------------------------------
# Scoring prompt template
# ---------------------------------------------------------------------------

_DEFAULT_SCORING_PROMPT = """You are an expert evaluator for AI-generated images. \
Evaluate the given image against the text prompt below. \
Score the image on three dimensions, each from 1 (worst) to 5 (best):

1. **Alignment**: How accurately the image matches the text description.
2. **Coherence**: Visual coherence — correct object shapes, spatial layout, \
absence of distortions or artifacts.
3. **Style**: Aesthetic quality — composition, color harmony, visual appeal.

Text prompt: "{prompt}"

Reply with ONLY three lines in exactly this format:
Alignment: X.X/5
Coherence: X.X/5
Style: X.X/5"""


# ---------------------------------------------------------------------------
# Backend: Qwen2.5-VL (v2.0)
# ---------------------------------------------------------------------------


def _load_v20(model_path: str, device, dtype) -> Tuple[object, object]:
    """Load UniReward v2.0 via Qwen2.5-VL transformers API."""
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor, process_vision_info


# ---------------------------------------------------------------------------
# Backend: LLaVA-NeXT (v1.5)
# ---------------------------------------------------------------------------


def _load_v15(model_path: str) -> Tuple[object, object, object]:
    """Load UniReward v1.5 via LLaVA-NeXT."""
    try:
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
    except ImportError as e:
        raise ImportError(
            "UniReward v1.5 requires the llava package. "
            "Install with: pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
        ) from e

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, "llava_qwen", device_map="auto"
    )
    model.eval()

    return model, tokenizer, image_processor


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

_SCORE_LINE_RE = re.compile(
    r"^\s*(Alignment|Coherence|Style)\s*:\s*([\d.]+)\s*/\s*5\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_scores(text: str) -> Tuple[float, float, float, float]:
    """Parse alignment, coherence, style scores from model output.

    Returns (alignment, coherence, style, overall) where overall is the mean.
    If parsing fails for a dimension, returns 0.0 for that dimension.
    """
    matches = {m.group(1).lower(): float(m.group(2)) for m in _SCORE_LINE_RE.finditer(text)}

    alignment = matches.get("alignment", 0.0)
    coherence = matches.get("coherence", 0.0)
    style = matches.get("style", 0.0)
    overall = (alignment + coherence + style) / 3.0 if matches else 0.0

    return alignment, coherence, style, overall


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class UniRewardModel(PointwiseRewardModel):
    """UniReward — VLM-based multi-dimensional image quality evaluation.

    Scores each (prompt, image) pair across three dimensions:
    alignment, coherence, and style.  The overall reward is the mean
    of the three scores (1–5 scale).

    .. attention::
        This model loads a ~7B VLM and runs full autoregressive generation
        per image.  It is **very expensive** for training-time reward
        computation.  Consider using it for held-out evaluation and smaller
        reward models (PickScore, HPSv2, AestheticScore) for training.

    Configuration (via RewardArguments extra_kwargs):
        model_path: HF model ID (default ``CodeGoat24/UnifiedReward-2.0-qwen-7b``).
        backend: ``"v2.0"`` (default, Qwen2.5-VL) or ``"v1.5"`` (LLaVA-NeXT).
        scoring_prompt: custom prompt template with ``{prompt}`` placeholder.
        max_new_tokens: max tokens to generate (default 128).
        return_dimensions: if True, return per-dimension scores in extra_info.
    """

    required_fields = ("prompt", "image", "video")

    DEFAULT_MODEL_PATH = _UNIREWARD_V20

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        extras = config.extra_kwargs or {}
        self._model_path = extras.get("model_path", self.DEFAULT_MODEL_PATH)
        self._backend_name = extras.get("backend", "v2.0")
        self._scoring_prompt = extras.get("scoring_prompt", _DEFAULT_SCORING_PROMPT)
        self._max_new_tokens = int(extras.get("max_new_tokens", 128))
        self._return_dimensions = bool(extras.get("return_dimensions", False))

        if self._backend_name == "v2.0":
            self._model, self._processor, self._process_vision_info = _load_v20(
                self._model_path, self.device, self.dtype
            )
        elif self._backend_name == "v1.5":
            self._model, self._processor, self._image_processor = _load_v15(self._model_path)
        else:
            raise ValueError(
                f"Unknown UniReward backend: {self._backend_name!r}. " "Expected 'v2.0' or 'v1.5'."
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # v2.0 scoring (Qwen2.5-VL)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_v20(self, prompt: str, image: Image.Image) -> Tuple[float, float, float, float]:
        from qwen_vl_utils import process_vision_info

        scoring_text = self._scoring_prompt.format(prompt=prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": scoring_text},
                ],
            }
        ]

        chat_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(text=[chat_input], images=image_inputs, return_tensors="pt").to(
            self.device
        )

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
        )
        # Strip input tokens
        output_ids = generated_ids[0][len(inputs.input_ids[0]) :]
        output_text = self._processor.decode(output_ids, skip_special_tokens=True)

        return _parse_scores(output_text)

    # ------------------------------------------------------------------
    # v1.5 scoring (LLaVA-NeXT)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_v15(self, prompt: str, image: Image.Image) -> Tuple[float, float, float, float]:
        import copy

        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        scoring_text = self._scoring_prompt.format(prompt=prompt)

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + scoring_text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        image_tensor = process_images([image], self._image_processor, self._model.config)
        image_tensor = [_img.to(dtype=self.dtype, device=self.device) for _img in image_tensor]

        input_ids = (
            tokenizer_image_token(
                prompt_question,
                self._processor,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.device)
        )

        cont = self._model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0,
            max_new_tokens=self._max_new_tokens,
        )
        output_text = self._processor.batch_decode(cont, skip_special_tokens=True)[0]

        return _parse_scores(output_text)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute UniReward scores for (prompt, image) pairs.

        Args:
            prompt: Text prompts.
            image: Generated PIL images.
            video: Optional videos (uses first frame of each).

        Returns:
            RewardModelOutput where ``rewards`` is the mean of alignment,
            coherence, and style scores (1–5 scale).  If ``return_dimensions``
            is enabled, ``extra_info`` contains per-dimension score lists.
        """
        if image is None and video is not None:
            image = [v[0] for v in video]
        if image is None:
            raise ValueError("UniReward requires either 'image' or 'video' input.")
        if len(prompt) != len(image):
            raise ValueError(f"prompt/image length mismatch: {len(prompt)} vs {len(image)}")

        score_fn = self._score_v20 if self._backend_name == "v2.0" else self._score_v15

        all_alignment: List[float] = []
        all_coherence: List[float] = []
        all_style: List[float] = []
        all_overall: List[float] = []

        for p, img in zip(prompt, image):
            alignment, coherence, style, overall = score_fn(p, img)
            all_alignment.append(alignment)
            all_coherence.append(coherence)
            all_style.append(style)
            all_overall.append(overall)

        rewards = torch.tensor(all_overall, device=self.device, dtype=torch.float32)

        extra_info: Dict = {}
        if self._return_dimensions:
            extra_info["alignment"] = all_alignment
            extra_info["coherence"] = all_coherence
            extra_info["style"] = all_style

        return RewardModelOutput(rewards=rewards, extra_info=extra_info)


def download_model():
    """Pre-download helper."""
    UniRewardModel(RewardArguments(device="cpu"), accelerator=None)


if __name__ == "__main__":
    download_model()
