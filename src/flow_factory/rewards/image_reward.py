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

"""ImageReward pointwise reward adapter.

The model and preprocessing stay in the upstream ``image-reward`` package so
that checkpoint loading and score semantics remain aligned with the authors'
reference implementation. Flow-Factory only adapts its scalar scorer to the
``PointwiseRewardModel`` contract.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional

import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from .abc import PointwiseRewardModel, RewardModelOutput

try:
    import ImageReward as _image_reward
except ImportError:
    _image_reward = None


class ImageRewardModel(PointwiseRewardModel):
    """Score prompt-image pairs with the upstream ImageReward-v1.0 model.

    Configuration (via ``RewardArguments`` extra keys):
        model_path: Upstream model name (default ``"ImageReward-v1.0"``) or
            a local checkpoint path.
        download_root: Optional cache directory passed to ``ImageReward.load``.
        med_config: Optional local BLIP medical configuration path passed to
            ``ImageReward.load``.

    ImageReward is a text-to-image reward and therefore accepts generated
    images only. Video, audio, and conditional-image inputs are intentionally
    not declared in ``required_fields``.
    """

    required_fields = ("prompt", "image")
    DEFAULT_MODEL = "ImageReward-v1.0"

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        if _image_reward is None:
            raise ImportError(
                "ImageReward reward requires the optional `image-reward` package. "
                'Install it with `pip install -e ".[image-reward]"`.'
            )

        extras = config.extra_kwargs or {}
        model_name = extras.get("model_path", extras.get("model_name_or_path", self.DEFAULT_MODEL))
        load_kwargs = {"device": self.device}
        if extras.get("download_root") is not None:
            load_kwargs["download_root"] = extras["download_root"]
        if extras.get("med_config") is not None:
            load_kwargs["med_config"] = extras["med_config"]

        try:
            self.model = _image_reward.load(model_name, **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ImageReward model {model_name!r}. Check the checkpoint "
                "path/cache and the optional image-reward installation."
            ) from exc
        self.model.eval()
        self.model_name = model_name

    def _autocast_context(self) -> contextlib.AbstractContextManager:
        """Return CUDA autocast for the configured reduced-precision dtype."""
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute one normalized ImageReward score for every prompt-image pair.

        Args:
            prompt: Text prompts, one per generated image.
            image: Generated PIL images, one per prompt.
            **kwargs: Ignored fields accepted for the common reward interface.

        Returns:
            ``RewardModelOutput`` with a CPU float32 tensor of shape ``(batch,)``.
        """
        if image is None:
            raise ValueError("ImageReward requires `image` input.")
        if len(prompt) != len(image):
            raise ValueError(f"ImageReward received {len(prompt)} prompts but {len(image)} images.")

        with self._autocast_context():
            scores = [self.model.score(text, sample) for text, sample in zip(prompt, image)]
        rewards = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
        if rewards.numel() != len(prompt):
            raise RuntimeError(
                f"ImageReward returned {rewards.numel()} scores for {len(prompt)} input pairs."
            )
        return RewardModelOutput(
            rewards=rewards,
            extra_info={"model_name": self.model_name},
        )
