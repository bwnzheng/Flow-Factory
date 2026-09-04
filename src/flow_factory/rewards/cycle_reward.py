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

"""CycleReward adapter for image-text alignment scoring.

The upstream project provides a BLIP-based scorer trained on cycle-consistency
preferences.  This module keeps the optional dependency and checkpoint loading
in the upstream package while adapting its ``model.score`` API to Flow-Factory's
pointwise reward contract.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from accelerate import Accelerator
from PIL import Image
from transformers import BertTokenizer
from transformers.modeling_utils import PreTrainedModel

from ..hparams import RewardArguments
from .abc import PointwiseRewardModel, RewardModelOutput


def _find_pruneable_heads_and_indices(
    heads: set[int] | list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.Tensor]:
    """Provide the removed Transformers helper used by CycleReward's BLIP fork."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head -= sum(1 for pruned_head in already_pruned_heads if pruned_head < head)
        mask[head] = 0
    index = mask.view(-1).contiguous().eq(1).nonzero().view(-1)
    return heads, index


def _init_cycle_tokenizer() -> object:
    """Initialize the BLIP tokenizer without removed Transformers attributes."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
    return tokenizer


def _get_head_mask(
    model: PreTrainedModel,
    head_mask: Optional[torch.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
) -> list[Optional[torch.Tensor]] | torch.Tensor:
    """Compatibility implementation for the removed Transformers helper."""
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).expand(num_hidden_layers, -1)
    if head_mask.dim() != 2 or head_mask.shape[0] != num_hidden_layers:
        raise ValueError(
            "head_mask must have shape [num_heads] or " "[num_hidden_layers, num_heads]"
        )
    converted = head_mask[:, None, :, None, None]
    if is_attention_chunked:
        converted = converted.unsqueeze(-1)
    return converted.to(dtype=model.dtype, device=model.device)


def _get_all_tied_weights_keys(model: PreTrainedModel) -> dict:
    """Read the expanded tied-weight mapping while supporting older Transformers models."""
    return (
        getattr(model, "_all_tied_weights_keys", None)
        or getattr(model, "_tied_weights_keys", None)
        or {}
    )


def _set_all_tied_weights_keys(model: PreTrainedModel, value: dict) -> None:
    """Store the expanded tied-weight mapping assigned by newer Transformers models."""
    model._all_tied_weights_keys = value


try:
    # CycleReward 0.1.7 vendors a BLIP fork written for older Transformers.
    # Transformers 5 moved two helpers and removed one; expose compatible
    # symbols before importing the optional package.
    import transformers.modeling_utils as _transformers_modeling_utils
    from transformers.pytorch_utils import apply_chunking_to_forward as _apply_chunking_to_forward
    from transformers.pytorch_utils import prune_linear_layer as _prune_linear_layer

    if not hasattr(_transformers_modeling_utils, "apply_chunking_to_forward"):
        _transformers_modeling_utils.apply_chunking_to_forward = _apply_chunking_to_forward
    if not hasattr(_transformers_modeling_utils, "prune_linear_layer"):
        _transformers_modeling_utils.prune_linear_layer = _prune_linear_layer
    if not hasattr(_transformers_modeling_utils, "find_pruneable_heads_and_indices"):
        _transformers_modeling_utils.find_pruneable_heads_and_indices = (
            _find_pruneable_heads_and_indices
        )
    all_tied_weights_keys = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if all_tied_weights_keys is None or (
        isinstance(all_tied_weights_keys, property) and all_tied_weights_keys.fset is None
    ):
        PreTrainedModel.all_tied_weights_keys = property(
            _get_all_tied_weights_keys,
            _set_all_tied_weights_keys,
        )
    if not hasattr(PreTrainedModel, "get_head_mask"):
        PreTrainedModel.get_head_mask = _get_head_mask
    import cyclereward.blip.blip_pretrain as _cycle_blip_pretrain
    from cyclereward import cyclereward as _cyclereward_factory

    _cycle_blip_pretrain.init_tokenizer = _init_cycle_tokenizer
except ImportError:
    _cyclereward_factory = None


class CycleRewardModel(PointwiseRewardModel):
    """Score generated image and prompt pairs with the upstream CycleReward model.

    Configuration is supplied through ``RewardArguments.extra_kwargs``:

    ``model_type``
        Upstream checkpoint variant: ``CycleReward-Combo`` (default),
        ``CycleReward-I2T``, or ``CycleReward-T2I``.
    ``cache_dir``
        Directory used by the upstream loader for model weights and BLIP
        configuration (default ``"./checkpoints"``).

    The upstream model reports uncalibrated alignment scores; Flow-Factory
    returns those scores unchanged so reward normalization remains controlled
    by the shared advantage processor.
    """

    required_fields = ("prompt", "image", "video")
    DEFAULT_MODEL_TYPE = "CycleReward-Combo"
    DEFAULT_CACHE_DIR = "./checkpoints"
    SUPPORTED_MODEL_TYPES = frozenset({"CycleReward-Combo", "CycleReward-I2T", "CycleReward-T2I"})

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        if _cyclereward_factory is None:
            raise ImportError(
                "CycleReward requires the optional `cyclereward` package. "
                'Install it with `pip install -e ".[cycle-reward]"`.'
            )

        extras = config.extra_kwargs or {}
        self.model_type = extras.get("model_type", self.DEFAULT_MODEL_TYPE)
        if self.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported CycleReward model_type {self.model_type!r}. "
                f"Expected one of {sorted(self.SUPPORTED_MODEL_TYPES)}."
            )
        self.cache_dir = extras.get("cache_dir", self.DEFAULT_CACHE_DIR)

        try:
            self.model, self.preprocess = _cyclereward_factory(
                device=self.device,
                model_type=self.model_type,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CycleReward checkpoint {self.model_type!r} in "
                f"cache directory {self.cache_dir!r}. Check the optional dependency, "
                "checkpoint cache, and network access."
            ) from exc

        self.model.eval()

    def _prepare_images(
        self,
        image: Optional[List[Image.Image]],
        video: Optional[List[List[Image.Image]]],
    ) -> List[Image.Image]:
        """Resolve image or video inputs to one PIL image per prompt."""
        if image is not None and video is not None:
            raise ValueError("CycleReward accepts either `image` or `video`, not both.")
        if image is not None:
            return image
        if video is None:
            raise ValueError("CycleReward requires either `image` or `video` input.")
        if any(len(frames) == 0 for frames in video):
            raise ValueError("CycleReward cannot score a video with no frames.")
        return [frames[0] for frames in video]

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute CycleReward scores for a batch of prompt-media pairs.

        Args:
            prompt: Text prompts, one per generated image or video.
            image: Generated PIL images.
            video: Generated videos; the first frame is scored for each video.
            **kwargs: Additional fields accepted by the common reward interface.

        Returns:
            ``RewardModelOutput`` containing a CPU float32 tensor of shape
            ``(batch_size,)``.
        """
        images = self._prepare_images(image, video)
        if len(prompt) != len(images):
            raise ValueError(
                f"CycleReward received {len(prompt)} prompts but {len(images)} images."
            )

        batch_size = getattr(self.config, "batch_size", len(prompt))
        if batch_size <= 0:
            raise ValueError(f"CycleReward batch_size must be positive, got {batch_size}.")

        scores: List[torch.Tensor] = []
        for start in range(0, len(prompt), batch_size):
            image_batch = torch.stack(
                [self.preprocess(sample) for sample in images[start : start + batch_size]]
            ).to(self.device)
            score_batch = self.model.score(
                image_batch,
                prompt[start : start + batch_size],
            )
            scores.append(torch.as_tensor(score_batch).reshape(-1).float().cpu())

        rewards = torch.cat(scores, dim=0) if scores else torch.empty(0, dtype=torch.float32)
        if rewards.numel() != len(prompt):
            raise RuntimeError(
                f"CycleReward returned {rewards.numel()} scores for {len(prompt)} input pairs."
            )
        return RewardModelOutput(
            rewards=rewards,
            extra_info={"model_type": self.model_type},
        )
