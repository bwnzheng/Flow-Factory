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
# The image checkpoint must be downloaded/extracted locally; see the evaluator
# README for the split-archive commands.

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__, rank_zero_only=True)

_DEFAULT_MODEL_PATH = "THUDM/VisionReward-Image-bf16"
_DEFAULT_TOKENIZER_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
_TOKENIZER_ENV = "VISIONREWARD_TOKENIZER"
_TOKENIZER_REPO_ENV = "VISIONREWARD_TOKENIZER_REPO"
_TOKENIZER_VOCAB_SIZE = 128256
_TOKENIZER_ARTIFACTS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
)
_DEFAULT_QUESTION_FILE = "VisionReward_Image/VisionReward_image_qa_select.txt"
_DEFAULT_WEIGHT_FILE = "VisionReward_Image/weight_select.json"
_DEFAULT_ALIGNMENT_MODEL = "clip-flant5-xxl"

# The first three questions are mask features in the official image scorer.
# Their values gate the remaining question features before the linear head.
_MASK_FEATURE_MAP = {
    0: (22, 23, 24, 28, 29),
    1: (25, 26),
    2: (27,),
}
_MASK_INDICES = frozenset(_MASK_FEATURE_MAP)


class _VisionRewardChatArgs(SimpleNamespace):
    """Expose the argparse fields expected by VisionReward's chat helper."""

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


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
    repo = Path(repo).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"VisionReward repo not found at {repo}")
    return repo


def _resolve_model_path(extras: dict) -> Path:
    """Resolve and validate an extracted local VisionReward checkpoint.

    SwissArmyTransformer accepts either one of its short built-in model names
    or a directory.  VisionReward's Hugging Face identifier is neither: its
    checkpoint is uploaded as split archive parts and must be merged/extracted
    before it can be passed to ``VisualLlamaEVA.from_pretrained``.
    """
    configured = extras.get("model_path")
    env_model = os.environ.get("VISIONREWARD_MODEL")
    # Allow the template's placeholder HF ID to be overridden without editing
    # every source entry, while preserving an explicitly configured path.
    if not configured or (configured == _DEFAULT_MODEL_PATH and env_model):
        configured = env_model or _DEFAULT_MODEL_PATH

    model_path = Path(str(configured)).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            "VisionReward requires an extracted local checkpoint directory, but "
            f"model_path={configured!r} does not exist. The value "
            f"{_DEFAULT_MODEL_PATH!r} is a Hugging Face repository ID, not a SAT "
            "model alias. Download the repository, merge and extract its split "
            "checkpoint files, then set `model_path` to the extracted directory. "
            "For example: `cat ckpts/split_part_* > ckpts/visionreward_image.tar` "
            "followed by `tar -xf ckpts/visionreward_image.tar -C <model_dir>`."
        )

    model_path = model_path.resolve()
    config_path = model_path / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "VisionReward checkpoint is incomplete: expected "
            f"{config_path}. Point `model_path` at the directory containing "
            "model_config.json after extracting the official checkpoint."
        )
    tracker_path = model_path / "latest"
    if not tracker_path.is_file():
        raise FileNotFoundError(
            "VisionReward checkpoint is incomplete: expected the SAT tracker "
            f"file {tracker_path}. Merge and extract the split checkpoint "
            "archive before starting evaluation."
        )
    tracker_value = tracker_path.read_text(encoding="utf-8").strip()
    checkpoint_dir_name = tracker_value
    if tracker_value != "release":
        try:
            if int(tracker_value) <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                f"Invalid VisionReward SAT tracker value {tracker_value!r} in "
                f"{tracker_path}; expected a positive iteration or 'release'."
            ) from exc
    checkpoint_file = model_path / checkpoint_dir_name / "mp_rank_00_model_states.pt"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            "VisionReward checkpoint is incomplete: expected "
            f"{checkpoint_file}. The split archive has not been extracted into "
            "the directory referenced by `latest`."
        )
    return model_path


def _is_hf_repo_id(value: Any) -> bool:
    """Return whether *value* is an unambiguous Hugging Face repo ID."""
    if not isinstance(value, str):
        return False
    spec = value.strip()
    if spec.startswith("hf://"):
        spec = spec[len("hf://") :]
    parts = spec.split("/")
    return len(parts) == 2 and all(
        part and part not in {".", ".."} and "\\" not in part for part in parts
    )


def _normalise_hf_repo_id(value: Any) -> str:
    """Validate and normalize a Hugging Face ``owner/repository`` ID."""
    if not isinstance(value, str):
        raise TypeError(f"VisionReward tokenizer repository must be a string, got {value!r}.")
    repo_id = value.strip()
    if repo_id.startswith("hf://"):
        repo_id = repo_id[len("hf://") :]
    if not _is_hf_repo_id(repo_id):
        raise ValueError(
            "VisionReward tokenizer repository must use the Hugging Face "
            f"'owner/repository' form, got {value!r}."
        )
    return repo_id


def _validate_tokenizer_directory(tokenizer_path: Path, source: str) -> Path:
    """Validate tokenizer files and the Llama-3 vocabulary size."""
    tokenizer_path = tokenizer_path.expanduser().resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"VisionReward tokenizer source {source!r} resolved to {tokenizer_path}, "
            "but that directory does not exist."
        )

    tokenizer_files = ("tokenizer.json", "tokenizer.model", "spiece.model")
    if not any((tokenizer_path / filename).is_file() for filename in tokenizer_files):
        raise FileNotFoundError(
            "VisionReward tokenizer directory is incomplete: expected one of "
            f"{', '.join(tokenizer_files)} under {tokenizer_path}. Copy the "
            "complete Llama-3 tokenizer snapshot rather than only model weights."
        )
    config_files = ("tokenizer_config.json", "config.json")
    if not any((tokenizer_path / filename).is_file() for filename in config_files):
        raise FileNotFoundError(
            "VisionReward tokenizer directory is incomplete: expected one of "
            f"{', '.join(config_files)} under {tokenizer_path}."
        )

    # Llama 3's embedding/output matrices use 128,256 token IDs.  A tokenizer
    # from another family can load successfully but silently produce invalid
    # IDs, so reject an explicitly incompatible metadata value before workers
    # construct the heavyweight VisionReward checkpoint.  Prefer config.json;
    # tokenizer-only repositories may omit it, in which case infer the size
    # from tokenizer.json's base and added-token IDs.
    metadata_path = tokenizer_path / "config.json"
    vocab_size = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not parse tokenizer metadata at {metadata_path} for {source!r}."
            ) from exc
        vocab_size = metadata.get("vocab_size") if isinstance(metadata, dict) else None
        if vocab_size is not None:
            try:
                vocab_size = int(vocab_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid tokenizer vocab_size={vocab_size!r} in {metadata_path}."
                ) from exc
    if vocab_size is None and (tokenizer_path / "tokenizer.json").is_file():
        tokenizer_json = tokenizer_path / "tokenizer.json"
        try:
            tokenizer_payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not parse tokenizer data at {tokenizer_json} for {source!r}."
            ) from exc
        token_ids: List[int] = []
        model_vocab = (
            tokenizer_payload.get("model", {}).get("vocab", {})
            if isinstance(tokenizer_payload, dict)
            else {}
        )
        if isinstance(model_vocab, dict):
            token_ids.extend(
                int(token_id) for token_id in model_vocab.values() if isinstance(token_id, int)
            )
        added_tokens = (
            tokenizer_payload.get("added_tokens", []) if isinstance(tokenizer_payload, dict) else []
        )
        if isinstance(added_tokens, list):
            token_ids.extend(
                int(token.get("id"))
                for token in added_tokens
                if isinstance(token, dict) and isinstance(token.get("id"), int)
            )
        if token_ids:
            vocab_size = max(token_ids) + 1

    if vocab_size is not None and vocab_size != _TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            "VisionReward requires a Llama-3-compatible tokenizer with "
            f"vocab_size={_TOKENIZER_VOCAB_SIZE}, but {source!r} declares or "
            f"implies vocab_size={vocab_size}. A Qwen/Mistral/Llama-2 tokenizer "
            "cannot be substituted for this checkpoint."
        )
    return tokenizer_path


def _download_tokenizer_snapshot(repo_id: Any, extras: dict) -> Path:
    """Download or resolve only tokenizer files from a Hugging Face repo."""
    repo_id = _normalise_hf_repo_id(repo_id)
    local_files_only = extras.get("tokenizer_local_files_only", False)
    if not isinstance(local_files_only, bool):
        raise TypeError(
            "VisionReward `tokenizer_local_files_only` must be a boolean, "
            f"got {local_files_only!r}."
        )

    revision = extras.get("tokenizer_revision")
    cache_dir = extras.get("tokenizer_cache_dir")
    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=str(revision) if revision else None,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
            allow_patterns=list(_TOKENIZER_ARTIFACTS),
        )
    except (
        EntryNotFoundError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        OSError,
    ) as exc:
        mode = "the local Hugging Face cache" if local_files_only else "the Hugging Face Hub"
        raise FileNotFoundError(
            f"Could not resolve VisionReward tokenizer repository {repo_id!r} from {mode}. "
            "If this repository is public, download only its tokenizer files on a "
            "networked host and set `tokenizer_path` to the copied directory. "
            "For an offline cache lookup, set `tokenizer_local_files_only: true`. "
            f"Original error: {exc}"
        ) from exc
    return _validate_tokenizer_directory(Path(str(snapshot)), f"HF repo {repo_id!r}")


def _resolve_tokenizer_path(extras: dict) -> Path:
    """Resolve a local path or an explicitly selected Hugging Face tokenizer repo.

    ``tokenizer_path`` remains compatible with the existing local-directory
    contract.  A public repository can be selected with ``tokenizer_repo`` (or
    by putting an unambiguous ``owner/repository`` ID in ``tokenizer_path``);
    only tokenizer/config files are downloaded.  The gated Meta default is
    never replaced implicitly, preserving the official checkpoint behavior.
    """
    configured = extras.get("tokenizer_path")
    env_tokenizer = os.environ.get(_TOKENIZER_ENV)
    configured_text = str(configured).strip() if configured else ""

    # A concrete local path or VISIONREWARD_TOKENIZER always wins over a repo
    # alias.  The template's gated ID is a placeholder and can be replaced by
    # either environment variable without editing every source entry.
    if configured_text and configured_text != _DEFAULT_TOKENIZER_PATH:
        candidate: Any = configured
    elif env_tokenizer:
        candidate = env_tokenizer
    else:
        candidate = None

    if candidate is None:
        candidate = extras.get("tokenizer_repo") or os.environ.get(_TOKENIZER_REPO_ENV)
        if candidate:
            return _download_tokenizer_snapshot(candidate, extras)
        candidate = _DEFAULT_TOKENIZER_PATH

    tokenizer_path = Path(str(candidate)).expanduser()
    if tokenizer_path.is_dir():
        return _validate_tokenizer_directory(tokenizer_path, f"local path {candidate!r}")

    # Explicitly selecting a public repo through tokenizer_path is convenient,
    # while the gated Meta placeholder still fails fast with an actionable
    # offline message instead of attempting an unauthorized download.
    if candidate != _DEFAULT_TOKENIZER_PATH and _is_hf_repo_id(candidate):
        return _download_tokenizer_snapshot(candidate, extras)

    raise FileNotFoundError(
        "VisionReward requires a local Llama-3 tokenizer directory, but "
        f"tokenizer_path={candidate!r} is not a directory. The default "
        f"{_DEFAULT_TOKENIZER_PATH!r} is a gated Hugging Face repository ID. "
        "Set `extra_kwargs.tokenizer_path` or `VISIONREWARD_TOKENIZER` to a "
        "local snapshot, or select a public compatible repository with "
        "`extra_kwargs.tokenizer_repo` / `VISIONREWARD_TOKENIZER_REPO`. The "
        "snapshot must contain tokenizer.json (or tokenizer.model) and "
        "tokenizer_config.json/config.json."
    )


def _resolve_repo_file(repo: Path, configured: Any, default_relative: str) -> Path:
    """Resolve a question/weight file relative to the VisionReward checkout."""
    path = Path(str(configured)).expanduser() if configured else repo / default_relative
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VisionReward resource not found: {path}")
    return path


def _load_image_score_head(
    question_file: Path, weight_file: Path
) -> Tuple[List[str], np.ndarray, float]:
    """Load the official selected questions and linear scoring head."""
    questions = [line.strip() for line in question_file.read_text(encoding="utf-8").splitlines()]
    questions = [question for question in questions if question]
    if not questions:
        raise ValueError(f"VisionReward question file is empty: {question_file}")
    max_feature_index = max(index for values in _MASK_FEATURE_MAP.values() for index in values)
    if len(questions) <= max_feature_index:
        raise ValueError(
            f"VisionReward question file {question_file} contains {len(questions)} "
            f"questions, but the official mask map requires at least "
            f"{max_feature_index + 1}."
        )

    payload = json.loads(weight_file.read_text(encoding="utf-8"))
    try:
        coefficients = np.asarray(payload["coef"], dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(payload["intercept"], dtype=np.float64).reshape(-1)[0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid VisionReward weight file {weight_file}; expected `coef` and "
            "a scalar `intercept`."
        ) from exc

    expected_features = 1 + len(questions) - len(_MASK_INDICES)
    if coefficients.size != expected_features:
        raise ValueError(
            f"VisionReward question/weight mismatch: {len(questions)} questions with "
            f"{len(_MASK_INDICES)} mask questions produce {expected_features} features, "
            f"but {weight_file} contains {coefficients.size} coefficients."
        )
    return questions, coefficients, intercept


def _import_visionreward(extras: dict):
    """Import VisionReward inference utilities from the cloned repo."""
    repo = _resolve_repo_path(extras)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        from sat.model.mixins import CachedAutoregressiveMixin
        from utils.models import VisualLlamaEVA
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
        VisualLlamaEVA,
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
    """VisionReward's official image scorer.

    The upstream project is a standalone SAT/CogVLM2 application.  Its model
    checkpoint must be downloaded and extracted locally before this adapter is
    constructed; passing the Hugging Face repository ID directly to SAT causes
    SAT to look up a nonexistent built-in ``MODEL_URLS`` entry.

    Configuration (via ``RewardArguments.extra_kwargs``):
        repo_path: Cloned VisionReward checkout (or ``VISIONREWARD_REPO``).
        model_path: Extracted checkpoint directory containing
            ``model_config.json`` (or ``VISIONREWARD_MODEL``).
        tokenizer_path: Local Llama-3 tokenizer directory (or
            ``VISIONREWARD_TOKENIZER``). An unambiguous public Hugging Face
            ``owner/repository`` ID is also accepted and downloads only the
            tokenizer/config files during preflight.
        tokenizer_repo: Optional public Hugging Face tokenizer repository (or
            ``VISIONREWARD_TOKENIZER_REPO``). This is useful when keeping
            ``tokenizer_path`` as the template placeholder.
        tokenizer_revision, tokenizer_cache_dir, tokenizer_local_files_only:
            Optional Hugging Face snapshot controls.
        question_file: Selected image QA file (defaults to the upstream file).
        weight_file: Selected linear head (defaults to the upstream file).
        alignment_model: VQAScore model used for prompt-image alignment.
        max_length, top_p, top_k, temperature, stream_chat, version: upstream
            chat-generation options.

    VisionReward pins an older Transformers/PyTorch stack than Flow-Factory.
    When those dependencies cannot coexist, run the official model in an
    isolated reward server and use the remote reward wrapper instead.
    """

    required_fields = ("prompt", "image")

    DEFAULT_MODEL_PATH = _DEFAULT_MODEL_PATH

    @classmethod
    def validate_config(cls, config: RewardArguments) -> None:
        """Validate files before the evaluator starts worker processes.

        Args:
            config: Reward configuration containing VisionReward paths.
        """
        extras = config.extra_kwargs or {}
        repo = _resolve_repo_path(extras)
        _resolve_model_path(extras)
        _resolve_tokenizer_path(extras)
        question_file = _resolve_repo_file(
            repo, extras.get("question_file"), _DEFAULT_QUESTION_FILE
        )
        weight_file = _resolve_repo_file(repo, extras.get("weight_file"), _DEFAULT_WEIGHT_FILE)
        _load_image_score_head(question_file, weight_file)

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        extras = config.extra_kwargs or {}

        # Resolve all local artifacts before importing the optional upstream
        # stack.  This keeps a bad path from producing a secondary dependency
        # error and avoids initializing SAT when the checkpoint is absent.
        model_path = _resolve_model_path(extras)
        tokenizer_path = _resolve_tokenizer_path(extras)
        repo = _resolve_repo_path(extras)
        question_file = _resolve_repo_file(
            repo, extras.get("question_file"), _DEFAULT_QUESTION_FILE
        )
        weight_file = _resolve_repo_file(repo, extras.get("weight_file"), _DEFAULT_WEIGHT_FILE)
        (
            self._questions,
            self._score_coefficients,
            self._score_intercept,
        ) = _load_image_score_head(question_file, weight_file)

        # Import VisionReward modules
        (
            VisualLlamaEVA,
            CachedAutoregressiveMixin,
            self._chat_fn,
            get_image_processor,
            llama2_text_processor_inference,
            llama3_tokenizer,
            VQAScore,
        ) = _import_visionreward(extras)

        self._max_length = int(extras.get("max_length", 3328))
        self._top_p = float(extras.get("top_p", 0.4))
        self._top_k = int(extras.get("top_k", 1))
        self._temperature = float(extras.get("temperature", 0.8))
        self._version = str(extras.get("version", "vqa"))
        self._stream_chat = bool(extras.get("stream_chat", False))
        self._bf16 = self.dtype == torch.bfloat16
        self._fp16 = self.dtype == torch.float16
        self._chat_args = _VisionRewardChatArgs(
            bf16=self._bf16,
            fp16=self._fp16,
            stream_chat=self._stream_chat,
            chinese=False,
        )

        # VisualLlamaEVA is the concrete class used by the upstream image
        # inference script.  AutoModel is intentionally not used: the released
        # checkpoint's model_config.json contains a registry name that is not
        # provided by the cloned VisionReward checkout.
        model_args = argparse.Namespace(
            deepspeed=None,
            local_rank=0,
            rank=0,
            world_size=1,
            model_parallel_size=1,
            mode="inference",
            skip_init=True,
            use_gpu_initialization=self.device.type == "cuda",
            device=str(self.device),
            max_length=self._max_length,
            top_p=self._top_p,
            top_k=self._top_k,
            temperature=self._temperature,
            version=self._version,
            tokenizer_path=str(tokenizer_path),
            bf16=self._bf16,
            fp16=self._fp16,
            stream_chat=self._stream_chat,
        )
        self._model, loaded_args = VisualLlamaEVA.from_pretrained(str(model_path), args=model_args)
        self._model = self._model.eval()
        self._model.add_mixin("auto-regressive", CachedAutoregressiveMixin())

        # Tokenizer & processors
        self._tokenizer = llama3_tokenizer(model_args.tokenizer_path, signal_type=self._version)
        image_size = loaded_args.eva_args["image_size"]
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        self._image_processor = get_image_processor(image_size)
        self._text_processor = llama2_text_processor_inference(
            self._tokenizer, self._max_length, self._model.image_length
        )

        # The alignment model is part of the official linear-head feature
        # vector.  It is loaded once per worker, alongside VisionReward.
        alignment_model = str(extras.get("alignment_model", _DEFAULT_ALIGNMENT_MODEL))
        alignment_device = str(extras.get("alignment_device", self.device))
        self._vqa_scorer = VQAScore(model=alignment_model, device=alignment_device)

        if accelerator is not None:
            accelerator.wait_for_everyone()

        logger.info(
            f"VisionReward loaded: model={model_path}, device={self.device}, "
            f"questions={len(self._questions)}, alignment={alignment_model}"
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    @torch.no_grad()
    def _score_single(self, prompt: str, image: Image.Image) -> Dict[str, float]:
        """Score a single image across all dimensions.

        Saves the image to a temporary file because both the official chat
        helper and VQAScore use file paths as their public input contract.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            with self._autocast():
                alignment_value = self._vqa_scorer(images=[tmp_path], texts=[prompt])[0][0]
                if isinstance(alignment_value, torch.Tensor):
                    alignment = alignment_value.detach().float().cpu().item()
                else:
                    alignment = float(alignment_value)
                answers: List[str] = []
                for question in self._questions:
                    response_result = self._chat_fn(
                        image_path=tmp_path,
                        image=None,
                        model=self._model,
                        text_processor=self._text_processor,
                        img_processor=self._image_processor,
                        query=question,
                        max_length=self._max_length,
                        top_p=self._top_p,
                        temperature=self._temperature,
                        top_k=self._top_k,
                        invalid_slices=self._text_processor.invalid_slices,
                        args=self._chat_args,
                    )
                    response = (
                        response_result[0]
                        if isinstance(response_result, tuple)
                        else response_result
                    )
                    answers.append(str(response).strip().lower())

            reward = np.asarray(
                [1.0 if answer.startswith("yes") else -1.0 for answer in answers],
                dtype=np.float64,
            )
            for mask_index, feature_indices in _MASK_FEATURE_MAP.items():
                for feature_index in feature_indices:
                    reward[feature_index] *= float(reward[mask_index] > 0)
            reward_filtered = [
                value for index, value in enumerate(reward) if index not in _MASK_INDICES
            ]
            features = np.asarray([alignment, *reward_filtered], dtype=np.float64)
            return {
                "overall": float(
                    np.dot(features, self._score_coefficients) + self._score_intercept
                ),
                "alignment": float(alignment),
            }
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
            RewardModelOutput where ``rewards`` is the official linear-head
            score. ``extra_info`` contains the prompt-image alignment feature.
        """
        if image is None and video is not None:
            image = [v[0] for v in video]
        if image is None:
            raise ValueError("VisionReward requires either 'image' or 'video' input.")
        if len(prompt) != len(image):
            raise ValueError(f"prompt/image length mismatch: {len(prompt)} vs {len(image)}")

        all_overall: List[float] = []
        alignments: List[float] = []

        for p, img in zip(prompt, image):
            score_data = self._score_single(p, img)
            all_overall.append(score_data["overall"])
            alignments.append(score_data["alignment"])

        rewards = torch.tensor(all_overall, device=self.device, dtype=torch.float32)
        return RewardModelOutput(rewards=rewards, extra_info={"alignment": alignments})


def download_model() -> None:
    """Validate the configured checkpoint and tokenizer without loading weights."""
    config = RewardArguments(
        device="cpu",
        extra_kwargs={
            "repo_path": os.environ.get("VISIONREWARD_REPO", "VisionReward"),
            "model_path": os.environ.get("VISIONREWARD_MODEL", _DEFAULT_MODEL_PATH),
            "tokenizer_path": os.environ.get(_TOKENIZER_ENV),
            "tokenizer_repo": os.environ.get(_TOKENIZER_REPO_ENV),
        },
    )
    VisionRewardModel.validate_config(config)
    logger.info("VisionReward checkpoint and scoring resources are ready.")


if __name__ == "__main__":
    download_model()
