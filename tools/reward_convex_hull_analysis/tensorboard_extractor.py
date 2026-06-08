"""TensorBoard event file image extraction.

Uses ``EventAccumulator`` to scan for logged rollout images (e.g.
``eval/<dataset>/samples/<index>``) and extract them as PIL Images grouped
by training step.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Known tag patterns for eval rollout images
_EVAL_TAG_PATTERNS = ["eval_samples", "eval/"]
_TRAIN_TAG_PATTERNS = ["train_samples", "train/"]


def _is_rollout_image_tag(tag: str) -> bool:
    """Heuristic: does this TensorBoard tag represent a rollout image?"""
    lower = tag.lower()
    # Match eval_samples/N, eval/<dataset>/samples/N, or similar
    for pat in _EVAL_TAG_PATTERNS:
        if lower.startswith(pat) or f"/{pat}" in lower:
            return True
    for pat in _TRAIN_TAG_PATTERNS:
        if lower.startswith(pat) or f"/{pat}" in lower:
            return True
    return False


def _parse_dataset_from_tag(tag: str) -> str:
    """Infer a dataset name from a TensorBoard image tag."""
    parts = tag.split("/")
    if len(parts) >= 3:
        # eval/<dataset>/samples/N
        return parts[1]
    if len(parts) == 2:
        # eval_samples/N → "eval", train_samples/N → "train"
        prefix = parts[0]
        if prefix.endswith("_samples"):
            return prefix[:-len("_samples")]
        return prefix
    return "unknown"


def _parse_tag_idx(tag: str) -> int:
    """Parse sample index from the final segment of a tag."""
    try:
        return int(tag.split("/")[-1])
    except (ValueError, IndexError):
        return 0


def extract_images_from_tensorboard(
    tb_dir: str,
    tag_filter: Optional[str] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
    """Extract all rollout images from a TensorBoard log directory.

    Args:
        tb_dir: Path to the TensorBoard log directory (contains
            ``events.out.tfevents.*`` files).
        tag_filter: Optional substring filter on image tags.  When ``None``,
            any tag matching ``eval/...samples...`` is kept.

    Returns:
        ``(images_by_step, datasets)`` where ``images_by_step`` maps step →
        list of ``{"image": PIL.Image, "tag": str, "tag_idx": int,
        "dataset": str}`` dicts, and ``datasets`` is the sorted list of
        discovered dataset names.
    """
    if not os.path.isdir(tb_dir):
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")

    ea = EventAccumulator(tb_dir, size_guidance={"images": 0})
    ea.Reload()

    available_tags = ea.Tags().get("images", [])
    if not available_tags:
        raise ValueError(
            f"No image tags found in TensorBoard dir: {tb_dir}\n"
            f"Available tags: {list(ea.Tags().keys())}"
        )

    # Filter to rollout image tags
    img_tags: List[str] = []
    for tag in available_tags:
        keep = _is_rollout_image_tag(tag)
        if keep and tag_filter:
            keep = tag_filter in tag
        if keep:
            img_tags.append(tag)

    if not img_tags:
        raise ValueError(
            f"No rollout image tags found. Available image tags: {available_tags}"
        )

    # Collect images per step
    images_by_step: Dict[int, List[Dict[str, Any]]] = {}

    # Also track which datasets we see
    datasets: set = set()

    for tag in img_tags:
        dataset = _parse_dataset_from_tag(tag)
        datasets.add(dataset)
        tag_idx = _parse_tag_idx(tag)

        for event in ea.Images(tag):
            step = event.step
            try:
                img = Image.open(io.BytesIO(event.encoded_image_string))
                img = img.convert("RGB")
            except Exception as exc:
                print(
                    f"  [WARN] Corrupt image at tag={tag}, step={step}: {exc} — skipping"
                )
                continue

            if step not in images_by_step:
                images_by_step[step] = []
            images_by_step[step].append({
                "image": img,
                "tag": tag,
                "tag_idx": tag_idx,
                "dataset": dataset,
            })

    # Sort images within each step by tag_idx so indices are stable
    for step in images_by_step:
        images_by_step[step].sort(key=lambda x: x["tag_idx"])

    return images_by_step, sorted(datasets)
