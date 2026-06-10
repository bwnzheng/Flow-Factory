"""TensorBoard event file image extraction with disk caching and resumability.

Uses ``EventAccumulator`` (C++ protobuf backend) for fast event parsing, then
decodes PNGs in parallel via a process pool.
"""

from __future__ import annotations

import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# Tag heuristics
# ---------------------------------------------------------------------------

_EVAL_TAG_PATTERNS = ["eval_samples", "eval/"]
_TRAIN_TAG_PATTERNS = ["train_samples", "train/"]


def _is_rollout_image_tag(tag: str) -> bool:
    lower = tag.lower()
    for pat in _EVAL_TAG_PATTERNS:
        if lower.startswith(pat) or f"/{pat}" in lower:
            return True
    for pat in _TRAIN_TAG_PATTERNS:
        if lower.startswith(pat) or f"/{pat}" in lower:
            return True
    return False


def _parse_dataset_from_tag(tag: str) -> str:
    """Normalize dataset name from tag.

    ``eval/default/samples/N``, ``eval_samples/N`` → ``"eval"``
    ``train/default/samples/N``, ``train_samples/N`` → ``"train"``
    """
    parts = tag.split("/")
    if len(parts) >= 3:
        # eval/<sub>/samples/N → "eval"
        return parts[0] if parts[0].endswith("eval") or parts[0].endswith("train") else parts[1]
    if len(parts) == 2:
        prefix = parts[0]
        if prefix.endswith("_samples"):
            return prefix[:-len("_samples")]
        return prefix
    return "unknown"


def _parse_tag_idx(tag: str) -> int:
    try:
        return int(tag.split("/")[-1])
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME = "manifest.json"


def _load_manifest(output_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(output_dir, _MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_manifest(output_dir: str, manifest: Dict[str, Any]):
    path = os.path.join(output_dir, _MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Worker: decode raw bytes → PNG file
# ---------------------------------------------------------------------------


def _decode_and_save(args: Tuple[bytes, str]) -> Optional[str]:
    img_bytes, output_path = args
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, "PNG")
        return output_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main decode-to-disk (EventAccumulator + parallel decode)
# ---------------------------------------------------------------------------


def decode_images_to_disk(
    tb_dir: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    max_per_step: int = 0,
    num_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Decode rollout images from TensorBoard event files and save as PNGs.

    Uses ``EventAccumulator`` (C++ protobuf parsing) for fast event loading,
    then dispatches PNG decoding to worker processes in parallel.

    Args:
        tb_dir: Path to TensorBoard log directory.
        output_dir: Root directory for decoded images and manifest.
        datasets: If non-empty, only keep images from these datasets.
        max_per_step: If > 0, cap images per ``(step, dataset)`` pair.
        num_workers: Number of parallel decoder processes.

    Returns:
        Manifest ``images`` list.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(tb_dir):
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")

    dataset_set = set(datasets) if datasets else None

    # ---- Phase 0: fast path — manifest already complete ----
    manifest = _load_manifest(output_dir)
    if manifest is not None and manifest.get("total", 0) > 0:
        if manifest.get("decoded", 0) >= manifest["total"]:
            sample = manifest["images"][:5]
            if all(os.path.isfile(os.path.join(output_dir, img["path"]))
                   for img in sample):
                # Check if stored datasets match current filter
                stored_datasets = set(img.get("dataset", "") for img in manifest["images"])
                if dataset_set is None or stored_datasets == dataset_set:
                    print(f"  All {manifest['total']} images already decoded — skipping.")
                    return manifest["images"]
                else:
                    # Filter changed — filter the existing manifest
                    filtered = [img for img in manifest["images"]
                                if img.get("dataset", "") in dataset_set]
                    print(f"  Using {len(filtered)}/{manifest['total']} images "
                          f"(datasets filter: {dataset_set})")
                    return filtered

    # ---- Phase 1: load events with EventAccumulator (C++ backend, fast) ----
    print(f"  Loading TensorBoard events from {tb_dir} ...")
    ea = EventAccumulator(tb_dir, size_guidance={"images": 0})
    ea.Reload()
    print(f"  Events loaded.")

    available = ea.Tags().get("images", [])
    if not available:
        raise ValueError(
            f"No image tags in {tb_dir}. Available: {list(ea.Tags().keys())}"
        )

    img_tags = [t for t in available if _is_rollout_image_tag(t)]
    if not img_tags:
        raise ValueError(f"No rollout image tags. Available: {available}")
    print(f"  Found {len(img_tags)} image tags.")

    # ---- Phase 2: collect metadata (no bytes) + build work plan ----
    work_items: List[Tuple[int, str, int, str, bytes]] = []
    # (step, dataset, tag_idx, tag, encoded_bytes)
    step_ds_counts: Dict[Tuple[int, str], int] = {}

    for tag in img_tags:
        dataset = _parse_dataset_from_tag(tag)
        if dataset_set is not None and dataset not in dataset_set:
            continue
        tag_idx = _parse_tag_idx(tag)

        for event in ea.Images(tag):
            step = event.step
            key = (step, dataset)
            if max_per_step > 0 and step_ds_counts.get(key, 0) >= max_per_step:
                continue
            work_items.append(
                (step, dataset, tag_idx, tag, event.encoded_image_string)
            )
            step_ds_counts[key] = step_ds_counts.get(key, 0) + 1

    total = len(work_items)
    print(f"  Total rollout images: {total}")
    if total == 0:
        return []

    # Sort for predictable order
    work_items.sort(key=lambda x: (x[0], x[1], x[2]))

    # ---- Phase 3: check manifest, skip already-decoded ----
    manifest = _load_manifest(output_dir)
    image_list: List[Dict[str, Any]] = []
    saved_paths: Set[str] = set()

    if manifest is not None:
        image_list = list(manifest["images"])
        for img in image_list:
            if os.path.isfile(os.path.join(output_dir, img["path"])):
                saved_paths.add(img["path"])

    # Prune missing
    image_list = [img for img in image_list if img["path"] in saved_paths]

    if len(image_list) >= total:
        print(f"  All {total} images already decoded — skipping.")
        return image_list

    already = len(image_list)
    if already > 0:
        print(f"  Resuming: {already}/{total} decoded, {total - already} remaining.")

    # ---- Phase 4: dispatch undecoded images to worker pool ----
    # Build flat dispatch list; tag-at-a-time bytes already collected above
    dispatch: List[Tuple[bytes, str, int, str, int]] = []
    # (img_bytes, rel_path, step, dataset, tag_idx)

    for i in range(already, total):
        step, dataset, tag_idx, tag, encoded_bytes = work_items[i]
        fname = f"step_{step}_idx_{tag_idx}.png"
        rel_path = os.path.join(dataset, fname)
        if rel_path in saved_paths:
            image_list.append({
                "step": step, "dataset": dataset, "tag_idx": tag_idx,
                "path": rel_path,
            })
            saved_paths.add(rel_path)
            continue
        dispatch.append((encoded_bytes, rel_path, step, dataset, tag_idx))

    if not dispatch:
        _save_manifest(output_dir, {
            "total": total, "decoded": total, "tb_dir": tb_dir,
            "images": image_list,
        })
        return image_list

    remaining = len(dispatch)
    print(f"  Decoding {remaining} images with {num_workers} workers ...")

    decoded = len(image_list)
    last_reported = decoded

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        for img_bytes, rel_path, step, dataset, tag_idx in dispatch:
            full_path = os.path.join(output_dir, rel_path)
            fut = pool.submit(_decode_and_save, (img_bytes, full_path))
            futures[fut] = (rel_path, step, dataset, tag_idx)

            # Collect completed results periodically
            if len(futures) >= 500:
                _harvest(futures, image_list, saved_paths)
                decoded = len(image_list)
                # Only print when progress actually changed
                if decoded != last_reported:
                    last_reported = decoded
                    _save_manifest(output_dir, {
                        "total": total, "decoded": decoded, "tb_dir": tb_dir,
                        "images": image_list,
                    })
                    pct = decoded * 100 // total
                    print(f"    {decoded}/{total} ({pct}%)")

        # Wait for all remaining futures, then harvest
        wait(futures)
        _harvest(futures, image_list, saved_paths)

    # Final manifest
    _save_manifest(output_dir, {
        "total": total, "decoded": len(image_list), "tb_dir": tb_dir,
        "images": image_list,
    })
    print(f"  Decoding complete: {len(image_list)} images saved.")
    return image_list


def _harvest(futures: dict, image_list: List[Dict[str, Any]], saved_paths: Set[str]):
    """Collect completed futures from the dict."""
    done = [f for f in futures if f.done()]
    for f in done:
        rel_path, step, dataset, tag_idx = futures.pop(f)
        if f.result() is not None:
            image_list.append({
                "step": step, "dataset": dataset, "tag_idx": tag_idx,
                "path": rel_path,
            })
            saved_paths.add(rel_path)


# ---------------------------------------------------------------------------
# Load decoded images from disk for reward scoring
# ---------------------------------------------------------------------------


def load_decoded_images(
    output_dir: str,
    image_index: Optional[List[Dict[str, Any]]] = None,
    max_per_step: int = 0,
) -> Dict[int, List[Dict[str, Any]]]:
    """Load decoded PNGs from disk, grouped by step."""
    if image_index is None:
        manifest = _load_manifest(output_dir)
        if manifest is None:
            raise FileNotFoundError(
                f"No manifest.json in {output_dir}. Run decode_images_to_disk first."
            )
        image_index = manifest["images"]

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    step_counts: Dict[int, int] = {}

    for entry in image_index:
        step = entry["step"]
        if max_per_step > 0:
            c = step_counts.get(step, 0)
            if c >= max_per_step:
                continue
            step_counts[step] = c + 1

        full_path = os.path.join(output_dir, entry["path"])
        if not os.path.isfile(full_path):
            continue

        try:
            img = Image.open(full_path)
            img.load()
        except (OSError, IOError):
            print(f"  [WARN] Corrupt image file: {full_path} — skipping")
            continue

        if step not in grouped:
            grouped[step] = []
        grouped[step].append({
            "image": img,
            "tag_idx": entry["tag_idx"],
            "dataset": entry["dataset"],
        })

    return grouped
