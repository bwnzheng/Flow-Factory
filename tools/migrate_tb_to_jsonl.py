#!/usr/bin/env python3
"""Migrate old TensorBoard event files to the new JSONL log format.

Always migrates ALL datasets.  Supports resume: re-running will skip
already-saved images and pick up where it left off.

Usage::

    python tools/migrate_tb_to_jsonl.py --run-name sd3-5_lora_nft_20260602_100414
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tag heuristics
# ---------------------------------------------------------------------------

def _is_rollout_image_tag(tag: str) -> bool:
    lower = tag.lower()
    for pat in ["eval_samples", "eval/", "train_samples", "train/"]:
        if lower.startswith(pat) or f"/{pat}" in lower:
            return True
    return False


def _tag_to_key(tag: str) -> str:
    return tag


def _key_to_filename(key: str) -> str:
    return key.replace("/", "_") + ".png"


# ---------------------------------------------------------------------------
# Prompt reconstruction (GroupContiguousSampler)
# ---------------------------------------------------------------------------

def _reconstruct_training_prompts(
    prompts_file: str,
    steps: List[int],
    seed: int = 42,
    group_size: int = 16,
    unique_sample_num: int = 48,
    world_size: int = 4,
    dataset_size: Optional[int] = None,
) -> Dict[int, List[str]]:
    if not os.path.isfile(prompts_file):
        return {}
    with open(prompts_file, "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    D = dataset_size if dataset_size else len(all_prompts)
    M = unique_sample_num
    G = M // world_size
    result: Dict[int, List[str]] = {}
    for epoch in steps:
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        indices = torch.randperm(D, generator=g)[:M].tolist()
        group_perm = torch.randperm(M, generator=g).tolist()
        shuffled_groups = [indices[i] for i in group_perm]
        my_groups = shuffled_groups[:G]
        result[epoch] = [all_prompts[my_groups[0]], all_prompts[my_groups[1]]]
    return result


# ---------------------------------------------------------------------------
# Migration (with resume)
# ---------------------------------------------------------------------------

def migrate_run(run_name: str, save_dir: str = "saves") -> None:
    tb_dir = os.path.join(save_dir, "tensorboard", run_name)
    if not os.path.isdir(tb_dir):
        print(f"ERROR: TensorBoard dir not found: {tb_dir}")
        sys.exit(1)

    log_dir = os.path.join(save_dir, run_name, "logs")
    images_dir = os.path.join(log_dir, "images")
    media_path = os.path.join(log_dir, "media.jsonl")
    metrics_path = os.path.join(log_dir, "metrics.jsonl")

    os.makedirs(images_dir, exist_ok=True)

    # ---- Phase 1: Load TensorBoard events ----
    print(f"[{run_name}] Loading TensorBoard events ...")
    ea = EventAccumulator(tb_dir, size_guidance={"images": 0, "scalars": 0})
    ea.Reload()
    print(f"[{run_name}]   Done.")

    # ---- Phase 2: Collect metadata (no bytes), check completeness ----
    img_tags = [t for t in ea.Tags().get("images", []) if _is_rollout_image_tag(t)]
    print(f"[{run_name}] Found {len(img_tags)} image tags.")

    tag_events: Dict[str, List[Tuple[int, str]]] = {}
    # tag -> [(step, key), ...]  (metadata only, no bytes)
    train_steps: set = set()

    for tag in img_tags:
        if tag.startswith("train_samples/"):
            ds = "train"
        elif tag.startswith("eval/"):
            ds = "eval"
        else:
            ds = "unknown"

        tag_entries = []
        for event in ea.Images(tag):
            key = _tag_to_key(tag)
            tag_entries.append((event.step, key))
            if ds == "train":
                train_steps.add(event.step)
        tag_events[tag] = tag_entries

    # Flatten for counting
    all_entries = [(step, key) for entries in tag_events.values() for step, key in entries]
    all_entries.sort(key=lambda x: (x[0], x[1]))
    total = len(all_entries)
    print(f"[{run_name}] Total entries: {total}")

    # Reconstruct training prompts
    prompt_file = "dataset/pickscore/train.txt"
    step_prompts: Dict[int, List[str]] = {}
    if train_steps and os.path.isfile(prompt_file):
        print(f"[{run_name}] Reconstructing training prompts ...")
        step_prompts = _reconstruct_training_prompts(prompt_file, sorted(train_steps),
                                                      dataset_size=1024)

    # ---- Phase 3: Check what's already done (no bytes needed) ----
    existing_jsonl: set = set()
    if os.path.isfile(media_path):
        with open(media_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing_jsonl.add((rec["step"], rec["key"]))
                except (json.JSONDecodeError, KeyError):
                    pass

    # Build work plan per tag: which (step, key) need what
    # Each entry: (step, key, need_png, need_jsonl)
    work_per_tag: Dict[str, List[Tuple[int, str, bool, bool]]] = {}
    complete = 0
    missing_png = 0
    missing_jsonl = 0

    for tag, tag_entries in tag_events.items():
        work = []
        for step, key in tag_entries:
            pk = (step, key)
            step_dir = os.path.join(images_dir, f"step_{step:06d}")
            fname = _key_to_filename(key)
            png_ok = os.path.isfile(os.path.join(step_dir, fname))
            jsonl_ok = pk in existing_jsonl

            if png_ok and jsonl_ok:
                complete += 1
                continue

            need_png = not png_ok
            need_jsonl = not jsonl_ok
            if need_png:
                missing_png += 1
            if need_jsonl:
                missing_jsonl += 1
            work.append((step, key, need_png, need_jsonl))
        if work:
            work_per_tag[tag] = work

    if not work_per_tag:
        print(f"[{run_name}] All {total} entries complete — skipping.")
    else:
        print(f"[{run_name}] {complete}/{total} complete, "
              f"{missing_png} PNGs needed, {missing_jsonl} JSONL entries needed.")

        # ---- Phase 4: Process tag-by-tag (only one tag's bytes in memory) ----
        pbar = tqdm(total=sum(len(w) for w in work_per_tag.values()),
                    desc=f"  [{run_name}] Media", unit="img")
        with open(media_path, "a") as media_f:
            for tag, work in work_per_tag.items():
                # Fetch bytes for this tag only
                events_by_step: Dict[int, bytes] = {}
                for event in ea.Images(tag):
                    events_by_step[event.step] = event.encoded_image_string

                for step, key, need_png, need_jsonl in work:
                    step_dir = os.path.join(images_dir, f"step_{step:06d}")
                    os.makedirs(step_dir, exist_ok=True)
                    fname = _key_to_filename(key)
                    img_full_path = os.path.join(step_dir, fname)
                    rel_path = os.path.join("images", f"step_{step:06d}", fname)

                    if need_png:
                        img_bytes = events_by_step.get(step)
                        if img_bytes:
                            try:
                                img = Image.open(io.BytesIO(img_bytes))
                                img = img.convert("RGB")
                                img.save(img_full_path, "PNG")
                            except Exception as exc:
                                print(f"  [WARN] Corrupt image step={step} key={key}: {exc}")

                    if need_jsonl:
                        prompt = ""
                        if key.startswith("train_samples/"):
                            try:
                                tag_idx = int(key.rsplit("/", 1)[-1])
                            except ValueError:
                                tag_idx = 0
                            prompts = step_prompts.get(step, [])
                            if tag_idx < 16 and len(prompts) > 0:
                                prompt = prompts[0]
                            elif tag_idx >= 16 and len(prompts) > 1:
                                prompt = prompts[1]

                        entry = {
                            "step": step, "key": key, "path": rel_path,
                            "caption": "", "reward": {}, "prompt": prompt,
                        }
                        media_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                    pbar.update(1)

                # Bytes for this tag are released when we move to next tag

        pbar.close()
        print(f"[{run_name}]   media.jsonl up to date.")

    # ---- Phase 4: Write metrics.jsonl (overwrite, fast) ----
    scalar_tags = ea.Tags().get("scalars", [])
    if scalar_tags and not os.path.isfile(metrics_path):
        print(f"[{run_name}] Found {len(scalar_tags)} scalar tags. Writing metrics.jsonl ...")
        step_scalars: Dict[int, Dict[str, float]] = defaultdict(dict)
        for tag in tqdm(scalar_tags, desc=f"  [{run_name}] Scalars", unit="tag"):
            for event in ea.Scalars(tag):
                step_scalars[event.step][tag] = event.value

        with open(metrics_path, "w") as metrics_f:
            for step in sorted(step_scalars.keys()):
                rec = {"step": step, **step_scalars[step]}
                metrics_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{run_name}]   metrics.jsonl written.")
    elif os.path.isfile(metrics_path):
        print(f"[{run_name}]   metrics.jsonl already exists — skipping.")

    print(f"[{run_name}] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate TensorBoard events to JSONL logs")
    parser.add_argument("--run-name", required=True, help="Run directory name under saves/")
    parser.add_argument("--save-dir", default="saves", help="Parent save directory")
    args = parser.parse_args()
    migrate_run(args.run_name, args.save_dir)
