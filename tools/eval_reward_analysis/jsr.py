# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Joint Success Rate (JSR) analysis for cached reward records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def reference_quantile(
    values: Sequence[float], q: float, weights: Sequence[float] | None = None
) -> float:
    """Return the weighted empirical generalized-inverse quantile."""
    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}.")
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError("Reference rewards must be a non-empty finite vector.")
    if q == 0:
        return float("-inf")
    if weights is None:
        weights_array = np.ones(data.size, dtype=float)
    else:
        weights_array = np.asarray(weights, dtype=float)
        if (
            weights_array.shape != data.shape
            or (weights_array <= 0).any()
            or not np.isfinite(weights_array).all()
        ):
            raise ValueError("Quantile weights must be positive and match values.")
    order = np.argsort(data, kind="stable")
    sorted_values, sorted_weights = data[order], weights_array[order]
    index = int(np.searchsorted(np.cumsum(sorted_weights), q * sorted_weights.sum(), side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def build_reference_thresholds(
    base_records: Sequence[Mapping[str, Any]], rewards: Sequence[str], q_grid: Sequence[float]
) -> Dict[str, List[float]]:
    """Build prompt-equally-weighted reference thresholds for each reward."""
    _validate_records(base_records, rewards)
    _validate_q_grid(q_grid)
    prompts = _group(base_records)
    if not prompts:
        raise ValueError("Reference records must contain at least one prompt.")
    thresholds: Dict[str, List[float]] = {}
    for reward in rewards:
        # Equal prompt weighting: each prompt contributes one normalized mass.
        values: List[float] = []
        weights: List[float] = []
        for rows in prompts.values():
            weight = 1.0 / len(rows)
            values.extend(float(row["rewards"][reward]) for row in rows)
            weights.extend([weight] * len(rows))
        thresholds[reward] = [reference_quantile(values, float(q), weights) for q in q_grid]
    return thresholds


def compute_jsr(
    records: Sequence[Mapping[str, Any]],
    rewards: Sequence[str],
    thresholds: Mapping[str, Sequence[float]],
    q_grid: Sequence[float],
) -> Dict[str, Any]:
    """Compute image-then-prompt-then-seed equally weighted JSR curves."""
    _validate_records(records, rewards)
    _validate_q_grid(q_grid)
    if any(len(thresholds[r]) != len(q_grid) for r in rewards):
        raise ValueError("Threshold vectors must match q_grid length.")
    groups = _group(records)
    per_prompt = {
        str(pid): _prompt_curve(rows, rewards, thresholds, q_grid) for pid, rows in groups.items()
    }
    curve = np.mean(np.asarray(list(per_prompt.values()), dtype=float), axis=0)
    return {"q": [float(q) for q in q_grid], "jsr": curve.tolist(), "per_prompt": per_prompt}


def _prompt_curve(
    rows: Sequence[Mapping[str, Any]],
    rewards: Sequence[str],
    thresholds: Mapping[str, Sequence[float]],
    q_grid: Sequence[float],
) -> List[float]:
    scores = np.asarray([[float(row["rewards"][r]) for r in rewards] for row in rows], dtype=float)
    threshold_matrix = np.asarray([[thresholds[r][i] for r in rewards] for i in range(len(q_grid))])
    return (scores[:, None, :] >= threshold_matrix[None, :, :]).all(axis=2).mean(axis=0).tolist()


def paired_prompt_bootstrap(
    base_records: Sequence[Mapping[str, Any]],
    model_records: Mapping[str, Sequence[Mapping[str, Any]]],
    rewards: Sequence[str],
    q_grid: Sequence[float],
    replicates: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Estimate paired prompt bootstrap percentile intervals for model curves."""
    if replicates <= 0:
        raise ValueError("replicates must be positive.")
    base_groups = _group(base_records)
    model_groups = {name: _group(rows) for name, rows in model_records.items()}
    prompt_ids = list(base_groups)
    if not prompt_ids or any(set(groups) != set(prompt_ids) for groups in model_groups.values()):
        raise ValueError("All models and the reference must contain the same prompt IDs.")
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in model_records}
    for _ in range(replicates):
        drawn = rng.choice(prompt_ids, size=len(prompt_ids), replace=True).tolist()
        base = [
            {**row, "image_id": f"{occurrence}:{row.get('image_id', row.get('sample_index', ''))}"}
            for occurrence, pid in enumerate(drawn)
            for row in base_groups[pid]
        ]
        thresholds = build_reference_thresholds(base, rewards, q_grid)
        for name, groups in model_groups.items():
            # Preserve duplicate prompt occurrences as equal-weight clusters.
            curves = [_prompt_curve(groups[pid], rewards, thresholds, q_grid) for pid in drawn]
            samples[name].append(np.mean(np.asarray(curves, dtype=float), axis=0).tolist())
    intervals = {
        name: np.percentile(np.asarray(values), [2.5, 97.5], axis=0).tolist()
        for name, values in samples.items()
    }
    return {"intervals": intervals, "replicates": replicates, "seed": seed}


def load_reward_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load cached per-image reward records from JSONL."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Reward record file does not exist: {file_path}")
    records = [
        json.loads(line)
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Reward record file must contain JSON objects: {file_path}")
    return records


def analyze_cached_results(
    reference_path: str | Path,
    model_paths: Mapping[str, str | Path],
    rewards: Sequence[str],
    q_grid: Sequence[float],
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    """Analyze a shared reference and already sampled/evaluated model results."""
    base_records = load_reward_records(reference_path)
    thresholds = build_reference_thresholds(base_records, rewards, q_grid)
    curves = {
        name: compute_jsr(load_reward_records(path), rewards, thresholds, q_grid)
        for name, path in model_paths.items()
    }
    result: Dict[str, Any] = {
        "q": [float(q) for q in q_grid],
        "thresholds": thresholds,
        "models": curves,
    }
    if bootstrap_replicates:
        result["bootstrap"] = paired_prompt_bootstrap(
            base_records,
            {name: load_reward_records(path) for name, path in model_paths.items()},
            rewards,
            q_grid,
            bootstrap_replicates,
            bootstrap_seed,
        )
    return result


def _group(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        pid = str(row.get("prompt_id", row.get("prompt_index", "")))
        if not pid:
            raise ValueError("Each record requires prompt_id or prompt_index.")
        grouped.setdefault(pid, []).append(row)
    return grouped


def _validate_records(records: Sequence[Mapping[str, Any]], rewards: Sequence[str]) -> None:
    if not rewards:
        raise ValueError("rewards must not be empty.")
    if not records:
        raise ValueError("records must not be empty.")
    seen: set[Tuple[str, str]] = set()
    for row in records:
        pid = str(row.get("prompt_id", row.get("prompt_index", "")))
        iid = str(row.get("image_id", row.get("sample_index", "")))
        key = (pid, iid)
        if key in seen:
            raise ValueError(f"Duplicate record key: {key}.")
        seen.add(key)
        values = row.get("rewards")
        if not isinstance(values, Mapping) or any(r not in values for r in rewards):
            raise ValueError("Every record must contain all declared rewards.")
        if not np.isfinite([float(values[r]) for r in rewards]).all():
            raise ValueError("Reward values must be finite.")


def _validate_q_grid(q_grid: Sequence[float]) -> None:
    """Validate a finite, non-decreasing percentile grid."""
    values = np.asarray(q_grid, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("q_grid must be a non-empty finite vector.")
    if np.any((values < 0) | (values > 1)) or np.any(np.diff(values) < 0):
        raise ValueError("q_grid must be non-decreasing and lie in [0, 1].")
