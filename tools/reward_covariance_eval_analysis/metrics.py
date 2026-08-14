# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure prompt-local reward geometry metrics."""

from __future__ import annotations

import numpy as np


def compute_group_metrics(
    rewards: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Compute covariance geometry for one prompt's repeated rollouts."""
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if rewards.ndim != 2 or rewards.shape[0] < 2 or rewards.shape[1] < 2:
        raise ValueError(f"rewards must be (samples >= 2, rewards >= 2), got {rewards.shape}.")
    if rewards.shape[1] != len(weights) or not np.isfinite(rewards).all():
        raise ValueError("Rewards must be finite and match the supplied weights.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be finite and strictly positive.")
    covariance = np.cov(rewards, rowvar=False, ddof=1)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = np.divide(covariance, scale, out=np.eye(len(weights)), where=scale > 0)
    concordance = weights * (covariance @ weights)
    upper = correlation[np.triu_indices(len(weights), k=1)]
    return {
        "mean": rewards.mean(0),
        "covariance": covariance,
        "correlation": correlation,
        "concordance": concordance,
        "weakest_concordance": float(concordance.min()),
        "negative_pairwise_correlation_ratio": float((upper < 0).mean()),
        "mean_negative_pairwise_correlation": (
            float(upper[upper < 0].mean()) if (upper < 0).any() else 0.0
        ),
    }


def aggregate_group_metrics(
    groups: list[dict[str, np.ndarray | float]],
) -> dict[str, np.ndarray | float]:
    """Macro-average prompt-local metrics without pooling prompts."""
    if not groups:
        raise ValueError("Cannot aggregate no prompt groups.")
    keys = (
        "mean",
        "covariance",
        "correlation",
        "concordance",
        "weakest_concordance",
        "negative_pairwise_correlation_ratio",
        "mean_negative_pairwise_correlation",
    )
    return {key: np.mean([group[key] for group in groups], axis=0) for key in keys}
