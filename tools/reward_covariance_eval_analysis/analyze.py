#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Analyze prompt-local covariance geometry from saved repeated rollout groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tools.reward_covariance_eval_analysis.metrics import (
    aggregate_group_metrics,
    compute_group_metrics,
)
from tools.reward_disagreement_analysis.analyze import RunSpec, _weights_for_group
from tools.reward_disagreement_analysis.reward_logs import (
    load_saved_reward_weight_context,
    load_train_reward_groups,
)


def main() -> None:
    """Write covariance metrics and heatmaps for configured saved runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    raw = yaml.safe_load(Path(parser.parse_args().config).read_text()) or {}
    output = Path(
        raw.get("output", {}).get("dir", "analysis_output/reward_covariance_eval_analysis")
    )
    output.mkdir(parents=True, exist_ok=True)
    rows, metadata = [], {
        "source": "saved_train_reward_pickles",
        "warning": "These are training rollout groups, not fresh evaluation rollouts.",
    }
    final_heatmaps = []
    for item in raw["runs"]:
        run = RunSpec(item["name"], item.get("label", item["name"]), item.get("reward_weights", {}))
        groups_by_step = load_train_reward_groups(
            Path(raw.get("save_dir", "saves")) / run.name / "logs/rewards"
        )
        context = load_saved_reward_weight_context(Path(raw.get("save_dir", "saves")) / run.name)
        for step, groups in groups_by_step.items():
            combos = {}
            for group in groups:
                combos.setdefault(group.reward_names, []).append(group)
            for names, selected in combos.items():
                weights, source = _weights_for_group(run, names, context)
                metric = aggregate_group_metrics(
                    [compute_group_metrics(g.rewards, weights) for g in selected]
                )
                base = {
                    "run_name": run.name,
                    "run_label": run.label,
                    "step": step,
                    "reward_combination": "__".join(names),
                    "n_groups": len(selected),
                    "weight_source": source,
                }
                for name, value in zip(names, metric["mean"]):
                    rows.append(
                        {**base, "metric": "reward_mean", "reward": name, "value": float(value)}
                    )
                for name, value in zip(names, metric["concordance"]):
                    rows.append(
                        {
                            **base,
                            "metric": "covariance_concordance",
                            "reward": name,
                            "value": float(value),
                        }
                    )
                for key in (
                    "weakest_concordance",
                    "negative_pairwise_correlation_ratio",
                    "mean_negative_pairwise_correlation",
                ):
                    rows.append({**base, "metric": key, "reward": "", "value": float(metric[key])})
                if step == max(groups_by_step):
                    final_heatmaps.append((run, step, names, metric))
    for run, step, names, metric in final_heatmaps:
        stem = f"{run.name}_step_{step:06d}_{'__'.join(names)}"
        for key in ("covariance", "correlation"):
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(metric[key], cmap="coolwarm", aspect="equal")
            ax.set(
                xticks=range(len(names)),
                yticks=range(len(names)),
                xticklabels=names,
                yticklabels=names,
                title=f"{run.label}: {key}",
            )
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(output / f"{stem}_{key}.png", dpi=180)
            plt.close(fig)
    with (output / "metrics.csv").open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader()
        csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
