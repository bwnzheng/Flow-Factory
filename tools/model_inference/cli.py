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

"""Command-line entry point for standalone evaluation-set inference."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from .config import load_inference_config
from .runner import (
    EvaluationRunner,
    ParallelEvaluationRunner,
    load_evaluation_prompts,
    resolve_checkpoints,
    run_evaluation_set,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone model-inference argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run YAML-configured checkpoint inference.")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "default.yaml"),
        help="Path to the model inference YAML configuration.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run standalone checkpoint inference.

    Args:
        argv: Optional argument list for programmatic invocation and tests.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    print(f"Loading config: {args.config}")
    config = load_inference_config(args.config)

    checkpoints = resolve_checkpoints(
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_path=config.checkpoint_path,
        checkpoint_step=config.checkpoint_step,
    )
    prompts = load_evaluation_prompts(
        config.evaluation_set,
        prompt_key=config.prompt_key,
        max_prompts=config.max_prompts,
    )
    runner_kwargs = {} if config.model_type == "sd3-5" else {"model_type": config.model_type}
    if config.num_processes > 1:
        runner = ParallelEvaluationRunner(
            config.base_model,
            config.dtype,
            num_processes=config.num_processes,
            device=config.device,
            **runner_kwargs,
        )
    else:
        runner = EvaluationRunner(
            config.base_model,
            config.dtype,
            device=config.device,
            **runner_kwargs,
        )

    try:
        run_evaluation_set(
            runner=runner,
            checkpoints=checkpoints,
            prompts=prompts,
            output_dir=config.output_dir,
            num_samples=config.num_samples,
            generation_kwargs=config.generation_kwargs,
            batch_size=config.batch_size,
            base_seed=config.seed,
        )
    finally:
        runner.close()
    print(f"Inference completed: {config.output_dir}")
    return 0
