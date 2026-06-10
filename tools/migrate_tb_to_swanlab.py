#!/usr/bin/env python3
"""Migrate TensorBoard TFEvent files to SwanLab local format.

Supports parallel migration across multiple runs.

Usage:
    python tools/migrate_tb_to_swanlab.py --dry-run
    python tools/migrate_tb_to_swanlab.py --run-name sd3-5_lora_grpo_20260522_132605
    python tools/migrate_tb_to_swanlab.py --workers 4
    swanlab watch saves/swanlog
"""

import argparse
import io
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from PIL import Image as PILImage
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def check_swanboard():
    try:
        import swanboard  # noqa: F401
        return True
    except ImportError:
        return False


def find_tfevents(logdir):
    """Find all TFEvent files under *logdir*, grouped by parent directory name."""
    result = {}
    for root, _dirs, files in os.walk(logdir):
        for f in files:
            if "tfevents" in f:
                dir_name = os.path.basename(root)
                if dir_name not in result:
                    result[dir_name] = []
                result[dir_name].append(os.path.join(root, f))
    return result


def format_size(size_bytes):
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate TensorBoard TFEvent files to SwanLab format")
    parser.add_argument("--tb-dir", default="saves/tensorboard", help="TensorBoard log directory")
    parser.add_argument("--output", default="saves/swanlog", help="SwanLab output directory")
    parser.add_argument("--project", default="Flow-Factory", help="SwanLab project name")
    parser.add_argument("--run-name", default=None, help="Only migrate a specific run")
    parser.add_argument(
        "--types",
        default="scalar,image",
        help="Comma-separated data types to migrate (scalar, image, audio, text)",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: min(4, runs))")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs to migrate (smallest first)")
    parser.add_argument("--dry-run", action="store_true", help="List runs without migrating")
    parser.add_argument("--no-images", action="store_true", help="Skip image migration (scalars only)")
    return parser.parse_args()


def collect_scalars_by_step(ea, scalar_tags):
    """Collect all scalar events into a step→{tag: value} dict."""
    step_data = defaultdict(dict)
    for tag in scalar_tags:
        for e in ea.Scalars(tag):
            step_data[e.step][tag] = e.value
    return step_data


def migrate_one_run(job):
    """Migrate a single run. Runs in a worker process."""
    dir_name, path, file_size, output_dir, project, types = job

    import swanlab

    run_start = time.time()
    prefix = f"[{dir_name}]"

    try:
        # Load metadata
        ea = EventAccumulator(path)
        ea.Reload()

        all_tags = ea.Tags()
        scalar_tags = all_tags.get("scalars", [])
        image_tags = all_tags.get("images", [])

        n_scalars = len(scalar_tags) if "scalar" in types else 0
        n_images = len(image_tags) if "image" in types else 0

        print(f"{prefix} Start  ({format_size(file_size)})  scalars={len(scalar_tags)} images={len(image_tags)}")

        run = swanlab.init(
            project=project,
            name=dir_name,
            mode="local",
            log_dir=output_dir,
            config={"source_tfevent": path},
        )

        run_scalar_steps = 0
        run_image_count = 0

        # Scalars
        if "scalar" in types and scalar_tags:
            step_data = collect_scalars_by_step(ea, scalar_tags)
            n_steps = len(step_data)
            for step in sorted(step_data):
                swanlab.log(step_data[step], step=step)
            run_scalar_steps = n_steps
            print(f"{prefix} Scalars done: {n_scalars} tags, {n_steps} steps")

        # Images
        if "image" in types and image_tags:
            n_img_tags = len(image_tags)
            report_interval = max(1, n_img_tags // 5)
            t0 = time.time()
            for i, tag in enumerate(image_tags):
                events = ea.Images(tag)
                for e in events:
                    img = PILImage.open(io.BytesIO(e.encoded_image_string))
                    swanlab.log({tag: swanlab.Image(img)}, step=e.step)
                    run_image_count += 1

                if (i + 1) % report_interval == 0 or i == n_img_tags - 1:
                    pct = (i + 1) / n_img_tags * 100
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"{prefix} Images: {i + 1}/{n_img_tags} ({pct:.0f}%)  "
                          f"[{format_duration(elapsed)}, {rate:.1f} tags/s]")

        run.finish()

        run_elapsed = time.time() - run_start
        rate_str = format_size(file_size / run_elapsed) if run_elapsed > 0 else ""
        print(f"{prefix} Done  {format_duration(run_elapsed)}  {rate_str}/s"
              + (f"  scalars={n_scalars}" if n_scalars else "")
              + (f"  images={n_images}/{run_image_count}" if n_images else ""))

        return {
            "dir_name": dir_name,
            "file_size": file_size,
            "scalar_tags": n_scalars,
            "image_tags": n_images,
            "scalar_steps": run_scalar_steps,
            "image_count": run_image_count,
            "elapsed": run_elapsed,
            "error": None,
        }

    except Exception as e:
        run_elapsed = time.time() - run_start
        print(f"{prefix} FAILED  {format_duration(run_elapsed)}: {e}")
        traceback.print_exc()
        return {
            "dir_name": dir_name,
            "file_size": file_size,
            "scalar_tags": 0,
            "image_tags": 0,
            "scalar_steps": 0,
            "image_count": 0,
            "elapsed": run_elapsed,
            "error": str(e),
        }


def main():
    args = parse_args()

    if not check_swanboard():
        print("swanboard not installed. Run: pip install swanlab[dashboard]")
        sys.exit(1)

    types = set(t.strip().lower() for t in args.types.split(","))
    if args.no_images:
        types.discard("image")

    valid_types = {"scalar", "image", "audio", "text"}
    invalid = types - valid_types
    if invalid:
        print(f"Unsupported types: {invalid}. Supported: {valid_types}")
        sys.exit(1)

    tb_dir = os.path.abspath(args.tb_dir)
    if not os.path.isdir(tb_dir):
        print(f"TensorBoard directory not found: {tb_dir}")
        sys.exit(1)

    path_dict = find_tfevents(tb_dir)
    if not path_dict:
        print(f"No TFEvent files found in {tb_dir}")
        sys.exit(1)

    if args.run_name:
        if args.run_name in path_dict:
            path_dict = {args.run_name: path_dict[args.run_name]}
        else:
            print(f"Run '{args.run_name}' not found. Available: {list(path_dict.keys())}")
            sys.exit(1)

    output_dir = os.path.abspath(args.output)

    jobs = []
    for dir_name, paths in path_dict.items():
        for p in paths:
            jobs.append((dir_name, p, os.path.getsize(p), output_dir, args.project, types))

    jobs.sort(key=lambda x: x[2])

    if args.max_runs and args.max_runs < len(jobs):
        jobs = jobs[:args.max_runs]

    total_size = sum(j[2] for j in jobs)
    total_runs = len(jobs)
    print(f"Found {total_runs} run(s), total {format_size(total_size)}:")
    for dir_name, path, size, *_ in jobs:
        print(f"  {dir_name}/  ({format_size(size)})")

    if args.dry_run:
        print("\nDry run — no data migrated.")
        return

    n_workers = args.workers or min(4, total_runs)
    n_workers = max(1, min(n_workers, total_runs))
    print(f"\nMigrating with {n_workers} worker(s) ...")

    os.makedirs(output_dir, exist_ok=True)

    overall_start = time.time()
    completed = 0
    failed = 0
    total_scalar_tags = 0
    total_image_tags = 0
    total_steps = 0
    total_images = 0

    # Pre-create output dir so swanlab can see it in all workers
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(migrate_one_run, job): job for job in jobs}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["error"]:
                failed += 1
            else:
                total_scalar_tags += result["scalar_tags"]
                total_image_tags += result["image_tags"]
                total_steps += result["scalar_steps"]
                total_images += result["image_count"]

            # Overall progress
            overall_elapsed = time.time() - overall_start
            avg_per_run = overall_elapsed / completed
            remaining = (total_runs - completed) * avg_per_run
            done_size = sum(
                f.result()["file_size"]
                for f in futures
                if f.done() and not f.result()["error"]
            )
            done_pct = done_size / total_size * 100 if total_size > 0 else 0
            print(f"\n  [{completed}/{total_runs}] {result['dir_name']} "
                  f"({'FAILED' if result['error'] else format_duration(result['elapsed'])})  "
                  f"done: {format_size(done_size)}/{format_size(total_size)} ({done_pct:.0f}%)  "
                  f"elapsed: {format_duration(overall_elapsed)}  "
                  f"ETA: {format_duration(remaining)}")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"Migration complete!")
    print(f"  Runs:        {total_runs}  ({failed} failed)" if failed else f"  Runs:        {total_runs}")
    print(f"  Total size:  {format_size(total_size)}")
    print(f"  Duration:    {format_duration(overall_elapsed)}")
    print(f"  Data rate:   {format_size(total_size / overall_elapsed)}/s" if overall_elapsed > 0 else "")
    print(f"  Scalars:     {total_scalar_tags} tags, {total_steps} steps")
    print(f"  Images:      {total_image_tags} tags, {total_images} images")
    print(f"  Output:      {output_dir}")
    print(f"{'=' * 60}")
    print(f"\nView with: swanlab watch {output_dir}")


if __name__ == "__main__":
    main()
