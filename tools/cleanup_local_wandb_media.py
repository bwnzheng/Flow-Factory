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

"""Remove local W&B media payloads and media references from summaries.

The tool is dry-run by default. With ``--execute`` it removes children of
``files/media`` using a NUL-safe ``find | xargs -P`` pipeline. Work is split at
``--split-depth`` so many independent subdirectories can be distributed across
workers; after the children are removed, the media directory itself is also removed.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

MEDIA_TYPES = {
    "audio-file",
    "html-file",
    "image-file",
    "video-file",
    "table-file",
}


def _find_paths(root: Path, expression: Sequence[str]) -> List[Path]:
    """Run GNU find once and decode its NUL-delimited output."""
    process = subprocess.Popen(["find", str(root), *expression], stdout=subprocess.PIPE)
    assert process.stdout is not None
    paths = [Path(os.fsdecode(raw)) for raw in process.stdout.read().split(b"\0") if raw]
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"find failed for {root} with exit code {return_code}")
    return paths


def _discover(root: Path) -> Tuple[List[Path], List[Path]]:
    """Find media directories and associated summaries below a run or W&B root."""
    root = root.resolve()
    media_dirs: List[Path] = []
    summaries: List[Path] = []

    if root.name == "media" and root.parent.name == "files":
        media_dirs.append(root)
        summary = root.parent / "wandb-summary.json"
        if summary.is_file():
            summaries.append(summary)
        return media_dirs, summaries

    candidates: Iterable[Path]
    if root.name == "files":
        candidates = root.glob("wandb-summary.json")
    else:
        entries = _find_paths(
            root,
            [
                "(",
                "-type",
                "d",
                "-path",
                "*/files/media",
                "-print0",
                "-prune",
                ")",
                "-o",
                "(",
                "-type",
                "f",
                "-path",
                "*/files/wandb-summary.json",
                "-print0",
                ")",
            ],
        )
        for entry in entries:
            if entry.is_dir():
                media_dirs.append(entry)
            else:
                summaries.append(entry)
        return sorted(set(media_dirs)), sorted(set(summaries))

    for summary in candidates:
        if summary.parent.name != "files":
            continue
        summaries.append(summary)
        media = summary.parent / "media"
        if media.is_dir():
            media_dirs.append(media)

    return sorted(set(media_dirs)), sorted(set(summaries))


def _find_xargs_delete(path: Path, depth: int, workers: int, execute: bool) -> None:
    """Delete a depth frontier and shallow files with find/xargs."""
    if depth < 1:
        raise ValueError(f"--split-depth must be positive, got {depth}")
    if workers < 1:
        raise ValueError(f"--workers must be positive, got {workers}")

    frontier = [
        "find",
        str(path),
        "-mindepth",
        str(depth),
        "-maxdepth",
        str(depth),
        "-print0",
    ]
    shallow = [
        "find",
        str(path),
        "-mindepth",
        "1",
        "-maxdepth",
        str(max(1, depth - 1)),
        "!",
        "-type",
        "d",
        "-print0",
    ]

    if not execute:
        print("DRY-RUN:", " ".join(frontier), "| xargs -0 -n 1 -P", workers, "rm -rf --")
        if depth > 1:
            print("DRY-RUN:", " ".join(shallow), "| xargs -0 -n 1 -P", workers, "rm -f --")
        print("DRY-RUN: rm -rf --", path)
        return

    for find_command, remove_flags in (
        (frontier, ["rm", "-rf", "--"]),
        (shallow, ["rm", "-f", "--"]),
    ):
        if find_command is shallow and depth == 1:
            continue
        finder = subprocess.Popen(find_command, stdout=subprocess.PIPE)
        assert finder.stdout is not None
        remover = subprocess.Popen(
            ["xargs", "-0", "-r", "-n", "1", "-P", str(workers), *remove_flags],
            stdin=finder.stdout,
        )
        finder.stdout.close()
        remove_code = remover.wait()
        find_code = finder.wait()
        if find_code != 0 or remove_code != 0:
            raise RuntimeError(
                f"find/xargs deletion failed for {path}: find={find_code}, xargs={remove_code}"
            )

    subprocess.run(["rm", "-rf", "--", str(path)], check=True)


def _count_files(path: Path) -> int:
    """Count regular files without retaining or printing their names."""
    process = subprocess.Popen(["find", str(path), "-type", "f", "-print0"], stdout=subprocess.PIPE)
    assert process.stdout is not None
    count = 0
    while chunk := process.stdout.read(1024 * 1024):
        count += chunk.count(b"\0")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"find failed while counting {path}: exit code {return_code}")
    return count


def _contains_media_reference(value: Any) -> bool:
    """Return whether a JSON value contains a W&B media-file reference."""
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return normalized.startswith("media/") or "/media/" in normalized
    if isinstance(value, dict):
        if value.get("_type") in MEDIA_TYPES:
            return True
        return any(_contains_media_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_media_reference(item) for item in value)
    return False


def _json_compliant(value: Any) -> Any:
    """Convert non-finite floating-point values to JSON null recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_compliant(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compliant(item) for item in value]
    return value


def _clean_summary(path: Path, backup_dir: Path, execute: bool) -> List[str]:
    """Remove top-level summary entries that reference W&B media."""
    try:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read summary {path}: {error}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"Expected summary object in {path}, got {type(summary).__name__}")

    removed = [
        key
        for key, value in summary.items()
        if key == "media" or key.startswith("media/") or _contains_media_reference(value)
    ]
    if not removed or not execute:
        return sorted(removed)

    backup_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    backup = backup_dir / f"{path.name}.{digest}.bak"
    backup.write_bytes(path.read_bytes())
    for key in removed:
        del summary[key]
    summary = _json_compliant(summary)

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"summary cleaned: {path} removed={len(removed)} backup={backup}")
    return sorted(removed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="A W&B root, run directory, files directory, or media directory."
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--split-depth", type=int, default=4)
    parser.add_argument("--backup-dir", type=Path, default=Path(".scratch/wandb_summary_backups"))
    parser.add_argument(
        "--execute", action="store_true", help="Actually delete media and rewrite summaries."
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Discover and clean local W&B media files and summary references."""
    args = args or parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    media_dirs, summaries = _discover(args.root)
    if not media_dirs and not summaries:
        print(f"No W&B files found below {args.root}")
        return 0

    summary_results = []
    for summary in summaries:
        removed = _clean_summary(summary, args.backup_dir, args.execute)
        if removed:
            summary_results.append((summary, len(removed)))

    media_run_dirs = sorted({media.parent.parent for media in media_dirs})
    summary_run_dirs = sorted({summary.parent.parent for summary, _ in summary_results})
    print(
        f"runs_with_media={len(media_run_dirs)} "
        f"runs_with_summary_media={len(summary_run_dirs)} "
        f"execute={args.execute}"
    )
    for run_dir in media_run_dirs:
        print(f"media run: {run_dir}")
    for run_dir in summary_run_dirs:
        print(f"summary-media run: {run_dir}")

    total_files = sum(_count_files(media_dir) for media_dir in media_dirs)
    if media_dirs:
        base, extra = divmod(total_files, args.workers)
        per_worker = [base + int(index < extra) for index in range(args.workers)]
        print(f"estimated_media_files={total_files} estimated_files_per_worker={per_worker}")

    for media_dir in media_dirs:
        _find_xargs_delete(media_dir, args.split_depth, args.workers, args.execute)
    if not args.execute:
        for summary, removed_count in summary_results:
            print(f"DRY-RUN: summary={summary} " f"would remove {removed_count} media-related keys")
    if not args.execute:
        print("Dry run only. Add --execute to apply changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
