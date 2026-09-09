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
workers while the media directory itself is preserved.
"""

import argparse
import hashlib
import json
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
    process = subprocess.Popen(["find", str(root), *expression, "-print0"], stdout=subprocess.PIPE)
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
        # Searching only for the media directory avoids a second full Python
        # tree walk. The summary lives next to it in W&B offline runs.
        media_dirs = _find_paths(root, ["-type", "d", "-path", "*/files/media", "-prune"])
        for media in media_dirs:
            summary = media.parent / "wandb-summary.json"
            if summary.is_file():
                summaries.append(summary)
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

    subprocess.run(
        ["find", str(path), "-mindepth", "1", "-depth", "-type", "d", "-empty", "-delete"],
        check=True,
    )


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
    media_dirs, summaries = _discover(args.root)
    run_dirs = sorted({media.parent.parent for media in media_dirs})
    print(f"runs_with_media={len(run_dirs)} summaries={len(summaries)} execute={args.execute}")
    for run_dir in run_dirs:
        print(f"run: {run_dir}")
    if not media_dirs and not summaries:
        print(f"No W&B files found below {args.root}")
        return 0

    for media_dir in media_dirs:
        _find_xargs_delete(media_dir, args.split_depth, args.workers, args.execute)
    for summary in summaries:
        removed = _clean_summary(summary, args.backup_dir, args.execute)
        if removed and not args.execute:
            print(f"DRY-RUN: summary={summary} would remove {len(removed)} media-related keys")
    if not args.execute:
        print("Dry run only. Add --execute to apply changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
