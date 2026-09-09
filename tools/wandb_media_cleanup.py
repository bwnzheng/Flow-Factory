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

"""Cache and delete W&B run media files with resumable retries.

Examples:
    python tools/wandb_media_cleanup.py scan \
        --entity ENTITY --project PROJECT --run-id RUN_ID
    python tools/wandb_media_cleanup.py status
    python tools/wandb_media_cleanup.py delete --execute --workers 8

The cache is append-only. A successful deletion is recorded immediately, so a
later invocation skips completed files. Files that still fail after their
retry budget are recorded as ``skipped`` and are not retried by default.

Authentication is delegated to the W&B SDK; set ``WANDB_API_KEY`` in the
environment before running this tool. The key is never written to the cache.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

SCHEMA_VERSION = 1
DEFAULT_CACHE = ".scratch/wandb_media_cleanup.jsonl"
PERMANENT_HTTP_CODES = {400, 401, 403}


@dataclass
class CacheState:
    """Represent the reconstructed state of an append-only cleanup cache."""

    entity: Optional[str] = None
    project: Optional[str] = None
    pattern: Optional[str] = None
    files: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    deleted: Set[Tuple[str, str]] = field(default_factory=set)
    skipped: Set[Tuple[str, str]] = field(default_factory=set)
    scan_complete: Set[str] = field(default_factory=set)


class EventCache:
    """Append cleanup events and reconstruct completed work."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> CacheState:
        """Load cache state, tolerating only a truncated final JSONL line."""
        state = CacheState()
        if not self.path.exists():
            return state

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # A crash can leave only the final append partially written.
                    # Confirm there is no later non-whitespace content before
                    # treating this line as truncation.
                    if not handle.read().strip():
                        tqdm.write(f"Ignoring truncated final cache line in {self.path}")
                        break
                    raise

                record_type = event.get("record_type")
                if record_type == "cache_header":
                    state.entity = event["entity"]
                    state.project = event["project"]
                    state.pattern = event["pattern"]
                elif record_type == "file":
                    key = (event["run_id"], event["name"])
                    state.files[key] = event
                elif record_type == "deleted":
                    state.deleted.add((event["run_id"], event["name"]))
                elif record_type == "skipped":
                    state.skipped.add((event["run_id"], event["name"]))
                elif record_type == "scan_complete":
                    state.scan_complete.add(event["run_id"])
        return state

    def initialize(self, entity: str, project: str, pattern: str) -> CacheState:
        """Create or validate the cache header."""
        state = self.load()
        if state.entity is None:
            self.append(
                {
                    "record_type": "cache_header",
                    "schema_version": SCHEMA_VERSION,
                    "entity": entity,
                    "project": project,
                    "pattern": pattern,
                }
            )
            state.entity = entity
            state.project = project
            state.pattern = pattern
            return state

        actual = (state.entity, state.project, state.pattern)
        expected = (entity, project, pattern)
        if actual != expected:
            raise ValueError(
                "Cache scope mismatch: "
                f"cache={actual!r}, requested={expected!r}. Use a different --cache path."
            )
        return state

    def append(self, event: Dict[str, Any]) -> None:
        """Append one event and flush it before returning."""
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()


def _load_wandb() -> Any:
    """Import the optional W&B dependency with an actionable error."""
    if wandb is None:
        raise RuntimeError('W&B is not installed. Install it with `pip install -e ".[wandb]"`.')
    return wandb


def _http_status(error: BaseException) -> Optional[int]:
    """Extract an HTTP status code from common W&B and requests exceptions."""
    current: Optional[BaseException] = error
    visited: Set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(current, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        current = current.__cause__ or current.__context__
    return None


def _is_permanent_error(error: BaseException) -> bool:
    """Return whether retrying the request cannot fix the reported error."""
    return _http_status(error) in PERMANENT_HTTP_CODES


def _backoff_seconds(attempt: int, base_delay: float, max_delay: float) -> float:
    """Compute bounded exponential backoff with jitter."""
    exponent = min(max(0, attempt - 1), 10)
    delay = min(max_delay, base_delay * (2**exponent))
    return delay * random.uniform(0.8, 1.2)


def _resolve_entity(
    entity: Optional[str],
    timeout: int,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> str:
    """Resolve the authenticated account's default entity when omitted."""
    if entity:
        return entity

    retries = 0
    while True:
        try:
            wandb = _load_wandb()
            resolved = wandb.Api(timeout=timeout).default_entity
            if not resolved:
                raise ValueError(
                    "W&B did not return a default entity for the authenticated account. "
                    "Pass --entity explicitly."
                )
            tqdm.write(f"Using W&B default entity: {resolved}")
            return resolved
        except ValueError:
            raise
        except Exception as error:
            if _is_permanent_error(error):
                raise
            if retries >= max_retries:
                raise
            retries += 1
            delay = _backoff_seconds(retries, base_delay, max_delay)
            tqdm.write(f"[entity] retry {retries} in {delay:.1f}s: {error}")
            time.sleep(delay)


def retry_call(
    operation: Callable[[], None],
    max_retries: int,
    base_delay: float,
    max_delay: float,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Tuple[str, int]:
    """Run a deletion operation with a bounded retry budget.

    Returns:
        A pair of final status and retry count. Status is ``deleted``,
        ``already_missing``, or ``skipped``.
    """
    retries = 0
    while True:
        try:
            operation()
            return "deleted", retries
        except Exception as error:
            status = _http_status(error)
            if status == 404 or "not found" in str(error).lower():
                return "already_missing", retries
            if _is_permanent_error(error):
                raise
            if retries >= max_retries:
                return "skipped", retries
            retries += 1
            delay = _backoff_seconds(retries, base_delay, max_delay)
            if on_retry is not None:
                on_retry(retries, error, delay)
            sleep(delay)


def _scan_run(
    cache: EventCache,
    state: CacheState,
    entity: str,
    project: str,
    run_id: str,
    pattern: str,
    per_page: int,
    timeout: int,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    position: int,
) -> Tuple[str, int, int]:
    """Scan one run and durably cache every unique matching filename."""
    known = {name for cached_run_id, name in state.files if cached_run_id == run_id}
    retry_count = 0

    with tqdm(
        desc=f"{run_id} scan",
        unit="file",
        initial=len(known),
        position=position,
        leave=True,
        dynamic_ncols=True,
    ) as progress:
        while True:
            try:
                wandb = _load_wandb()
                api = wandb.Api(timeout=timeout)
                run = api.run(f"{entity}/{project}/{run_id}")
                for remote_file in run.files(pattern=pattern, per_page=per_page):
                    if remote_file.name in known:
                        continue
                    event = {
                        "record_type": "file",
                        "run_id": run_id,
                        "run_name": run.name,
                        "name": remote_file.name,
                        "size": int(remote_file.size or 0),
                    }
                    cache.append(event)
                    state.files[(run_id, remote_file.name)] = event
                    known.add(remote_file.name)
                    progress.update(1)

                cache.append({"record_type": "scan_complete", "run_id": run_id})
                state.scan_complete.add(run_id)
                return run.name, len(known), retry_count
            except Exception as error:
                if _is_permanent_error(error):
                    raise
                if retry_count >= max_retries:
                    raise
                retry_count += 1
                delay = _backoff_seconds(retry_count, base_delay, max_delay)
                tqdm.write(f"[{run_id}] scan retry {retry_count} in {delay:.1f}s: {error}")
                time.sleep(delay)


def scan_command(args: argparse.Namespace) -> int:
    """Scan selected runs into the append-only cache."""
    entity = _resolve_entity(
        args.entity,
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
    )
    cache = EventCache(args.cache)
    state = cache.initialize(entity, args.project, args.pattern)
    run_ids = list(dict.fromkeys(args.run_id))
    if not args.rescan:
        run_ids = [run_id for run_id in run_ids if run_id not in state.scan_complete]
    if not run_ids:
        print("All selected runs already have scan_complete records. Use --rescan to scan again.")
        return 0

    with ThreadPoolExecutor(max_workers=min(args.workers, len(run_ids))) as executor:
        futures = {
            executor.submit(
                _scan_run,
                cache,
                state,
                entity,
                args.project,
                run_id,
                args.pattern,
                args.per_page,
                args.timeout,
                args.max_retries,
                args.base_delay,
                args.max_delay,
                position,
            ): run_id
            for position, run_id in enumerate(run_ids)
        }
        for future in as_completed(futures):
            run_id = futures[future]
            run_name, count, retries = future.result()
            tqdm.write(
                f"[scan complete] run={run_id} name={run_name} files={count} retries={retries}"
            )
    return 0


class _ThreadClients:
    """Hold one W&B API and run cache per deletion worker thread."""

    def __init__(self, entity: str, project: str, timeout: int):
        self.entity = entity
        self.project = project
        self.timeout = timeout
        self.local = threading.local()

    def run(self, run_id: str) -> Any:
        """Return a thread-local W&B run handle."""
        if not hasattr(self.local, "api"):
            wandb = _load_wandb()
            self.local.api = wandb.Api(timeout=self.timeout)
            self.local.runs = {}
        if run_id not in self.local.runs:
            self.local.runs[run_id] = self.local.api.run(f"{self.entity}/{self.project}/{run_id}")
        return self.local.runs[run_id]


def _delete_cached_file(
    clients: _ThreadClients,
    run_id: str,
    name: str,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> Tuple[str, int]:
    """Delete one cached filename with retry accounting."""

    def operation() -> None:
        run = clients.run(run_id)
        remote_file = run.file(name)
        try:
            remote_file.delete()
        except Exception as delete_error:
            # The mutation may have succeeded even if its response was lost.
            # Re-check before retrying, so an already-removed file is not
            # submitted repeatedly.
            try:
                run.file(name)
            except Exception as verify_error:
                if _http_status(verify_error) == 404 or "not found" in str(verify_error).lower():
                    raise ValueError(f"file no longer exists: {name}") from verify_error
            raise delete_error

    def report_retry(attempt: int, error: BaseException, delay: float) -> None:
        if attempt == 1 or attempt % 10 == 0:
            tqdm.write(f"[{run_id}] retry={attempt} delay={delay:.1f}s file={name}: {error}")

    return retry_call(
        operation,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        on_retry=report_retry,
    )


def _pending_files(
    state: CacheState,
    run_ids: Optional[Iterable[str]],
    retry_skipped: bool = False,
) -> List[Tuple[str, str]]:
    """Return stable pending cache keys, optionally filtered by run ID."""
    selected = set(run_ids) if run_ids else None
    return sorted(
        key
        for key in state.files
        if key not in state.deleted
        and (retry_skipped or key not in state.skipped)
        and (selected is None or key[0] in selected)
    )


def delete_command(args: argparse.Namespace) -> int:
    """Delete cached media filenames with durable success records."""
    cache = EventCache(args.cache)
    state = cache.load()
    if state.entity is None or state.project is None:
        raise ValueError(f"Cache has no valid header: {args.cache}")

    pending = _pending_files(state, args.run_id, retry_skipped=args.retry_skipped)
    print(
        f"cache={args.cache} pending={len(pending)} deleted={len(state.deleted)} "
        f"skipped={len(state.skipped)} "
        f"workers={args.workers} max_retries={args.max_retries}"
    )
    if not args.execute:
        print("Dry run only. Add --execute to delete the cached remote files.")
        return 0
    if not pending:
        return 0

    clients = _ThreadClients(state.entity, state.project, args.timeout)
    failed: List[Tuple[str, str, str]] = []
    skipped_count = 0
    total_retries = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _delete_cached_file,
                clients,
                run_id,
                name,
                args.max_retries,
                args.base_delay,
                args.max_delay,
            ): (run_id, name)
            for run_id, name in pending
        }
        with tqdm(total=len(futures), desc="delete", unit="file", dynamic_ncols=True) as progress:
            for future in as_completed(futures):
                run_id, name = futures[future]
                try:
                    status, retries = future.result()
                    total_retries += retries
                    record_type = "skipped" if status == "skipped" else "deleted"
                    cache.append(
                        {
                            "record_type": record_type,
                            "run_id": run_id,
                            "name": name,
                            "status": status,
                            "retries": retries,
                        }
                    )
                    if status == "skipped":
                        skipped_count += 1
                except Exception as error:
                    failed.append((run_id, name, str(error)))
                    cache.append(
                        {
                            "record_type": "skipped",
                            "run_id": run_id,
                            "name": name,
                            "status": "skipped",
                            "error": str(error),
                        }
                    )
                    tqdm.write(f"[permanent failure] run={run_id} file={name}: {error}")
                progress.set_postfix(retries=total_retries, failed=len(failed))
                progress.update(1)

    print(
        f"Deletion finished: completed={len(pending) - len(failed) - skipped_count} "
        f"skipped={skipped_count} failed={len(failed)} retries={total_retries}"
    )
    return 1 if failed else 0


def status_command(args: argparse.Namespace) -> int:
    """Print local cache counts without accessing W&B."""
    state = EventCache(args.cache).load()
    per_run: Dict[str, Dict[str, int]] = {}
    for run_id, name in state.files:
        counts = per_run.setdefault(run_id, {"cached": 0, "deleted": 0, "skipped": 0})
        counts["cached"] += 1
        if (run_id, name) in state.deleted:
            counts["deleted"] += 1
        if (run_id, name) in state.skipped:
            counts["skipped"] += 1

    print(
        f"cache={args.cache} entity={state.entity} project={state.project} "
        f"pattern={state.pattern}"
    )
    for run_id, counts in sorted(per_run.items()):
        complete = "yes" if run_id in state.scan_complete else "no"
        pending = counts["cached"] - counts["deleted"] - counts["skipped"]
        print(
            f"run={run_id} cached={counts['cached']} deleted={counts['deleted']} "
            f"skipped={counts['skipped']} pending={pending} scan_complete={complete}"
        )
    return 0


def _add_retry_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared network retry options to a subcommand."""
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Maximum transient retries per file before recording it as skipped (0 = no retry).",
    )
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument("--max-delay", type=float, default=120.0)
    parser.add_argument("--timeout", type=int, default=60, help="W&B API timeout in seconds.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE))
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="List W&B media and append names to cache.")
    scan.add_argument(
        "--entity",
        help="W&B entity; defaults to the authenticated account's default entity.",
    )
    scan.add_argument("--project", required=True)
    scan.add_argument("--run-id", action="append", required=True)
    scan.add_argument("--pattern", default="media/%")
    scan.add_argument("--per-page", type=int, default=1000)
    scan.add_argument("--workers", type=int, default=4)
    scan.add_argument("--rescan", action="store_true")
    _add_retry_arguments(scan)
    scan.set_defaults(func=scan_command)

    delete = subparsers.add_parser("delete", help="Delete filenames already stored in cache.")
    delete.add_argument("--run-id", action="append")
    delete.add_argument("--workers", type=int, default=8)
    delete.add_argument("--execute", action="store_true")
    delete.add_argument(
        "--retry-skipped",
        action="store_true",
        help="Retry files previously recorded as skipped.",
    )
    _add_retry_arguments(delete)
    delete.set_defaults(func=delete_command)

    status = subparsers.add_parser("status", help="Show local cache progress without network.")
    status.set_defaults(func=status_command)
    return parser.parse_args()


def main() -> int:
    """Run the selected cleanup command."""
    args = parse_args()
    if hasattr(args, "workers") and args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}.")
    if hasattr(args, "max_retries") and args.max_retries < 0:
        raise ValueError(f"--max-retries must be non-negative, got {args.max_retries}.")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
