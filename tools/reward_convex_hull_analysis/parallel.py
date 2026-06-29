"""Lightweight parallel-execution helpers.

Provides two patterns used throughout the convex-hull analysis:

* ``run_tasks`` — fire-and-forget parallelism (plot generation).
* ``compute_map`` — embarrassingly-parallel map over items (step/epoch
  metrics).  Uses ``ProcessPoolExecutor`` for CPU-bound work (numpy /
  scipy) and falls back to sequential execution when there are too few
  items to amortise the spawn overhead.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Fire-and-forget task runner (threads — plot generation etc.)
# ---------------------------------------------------------------------------


def run_tasks(
    tasks: List[Tuple[str, Callable[..., Any], tuple, dict]],
    *,
    max_workers: int = 0,
) -> None:
    """Execute *tasks* in parallel using a thread pool.

    Each task is ``(label, func, args, kwargs)``.  Matplotlib Agg backend
    is thread-safe, so we can overlap figure generation and PNG I/O.
    When there is only one task it runs inline so tracebacks are easier
    to read.
    """
    if len(tasks) <= 1:
        for label, func, args, kwargs in tasks:
            print(f"  [Plot] Generating {label} ...")
            func(*args, **kwargs)
        return

    workers = max_workers or len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(func, *args, **kwargs): label for label, func, args, kwargs in tasks}
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
                print(f"  [Plot] ✓ {label}")
            except Exception as exc:
                print(f"  [Plot] ✗ {label}: {exc}")


# ---------------------------------------------------------------------------
# Embarrassingly-parallel map (processes — CPU-bound metrics)
# ---------------------------------------------------------------------------


def compute_map(
    func: Callable[..., R],
    items: Iterable[Tuple[Any, ...]],
    *,
    max_workers: int = 0,
    min_items: int = 5,
) -> Dict[int, R]:
    """Map *func* over *items* in parallel, keyed by the first element.

    Each item is ``(key, *args)`` — *key* is used to index the returned
    dict and is NOT passed to *func*.  Uses ``ProcessPoolExecutor`` for
    CPU-bound workloads (numpy / scipy); falls back to sequential
    execution when there are fewer than *min_items* items.

    Returns ``{key: func(*args)}``.  Items that raise are silently
    dropped (returns ``None`` for that key).
    """
    items_list = list(items)
    if len(items_list) < min_items:
        result: Dict[int, R] = {}
        for key, *args in items_list:
            result[key] = func(*args)
        return result

    workers = max_workers or min(os.cpu_count() or 4, len(items_list), 8)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(func, *args): key for key, *args in items_list}
        result = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result[key] = future.result()
            except Exception:
                result[key] = None  # type: ignore[assignment]
        return result
