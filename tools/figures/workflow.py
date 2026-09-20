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

"""The figure stage both analysis tools share: data files, index, and rendering.

Every figure is written twice: as an image, and as the spec that produced it.
The spec is the only input the renderer takes, so re-drawing a whole analysis
means reading the specs back -- which is what the reuse figure mode does -- and
never re-running the analysis behind them.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from tools.figures.render import render_figure
from tools.figures.spec import SPEC_VERSION, Figure, read_spec, write_spec

# Whether the figure stage rebuilds its data from the analysis, or redraws the
# figures an earlier run already described.
FIGURE_MODES = ("regenerate", "reuse")

# One spec per figure, addressed by its path stem inside the output directory.
FigureOutput = tuple[str, Figure]

# The index that lists the figures written into one directory. An analysis tool
# may instead fold this list into a document of its own, which is why the
# readers below also accept a caller-supplied list.
FIGURE_INDEX_NAME = "figures.json"


def spec_path(directory: str | Path, stem: str) -> Path:
    """Return the data file that sits beside one figure's image."""
    return Path(directory) / f"{stem}.json"


def write_figure_data(outputs: Sequence[FigureOutput], output_dir: str | Path) -> list[str]:
    """Write one spec beside each image, returning the stems that were written."""
    stems = [stem for stem, _ in outputs]
    if len(stems) != len(set(stems)):
        duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
        raise ValueError(f"Figure builders produced duplicate output stems: {duplicates}")

    written: list[str] = []
    for stem, spec in outputs:
        write_spec(spec, Path(output_dir) / Path(stem).parent, Path(stem).name)
        written.append(stem)
    return written


def write_figure_index(
    output_dir: str | Path,
    stems: Sequence[str],
    figure_mode: str,
    plot_format: str,
) -> Path:
    """Record which figures this directory holds, so reuse never has to guess."""
    if figure_mode not in FIGURE_MODES:
        raise ValueError(f"figure_mode must be one of {list(FIGURE_MODES)}.")
    document = {
        "format_version": SPEC_VERSION,
        "figure_mode": figure_mode,
        "plot_format": plot_format,
        "figures": list(stems),
    }
    path = Path(output_dir) / FIGURE_INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_figure_index(directory: str | Path) -> dict[str, Any]:
    """Read one directory's figure index."""
    path = Path(directory) / FIGURE_INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No {FIGURE_INDEX_NAME} in {directory}. Re-run with the regenerate figure mode to "
            "write the figure data first."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return document


def figure_stems(stems: Any, directory: str | Path, source: str) -> list[str]:
    """Validate an indexed figure list and require every spec it names to exist.

    Reading the index rather than globbing means a figure whose data was deleted
    is reported instead of silently disappearing, and a leftover file from an
    older configuration is never redrawn.
    """
    if not stems:
        raise ValueError(
            f"{source} lists no figures, so there is nothing to redraw. Re-run with the "
            "regenerate figure mode."
        )
    if not isinstance(stems, list) or not all(isinstance(stem, str) and stem for stem in stems):
        raise ValueError(
            f"{source} must list figure path stems as non-empty strings. Re-run with the "
            "regenerate figure mode."
        )
    if len(stems) != len(set(stems)):
        raise ValueError(
            f"{source} lists duplicate figures. Re-run with the regenerate figure mode."
        )
    missing = [stem for stem in stems if not spec_path(directory, stem).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Figure data missing for {missing} in {directory}. Re-run with the regenerate "
            "figure mode."
        )
    return [str(stem) for stem in stems]


def read_indexed_figures(directory: str | Path) -> list[FigureOutput]:
    """Load every figure spec one directory's index names."""
    index_path = Path(directory) / FIGURE_INDEX_NAME
    document = read_figure_index(directory)
    return [
        (stem, read_spec(spec_path(directory, stem)))
        for stem in figure_stems(document.get("figures"), directory, str(index_path))
    ]


def plot_worker_count(figure_count: int) -> int:
    """Pick how many worker processes the figure stage should use."""
    return max(1, min(figure_count, os.cpu_count() or 1))


def render_figures(
    outputs: Sequence[FigureOutput], output_dir: str | Path, plot_format: str
) -> None:
    """Render every figure, spreading them over worker processes.

    Matplotlib is imported by this module and keeps global state, so the pool is
    spawned rather than forked: forking a process that already loaded extension
    modules and started threads risks deadlocking the children. Worker startup
    costs a fresh interpreter import, which the parallel figures amortize.
    """
    tasks = [(spec, str(output_dir), stem, plot_format) for stem, spec in outputs]
    if not tasks:
        return
    workers = plot_worker_count(len(tasks))
    if workers == 1:
        for task in tasks:
            _render_figure_task(task)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        list(executor.map(_render_figure_task, tasks))


def _render_figure_task(task: tuple[Any, ...]) -> None:
    """Draw one figure inside a worker process."""
    spec, output_dir, stem, plot_format = task
    render_figure(spec, output_dir, stem, plot_format)
