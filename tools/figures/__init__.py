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

"""Figure specs, the renderer that draws them, and the reuse workflow around them.

Shared by the analysis tools under ``tools/``: a tool builds specs from its own
metrics, and this package writes, indexes, reads, and draws them.
"""

from .render import render_figure
from .spec import (
    BAR_SERIES_KIND,
    GRID_AXES,
    LINE_SERIES_KIND,
    MATRIX_FIGURE_KIND,
    SPEC_VERSION,
    SUPPORTED_SPEC_VERSIONS,
    Figure,
    FigureAxis,
    FigureFontSizes,
    FigureHLine,
    FigureLegend,
    FigureMatrix,
    FigureSeries,
    FigureSpec,
    LegendEntry,
    MatrixFigureSpec,
    read_spec,
    to_json,
    validate_spec,
    write_spec,
)
from .style import CATEGORICAL_COLORS, RUN_MARKERS
from .workflow import (
    FIGURE_INDEX_NAME,
    FIGURE_MODES,
    FigureOutput,
    figure_stems,
    plot_worker_count,
    read_figure_index,
    read_indexed_figures,
    render_figures,
    spec_path,
    write_figure_data,
    write_figure_index,
)

__all__ = [
    "BAR_SERIES_KIND",
    "CATEGORICAL_COLORS",
    "FIGURE_INDEX_NAME",
    "FIGURE_MODES",
    "GRID_AXES",
    "LINE_SERIES_KIND",
    "MATRIX_FIGURE_KIND",
    "RUN_MARKERS",
    "SPEC_VERSION",
    "SUPPORTED_SPEC_VERSIONS",
    "Figure",
    "FigureAxis",
    "FigureFontSizes",
    "FigureHLine",
    "FigureLegend",
    "FigureMatrix",
    "FigureOutput",
    "FigureSeries",
    "FigureSpec",
    "LegendEntry",
    "MatrixFigureSpec",
    "figure_stems",
    "plot_worker_count",
    "read_figure_index",
    "read_indexed_figures",
    "read_spec",
    "render_figure",
    "render_figures",
    "spec_path",
    "to_json",
    "validate_spec",
    "write_figure_data",
    "write_figure_index",
    "write_spec",
]
