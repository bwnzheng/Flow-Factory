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

"""Shared categorical palette and marker cycle for analysis figures.

Both analysis tools draw from these slots, so their figures read as one system
even when they appear side by side.
"""

# Fixed-order categorical slots. The slot *order* is what keeps neighbouring
# series separable under colour-vision deficiency, so slots are taken in
# sequence and never cycled or reordered; the opening three validate on every
# pair, not only adjacent ones, so any of the three reads against any other. The
# slots sit below 3:1 contrast on a light surface, so every figure ships a legend
# and every plotted value also sits in the figure's own data file.
CATEGORICAL_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

RUN_MARKERS = ("o", "s", "^", "D", "v", "X", "P", "*")
