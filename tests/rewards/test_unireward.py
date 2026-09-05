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

import pytest

from flow_factory.rewards.unireward import _parse_scores


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """Alignment Score (1-5): 2.694700002670288
Coherence Score (1-5): 2.371500015258789
Style Score (1-5): 2.802500009536743""",
            (2.694700002670288, 2.371500015258789, 2.802500009536743),
        ),
        (
            """Alignment: 2.7/5
Coherence: 2.4/5
Style: 2.8/5""",
            (2.7, 2.4, 2.8),
        ),
        (
            """- **Alignment**: 4.0 / 5
- **Coherence**: 3.5 / 5
- **Style**: 4 / 5""",
            (4.0, 3.5, 4.0),
        ),
    ],
)
def test_parse_scores_supports_unireward_output_formats(text, expected):
    alignment, coherence, style, overall = _parse_scores(text)

    assert (alignment, coherence, style) == expected
    assert overall == pytest.approx(sum(expected) / 3.0)


def test_parse_scores_reports_missing_dimensions_with_raw_output():
    text = "Alignment Score (1-5): 3.0\nThe image is well composed."

    with pytest.raises(ValueError, match="missing score fields.*coherence.*style") as exc_info:
        _parse_scores(text)

    assert "Alignment Score (1-5): 3.0" in str(exc_info.value)


def test_parse_scores_accepts_upstream_all_zero_sentinel():
    assert _parse_scores(
        "Alignment: 0.0/5\nCoherence: 0.0/5\nStyle: 0.0/5"
    ) == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("score", [0, 5.1, 10])
def test_parse_scores_rejects_values_outside_the_declared_scale(score):
    text = f"Alignment Score (1-5): {score}\nCoherence Score (1-5): 3\nStyle Score (1-5): 3"

    with pytest.raises(ValueError, match="outside the expected 1-5 range"):
        _parse_scores(text)
