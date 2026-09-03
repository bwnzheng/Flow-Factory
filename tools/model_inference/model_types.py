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

"""Define model families supported by standalone image inference."""

from typing import Tuple

SUPPORTED_MODEL_TYPES: Tuple[str, ...] = ("sdxl", "sd3-5", "flux1")
_MODEL_TYPE_ALIASES = {
    "sd-xl": "sdxl",
    "sd_xl": "sdxl",
    "sd3.5": "sd3-5",
    "sd3.5l": "sd3-5",
    "sd3-5l": "sd3-5",
    "sd3_5": "sd3-5",
    "flux.1-dev": "flux1",
    "flux1-dev": "flux1",
    "flux1_dev": "flux1",
}


def validate_model_type(model_type: str) -> str:
    """Validate and normalize a standalone inference model type.

    Args:
        model_type: User-facing model-family identifier.

    Returns:
        Normalized model-family identifier.

    Raises:
        ValueError: If the model family is not supported.
    """
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a non-empty string.")
    normalized = model_type.strip().lower()
    normalized = _MODEL_TYPE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Options: {list(SUPPORTED_MODEL_TYPES)}"
        )
    return normalized
