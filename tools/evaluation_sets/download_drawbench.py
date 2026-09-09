# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Download and materialize the canonical DrawBench prompt set as JSONL."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen


DRAWBENCH_URL = (
    "https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/"
    "gviz/tq?tqx=out:csv"
)
EXPECTED_PROMPT_COUNT = 200
EXPECTED_CATEGORY_COUNT = 11


def download_drawbench(output_path: str | Path, url: str = DRAWBENCH_URL) -> Path:
    """Download DrawBench CSV and write validated prompt records to JSONL.

    Args:
        output_path: Destination JSONL path, normally ``dataset/drawbench/test.jsonl``.
        url: Source CSV URL. Defaults to the Google Research spreadsheet export.

    Returns:
        The written output path.

    Raises:
        ValueError: If the source does not contain the canonical 200 prompts.
        OSError: If the source cannot be downloaded or the output cannot be written.
    """
    with urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    records = []
    for index, row in enumerate(rows):
        prompt = (row.get("Prompts") or row.get("prompt") or "").strip()
        category = (row.get("Category") or row.get("category") or "").strip()
        if not prompt or not category:
            raise ValueError(f"DrawBench row {index + 2} has an empty prompt or category.")
        records.append({"prompt": prompt, "category": category, "drawbench_index": index})

    categories = {record["category"] for record in records}
    if len(records) != EXPECTED_PROMPT_COUNT or len(categories) != EXPECTED_CATEGORY_COUNT:
        raise ValueError(
            "Unexpected DrawBench source shape: "
            f"prompts={len(records)} (expected {EXPECTED_PROMPT_COUNT}), "
            f"categories={len(categories)} (expected {EXPECTED_CATEGORY_COUNT})."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return output


def main() -> None:
    """Parse command-line arguments and materialize DrawBench."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="dataset/drawbench/test.jsonl")
    parser.add_argument("--url", default=DRAWBENCH_URL)
    args = parser.parse_args()
    output = download_drawbench(args.output, args.url)
    print(f"Wrote {EXPECTED_PROMPT_COUNT} DrawBench prompts to {output}")


if __name__ == "__main__":
    main()

