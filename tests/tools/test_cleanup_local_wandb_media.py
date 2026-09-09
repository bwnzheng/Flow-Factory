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

import json
from pathlib import Path
from types import SimpleNamespace

from tools.cleanup_local_wandb_media import _clean_summary, _discover, main


def test_discover_run_media_and_summary(tmp_path: Path):
    files = tmp_path / "offline-run-x" / "files"
    (files / "media" / "images" / "nested").mkdir(parents=True)
    (files / "wandb-summary.json").write_text("{}", encoding="utf-8")

    media, summaries = _discover(tmp_path)

    assert media == [files / "media"]
    assert summaries == [files / "wandb-summary.json"]


def test_clean_summary_removes_media_references_and_backs_up(tmp_path: Path):
    summary = tmp_path / "files" / "wandb-summary.json"
    summary.parent.mkdir()
    summary.write_text(
        json.dumps(
            {
                "loss": 0.1,
                "media/evaluation": {"_type": "image-file", "path": "media/images/a.jpg"},
                "generated": {"path": "media/images/b.jpg"},
            }
        ),
        encoding="utf-8",
    )

    removed = _clean_summary(summary, tmp_path / "backup", execute=True)

    assert removed == ["generated", "media/evaluation"]
    assert json.loads(summary.read_text(encoding="utf-8")) == {"loss": 0.1}
    assert list((tmp_path / "backup").glob("*.bak"))


def test_main_deletes_media_children_but_preserves_media_dir(tmp_path: Path):
    files = tmp_path / "offline-run-x" / "files"
    media = files / "media"
    (media / "images" / "a" / "nested").mkdir(parents=True)
    (media / "images" / "a" / "nested" / "x.jpg").write_bytes(b"x")
    (media / "root.jpg").write_bytes(b"x")
    (files / "wandb-summary.json").write_text(json.dumps({"loss": 1}), encoding="utf-8")

    main(
        SimpleNamespace(
            root=tmp_path,
            workers=2,
            split_depth=3,
            backup_dir=tmp_path / "backup",
            execute=True,
        )
    )

    assert media.is_dir()
    assert list(media.iterdir()) == []


def test_main_dry_run_prints_summary_count_without_keys(tmp_path: Path, capsys):
    files = tmp_path / "offline-run-x" / "files"
    (files / "media").mkdir(parents=True)
    (files / "wandb-summary.json").write_text(
        json.dumps({"media/private-key": {"path": "media/images/x.jpg"}}),
        encoding="utf-8",
    )

    main(
        SimpleNamespace(
            root=tmp_path,
            workers=2,
            split_depth=3,
            backup_dir=tmp_path / "backup",
            execute=False,
        )
    )

    output = capsys.readouterr().out
    assert "would remove 1 media-related keys" in output
    assert "media/private-key" not in output
