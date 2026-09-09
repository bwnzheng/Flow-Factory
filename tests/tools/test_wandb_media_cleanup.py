import json
from pathlib import Path

import pytest

from tools.wandb_media_cleanup import EventCache, _pending_files, retry_call


class _HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_event_cache_reconstructs_pending_deletions(tmp_path: Path):
    cache = EventCache(tmp_path / "media.jsonl")
    cache.initialize("entity", "project", "media/%")
    cache.append({"record_type": "file", "run_id": "run-a", "name": "media/a.png"})
    cache.append({"record_type": "file", "run_id": "run-a", "name": "media/b.png"})
    cache.append({"record_type": "deleted", "run_id": "run-a", "name": "media/a.png"})
    cache.append({"record_type": "scan_complete", "run_id": "run-a"})

    state = cache.load()

    assert state.entity == "entity"
    assert state.project == "project"
    assert state.scan_complete == {"run-a"}
    assert _pending_files(state, None) == [("run-a", "media/b.png")]


def test_event_cache_rejects_scope_mismatch(tmp_path: Path):
    cache = EventCache(tmp_path / "media.jsonl")
    cache.initialize("entity", "project", "media/%")

    with pytest.raises(ValueError, match="Cache scope mismatch"):
        cache.initialize("other", "project", "media/%")


def test_event_cache_tolerates_truncated_final_line(tmp_path: Path):
    path = tmp_path / "media.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_type": "cache_header",
                "entity": "entity",
                "project": "project",
                "pattern": "media/%",
            }
        )
        + "\n{",
        encoding="utf-8",
    )

    state = EventCache(path).load()

    assert state.entity == "entity"


def test_retry_call_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("tools.wandb_media_cleanup.random.uniform", lambda _a, _b: 1.0)
    attempts = []
    sleeps = []

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise _HTTPError(503)

    status, retries = retry_call(
        operation,
        max_retries=0,
        base_delay=1.0,
        max_delay=10.0,
        sleep=sleeps.append,
    )

    assert status == "deleted"
    assert retries == 2
    assert sleeps == [1.0, 2.0]


def test_retry_call_treats_not_found_as_completed():
    def operation():
        raise _HTTPError(404)

    assert retry_call(operation, 0, 1.0, 10.0, sleep=lambda _delay: None) == (
        "already_missing",
        0,
    )


def test_retry_call_stops_on_permanent_error():
    def operation():
        raise _HTTPError(403)

    with pytest.raises(_HTTPError, match="HTTP 403"):
        retry_call(operation, 0, 1.0, 10.0, sleep=lambda _delay: None)
