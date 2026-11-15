import os
import time
from pathlib import Path

from local_call_assistant.storage.ingest import ingest_directory, ingest_file
from local_call_assistant.storage.spool import RecordingSpool, sha256_of


def _old_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    past = time.time() - 3_600
    os.utime(path, (past, past))
    return path


def test_ingest_copies_verifies_and_leaves_the_source_untouched(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path / "spool")
    source = _old_file(tmp_path / "meeting.m4a", b"fake audio bytes")
    before = (sha256_of(source), source.stat().st_mtime)
    result = ingest_file(spool, source)
    assert result.outcome == "ingested"
    manifest = spool.load(result.job_id)
    assert manifest.sha256 == before[0] == sha256_of(spool.path_of(manifest))
    assert manifest.source_kind == "ingest"
    assert (sha256_of(source), source.stat().st_mtime) == before


def test_same_file_is_not_ingested_twice(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path / "spool")
    source = _old_file(tmp_path / "a.wav", b"same")
    first = ingest_file(spool, source)
    copy = _old_file(tmp_path / "b.wav", b"same")
    second = ingest_file(spool, copy)
    assert second.outcome == "duplicate" and second.job_id == first.job_id
    assert len(list(spool.iter_manifests())) == 1


def test_recent_files_wait_until_the_recorder_is_done(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path / "spool")
    source = tmp_path / "still-writing.wav"
    source.write_bytes(b"partial")
    assert ingest_file(spool, source, min_age_seconds=60).outcome == "too-new"
    assert ingest_file(spool, source, min_age_seconds=0).outcome == "ingested"


def test_ingest_directory_takes_only_audio(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path / "spool")
    folder = tmp_path / "recorder"
    folder.mkdir()
    _old_file(folder / "one.wav", b"1")
    _old_file(folder / "notes.txt", b"not audio")
    results = ingest_directory(spool, folder)
    assert [(r.path.name, r.outcome) for r in results] == [("one.wav", "ingested")]
