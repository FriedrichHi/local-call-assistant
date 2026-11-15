"""Ingest recordings made by another recorder (for example Meetily) into the spool.

The source file is only read, never moved or modified. A copy is verified by
checksum before its manifest is written, and a file already ingested (same
SHA-256) is not ingested twice.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_call_assistant.models import JobStatus, Manifest, utc_now
from local_call_assistant.storage.atomic import fsync_directory
from local_call_assistant.storage.filelock import JobLock
from local_call_assistant.storage.spool import RecordingSpool, SpoolError, sha256_of

INGEST_LOCK_NAME = "ingest.lock"
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm", ".aac"})
DEFAULT_MIN_AGE_SECONDS = 60.0


class SourceChanged(SpoolError):
    pass


@dataclass(frozen=True)
class IngestResult:
    path: Path
    outcome: str  # "ingested", "duplicate", "too-new", "busy" or "not-audio"
    job_id: str | None = None


def ingest_file(spool: RecordingSpool, path: Path, *,
                min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS) -> IngestResult:
    if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
        return IngestResult(path, "not-audio")
    # Heuristic: a recorder may still be writing a recent file. min_age <= 0
    # disables the check (file times can run slightly ahead of the clock).
    if min_age_seconds > 0 and time.time() - path.stat().st_mtime < min_age_seconds:
        return IngestResult(path, "too-new")
    lock = JobLock(spool.locks / INGEST_LOCK_NAME)
    if not lock.acquire():
        return IngestResult(path, "busy")  # another ingest is running
    try:
        return _ingest_locked(spool, path)
    finally:
        lock.release()


def _ingest_locked(spool: RecordingSpool, path: Path) -> IngestResult:
    before = path.stat()
    digest = sha256_of(path)
    existing = _find_by_source_digest(spool, digest)
    if existing is not None:
        return IngestResult(path, "duplicate", existing)
    job_id = uuid.uuid4().hex
    target_name = f"recordings/completed/{job_id}{path.suffix.lower()}"
    try:
        _copy_verified(path, spool.root / target_name, digest)
    except SourceChanged:
        return IngestResult(path, "too-new")  # the recorder is still writing it
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        (spool.root / target_name).unlink(missing_ok=True)
        return IngestResult(path, "too-new")  # the recorder is still writing it
    manifest = Manifest(
        job_id=job_id, status=JobStatus.COMPLETE, source_kind="ingest",
        started_at=_iso_from_timestamp(stat.st_mtime), ended_at=utc_now(),
        audio_file=target_name, sha256=digest, audio_bytes=stat.st_size,
        source={"original_path": str(path.resolve()), "original_sha256": digest},
    )
    spool.save(manifest)
    return IngestResult(path, "ingested", job_id)


def ingest_directory(spool: RecordingSpool, directory: Path, *, recursive: bool = False,
                     min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS) -> list[IngestResult]:
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))
    candidates = directory.rglob("*") if recursive else directory.glob("*")
    files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    return [ingest_file(spool, p, min_age_seconds=min_age_seconds) for p in files]


def _copy_verified(source: Path, target: Path, digest: str) -> None:
    temporary = target.with_name(target.name + ".part")
    shutil.copyfile(source, temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    if sha256_of(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise SourceChanged(f"{source.name} changed while it was being copied")
    os.replace(temporary, target)
    fsync_directory(target.parent)


def _find_by_source_digest(spool: RecordingSpool, digest: str) -> str | None:
    for manifest in spool.iter_manifests():
        if manifest.source.get("original_sha256") == digest:
            return manifest.job_id
    return None


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
