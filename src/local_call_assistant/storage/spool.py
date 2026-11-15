"""Crash-safe local spool for recordings.

Ordering rules that keep audio safe across a crash at any point:

1. The manifest is written before the first audio byte.
2. Raw PCM is appended and flushed on every write, and fsynced at least once a
   second, while the job lock is held.
3. On stop, still holding the lock, the WAV is built from the raw file into a
   temp name, fsynced and renamed into place; the manifest is then marked
   complete; only after both are durable is the raw file removed. A crash
   between any two steps leaves the raw file, so recovery can rebuild the WAV.
4. Recovery takes the job lock first, never replaces an existing WAV with an
   empty one, and also finishes seals interrupted after the manifest update.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from local_call_assistant.models import (
    JobStatus, LIVE_STATUSES, Manifest, utc_now,
)
from local_call_assistant.storage.atomic import atomic_write_text, fsync_directory
from local_call_assistant.storage.filelock import JobLock

SAMPLE_WIDTH_BYTES = 2
FSYNC_INTERVAL_SECONDS = 1.0
COPY_CHUNK_BYTES = 1024 * 1024


class SpoolError(RuntimeError):
    pass


class RecordingSpool:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending = root / "recordings" / "pending"
        self.completed = root / "recordings" / "completed"
        self.manifests = root / "manifests"
        self.locks = root / "locks"
        for directory in (self.pending, self.completed, self.manifests, self.locks):
            directory.mkdir(parents=True, exist_ok=True)
        for directory in (root / "recordings", root, root.parent):
            fsync_directory(directory)
        self._open: dict[str, _OpenRecording] = {}

    # ----- paths -------------------------------------------------------
    def path_of(self, manifest: Manifest) -> Path:
        return self.root / manifest.audio_file

    def raw_path(self, job_id: str) -> Path:
        return self.pending / f"{job_id}.raw"

    def manifest_path(self, job_id: str) -> Path:
        return self.manifests / f"{job_id}.json"

    def lock_path(self, job_id: str) -> Path:
        return self.locks / f"{job_id}.lock"

    # ----- recording ---------------------------------------------------
    def start(self, *, sample_rate: int, channels: int) -> Manifest:
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        job_id = uuid.uuid4().hex
        lock = JobLock(self.lock_path(job_id))
        if not lock.acquire():
            raise SpoolError(f"could not lock new job {job_id}")
        try:
            manifest = Manifest(
                job_id=job_id, status=JobStatus.RECORDING, source_kind="session",
                started_at=utc_now(), sample_rate=sample_rate, channels=channels,
                audio_file=f"recordings/completed/{job_id}.wav",
            )
            self.save(manifest)
            handle = self.raw_path(job_id).open("ab")
            fsync_directory(self.pending)  # the raw file's name must survive a power cut
        except BaseException:
            lock.release()
            raise
        self._open[job_id] = _OpenRecording(handle=handle, lock=lock)
        return manifest

    def append(self, job_id: str, data: bytes) -> None:
        recording = self._open.get(job_id)
        if recording is None:
            raise ValueError("recording is not recording")
        if data:
            recording.write(data)

    def finalize(self, job_id: str) -> Manifest:
        manifest = self.load(job_id)
        recording = self._open.pop(job_id, None)
        if manifest.status not in LIVE_STATUSES or recording is None:
            raise ValueError("recording is not recording")
        try:
            recording.close_file()  # the lock stays held until sealing is done
            return self._seal(manifest, JobStatus.COMPLETE)
        finally:
            recording.lock.release()

    # ----- recovery ----------------------------------------------------
    def recover_incomplete(self) -> list[Manifest]:
        """Seal recordings whose owning process is gone; finish interrupted seals.

        Each job is handled only while holding its lock, so a live recorder,
        a finalizer or another recoverer is never raced.
        """
        recovered = []
        for candidate in self.iter_manifests():
            if candidate.job_id in self._open or not self._may_need_recovery(candidate):
                continue
            lock = JobLock(self.lock_path(candidate.job_id))
            if not lock.acquire():
                continue  # owned by a live process
            try:
                manifest = self.load(candidate.job_id)  # re-read under the lock
                if manifest.status in LIVE_STATUSES:
                    recovered.append(self._seal(manifest, JobStatus.INTERRUPTED))
                else:
                    self._reconcile_sealed(manifest)
            finally:
                lock.release()
        return recovered

    def _may_need_recovery(self, manifest: Manifest) -> bool:
        return manifest.status in LIVE_STATUSES or self.raw_path(manifest.job_id).exists()

    def _reconcile_sealed(self, manifest: Manifest) -> None:
        """A crash after the manifest was sealed but before the raw file was removed."""
        raw, target = self.raw_path(manifest.job_id), self.path_of(manifest)
        if target.exists() and sha256_of(target) == manifest.sha256:
            raw.unlink(missing_ok=True)
            fsync_directory(raw.parent)
            return
        manifest.audio_bytes = _build_wav(raw, target, manifest)
        manifest.sha256 = sha256_of(target)
        manifest.notes.append("audio rebuilt from the retained raw file during recovery")
        self.save(manifest)
        raw.unlink(missing_ok=True)
        fsync_directory(raw.parent)

    def _seal(self, manifest: Manifest, status: JobStatus) -> Manifest:
        raw = self.raw_path(manifest.job_id)
        target = self.path_of(manifest)
        if raw.exists():
            manifest.audio_bytes = _build_wav(raw, target, manifest)
        elif not target.exists():
            manifest.notes.append("no audio was captured before the recording ended")
            manifest.audio_bytes = _build_wav(None, target, manifest)
        else:
            manifest.audio_bytes = _wav_data_bytes(target)
        manifest.status = status
        manifest.ended_at = manifest.ended_at or utc_now()
        manifest.sha256 = sha256_of(target)
        self.save(manifest)  # durable (file and directory) before the raw backup goes
        raw.unlink(missing_ok=True)
        fsync_directory(raw.parent)
        # Lock files are kept: deleting one another process may hold breaks POSIX locking.
        return manifest

    # ----- manifests ---------------------------------------------------
    def save(self, manifest: Manifest) -> None:
        text = json.dumps(manifest.to_dict(), indent=2) + "\n"
        atomic_write_text(self.manifest_path(manifest.job_id), text)

    def load(self, job_id: str) -> Manifest:
        path = self.manifest_path(job_id)
        if not path.exists():
            raise KeyError(f"unknown job: {job_id}")
        return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def iter_manifests(self) -> Iterator[Manifest]:
        for path in sorted(self.manifests.glob("*.json")):
            yield Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _close(self, job_id: str) -> None:
        recording = self._open.pop(job_id, None)
        if recording is not None:
            try:
                recording.close_file()
            finally:
                recording.lock.release()


class _OpenRecording:
    def __init__(self, handle: IO[bytes], lock: JobLock) -> None:
        self.handle = handle
        self.lock = lock
        self._last_sync = time.monotonic()

    def write(self, data: bytes) -> None:
        self.handle.write(data)
        self.handle.flush()
        now = time.monotonic()
        if now - self._last_sync >= FSYNC_INTERVAL_SECONDS:
            os.fsync(self.handle.fileno())
            self._last_sync = now

    def close_file(self) -> None:
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.handle.close()


def _build_wav(raw: Path | None, target: Path, manifest: Manifest) -> int:
    if manifest.sample_rate is None or manifest.channels is None:
        raise SpoolError(f"job {manifest.job_id} has no audio format")
    temporary = target.with_name(target.name + ".part")
    frame_bytes = SAMPLE_WIDTH_BYTES * manifest.channels
    written = 0
    with wave.open(str(temporary), "wb") as wav_file:
        wav_file.setnchannels(manifest.channels)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(manifest.sample_rate)
        for chunk in _whole_frames(raw, frame_bytes):
            wav_file.writeframes(chunk)
            written += len(chunk)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    fsync_directory(target.parent)
    return written


def _whole_frames(raw: Path | None, frame_bytes: int) -> Iterator[bytes]:
    """Yield raw PCM, dropping a trailing partial frame left by a crash mid-write."""
    if raw is None:
        return
    usable = raw.stat().st_size // frame_bytes * frame_bytes
    remaining = usable
    with raw.open("rb") as stream:
        while remaining > 0:
            chunk = stream.read(min(COPY_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _wav_data_bytes(path: Path) -> int:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() * wav_file.getnchannels() * wav_file.getsampwidth()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
