"""Runs the post-recording stages for spooled jobs and records every attempt.

Stages: transcription -> analysis -> note. A stage only reads the spooled
audio (whose checksum is verified first) and writes its own outputs, so a
failure anywhere leaves the recording untouched and the stage retryable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_call_assistant.models import (
    Attempt, Manifest, PROCESSABLE_STATUSES, Stage, StageStatus, utc_now,
)
from local_call_assistant.storage.filelock import JobLock
from local_call_assistant.storage.spool import RecordingSpool, sha256_of

STAGE_ORDER = ("transcription", "analysis", "note")
RESET_OUTCOME = "reset"
MAX_ERROR_CHARS = 500


class StageSkipped(Exception):
    """Raised by a stage handler that is not configured to run."""


@dataclass(frozen=True)
class StageContext:
    spool: RecordingSpool
    manifest: Manifest
    audio_path: Path


# A handler returns the outputs to record, or raises StageSkipped / any error.
StageHandler = Callable[[StageContext], dict[str, Any]]

_REQUIRES: dict[str, tuple[str, ...]] = {
    "transcription": (),
    "analysis": ("transcription",),
    "note": ("transcription", "analysis"),
}
_DEPENDENTS: dict[str, tuple[str, ...]] = {
    "transcription": ("analysis", "note"),
    "analysis": ("note",),
    "note": (),
}
_SATISFIED = {"transcription": {StageStatus.DONE},
              "analysis": {StageStatus.DONE, StageStatus.SKIPPED}}


class Pipeline:
    def __init__(self, spool: RecordingSpool, handlers: dict[str, StageHandler],
                 max_attempts: int) -> None:
        self.spool = spool
        self.handlers = handlers
        self.max_attempts = max_attempts

    def recover_stages(self) -> list[str]:
        """Mark stages left 'running' by a dead process as failed attempts.

        A job whose lock is held belongs to a live worker and is left alone.
        """
        touched = []
        for candidate in self.spool.iter_manifests():
            if not any(s.status is StageStatus.RUNNING for s in candidate.stages.values()):
                continue
            with _job_lock(self.spool, candidate.job_id) as owned:
                if not owned:
                    continue
                manifest = self.spool.load(candidate.job_id)
                for stage in manifest.stages.values():
                    if stage.status is StageStatus.RUNNING:
                        _finish(stage, StageStatus.FAILED, "interrupted",
                                "process stopped while the stage was running")
                self.spool.save(manifest)
                touched.append(manifest.job_id)
        return touched

    def process(self, job_id: str) -> Manifest:
        """Run every ready stage of one job. A job busy elsewhere is returned unchanged."""
        with _job_lock(self.spool, job_id) as owned:
            manifest = self.spool.load(job_id)
            if not owned or manifest.status not in PROCESSABLE_STATUSES:
                return manifest
            for name in STAGE_ORDER:
                if self._ready(manifest, name) and not self._run_stage(manifest, name):
                    break
            return manifest

    def retry(self, job_id: str, stage_name: str) -> Manifest:
        """Reset a failed or skipped stage. Refused when it would leave outputs stale."""
        if stage_name not in STAGE_ORDER:
            raise ValueError(f"unknown stage: {stage_name}")
        with _job_lock(self.spool, job_id) as owned:
            if not owned:
                raise ValueError(f"job {job_id} is busy in another process")
            manifest = self.spool.load(job_id)
            stage = manifest.stage(stage_name)
            if stage.status not in {StageStatus.FAILED, StageStatus.SKIPPED}:
                raise ValueError(f"{stage_name} is {stage.status}; only failed or skipped "
                                 "stages can be retried")
            # A dependent that has ever run may have published output from the
            # earlier result. Judged from its attempt history, which resets keep.
            ran = [d for d in _DEPENDENTS[stage_name] if _has_ever_run(manifest.stage(d))]
            if ran:
                raise ValueError(f"{', '.join(ran)} already ran on the earlier result; "
                                 "retrying would leave its output stale")
            stage.attempts.append(Attempt(started_at=utc_now(), ended_at=utc_now(),
                                          outcome=RESET_OUTCOME))
            stage.status, stage.reason = StageStatus.PENDING, "manual retry"
            for dependent in _DEPENDENTS[stage_name]:
                if manifest.stage(dependent).status is StageStatus.SKIPPED:
                    manifest.stage(dependent).status = StageStatus.PENDING
            self.spool.save(manifest)
            return manifest

    def _ready(self, manifest: Manifest, name: str) -> bool:
        stage = manifest.stage(name)
        if stage.status in {StageStatus.DONE, StageStatus.SKIPPED, StageStatus.RUNNING}:
            return False
        if stage.status is StageStatus.FAILED and _failures_since_reset(stage) >= self.max_attempts:
            return False
        return all(manifest.stage(dep).status in _SATISFIED[dep] for dep in _REQUIRES[name])

    def _run_stage(self, manifest: Manifest, name: str) -> bool:
        """Run one stage. Returns False when later stages must not run now."""
        stage = manifest.stage(name)
        stage.status = StageStatus.RUNNING
        stage.attempts.append(Attempt(started_at=utc_now()))
        self.spool.save(manifest)  # persisted so a crash here is visible on recovery
        try:
            audio_path = self.spool.path_of(manifest)  # inside try: a bad path is a failed attempt
            _verify_audio(audio_path, manifest)
            stage.outputs = self.handlers[name](StageContext(self.spool, manifest, audio_path))
        except StageSkipped as skipped:
            _finish(stage, StageStatus.SKIPPED, "skipped", None, reason=str(skipped))
        except Exception as error:  # recorded and retryable, never fatal to the spool
            _finish(stage, StageStatus.FAILED, "error", _describe(error))
            self.spool.save(manifest)
            return False
        else:
            _finish(stage, StageStatus.DONE, "done", None)
        self.spool.save(manifest)
        return True


class AudioVerificationError(RuntimeError):
    pass


def _verify_audio(audio_path: Path, manifest: Manifest) -> None:
    try:
        matches = sha256_of(audio_path) == manifest.sha256
    except OSError as error:
        raise AudioVerificationError(f"spooled audio cannot be read ({error.strerror})") from None
    if not matches:
        raise AudioVerificationError("spooled audio does not match its recorded checksum; "
                                     "not processing it")


@contextmanager
def _job_lock(spool: RecordingSpool, job_id: str) -> Iterator[bool]:
    lock = JobLock(spool.lock_path(job_id))
    owned = lock.acquire()
    try:
        yield owned
    finally:
        if owned:
            lock.release()


def _finish(stage: Stage, status: StageStatus, outcome: str, error: str | None,
            reason: str | None = None) -> None:
    stage.status, stage.reason = status, reason
    if stage.attempts and stage.attempts[-1].ended_at is None:
        attempt = stage.attempts[-1]
        attempt.ended_at, attempt.outcome, attempt.error = utc_now(), outcome, error


def _has_ever_run(stage: Stage) -> bool:
    """True if the stage ever ran (so output may exist). Skips and resets don't count."""
    return any(a.outcome not in {"skipped", RESET_OUTCOME} for a in stage.attempts)


def _failures_since_reset(stage: Stage) -> int:
    count = 0
    for attempt in reversed(stage.attempts):
        if attempt.outcome == RESET_OUTCOME:
            break
        if attempt.outcome in {"error", "interrupted"}:
            count += 1
    return count


def _describe(error: Exception) -> str:
    # Handlers that talk to remote services map errors to fixed texts first
    # (see integrations/hermes.py); this is a last line of defence.
    text = " ".join(f"{type(error).__name__}: {error}".split())
    return text[:MAX_ERROR_CHARS]
