"""Manifest: the durable record of one recording and its processing stages.

Paths inside a manifest are relative to the runtime root, so a spool can be
moved or restored without rewriting every manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(StrEnum):
    RECORDING = "recording"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"


LIVE_STATUSES = frozenset({JobStatus.RECORDING})
PROCESSABLE_STATUSES = frozenset({JobStatus.COMPLETE, JobStatus.INTERRUPTED})


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Attempt:
    started_at: str
    ended_at: str | None = None
    outcome: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"started_at": self.started_at, "ended_at": self.ended_at,
                "outcome": self.outcome, "error": self.error}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attempt:
        return cls(started_at=str(data["started_at"]), ended_at=data.get("ended_at"),
                   outcome=data.get("outcome"), error=data.get("error"))


@dataclass
class Stage:
    status: StageStatus = StageStatus.PENDING
    attempts: list[Attempt] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": str(self.status), "attempts": [a.to_dict() for a in self.attempts],
                "outputs": self.outputs, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage:
        return cls(status=StageStatus(data.get("status", StageStatus.PENDING)),
                   attempts=[Attempt.from_dict(a) for a in data.get("attempts", [])],
                   outputs=dict(data.get("outputs", {})), reason=data.get("reason"))


@dataclass
class Manifest:
    job_id: str
    status: JobStatus
    source_kind: str
    started_at: str
    audio_file: str
    sample_rate: int | None = None
    channels: int | None = None
    ended_at: str | None = None
    sha256: str | None = None
    audio_bytes: int = 0
    source: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Stage] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def stage(self, name: str) -> Stage:
        return self.stages.setdefault(name, Stage())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "job_id": self.job_id, "status": str(self.status),
            "source_kind": self.source_kind, "source": self.source,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "audio_file": self.audio_file, "sample_rate": self.sample_rate,
            "channels": self.channels, "sha256": self.sha256,
            "audio_bytes": self.audio_bytes, "notes": self.notes,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        version = data.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema version: {version!r}")
        return cls(
            job_id=str(data["job_id"]), status=JobStatus(data["status"]),
            source_kind=str(data["source_kind"]), source=dict(data.get("source", {})),
            started_at=str(data["started_at"]), ended_at=data.get("ended_at"),
            audio_file=str(data["audio_file"]), sample_rate=data.get("sample_rate"),
            channels=data.get("channels"), sha256=data.get("sha256"),
            audio_bytes=int(data.get("audio_bytes", 0)), notes=list(data.get("notes", [])),
            stages={name: Stage.from_dict(s) for name, s in data.get("stages", {}).items()},
        )
