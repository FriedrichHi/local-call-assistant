from __future__ import annotations

from local_call_assistant.domain import RecordingState, can_transition
from local_call_assistant.models import Manifest
from local_call_assistant.storage.spool import RecordingSpool


class RecordingSession:
    def __init__(self, spool: RecordingSpool) -> None:
        self.spool = spool
        self.state = RecordingState.READY
        self._manifest: Manifest | None = None

    def start(self, *, sample_rate: int, channels: int) -> None:
        self._require(RecordingState.RECORDING)
        self._manifest = self.spool.start(sample_rate=sample_rate, channels=channels)
        self.state = RecordingState.RECORDING

    def write(self, data: bytes) -> None:
        if self.state != RecordingState.RECORDING or self._manifest is None:
            raise ValueError("session is not recording")
        self.spool.append(self._manifest.job_id, data)

    def pause(self) -> None:
        self._require(RecordingState.PAUSED)
        self.state = RecordingState.PAUSED

    def resume(self) -> None:
        self._require(RecordingState.RECORDING)
        self.state = RecordingState.RECORDING

    def stop(self) -> Manifest:
        self._require(RecordingState.STOPPED)
        if self._manifest is None:
            raise ValueError("session has not started")
        manifest = self.spool.finalize(self._manifest.job_id)
        self.state = RecordingState.STOPPED
        return manifest

    def _require(self, target: RecordingState) -> None:
        if not can_transition(self.state, target):
            raise ValueError(f"cannot transition from {self.state} to {target}")
