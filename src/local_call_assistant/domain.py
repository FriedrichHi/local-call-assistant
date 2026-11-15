from __future__ import annotations

from enum import StrEnum


class RecordingState(StrEnum):
    READY = "ready"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


_TRANSITIONS = {
    RecordingState.READY: {RecordingState.RECORDING},
    RecordingState.RECORDING: {RecordingState.PAUSED, RecordingState.STOPPED},
    RecordingState.PAUSED: {RecordingState.RECORDING, RecordingState.STOPPED},
    RecordingState.STOPPED: set(),
}


def can_transition(current: RecordingState, target: RecordingState) -> bool:
    return target in _TRANSITIONS[current]
