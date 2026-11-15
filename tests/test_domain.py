from local_call_assistant.domain import RecordingState, can_transition


def test_valid_recording_state_transitions() -> None:
    assert can_transition(RecordingState.READY, RecordingState.RECORDING)
    assert can_transition(RecordingState.RECORDING, RecordingState.PAUSED)
    assert can_transition(RecordingState.PAUSED, RecordingState.RECORDING)
    assert can_transition(RecordingState.RECORDING, RecordingState.STOPPED)


def test_invalid_recording_state_transition() -> None:
    assert not can_transition(RecordingState.READY, RecordingState.STOPPED)
    assert not can_transition(RecordingState.STOPPED, RecordingState.RECORDING)
