from pathlib import Path

import pytest

from local_call_assistant.recording.session import RecordingSession
from local_call_assistant.storage.spool import RecordingSpool


def test_session_controls_write_and_finalize_recording(tmp_path: Path) -> None:
    session = RecordingSession(RecordingSpool(tmp_path))
    session.start(sample_rate=8_000, channels=1)
    session.write(b"\x00\x01" * 3)
    session.pause()
    with pytest.raises(ValueError, match="not recording"):
        session.write(b"\x00\x01")
    session.resume()
    session.write(b"\x00\x01" * 4)
    manifest = session.stop()
    assert manifest.status == "complete"
    assert manifest.audio_bytes == 14
    assert session.state == "stopped"
