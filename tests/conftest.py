from pathlib import Path

import pytest

from local_call_assistant.models import Manifest
from local_call_assistant.storage.spool import RecordingSpool
from local_call_assistant.transcription import Segment, Transcript


class FakeTranscriber:
    """Stands in for faster-whisper; can be told to fail a number of times."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    def transcribe(self, audio_path: Path) -> Transcript:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("model crashed")
        segments = [Segment(0.0, 2.0, " We agreed to ship on Friday."),
                    Segment(65.0, 67.5, " Anna will send the invoice.")]
        return Transcript(language="en", duration=67.5, segments=segments,
                          model="fake", device="cpu", compute_type="int8")


@pytest.fixture
def spool(tmp_path: Path) -> RecordingSpool:
    return RecordingSpool(tmp_path / "runtime")


@pytest.fixture
def recorded(spool: RecordingSpool) -> Manifest:
    manifest = spool.start(sample_rate=16_000, channels=1)
    spool.append(manifest.job_id, b"\x00\x01" * 16_000)
    return spool.finalize(manifest.job_id)
