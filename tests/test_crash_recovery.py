"""Kill a real recording process mid-write, then recover in a fresh process."""

import subprocess
import sys
import time
import wave
from pathlib import Path

from local_call_assistant.models import JobStatus
from local_call_assistant.storage.spool import RecordingSpool, sha256_of

CHUNK_BYTES = 3_200  # 100 ms of 16 kHz mono 16-bit audio
READY_AFTER_CHUNKS = 20
LOCK_RELEASE_TIMEOUT_SECONDS = 5.0

CHILD = f"""
import sys, time
from pathlib import Path
from local_call_assistant.recording.session import RecordingSession
from local_call_assistant.storage.spool import RecordingSpool

session = RecordingSession(RecordingSpool(Path(sys.argv[1])))
session.start(sample_rate=16_000, channels=1)
for index in range(10**9):
    session.write(b"\\x10\\x00" * ({CHUNK_BYTES} // 2))
    if index == {READY_AFTER_CHUNKS} - 1:
        print(session._manifest.job_id, flush=True)
    time.sleep(0.005)
"""


def _recover_once_lock_is_released(spool: RecordingSpool) -> list:
    # Windows can release a killed process's file lock a moment after wait()
    # returns; until then recovery correctly treats the job as live.
    deadline = time.monotonic() + LOCK_RELEASE_TIMEOUT_SECONDS
    while True:
        recovered = spool.recover_incomplete()
        if recovered or time.monotonic() > deadline:
            return recovered
        time.sleep(0.05)


def test_killed_recorder_keeps_its_audio_and_is_recovered(tmp_path: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(tmp_path)],
                             stdout=subprocess.PIPE, text=True)
    try:
        job_id = child.stdout.readline().strip()
        assert job_id, "recorder child did not start"
        # While the recorder lives, recovery must not touch its job.
        assert RecordingSpool(tmp_path).recover_incomplete() == []
    finally:
        child.kill()
        child.wait(timeout=10)

    spool = RecordingSpool(tmp_path)
    assert spool.load(job_id).status is JobStatus.RECORDING
    recovered = _recover_once_lock_is_released(spool)
    assert [m.job_id for m in recovered] == [job_id]
    manifest = recovered[0]
    audio = spool.path_of(manifest)
    assert manifest.status is JobStatus.INTERRUPTED
    assert manifest.audio_bytes >= CHUNK_BYTES * READY_AFTER_CHUNKS
    assert manifest.sha256 == sha256_of(audio)
    with wave.open(str(audio), "rb") as wav_file:
        assert wav_file.getnframes() * 2 == manifest.audio_bytes
    # A second recovery is a no-op.
    assert spool.recover_incomplete() == []
