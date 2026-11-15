import wave
from pathlib import Path

import pytest

from local_call_assistant.models import JobStatus
from local_call_assistant.storage.spool import RecordingSpool, sha256_of


def _frames(path: Path) -> int:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes()


def test_finalize_writes_wav_and_manifest_with_checksum(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path)
    recording = spool.start(sample_rate=16_000, channels=1)
    spool.append(recording.job_id, b"\x00\x01" * 100)
    manifest = spool.finalize(recording.job_id)
    audio = spool.path_of(manifest)
    assert manifest.status is JobStatus.COMPLETE
    assert audio.suffix == ".wav" and _frames(audio) == 100
    assert manifest.sha256 == sha256_of(audio)
    assert manifest.audio_bytes == 200
    assert spool.load(recording.job_id).sha256 == manifest.sha256
    assert not spool.raw_path(recording.job_id).exists()


def test_manifest_paths_are_relative_to_the_runtime_root(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path)
    recording = spool.start(sample_rate=16_000, channels=1)
    assert not Path(recording.audio_file).is_absolute()


def test_append_after_finalize_is_rejected(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path)
    recording = spool.start(sample_rate=16_000, channels=1)
    spool.finalize(recording.job_id)
    with pytest.raises(ValueError, match="not recording"):
        spool.append(recording.job_id, b"data")


def test_recovery_seals_an_abandoned_recording(tmp_path: Path) -> None:
    owner = RecordingSpool(tmp_path)
    recording = owner.start(sample_rate=16_000, channels=1)
    owner.append(recording.job_id, b"\x01\x02" * 50)
    owner._close(recording.job_id)  # owner goes away without finalizing
    recovered = RecordingSpool(tmp_path).recover_incomplete()
    assert [m.job_id for m in recovered] == [recording.job_id]
    assert recovered[0].status is JobStatus.INTERRUPTED
    assert recovered[0].audio_bytes == 100


def test_recovery_leaves_a_live_recording_alone(tmp_path: Path) -> None:
    owner = RecordingSpool(tmp_path)
    recording = owner.start(sample_rate=16_000, channels=1)
    owner.append(recording.job_id, b"\x00\x00" * 10)
    assert RecordingSpool(tmp_path).recover_incomplete() == []
    assert owner.load(recording.job_id).status is JobStatus.RECORDING
    owner.append(recording.job_id, b"\x00\x00" * 10)
    assert owner.finalize(recording.job_id).audio_bytes == 40


def test_recovery_never_replaces_a_built_wav_with_an_empty_one(tmp_path: Path) -> None:
    # Crash after the WAV was renamed into place and the raw file removed,
    # but before the manifest was marked complete.
    spool = RecordingSpool(tmp_path)
    recording = spool.start(sample_rate=16_000, channels=1)
    spool.append(recording.job_id, b"\x05\x06" * 80)
    spool._close(recording.job_id)
    from local_call_assistant.storage import spool as spool_module
    spool_module._build_wav(spool.raw_path(recording.job_id), spool.path_of(recording), recording)
    spool.raw_path(recording.job_id).unlink()
    recovered = RecordingSpool(tmp_path).recover_incomplete()
    assert recovered[0].audio_bytes == 160
    assert _frames(spool.path_of(recovered[0])) == 80


def test_trailing_partial_frame_is_dropped(tmp_path: Path) -> None:
    spool = RecordingSpool(tmp_path)
    recording = spool.start(sample_rate=16_000, channels=2)
    spool.append(recording.job_id, b"\x00" * 4 * 10 + b"\x01\x02\x03")
    manifest = spool.finalize(recording.job_id)
    assert manifest.audio_bytes == 40
    assert _frames(spool.path_of(manifest)) == 10
