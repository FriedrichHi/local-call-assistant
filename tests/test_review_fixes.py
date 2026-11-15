"""Regression tests for the round-1 review findings (locking, publication, privacy)."""

import socket
import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import FakeTranscriber

from local_call_assistant.config import HermesConfig, ObsidianConfig, TranscriptionConfig
from local_call_assistant.integrations.hermes import HermesError, analyse, analysis_handler
from local_call_assistant.integrations.obsidian import note_handler, render_note, write_note
from local_call_assistant.models import Attempt, JobStatus, Manifest, StageStatus
from local_call_assistant.processing import Pipeline
from local_call_assistant.storage import ingest as ingest_module
from local_call_assistant.storage import spool as spool_module
from local_call_assistant.storage.atomic import create_new_text
from local_call_assistant.storage.filelock import JobLock
from local_call_assistant.storage.ingest import ingest_file
from local_call_assistant.storage.spool import RecordingSpool, sha256_of
from local_call_assistant.transcription import Segment, transcription_handler

KEY_ENV = "LCA_TEST_HERMES_KEY"
SECRET_PHRASE = "the merger closes on the ninth"


def _pipeline(spool: RecordingSpool, inbox: Path | None = None) -> Pipeline:
    return Pipeline(spool, {
        "transcription": transcription_handler(TranscriptionConfig(enabled=True), FakeTranscriber()),
        "analysis": analysis_handler(HermesConfig(enabled=False)),
        "note": note_handler(ObsidianConfig(inbox=inbox)),
    }, max_attempts=3)


# ----- spool: sealing under the lock, reconciling interrupted seals --------

def _sealed_with_raw_retained(spool: RecordingSpool) -> Manifest:
    """Manifest marked complete, but the raw backup was never removed."""
    manifest = spool.start(sample_rate=16_000, channels=1)
    spool.append(manifest.job_id, b"\x07\x00" * 40)
    spool._close(manifest.job_id)
    spool_module._build_wav(spool.raw_path(manifest.job_id), spool.path_of(manifest), manifest)
    manifest.status, manifest.sha256 = JobStatus.COMPLETE, sha256_of(spool.path_of(manifest))
    manifest.audio_bytes = 80
    spool.save(manifest)
    return manifest


def test_recovery_waits_for_the_lock_holder(spool: RecordingSpool) -> None:
    manifest = _sealed_with_raw_retained(spool)
    holder = JobLock(spool.lock_path(manifest.job_id))
    assert holder.acquire()
    try:
        RecordingSpool(spool.root).recover_incomplete()
        assert spool.raw_path(manifest.job_id).exists()  # untouched while held
    finally:
        holder.release()
    RecordingSpool(spool.root).recover_incomplete()
    assert not spool.raw_path(manifest.job_id).exists()
    assert sha256_of(spool.path_of(manifest)) == manifest.sha256


def test_missing_wav_is_rebuilt_from_the_retained_raw_file(spool: RecordingSpool) -> None:
    manifest = _sealed_with_raw_retained(spool)
    spool.path_of(manifest).unlink()
    RecordingSpool(spool.root).recover_incomplete()
    rebuilt = spool.load(manifest.job_id)
    assert rebuilt.audio_bytes == 80
    assert rebuilt.sha256 == sha256_of(spool.path_of(rebuilt))
    assert not spool.raw_path(manifest.job_id).exists()


def test_lock_files_are_kept_after_sealing(spool: RecordingSpool, recorded: Manifest) -> None:
    assert spool.lock_path(recorded.job_id).exists()


def test_finalize_keeps_its_lock_while_sealing(
        spool: RecordingSpool, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = spool.start(sample_rate=16_000, channels=1)
    spool.append(manifest.job_id, b"\x01\x00" * 30)
    real_build = spool_module._build_wav
    observed: list[list[Manifest]] = []

    def build_while_recovery_tries(raw: Path | None, target: Path, job: Manifest) -> int:
        observed.append(RecordingSpool(spool.root).recover_incomplete())
        return real_build(raw, target, job)

    monkeypatch.setattr(spool_module, "_build_wav", build_while_recovery_tries)
    sealed = spool.finalize(manifest.job_id)
    assert observed == [[]]  # recovery could not take the job mid-seal
    assert sealed.status is JobStatus.COMPLETE and sealed.audio_bytes == 60


def test_failed_close_still_releases_the_lock(
        spool: RecordingSpool, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = spool.start(sample_rate=16_000, channels=1)
    spool.append(manifest.job_id, b"\x01\x00" * 30)

    def failing_fsync(descriptor: int) -> None:
        raise OSError(5, "simulated I/O error")

    monkeypatch.setattr(spool_module.os, "fsync", failing_fsync)
    # Keep the exception (and so its traceback and frames) alive, as a caller
    # that logs or re-raises it would: the lock must not depend on GC.
    with pytest.raises(OSError) as caught:
        spool.finalize(manifest.job_id)
    monkeypatch.undo()
    recovered = RecordingSpool(spool.root).recover_incomplete()
    assert [m.job_id for m in recovered] == [manifest.job_id]
    assert recovered[0].audio_bytes == 60
    assert caught.value.errno == 5


# ----- pipeline: live workers are not disturbed ----------------------------

def _mark_running(spool: RecordingSpool, job_id: str) -> None:
    manifest = spool.load(job_id)
    manifest.stage("transcription").status = StageStatus.RUNNING
    manifest.stage("transcription").attempts.append(Attempt(started_at="2026-11-15T00:00:00+00:00"))
    spool.save(manifest)


def test_running_stage_of_a_live_worker_is_left_alone(
        spool: RecordingSpool, recorded: Manifest) -> None:
    _mark_running(spool, recorded.job_id)
    worker = JobLock(spool.lock_path(recorded.job_id))
    assert worker.acquire()
    try:
        pipeline = _pipeline(spool)
        assert pipeline.recover_stages() == []
        assert pipeline.process(recorded.job_id).stage("transcription").status is StageStatus.RUNNING
        with pytest.raises(ValueError, match="busy"):
            pipeline.retry(recorded.job_id, "transcription")
    finally:
        worker.release()
    assert _pipeline(spool).recover_stages() == [recorded.job_id]


def test_retry_refuses_to_leave_dependent_outputs_stale(
        spool: RecordingSpool, recorded: Manifest, tmp_path: Path) -> None:
    pipeline = _pipeline(spool, inbox=tmp_path / "inbox")
    pipeline.process(recorded.job_id)
    with pytest.raises(ValueError, match="only failed or skipped"):
        pipeline.retry(recorded.job_id, "transcription")
    with pytest.raises(ValueError, match="note already ran"):
        pipeline.retry(recorded.job_id, "analysis")


def test_retry_is_refused_once_a_dependent_has_run_even_if_it_failed(
        spool: RecordingSpool, recorded: Manifest, tmp_path: Path) -> None:
    # The note was published, then the process died before recording success.
    pipeline = _pipeline(spool, inbox=tmp_path / "inbox")
    pipeline.process(recorded.job_id)
    manifest = spool.load(recorded.job_id)
    manifest.stage("note").status = StageStatus.FAILED
    spool.save(manifest)
    with pytest.raises(ValueError, match="note already ran"):
        pipeline.retry(recorded.job_id, "analysis")


def test_unreadable_audio_is_recorded_as_a_failed_attempt(
        spool: RecordingSpool, recorded: Manifest) -> None:
    spool.path_of(recorded).unlink()
    stage = _pipeline(spool).process(recorded.job_id).stage("transcription")
    assert stage.status is StageStatus.FAILED
    assert stage.attempts[-1].outcome == "error"
    assert "cannot be read" in stage.attempts[-1].error


# ----- no-overwrite publication -------------------------------------------

def test_create_new_text_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    assert create_new_text(target, "first")
    assert not create_new_text(target, "second")
    assert target.read_text(encoding="utf-8") == "first"
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_note_is_found_again_under_another_name(recorded: Manifest, tmp_path: Path) -> None:
    # Same job written earlier under a stem from another time zone.
    text = render_note(recorded, [Segment(0, 1, "x")], None, Path("t.json"))
    earlier = tmp_path / f"2026-01-01 0000 call {recorded.job_id[:8]}.md"
    earlier.write_text(text, encoding="utf-8")
    assert write_note(tmp_path, recorded, text) == earlier
    assert len(list(tmp_path.glob("*.md"))) == 1


# ----- ingest --------------------------------------------------------------

def test_concurrent_ingest_is_refused(spool: RecordingSpool, tmp_path: Path) -> None:
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio")
    other = JobLock(spool.locks / ingest_module.INGEST_LOCK_NAME)
    assert other.acquire()
    try:
        assert ingest_file(spool, source, min_age_seconds=0).outcome == "busy"
    finally:
        other.release()


def test_source_changing_during_copy_is_not_ingested(
        spool: RecordingSpool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "growing.wav"
    source.write_bytes(b"first part")
    real_copy = ingest_module._copy_verified

    def copy_then_grow(src: Path, target: Path, digest: str) -> None:
        real_copy(src, target, digest)
        with src.open("ab") as stream:
            stream.write(b" and more")

    monkeypatch.setattr(ingest_module, "_copy_verified", copy_then_grow)
    assert ingest_file(spool, source, min_age_seconds=0).outcome == "too-new"
    assert list(spool.iter_manifests()) == []
    assert list(spool.completed.iterdir()) == []


def test_source_growing_before_the_copy_is_reported_not_raised(
        spool: RecordingSpool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "growing.wav"
    source.write_bytes(b"first part")
    real_copyfile = ingest_module.shutil.copyfile

    def grow_then_copy(src: Path, dst: Path) -> None:
        with Path(src).open("ab") as stream:
            stream.write(b" and more")
        real_copyfile(src, dst)

    monkeypatch.setattr(ingest_module.shutil, "copyfile", grow_then_copy)
    assert ingest_file(spool, source, min_age_seconds=0).outcome == "too-new"
    assert list(spool.completed.iterdir()) == []


# ----- Hermes: redirects and error privacy ---------------------------------

def _read_request(connection: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        data += connection.recv(65536)
    head, _, body = data.partition(b"\r\n\r\n")
    length = next((int(line.split(b":")[1]) for line in head.split(b"\r\n")
                   if line.lower().startswith(b"content-length:")), 0)
    while len(body) < length:
        body += connection.recv(65536)
    return head + b"\r\n\r\n" + body


def _raw_server(response: bytes) -> Iterator[str]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    seen: list[bytes] = []

    def serve() -> None:
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            with connection:
                seen.append(_read_request(connection))
                connection.sendall(response)

    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    listener.close()


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(KEY_ENV, "test-key")
    return "test-key"


def _config(url: str) -> HermesConfig:
    return HermesConfig(enabled=True, base_url=url, api_key_env=KEY_ENV, timeout_seconds=5)


def test_redirects_are_refused(key: str) -> None:
    for url in _raw_server(b"HTTP/1.1 302 Found\r\nLocation: http://example.com/steal\r\n"
                           b"Content-Length: 0\r\n\r\n"):
        with pytest.raises(HermesError, match="redirect"):
            analyse(_config(url), "job", "hello")


def test_malformed_response_does_not_leak_into_the_error(key: str) -> None:
    for url in _raw_server(f"GARBAGE {SECRET_PHRASE}\r\n\r\n".encode()):
        with pytest.raises(HermesError) as caught:
            analyse(_config(url), "job", SECRET_PHRASE)
    assert SECRET_PHRASE not in str(caught.value)


def test_hermes_client_ignores_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Checked directly: on Windows the platform may already bypass proxies for
    # loopback, which would hide a regression in the end-to-end test below.
    from local_call_assistant.integrations.hermes import _opener
    monkeypatch.setenv("HTTPS_PROXY", "http://203.0.113.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.1:9")
    proxies = [h.proxies for h in _opener().handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert all(not table for table in proxies)


def test_system_proxy_settings_are_ignored(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    reply = b'{"choices": [{"message": {"content": "ok"}}]}'
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    # A proxy that does not exist: going through it would fail the request.
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(name, "http://203.0.113.1:9")
    response = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(reply)).encode() + b"\r\n\r\n" + reply)
    for url in _raw_server(response):
        assert analyse(_config(url), "job", "hello") == "ok"


def test_key_with_control_characters_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "abc\r\nX-Injected: 1")
    with pytest.raises(HermesError, match="invalid characters") as caught:
        analyse(_config("http://127.0.0.1:9"), "job", "hello")
    assert "abc" not in str(caught.value)
