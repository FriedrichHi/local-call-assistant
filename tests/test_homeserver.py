import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from conftest import FakeTranscriber

from local_call_assistant import cli
from local_call_assistant.config import HomeserverConfig, TranscriptionConfig, load_config
from local_call_assistant.integrations import homeserver, homeserver_server
from local_call_assistant.integrations.homeserver import (
    HomeserverError, homeserver_handler, transcribe_remote,
)
from local_call_assistant.models import Manifest, StageStatus
from local_call_assistant.processing import Pipeline
from local_call_assistant.storage.spool import RecordingSpool, sha256_of
from local_call_assistant.transcription import (
    FasterWhisperBackend, Segment, Transcript, transcription_handler,
)

FAKE_SSH = str(Path(__file__).with_name("fake_ssh.py"))
SECRET = "server-secret-quote"
REMOTE = "/srv/lca/.venv/bin/lca"
WATCHDOG_SECONDS = 15
TRANSCRIPTION = TranscriptionConfig(enabled=True, model="small", language="de")


@pytest.fixture
def fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route ssh to the fake: argv becomes [python, fake_ssh.py, <the real options...>]."""
    record = tmp_path / "ssh-record.json"
    monkeypatch.setenv("FAKE_SSH_RECORD", str(record))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "server-cache"))
    monkeypatch.setattr(homeserver, "SSH_SAFETY_OPTIONS", (FAKE_SSH, *homeserver.SSH_SAFETY_OPTIONS))
    return record


def _resolve(name: str) -> str:
    return sys.executable


def _config(**overrides: object) -> HomeserverConfig:
    values = {"enabled": True, "host": "homeserver", "remote_command": REMOTE,
              "timeout_seconds": 20.0}
    values.update(overrides)
    return HomeserverConfig(**values)


def _pipeline(spool: RecordingSpool, config: HomeserverConfig | None = None,
              resolve=_resolve) -> Pipeline:
    return Pipeline(spool, {
        "transcription": homeserver_handler(TRANSCRIPTION, config or _config(), resolve),
        "analysis": lambda context: {}, "note": lambda context: {},
    }, max_attempts=3)


def test_transcript_comes_back_from_the_server(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path) -> None:
    manifest = _pipeline(spool).process(recorded.job_id)
    stage = manifest.stage("transcription")
    assert stage.status is StageStatus.DONE and stage.outputs["where"] == "homeserver"
    text = (spool.root / stage.outputs["transcript_text"]).read_text(encoding="utf-8")
    size = spool.path_of(recorded).stat().st_size
    assert f"heard {size} bytes" in text  # the server read the whole, rewound file
    seen = json.loads(fake_ssh.read_text(encoding="utf-8"))
    assert seen["stdin_sha256"] == recorded.sha256
    argv = seen["argv"]
    for option in ("BatchMode=yes", "ForwardAgent=no", "ClearAllForwardings=yes",
                   "PermitLocalCommand=no", "RemoteCommand=none"):
        assert argv[argv.index(option) - 1] == "-o"
    assert argv[argv.index("--") + 1] == "homeserver"
    assert argv[-1].startswith(f"{REMOTE} transcribe-stdin --sha256 {recorded.sha256} ")


def test_remote_command_is_shell_quoted(fake_ssh: Path) -> None:
    command = homeserver.build_command("ssh", _config(), TRANSCRIPTION, "a" * 64)
    assert command[command.index("--") + 1:] == [
        "homeserver", f"{REMOTE} transcribe-stdin --sha256 {'a' * 64} --model small --device cpu "
                      "--compute-type int8 --timeout 18 --language de"]


def test_remote_values_are_quoted_even_if_validation_were_bypassed() -> None:
    hostile = TranscriptionConfig(enabled=True, model="small", language="de;touch x")
    remote = homeserver.build_command("ssh", _config(), hostile, "a" * 64)[-1]
    assert remote.endswith("--language 'de;touch x'")


def test_a_file_swapped_after_the_containment_check_is_not_sent(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other.wav"
    other.write_bytes(b"x")
    audio_path, real_stat = spool.path_of(recorded), os.stat
    monkeypatch.setattr(homeserver.os, "stat",
                        lambda path, **kw: real_stat(other if path == audio_path else path, **kw))
    with pytest.raises(HomeserverError, match="changed"):
        transcribe_remote(_config(), TRANSCRIPTION, audio_path, recorded.sha256, _resolve)
    assert not fake_ssh.exists()


def test_ssh_batch_launchers_are_never_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(homeserver.shutil, "which", lambda name: r"C:\tools\ssh.CMD")
    with pytest.raises(HomeserverError, match=r"\.cmd/\.bat launcher"):
        homeserver.resolve_ssh("ssh")


def test_server_deadline_is_below_the_client_timeout() -> None:
    assert homeserver.remote_timeout(_config(timeout_seconds=3600)) == 3540
    assert homeserver.remote_timeout(_config(timeout_seconds=20)) == 18


@pytest.mark.parametrize("mode, message", [
    ("garbage", "not a valid transcript"),
    ("wrong-sha", "not a valid transcript"),
    ("exit:255", "could not connect"),
    ("exit:4", "busy"),
    ("exit:7", "exit 7"),
])
def test_failures_keep_the_audio_and_are_retryable(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path,
        monkeypatch: pytest.MonkeyPatch, mode: str, message: str) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", mode)
    manifest = _pipeline(spool).process(recorded.job_id)
    stage = manifest.stage("transcription")
    assert stage.status is StageStatus.FAILED
    assert message in stage.attempts[-1].error and SECRET not in stage.attempts[-1].error
    assert sha256_of(spool.path_of(recorded)) == recorded.sha256
    monkeypatch.setenv("FAKE_SSH_MODE", "serve")
    assert _pipeline(spool).process(recorded.job_id).stage("transcription").status is StageStatus.DONE


def test_oversized_response_is_rejected(
        recorded: Manifest, spool: RecordingSpool, fake_ssh: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(homeserver, "MAX_RESPONSE_BYTES", 1000)
    monkeypatch.setenv("FAKE_SSH_MODE", "huge")
    monkeypatch.setenv("FAKE_SSH_HUGE_BYTES", "5000")
    with pytest.raises(HomeserverError, match="too large"):
        transcribe_remote(_config(), TRANSCRIPTION, spool.path_of(recorded), recorded.sha256, _resolve)


def test_a_server_that_keeps_writing_is_stopped_early(
        recorded: Manifest, spool: RecordingSpool, fake_ssh: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(homeserver, "MAX_RESPONSE_BYTES", 1024 * 1024)
    monkeypatch.setenv("FAKE_SSH_MODE", "stream")
    outcome: list[BaseException] = []

    def call() -> None:
        try:
            transcribe_remote(_config(timeout_seconds=600.0), TRANSCRIPTION,
                              spool.path_of(recorded), recorded.sha256, _resolve)
        except BaseException as error:  # noqa: BLE001 - handed to the main thread
            outcome.append(error)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(WATCHDOG_SECONDS)  # long before the 600 s timeout
    assert not worker.is_alive(), "a server that keeps writing was not stopped"
    assert isinstance(outcome[0], HomeserverError) and "too large" in str(outcome[0])


def _envelope(**changes: object) -> bytes:
    document = {"protocol": 1, "sha256": "a" * 64, "transcript": {
        "language": "de", "duration": 2.0, "model": "small", "device": "cpu",
        "compute_type": "int8", "segments": [{"start": 0.0, "end": 2.0, "text": " Hallo"}]}}
    segment_changes = changes.pop("segment", {})
    document["transcript"]["segments"][0].update(segment_changes)
    document.update(changes)
    return json.dumps(document).encode("utf-8")


def test_a_valid_response_parses() -> None:
    assert homeserver._parse(_envelope(), "a" * 64).segments == [Segment(0.0, 2.0, " Hallo")]


@pytest.mark.parametrize("changes", [
    {"protocol": True}, {"protocol": 1.0}, {"protocol": "1"}, {"protocol": 2},
    {"segment": {"start": True}}, {"segment": {"end": float("nan")}},
    {"segment": {"start": -1.0}}, {"segment": {"start": 3.0}}, {"segment": {"text": 5}},
    {"transcript": []}, {"transcript": {"segments": "x"}},
])
def test_malformed_responses_are_rejected(changes: dict) -> None:
    with pytest.raises(HomeserverError, match="not a valid transcript"):
        homeserver._parse(_envelope(**changes), "a" * 64)


def test_cpu_retry_starts_from_the_beginning_of_the_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FasterWhisperBackend(TranscriptionConfig(enabled=True, device="auto"))
    reads: list[bytes] = []

    def run(audio, device, compute_type):
        reads.append(audio.read())
        if device == "cuda":
            raise RuntimeError("cuBLAS missing")
        return Transcript(None, None, [], "small", device, compute_type)

    monkeypatch.setattr(backend, "_run", run)
    backend.transcribe(io.BytesIO(b"audio"))
    assert reads == [b"audio", b"audio"]


def _redirect_directory(directory: Path, target: Path) -> None:
    """Replace a folder with a link to another one (a junction on Windows)."""
    shutil.copytree(directory, target)
    shutil.rmtree(directory)
    try:
        directory.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(directory), str(target)],
                       check=True, capture_output=True)


@pytest.mark.parametrize("redirected", ["recordings", "recordings/completed"])
def test_redirected_spool_folders_are_refused(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path, tmp_path: Path,
        redirected: str) -> None:
    _redirect_directory(spool.root / redirected, tmp_path / "elsewhere")
    manifest = _pipeline(spool).process(recorded.job_id)
    stage = manifest.stage("transcription")
    assert stage.status is StageStatus.FAILED and "not inside the spool" in stage.attempts[-1].error
    assert len(stage.attempts) == 1  # an ordinary failed attempt, retryable within the budget
    assert not fake_ssh.exists()


def test_timeout_holds_when_ssh_never_reads(
        spool: RecordingSpool, fake_ssh: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = spool.start(sample_rate=16_000, channels=1)
    spool.append(manifest.job_id, b"\x00\x01" * 2_000_000)  # 4 MB, beyond any pipe buffer
    manifest = spool.finalize(manifest.job_id)
    monkeypatch.setenv("FAKE_SSH_MODE", "never-reads")
    outcome: list[BaseException] = []

    def call() -> None:
        try:
            transcribe_remote(_config(timeout_seconds=1.0), TRANSCRIPTION,
                              spool.path_of(manifest), manifest.sha256, _resolve)
        except BaseException as error:  # noqa: BLE001 - handed to the main thread
            outcome.append(error)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(WATCHDOG_SECONDS)
    assert not worker.is_alive(), "transcribe_remote blocked past its timeout"
    assert isinstance(outcome[0], HomeserverError) and "timeout" in str(outcome[0])


@pytest.mark.parametrize("audio_file", [
    "C:/Windows/win.ini", "/etc/passwd", "recordings/completed/../../manifests/x.json",
    "recordings/pending/{job}.wav", "recordings/completed/sub/{job}.wav",
    "recordings/completed/..\\{job}.wav", "recordings/completed/{job}.wav:secret",
    "recordings/completed/con.wav", "recordings/completed/" + "a" * 32 + ".wav",
])
def test_only_spooled_audio_can_be_sent(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path, audio_file: str) -> None:
    audio_file = audio_file.format(job=recorded.job_id)
    original = spool.path_of(recorded)
    tampered = Manifest.from_dict({**recorded.to_dict(), "audio_file": audio_file})
    spool.save(tampered)
    manifest = _pipeline(spool).process(recorded.job_id)
    stage = manifest.stage("transcription")
    assert stage.status is StageStatus.FAILED and "not inside the spool" in stage.attempts[-1].error
    assert len(stage.attempts) == 1  # an ordinary failed attempt, retryable within the budget
    assert not fake_ssh.exists()  # ssh was never started
    assert sha256_of(original) == recorded.sha256


def test_symlinked_audio_is_refused(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path, tmp_path: Path) -> None:
    outside = tmp_path / "private.wav"
    outside.write_bytes(b"private")
    link = spool.completed / f"{recorded.job_id}.flac"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks needs extra rights on this system")
    spool.save(Manifest.from_dict({**recorded.to_dict(), "audio_file": f"recordings/completed/{recorded.job_id}.flac"}))
    manifest = _pipeline(spool).process(recorded.job_id)
    assert manifest.stage("transcription").status is StageStatus.FAILED
    assert not fake_ssh.exists()


def test_symlink_to_another_jobs_audio_inside_the_spool_is_refused(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path) -> None:
    other = spool.start(sample_rate=16_000, channels=1)
    spool.append(other.job_id, b"\x00\x02" * 100)
    other = spool.finalize(other.job_id)
    link = spool.completed / f"{recorded.job_id}.flac"
    try:
        link.symlink_to(spool.path_of(other))
    except OSError:
        pytest.skip("creating symlinks needs extra rights on this system")
    spool.save(Manifest.from_dict({**recorded.to_dict(), "sha256": other.sha256,
                                   "audio_file": f"recordings/completed/{recorded.job_id}.flac"}))
    manifest = _pipeline(spool).process(recorded.job_id)
    stage = manifest.stage("transcription")
    assert stage.status is StageStatus.FAILED and "not inside the spool" in stage.attempts[-1].error
    assert not fake_ssh.exists()


def test_audio_changed_after_verification_is_not_sent(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path) -> None:
    with pytest.raises(HomeserverError, match="changed"):
        transcribe_remote(_config(), TRANSCRIPTION, spool.path_of(recorded), "f" * 64, _resolve)
    assert not fake_ssh.exists()


# ----- offline independence --------------------------------------------

def _homeserver_config_file(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(
        '[transcription]\nenabled = true\nmodel = "small"\n'
        '[homeserver]\nenabled = true\nhost = "homeserver"\n'
        f'ssh_command = "no-such-ssh-binary-xyz"\nremote_command = "{REMOTE}"\n',
        encoding="utf-8")


def test_everything_but_transcription_works_without_ssh(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runtime"
    _homeserver_config_file(root)
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)
    os.utime(audio, (0, 0))
    home = ["--home", str(root)]
    assert cli.main([*home, "ingest", str(audio)]) == 0
    assert cli.main([*home, "recover"]) == 0
    assert cli.main([*home, "status"]) == 0
    spool = RecordingSpool(root)
    session = spool.start(sample_rate=16_000, channels=1)
    spool.append(session.job_id, b"\x00\x01" * 100)
    assert spool.finalize(session.job_id).sha256
    for _ in range(3):
        assert cli.main([*home, "process"]) == 1  # stages failed, and that is reported
    for manifest in spool.iter_manifests():
        stage = manifest.stage("transcription")
        assert stage.status is StageStatus.FAILED and len(stage.attempts) == 3
        assert "ssh command was not found" in stage.attempts[-1].error
        assert sha256_of(spool.path_of(manifest)) == manifest.sha256


def test_retry_after_the_budget_is_used_up(
        spool: RecordingSpool, recorded: Manifest, fake_ssh: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "exit:255")
    for _ in range(4):
        _pipeline(spool).process(recorded.job_id)
    pipeline = _pipeline(spool)
    assert len(spool.load(recorded.job_id).stage("transcription").attempts) == 3
    monkeypatch.setenv("FAKE_SSH_MODE", "serve")
    pipeline.retry(recorded.job_id, "transcription")
    assert pipeline.process(recorded.job_id).stage("transcription").status is StageStatus.DONE


def test_local_transcription_is_unchanged_when_homeserver_is_off(
        spool: RecordingSpool, recorded: Manifest) -> None:
    pipeline = Pipeline(spool, {
        "transcription": transcription_handler(TRANSCRIPTION, FakeTranscriber()),
        "analysis": lambda context: {}, "note": lambda context: {},
    }, max_attempts=3)
    assert pipeline.process(recorded.job_id).stage("transcription").outputs["where"] == "local"


# ----- server --------------------------------------------------------------

class RecordingBackend:
    seen: dict = {}

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config

    def transcribe(self, audio) -> Transcript:
        RecordingBackend.seen = {
            "content": audio.read(), "config": self.config,
            "cache_entries": sorted(p.name for p in homeserver_server.cache_dir().iterdir()),
        }
        return Transcript(language="de", duration=2.0, segments=[Segment(0.0, 2.0, " Hallo")],
                          model=self.config.model, device=self.config.device,
                          compute_type=self.config.compute_type)


def _server_args(audio: bytes, **overrides: object) -> argparse.Namespace:
    import hashlib
    values = {"sha256": hashlib.sha256(audio).hexdigest(), "model": "small", "language": "de",
              "device": "cpu", "compute_type": "int8", "timeout": 60.0}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def server_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(homeserver_server, "_arm_deadline", lambda seconds: None)
    return tmp_path / "cache" / homeserver_server.CACHE_SUBDIR


def test_server_transcribes_and_keeps_nothing(server_cache: Path) -> None:
    audio = b"RIFF audio"
    code, document = homeserver_server.serve(_server_args(audio), io.BytesIO(audio), RecordingBackend)
    assert code == 0 and document["sha256"] == _server_args(audio).sha256
    assert RecordingBackend.seen["content"] == audio  # rewound before decoding
    assert RecordingBackend.seen["config"].device == "cpu"
    assert document["transcript"]["segments"] == [{"start": 0.0, "end": 2.0, "text": " Hallo"}]
    assert sorted(p.name for p in server_cache.iterdir()) == [homeserver_server.LOCK_NAME]


@pytest.mark.skipif(os.name == "nt", reason="Windows temporary files keep a name until closed")
def test_server_audio_has_no_name_while_transcribing(server_cache: Path) -> None:
    audio = b"RIFF audio"
    homeserver_server.serve(_server_args(audio), io.BytesIO(audio), RecordingBackend)
    assert RecordingBackend.seen["cache_entries"] == [homeserver_server.LOCK_NAME]


@pytest.mark.skipif(os.name == "nt", reason="Windows temporary files keep a name until closed")
def test_server_refuses_to_write_audio_into_a_named_file(
        server_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile
    monkeypatch.setattr(homeserver_server.tempfile, "TemporaryFile",
                        lambda dir: tempfile.NamedTemporaryFile(dir=dir))
    audio = b"x"
    assert homeserver_server.serve(_server_args(audio), io.BytesIO(audio), RecordingBackend) == (6, None)


def test_server_refuses_different_audio(server_cache: Path) -> None:
    args = _server_args(b"what the client hashed")
    assert homeserver_server.serve(args, io.BytesIO(b"what arrived"), RecordingBackend) == (3, None)


def test_server_refuses_oversized_audio(server_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(homeserver_server, "MAX_AUDIO_BYTES", 4)
    audio = b"too long"
    assert homeserver_server.serve(_server_args(audio), io.BytesIO(audio), RecordingBackend) == (5, None)


def test_server_runs_one_transcription_at_a_time(server_cache: Path) -> None:
    from local_call_assistant.storage.filelock import JobLock
    lock = JobLock(server_cache / homeserver_server.LOCK_NAME)
    assert lock.acquire()
    try:
        audio = b"x"
        assert homeserver_server.serve(_server_args(audio), io.BytesIO(audio), RecordingBackend) == (4, None)
    finally:
        lock.release()


def test_server_backend_failure_is_a_fixed_exit_code(server_cache: Path) -> None:
    class Broken(RecordingBackend):
        def transcribe(self, audio) -> Transcript:
            raise RuntimeError("model exploded")
    audio = b"x"
    assert homeserver_server.serve(_server_args(audio), io.BytesIO(audio), Broken) == (6, None)


@pytest.mark.skipif(os.name == "nt", reason="SIGALRM exists on POSIX only")
def test_server_deadline_ends_a_stalled_call_and_frees_the_lock(tmp_path: Path) -> None:
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    command = [sys.executable, "-m", "local_call_assistant", "transcribe-stdin",
               "--sha256", "a" * 64, "--model", "small", "--timeout", "1"]
    stalled = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    try:
        stalled.stdin.write(b"partial upload, never finished")
        stalled.stdin.flush()
        assert stalled.wait(timeout=WATCHDOG_SECONDS) < 0  # ended by its own alarm
    finally:
        stalled.kill()
        stalled.stdin.close()
        stalled.stdout.close()
    second = subprocess.run(command, input=b"x", capture_output=True, env=env, timeout=60, check=False)
    assert second.returncode == 3  # the lock was free again; reached the checksum check


@pytest.mark.parametrize("override", [
    {"sha256": "A" * 64}, {"sha256": "abc"}, {"model": "Systran/faster-whisper-small"},
    {"model": "../models"}, {"language": "de;id"}, {"device": "tpu"},
    {"compute_type": "fast"}, {"timeout": float("inf")}, {"timeout": 0.0}, {"timeout": 1e9},
])
def test_server_revalidates_every_argument(server_cache: Path, override: dict) -> None:
    audio = b"x"
    assert homeserver_server.serve(_server_args(audio, **override), io.BytesIO(audio),
                                   RecordingBackend) == (2, None)


def test_server_entry_point_reads_no_configuration(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "config.toml").write_text("this is [not valid toml", encoding="utf-8")
    env = {**os.environ, "LCA_HOME": str(root), "XDG_CACHE_HOME": str(tmp_path / "cache")}
    completed = subprocess.run(
        [sys.executable, "-m", "local_call_assistant", "transcribe-stdin", "--sha256", "a" * 64,
         "--model", "small", "--timeout", "60"],
        input=b"not the hashed audio", capture_output=True, env=env, timeout=60, check=False)
    assert completed.returncode == 3  # reached the checksum check; the bad config was never read
    assert completed.stdout == b""


# ----- configuration ---------------------------------------------------------

def _toml(transcription: str = "enabled = true", **fields: str) -> str:
    values = {"enabled": "true", "host": '"hs"', "remote_command": f'"{REMOTE}"', **fields}
    lines = [f"{key} = {value}" for key, value in values.items()]
    return f"[transcription]\n{transcription}\n[homeserver]\n" + "\n".join(lines) + "\n"


@pytest.mark.parametrize("field, value, message", [
    ("host", '"-oProxyCommand=x"', "host"),
    ("host", '"a b"', "host"),
    ("ssh_command", '["ssh", "otherhost"]', "one executable"),
    ("ssh_command", '"ssh otherhost"', "one executable"),
    ("ssh_command", '"-x"', "one executable"),
    ("remote_command", '"lca"', "absolute path"),
    ("remote_command", '"~/lca"', "absolute path"),
    ("remote_command", '"/srv/lca; rm -rf ~"', "absolute path"),
    ("remote_command", '"/srv/../etc/lca"', "absolute path"),
    ("timeout_seconds", "-1", "positive number"),
    ("timeout_seconds", "1e9", "at most one day"),
    ("ssh_command", '"my ssh"', "one executable"),
    ("mode", '"fallback"', "unknown"),
])
def test_invalid_homeserver_config_is_rejected(
        tmp_path: Path, field: str, value: str, message: str) -> None:
    (tmp_path / "config.toml").write_text(_toml(**{field: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(tmp_path)


def test_homeserver_needs_transcription_enabled(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(_toml(transcription="enabled = false"), encoding="utf-8")
    with pytest.raises(ValueError, match="transcription"):
        load_config(tmp_path)


def test_remote_model_must_be_a_plain_name_but_local_may_be_a_path(tmp_path: Path) -> None:
    repo_model = 'enabled = true\nmodel = "Systran/faster-whisper-small"'
    (tmp_path / "config.toml").write_text(f"[transcription]\n{repo_model}\n", encoding="utf-8")
    assert load_config(tmp_path).transcription.model == "Systran/faster-whisper-small"
    (tmp_path / "config.toml").write_text(_toml(transcription=repo_model), encoding="utf-8")
    with pytest.raises(ValueError, match="plain name"):
        load_config(tmp_path)


def test_absolute_ssh_path_may_contain_spaces(tmp_path: Path) -> None:
    path = r"C:\Program Files\OpenSSH\ssh.exe" if os.name == "nt" else "/opt/open ssh/ssh"
    toml_value = path.replace("\\", "\\\\")
    (tmp_path / "config.toml").write_text(_toml(ssh_command=f'"{toml_value}"'), encoding="utf-8")
    assert load_config(tmp_path).homeserver.ssh_command == path


def test_valid_homeserver_config(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(_toml(), encoding="utf-8")
    config = load_config(tmp_path).homeserver
    assert config.enabled and config.host == "hs" and config.ssh_command == "ssh"
