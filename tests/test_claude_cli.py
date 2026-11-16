import json
import sys
import threading
from pathlib import Path

import pytest
from conftest import FakeTranscriber

from local_call_assistant.config import (
    ClaudeConfig, ObsidianConfig, TranscriptionConfig, load_config,
)
from local_call_assistant.integrations import claude_cli
from local_call_assistant.integrations.claude_cli import ClaudeError, analyse, claude_handler
from local_call_assistant.integrations.obsidian import note_handler
from local_call_assistant.models import Manifest, StageStatus
from local_call_assistant.processing import Pipeline
from local_call_assistant.storage.spool import RecordingSpool, sha256_of
from local_call_assistant.transcription import transcription_handler

FAKE = str(Path(__file__).with_name("fake_claude.py"))
SECRET = "stderr-secret-quote"
WATCHDOG_SECONDS = 15


UNSAFE_VARIABLES = {
    "ANTHROPIC_API_KEY": "would-bill-the-api",
    "ANTHROPIC_AUTH_TOKEN": "would-bill-the-api",
    "ANTHROPIC_BASE_URL": "https://elsewhere.example",
    "ANTHROPIC_CUSTOM_HEADERS": "X-Token: abc",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "CLAUDE_CODE_USE_VERTEX": "1",
    "CLAUDE_CODE_PLUGIN_DIRS": "/tmp/plugins",
    "HTTPS_PROXY": "http://203.0.113.1:9",
}


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_CLAUDE_RECORD", str(path))
    for name, value in UNSAFE_VARIABLES.items():
        monkeypatch.setenv(name, value)
    # Let the test double's own control variables through the allowlist.
    monkeypatch.setattr(claude_cli, "ENV_ALLOWLIST",
                        claude_cli.ENV_ALLOWLIST | {"FAKE_CLAUDE_RECORD", "FAKE_CLAUDE_MODE"})
    return path


def _config(**overrides: object) -> ClaudeConfig:
    values = {"enabled": True, "command": (sys.executable, FAKE), "timeout_seconds": 20.0}
    values.update(overrides)
    return ClaudeConfig(**values)


def _pipeline(spool: RecordingSpool, config: ClaudeConfig, inbox: Path) -> Pipeline:
    return Pipeline(spool, {
        "transcription": transcription_handler(TranscriptionConfig(enabled=True), FakeTranscriber()),
        "analysis": claude_handler(config),
        "note": note_handler(ObsidianConfig(inbox=inbox)),
    }, max_attempts=3)


def test_analysis_reaches_the_note_through_an_isolated_call(
        spool: RecordingSpool, recorded: Manifest, record: Path, tmp_path: Path) -> None:
    manifest = _pipeline(spool, _config(model="sonnet"), tmp_path / "inbox").process(recorded.job_id)
    assert manifest.stage("analysis").status is StageStatus.DONE
    assert manifest.stage("analysis").outputs["provider"] == "claude-cli"
    seen = json.loads(record.read_text(encoding="utf-8"))
    # Allowlisted environment: nothing can switch billing, provider, endpoint,
    # headers, proxy or plugins.
    leaked = {k.upper() for k in seen["env_keys"]} & set(UNSAFE_VARIABLES)
    assert leaked == set()
    # The transcript arrives on stdin only, never as an argument.
    assert "Anna will send the invoice." in seen["stdin"]
    assert not any("Anna" in arg for arg in seen["argv"])
    # Isolation flags: no tools, no MCP, no settings/hooks, no saved session.
    argv = seen["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    for flag in ("-p", "--strict-mcp-config", "--no-session-persistence",
                 "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert seen["cwd_entries"] == []  # fresh empty working directory
    note = Path(manifest.stage("note").outputs["note_path"]).read_text(encoding="utf-8")
    assert "Ship on Friday." in note


@pytest.mark.parametrize("mode", ["is-error", "exit", "garbage"])
def test_failures_are_recorded_without_cli_output(
        spool: RecordingSpool, recorded: Manifest, record: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    audio_before = sha256_of(spool.path_of(recorded))
    manifest = _pipeline(spool, _config(), tmp_path / "inbox").process(recorded.job_id)
    stage = manifest.stage("analysis")
    assert stage.status is StageStatus.FAILED
    assert SECRET not in stage.attempts[-1].error
    assert manifest.stage("note").status is StageStatus.PENDING
    assert (spool.root / manifest.stage("transcription").outputs["transcript_json"]).exists()
    assert sha256_of(spool.path_of(recorded)) == audio_before


def test_timeout_is_reported(record: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    with pytest.raises(ClaudeError, match="timeout"):
        analyse(_config(timeout_seconds=1.0), "job", "hello")


def test_timeout_holds_when_the_child_never_reads_a_large_transcript(
        record: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "never-reads")
    outcome: list[BaseException] = []

    def call() -> None:
        try:
            analyse(_config(timeout_seconds=1.0), "job", "x" * 4_000_000)
        except BaseException as error:  # noqa: BLE001 - handed to the main thread
            outcome.append(error)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(WATCHDOG_SECONDS)  # external watchdog: a hang must not hang the suite
    assert not worker.is_alive(), "analyse blocked past its timeout"
    assert len(outcome) == 1 and isinstance(outcome[0], ClaudeError)
    assert "timeout" in str(outcome[0])


def test_missing_command_is_reported() -> None:
    with pytest.raises(ClaudeError, match="not found"):
        analyse(_config(command=("no-such-claude-binary-xyz",)), "job", "hello")


def test_disabled_claude_skips(spool: RecordingSpool, recorded: Manifest, tmp_path: Path) -> None:
    manifest = _pipeline(spool, _config(enabled=False), tmp_path / "inbox").process(recorded.job_id)
    assert manifest.stage("analysis").status is StageStatus.SKIPPED


def test_batch_launchers_are_never_executed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = tmp_path / "claude.cmd"
    launcher.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: str(launcher))
    with pytest.raises(ClaudeError, match=r"\.cmd/\.bat launcher"):
        claude_cli.resolve_executable("claude")


def test_npm_launcher_resolves_to_its_native_executable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = tmp_path / "claude.CMD"
    launcher.write_text("@echo off\n", encoding="utf-8")
    native = tmp_path / claude_cli.NPM_NATIVE_EXECUTABLE
    native.parent.mkdir(parents=True)
    native.write_bytes(b"MZ")
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: str(launcher))
    assert claude_cli.resolve_executable("claude") == str(native)


@pytest.mark.parametrize("snippet, message", [
    ('enabled = "false"', "true or false"),
    ('command = "claude"', "non-empty list"),
    ('command = []', "non-empty list"),
    ('command = ["claude", "--settings", "x.json"]', "must not contain options"),
    ('model = "sonnet&whoami"', "model name"),
    ('timeout_seconds = -1', "positive number"),
    ('timeout_seconds = inf', "positive number"),
    ('plugin_dir = "x"', "unknown"),
])
def test_invalid_claude_config_is_rejected(tmp_path: Path, snippet: str, message: str) -> None:
    (tmp_path / "config.toml").write_text(f"[claude]\n{snippet}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(tmp_path)


def test_hermes_enabled_must_be_a_boolean(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[hermes]\nenabled = "no"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="true or false"):
        load_config(tmp_path)


def test_claude_and_hermes_cannot_both_be_enabled(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        "[claude]\nenabled = true\n[hermes]\nenabled = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not both"):
        load_config(tmp_path)


def test_command_list_from_toml(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[claude]\nenabled = true\ncommand = ["claude"]\nmodel = "sonnet"\n', encoding="utf-8")
    config = load_config(tmp_path)
    assert config.claude.command == ("claude",) and config.claude.model == "sonnet"
