import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from conftest import FakeTranscriber

from local_call_assistant.config import HermesConfig, ObsidianConfig, TranscriptionConfig
from local_call_assistant.integrations.hermes import HermesError, analysis_handler, check_endpoint
from local_call_assistant.integrations.obsidian import note_handler
from local_call_assistant.models import Manifest, StageStatus
from local_call_assistant.processing import Pipeline
from local_call_assistant.storage.spool import RecordingSpool
from local_call_assistant.transcription import transcription_handler

KEY_ENV = "LCA_TEST_HERMES_KEY"
ANALYSIS = "## Summary\nShip on Friday.\n\n## Decisions\nShip on Friday."


class MockHermes(BaseHTTPRequestHandler):
    received: list[dict] = []
    status = 200

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers["Content-Length"])
        MockHermes.received.append({"path": self.path,
                                    "auth": self.headers.get("Authorization"),
                                    "body": json.loads(self.rfile.read(length))})
        reply = {"choices": [{"message": {"role": "assistant", "content": ANALYSIS}}]}
        data = json.dumps(reply).encode()
        self.send_response(MockHermes.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def hermes(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    MockHermes.received, MockHermes.status = [], 200
    monkeypatch.setenv(KEY_ENV, "test-key")
    server = HTTPServer(("127.0.0.1", 0), MockHermes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _pipeline(spool: RecordingSpool, base_url: str, inbox: Path) -> Pipeline:
    hermes = HermesConfig(enabled=True, base_url=base_url, api_key_env=KEY_ENV)
    return Pipeline(spool, {
        "transcription": transcription_handler(TranscriptionConfig(enabled=True), FakeTranscriber()),
        "analysis": analysis_handler(hermes),
        "note": note_handler(ObsidianConfig(inbox=inbox)),
    }, max_attempts=3)


def test_transcript_text_only_is_sent_and_analysis_lands_in_the_note(
        spool: RecordingSpool, recorded: Manifest, hermes: str, tmp_path: Path) -> None:
    manifest = _pipeline(spool, hermes, tmp_path / "inbox").process(recorded.job_id)
    assert manifest.stage("analysis").status is StageStatus.DONE
    [request] = MockHermes.received
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == "Bearer test-key"
    assert request["body"]["stream"] is False
    user_message = request["body"]["messages"][1]["content"]
    assert "Anna will send the invoice." in user_message
    note = Path(manifest.stage("note").outputs["note_path"]).read_text(encoding="utf-8")
    assert "Ship on Friday." in note


def test_hermes_failure_keeps_transcript_and_holds_the_note(
        spool: RecordingSpool, recorded: Manifest, hermes: str, tmp_path: Path) -> None:
    MockHermes.status = 500
    manifest = _pipeline(spool, hermes, tmp_path / "inbox").process(recorded.job_id)
    assert manifest.stage("analysis").status is StageStatus.FAILED
    assert manifest.stage("analysis").attempts[-1].error == "HermesError: Hermes returned HTTP 500"
    assert manifest.stage("note").status is StageStatus.PENDING
    assert (spool.root / manifest.stage("transcription").outputs["transcript_json"]).exists()


def test_missing_key_fails_without_sending(
        spool: RecordingSpool, recorded: Manifest, hermes: str, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV)
    manifest = _pipeline(spool, hermes, tmp_path / "inbox").process(recorded.job_id)
    assert "is not set" in manifest.stage("analysis").attempts[-1].error
    assert MockHermes.received == []


@pytest.mark.parametrize("url", ["http://hermes.example.com", "ftp://127.0.0.1", "https://"])
def test_plain_http_is_refused_except_to_loopback(url: str) -> None:
    with pytest.raises(HermesError):
        check_endpoint(url)


def test_https_and_loopback_http_are_accepted() -> None:
    assert check_endpoint("https://hermes.example.com/").endswith("/v1/chat/completions")
    assert check_endpoint("http://localhost:8642").endswith("/v1/chat/completions")
