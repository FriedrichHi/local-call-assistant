import json
import os
import time
from pathlib import Path

import pytest

from local_call_assistant.cli import main


def test_ingest_status_and_process_through_the_cli(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = tmp_path / "home"
    source = tmp_path / "call.wav"
    source.write_bytes(b"RIFF fake")
    past = time.time() - 3_600
    os.utime(source, (past, past))
    assert main(["--home", str(home), "ingest", str(source)]) == 0
    assert "ingested" in capsys.readouterr().out
    assert main(["--home", str(home), "status", "--json"]) == 0
    [job] = json.loads(capsys.readouterr().out)
    assert job["status"] == "complete" and job["source_kind"] == "ingest"
    # Default config: transcription disabled, so processing skips it and succeeds.
    assert main(["--home", str(home), "process"]) == 0
    assert "transcription=skipped" in capsys.readouterr().out


def test_unknown_job_is_reported_not_raised(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--home", str(tmp_path), "retry", "nope"]) == 2
    assert "unknown job" in capsys.readouterr().err
