from pathlib import Path

from conftest import FakeTranscriber

from local_call_assistant.config import ObsidianConfig, TranscriptionConfig, HermesConfig
from local_call_assistant.integrations.hermes import analysis_handler
from local_call_assistant.integrations.obsidian import note_handler
from local_call_assistant.models import Manifest, StageStatus
from local_call_assistant.processing import Pipeline
from local_call_assistant.storage.spool import RecordingSpool, sha256_of
from local_call_assistant.transcription import transcription_handler


def _pipeline(spool: RecordingSpool, transcriber: FakeTranscriber, *, enabled: bool = True,
              inbox: Path | None = None, max_attempts: int = 3) -> Pipeline:
    handlers = {
        "transcription": transcription_handler(TranscriptionConfig(enabled=enabled), transcriber),
        "analysis": analysis_handler(HermesConfig(enabled=False)),
        "note": note_handler(ObsidianConfig(inbox=inbox)),
    }
    return Pipeline(spool, handlers, max_attempts)


def test_full_local_flow_writes_transcript_and_draft_note(
        spool: RecordingSpool, recorded: Manifest, tmp_path: Path) -> None:
    inbox = tmp_path / "vault" / "Inbox"
    manifest = _pipeline(spool, FakeTranscriber(), inbox=inbox).process(recorded.job_id)
    assert manifest.stage("transcription").status is StageStatus.DONE
    assert manifest.stage("analysis").status is StageStatus.SKIPPED
    assert manifest.stage("note").status is StageStatus.DONE
    note = Path(manifest.stage("note").outputs["note_path"])
    text = note.read_text(encoding="utf-8")
    assert "status: draft" in text and f"job_id: {recorded.job_id}" in text
    assert "- [00:01:05] Anna will send the invoice." in text
    assert (spool.root / manifest.stage("transcription").outputs["transcript_text"]).exists()


def test_failure_keeps_audio_records_attempts_and_retries(
        spool: RecordingSpool, recorded: Manifest) -> None:
    audio_before = sha256_of(spool.path_of(recorded))
    transcriber = FakeTranscriber(failures=1)
    pipeline = _pipeline(spool, transcriber)
    first = pipeline.process(recorded.job_id)
    stage = first.stage("transcription")
    assert stage.status is StageStatus.FAILED
    assert stage.attempts[-1].error == "RuntimeError: model crashed"
    assert sha256_of(spool.path_of(recorded)) == audio_before
    assert pipeline.process(recorded.job_id).stage("transcription").status is StageStatus.DONE
    assert transcriber.calls == 2


def test_attempt_budget_stops_retries_until_manual_reset(
        spool: RecordingSpool, recorded: Manifest) -> None:
    transcriber = FakeTranscriber(failures=5)
    pipeline = _pipeline(spool, transcriber, max_attempts=2)
    for _ in range(4):
        pipeline.process(recorded.job_id)
    assert transcriber.calls == 2
    pipeline.retry(recorded.job_id, "transcription")
    pipeline.process(recorded.job_id)
    assert transcriber.calls == 3


def test_disabled_transcription_is_skipped_and_recording_unaffected(
        spool: RecordingSpool, recorded: Manifest) -> None:
    transcriber = FakeTranscriber()
    manifest = _pipeline(spool, transcriber, enabled=False).process(recorded.job_id)
    assert manifest.stage("transcription").status is StageStatus.SKIPPED
    assert manifest.stage("note").status is StageStatus.PENDING
    assert transcriber.calls == 0
    assert spool.path_of(recorded).exists()


def test_changed_audio_is_not_processed(spool: RecordingSpool, recorded: Manifest) -> None:
    with spool.path_of(recorded).open("ab") as stream:
        stream.write(b"tampered")
    transcriber = FakeTranscriber()
    manifest = _pipeline(spool, transcriber).process(recorded.job_id)
    assert manifest.stage("transcription").status is StageStatus.FAILED
    assert "checksum" in manifest.stage("transcription").attempts[-1].error
    assert transcriber.calls == 0


def test_stage_left_running_by_a_crash_is_recovered(
        spool: RecordingSpool, recorded: Manifest) -> None:
    manifest = spool.load(recorded.job_id)
    stage = manifest.stage("transcription")
    stage.status = StageStatus.RUNNING
    from local_call_assistant.models import Attempt
    stage.attempts.append(Attempt(started_at="2026-11-15T00:00:00+00:00"))
    spool.save(manifest)
    pipeline = _pipeline(spool, FakeTranscriber())
    assert pipeline.recover_stages() == [recorded.job_id]
    reloaded = spool.load(recorded.job_id).stage("transcription")
    assert reloaded.status is StageStatus.FAILED
    assert reloaded.attempts[-1].outcome == "interrupted"
    assert pipeline.process(recorded.job_id).stage("transcription").status is StageStatus.DONE
