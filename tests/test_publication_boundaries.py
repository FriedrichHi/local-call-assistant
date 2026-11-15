"""Fault injection at publication boundaries (round-3 review)."""

from pathlib import Path

import pytest
from conftest import FakeTranscriber

from local_call_assistant.config import HermesConfig, ObsidianConfig, TranscriptionConfig
from local_call_assistant.integrations import obsidian as obsidian_module
from local_call_assistant.integrations.hermes import analysis_handler
from local_call_assistant.integrations.obsidian import note_handler
from local_call_assistant.models import Manifest, StageStatus
from local_call_assistant.processing import Pipeline
from local_call_assistant.storage import spool as spool_module
from local_call_assistant.storage.spool import RecordingSpool
from local_call_assistant.transcription import transcription_handler


class SimulatedCrash(Exception):
    pass


def _pipeline(spool: RecordingSpool, inbox: Path) -> Pipeline:
    return Pipeline(spool, {
        "transcription": transcription_handler(TranscriptionConfig(enabled=True), FakeTranscriber()),
        "analysis": analysis_handler(HermesConfig(enabled=False)),
        "note": note_handler(ObsidianConfig(inbox=inbox)),
    }, max_attempts=3)


def _crash_when_note_is_marked_done(spool: RecordingSpool,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    real_save = spool.save

    def save(manifest: Manifest) -> None:
        if manifest.stage("note").status is StageStatus.DONE:
            raise SimulatedCrash("process died after publishing the note")
        real_save(manifest)

    monkeypatch.setattr(spool, "save", save)


def test_crash_after_note_publication(spool: RecordingSpool, recorded: Manifest,
                                      tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inbox = tmp_path / "inbox"
    _crash_when_note_is_marked_done(spool, monkeypatch)
    with pytest.raises(SimulatedCrash):
        _pipeline(spool, inbox).process(recorded.job_id)
    monkeypatch.undo()
    [published] = list(inbox.glob("*.md"))

    pipeline = _pipeline(spool, inbox)
    assert pipeline.recover_stages() == [recorded.job_id]
    assert spool.load(recorded.job_id).stage("note").status is StageStatus.FAILED

    # Resetting the note first must not open the way to a stale upstream retry.
    pipeline.retry(recorded.job_id, "note")
    with pytest.raises(ValueError, match="note already ran"):
        pipeline.retry(recorded.job_id, "analysis")

    synced: list[Path] = []
    monkeypatch.setattr(obsidian_module, "fsync_directory", synced.append)
    manifest = pipeline.process(recorded.job_id)
    assert manifest.stage("note").status is StageStatus.DONE
    assert Path(manifest.stage("note").outputs["note_path"]) == published
    assert list(inbox.glob("*.md")) == [published]  # reused, not duplicated
    assert synced == [inbox]  # durability finished before the note was accepted


def test_skipped_dependent_does_not_block_retry(spool: RecordingSpool, recorded: Manifest) -> None:
    # No inbox configured: the note stage was only ever skipped, nothing was published.
    pipeline = Pipeline(spool, {
        "transcription": transcription_handler(TranscriptionConfig(enabled=True), FakeTranscriber()),
        "analysis": analysis_handler(HermesConfig(enabled=False)),
        "note": note_handler(ObsidianConfig(inbox=None)),
    }, max_attempts=3)
    pipeline.process(recorded.job_id)
    assert spool.load(recorded.job_id).stage("note").status is StageStatus.SKIPPED
    pipeline.retry(recorded.job_id, "analysis")  # allowed
    assert spool.load(recorded.job_id).stage("analysis").status is StageStatus.PENDING


def test_raw_file_name_is_synced_before_recording_starts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spool = RecordingSpool(tmp_path)
    synced: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(spool_module, "fsync_directory",
                        lambda d: synced.append((d, sorted(p.name for p in d.iterdir()))))
    manifest = spool.start(sample_rate=16_000, channels=1)
    raw_name = spool.raw_path(manifest.job_id).name
    assert (spool.pending, [raw_name]) in synced
    spool.finalize(manifest.job_id)
