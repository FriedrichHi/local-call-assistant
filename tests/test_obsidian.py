from pathlib import Path

from local_call_assistant.integrations.obsidian import render_note, write_note
from local_call_assistant.models import Manifest
from local_call_assistant.transcription import Segment

SEGMENTS = [Segment(3_725.0, 3_727.0, " Late point.")]


def test_existing_files_are_never_overwritten(recorded: Manifest, tmp_path: Path) -> None:
    other = recorded.__class__.from_dict({**recorded.to_dict(), "job_id": recorded.job_id[:8] + "f" * 24})
    first = write_note(tmp_path, other, render_note(other, SEGMENTS, None, Path("t.json")))
    first_text = first.read_text(encoding="utf-8")
    # A different job whose name collides gets a numbered name instead.
    second = write_note(tmp_path, recorded, render_note(recorded, SEGMENTS, None, Path("t.json")))
    assert second != first and second.name.endswith("(2).md")
    assert first.read_text(encoding="utf-8") == first_text


def test_rewriting_the_same_job_reuses_its_note(recorded: Manifest, tmp_path: Path) -> None:
    text = render_note(recorded, SEGMENTS, None, Path("t.json"))
    first = write_note(tmp_path, recorded, text)
    again = write_note(tmp_path, recorded, text)
    assert again == first
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_note_is_a_draft_with_provenance(recorded: Manifest) -> None:
    text = render_note(recorded, SEGMENTS, None, Path("C:/runtime/transcripts/x.json"))
    assert text.startswith("---\ntype: call-note\nstatus: draft\n")
    assert f"recording_sha256: {recorded.sha256}" in text
    assert "- [01:02:05] Late point." in text
    assert "_No analysis yet." in text
