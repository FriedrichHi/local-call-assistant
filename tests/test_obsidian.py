from pathlib import Path

import pytest

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


def test_analysis_headings_nest_under_the_analysis_section(recorded: Manifest) -> None:
    analysis = "# Summary\nShort.\n## Detail\n```\n# not a heading\n```\n# Decisions"
    text = render_note(recorded, SEGMENTS, analysis, Path("t.json"))
    assert ("## Analysis\n\n### Summary\nShort.\n#### Detail\n```\n# not a heading\n```\n"
            "### Decisions") in text


@pytest.mark.parametrize("long, short", [("````", "```"), ("~~~~", "~~~"), ("```", "~~~")])
def test_longer_fences_are_not_closed_by_shorter_or_other_markers(
        recorded: Manifest, long: str, short: str) -> None:
    analysis = f"# Summary\n{long}\n{short}\n# code, not a heading\n{short}\n{long}\n# Decisions"
    text = render_note(recorded, SEGMENTS, analysis, Path("t.json"))
    assert (f"### Summary\n{long}\n{short}\n# code, not a heading\n{short}\n{long}\n"
            "### Decisions") in text


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_closing_fence_may_end_with_a_tab(recorded: Manifest, marker: str) -> None:
    text = render_note(recorded, SEGMENTS, f"# A\n{marker}\n# code\n{marker}\t\n# B", Path("t.json"))
    assert f"### A\n{marker}\n# code\n{marker}\t\n### B" in text


def test_backtick_line_with_backticks_after_is_not_a_fence(recorded: Manifest) -> None:
    text = render_note(recorded, SEGMENTS, "# A\n```x`y\n# B", Path("t.json"))
    assert "### A\n```x`y\n### B" in text


def test_already_deep_headings_stay_below_the_section(recorded: Manifest) -> None:
    text = render_note(recorded, SEGMENTS, "## Summary\n### Sub", Path("t.json"))
    assert "### Summary\n#### Sub" in text
