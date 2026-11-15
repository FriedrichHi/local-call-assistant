"""Write a draft call note into an Obsidian inbox folder.

Notes are draft-only and never overwrite anything: a name collision gets a
numbered suffix, and a job whose note was already written is not written
again.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from local_call_assistant.config import ObsidianConfig
from local_call_assistant.models import Manifest, StageStatus
from local_call_assistant.processing import StageContext, StageSkipped
from local_call_assistant.storage.atomic import create_new_text, fsync_directory
from local_call_assistant.transcription import Segment, load_segments

MAX_NAME_ATTEMPTS = 100
NO_ANALYSIS = "_No analysis yet. Hermes analysis is disabled or has not run._"


def render_note(manifest: Manifest, segments: list[Segment], analysis: str | None,
                transcript_path: Path) -> str:
    frontmatter = "\n".join([
        "---", "type: call-note", "status: draft",
        f"job_id: {manifest.job_id}", f"recorded_at: {manifest.started_at}",
        f"recording_sha256: {manifest.sha256}", f"source: {manifest.source_kind}",
        f"transcript: \"{transcript_path.as_posix()}\"", "---",
    ])
    lines = [f"- [{_clock(s.start)}] {s.text.strip()}" for s in segments]
    return "\n".join([
        frontmatter, "", f"# Call {_title_time(manifest.started_at)} (draft)", "",
        "## Analysis", "", (analysis or NO_ANALYSIS).strip(), "",
        "## Transcript", "", *lines, "",
    ])


def write_note(inbox: Path, manifest: Manifest, text: str) -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    stem = f"{_file_time(manifest.started_at)} call {manifest.job_id[:8]}"
    existing = _existing_note(inbox, manifest.job_id)
    if existing is not None:
        # Written before a crash stopped the manifest update: finish its durability.
        fsync_directory(inbox)
        return existing
    for index in range(MAX_NAME_ATTEMPTS):
        candidate = inbox / (f"{stem}.md" if index == 0 else f"{stem} ({index + 1}).md")
        if create_new_text(candidate, text):
            return candidate
    raise FileExistsError(f"no free note name for job {manifest.job_id}")


def note_handler(config: ObsidianConfig):
    def handle(context: StageContext) -> dict[str, Any]:
        if config.inbox is None:
            raise StageSkipped("no Obsidian inbox is configured")
        root, manifest = context.spool.root, context.manifest
        transcription = manifest.stage("transcription").outputs
        analysis_stage = manifest.stage("analysis")
        analysis = None
        if analysis_stage.status is StageStatus.DONE:
            analysis = (root / analysis_stage.outputs["analysis_markdown"]).read_text(encoding="utf-8")
        text = render_note(manifest, load_segments(root, transcription), analysis,
                           root / transcription["transcript_json"])
        path = write_note(config.inbox, manifest, text)
        return {"note_path": str(path)}
    return handle


def _existing_note(inbox: Path, job_id: str) -> Path | None:
    # Matched on the job ID only, so a change of time zone cannot hide a note.
    marker = f"\njob_id: {job_id}\n"
    for candidate in sorted(inbox.glob(f"* call {job_id[:8]}*.md")):
        if marker in candidate.read_text(encoding="utf-8"):
            return candidate
    return None


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _title_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


def _file_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H%M")
