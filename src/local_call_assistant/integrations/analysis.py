"""Shared pieces of the analysis stage, whichever service performs it."""

from __future__ import annotations

from typing import Any

from local_call_assistant.processing import StageContext
from local_call_assistant.storage.atomic import atomic_write_text
from local_call_assistant.transcription import load_segments

PAYLOAD_VERSION = 1
ANALYSIS_DIR = "analysis"
SYSTEM_PROMPT = (
    "You analyse a call transcript for a draft note. Reply in Markdown with the sections "
    "'Summary', 'Decisions', 'Action items', and 'Open questions'. Use only what the "
    "transcript says; write 'None' for an empty section. Do not take any other action."
)


def transcript_text(context: StageContext) -> str:
    outputs = context.manifest.stage("transcription").outputs
    return "\n".join(s.text.strip() for s in load_segments(context.spool.root, outputs))


def user_message(job_id: str, text: str) -> str:
    return f"Transcript (payload v{PAYLOAD_VERSION}, job {job_id}):\n\n{text}"


def write_analysis(context: StageContext, markdown: str, provider: str,
                   model: str | None) -> dict[str, Any]:
    name = f"{ANALYSIS_DIR}/{context.manifest.job_id}.md"
    (context.spool.root / ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    atomic_write_text(context.spool.root / name, markdown.strip() + "\n")
    return {"analysis_markdown": name, "provider": provider, "model": model,
            "payload_version": PAYLOAD_VERSION}
