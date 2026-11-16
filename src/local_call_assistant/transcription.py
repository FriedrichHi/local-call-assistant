"""Optional local transcription with faster-whisper.

faster-whisper is an optional dependency (``pip install .[transcription]``);
recording, ingest and replay never import it. With ``device = "auto"`` the
backend tries CUDA and falls back to the CPU configuration if CUDA cannot be
used (no GPU, or the CUDA runtime libraries are missing).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from local_call_assistant.config import TranscriptionConfig
from local_call_assistant.processing import StageContext, StageSkipped
from local_call_assistant.storage.atomic import atomic_write_text

TRANSCRIPTS_DIR = "transcripts"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    language: str | None
    duration: float | None
    segments: list[Segment]
    model: str
    device: str
    compute_type: str

    @property
    def text(self) -> str:
        return "\n".join(segment.text.strip() for segment in self.segments)


class TranscriptionBackend(Protocol):
    def transcribe(self, audio_path: Path) -> Transcript: ...


class FasterWhisperBackend:
    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config

    def transcribe(self, audio_path: Path | BinaryIO) -> Transcript:
        if self.config.device != "auto":
            return self._run(audio_path, self.config.device, self.config.compute_type)
        try:
            return self._run(audio_path, "cuda", self.config.compute_type)
        except (RuntimeError, OSError, ValueError):
            # CUDA unavailable or its libraries missing: use the CPU profile.
            return self._run(audio_path, "cpu", self.config.cpu_compute_type)

    def _run(self, audio: Path | BinaryIO, device: str, compute_type: str) -> Transcript:
        from faster_whisper import WhisperModel  # optional dependency

        model = WhisperModel(self.config.model, device=device, compute_type=compute_type)
        source = str(audio) if isinstance(audio, Path) else audio
        segments, info = model.transcribe(source, language=self.config.language,
                                          vad_filter=True)
        collected = [Segment(s.start, s.end, s.text) for s in segments]
        return Transcript(language=info.language, duration=info.duration, segments=collected,
                          model=self.config.model, device=device, compute_type=compute_type)


def transcription_handler(config: TranscriptionConfig,
                          backend: TranscriptionBackend | None = None):
    def handle(context: StageContext) -> dict[str, Any]:
        if not config.enabled:
            raise StageSkipped("transcription is disabled in the configuration")
        result = (backend or FasterWhisperBackend(config)).transcribe(context.audio_path)
        return write_transcript(context.spool.root, context.manifest.job_id, result, "local")
    return handle


def transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    return {
        "language": transcript.language, "duration": transcript.duration,
        "model": transcript.model, "device": transcript.device,
        "compute_type": transcript.compute_type,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in transcript.segments],
    }


def transcript_from_dict(data: Any) -> Transcript:
    """Strictly rebuild a transcript received from elsewhere; ValueError if malformed."""
    if not isinstance(data, dict):
        raise ValueError("transcript is not an object")
    language, duration = data.get("language"), data.get("duration")
    if language is not None and not isinstance(language, str):
        raise ValueError("transcript language is not text")
    if duration is not None:
        _time(duration)
    texts = [data.get(key) for key in ("model", "device", "compute_type")]
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("transcript model, device or compute type is not text")
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("transcript segments are not a list")
    segments = []
    for raw in raw_segments:
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise ValueError("transcript segment is malformed")
        start, end = _time(raw.get("start")), _time(raw.get("end"))
        if end < start:
            raise ValueError("transcript segment ends before it starts")
        segments.append(Segment(start, end, raw["text"]))
    return Transcript(language=language, duration=None if duration is None else float(duration),
                      segments=segments, model=texts[0], device=texts[1], compute_type=texts[2])


def _time(value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("transcript time is not a non-negative number")
    return float(value)


def write_transcript(root: Path, job_id: str, transcript: Transcript,
                     where: str) -> dict[str, Any]:
    directory = root / TRANSCRIPTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    json_name, text_name = f"{TRANSCRIPTS_DIR}/{job_id}.json", f"{TRANSCRIPTS_DIR}/{job_id}.txt"
    document = {"job_id": job_id, "where": where, **transcript_to_dict(transcript)}
    atomic_write_text(root / json_name, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(root / text_name, transcript.text + "\n")
    return {"transcript_json": json_name, "transcript_text": text_name,
            "language": transcript.language, "model": transcript.model,
            "device": transcript.device, "where": where, "segments": len(transcript.segments)}


def load_segments(root: Path, outputs: dict[str, Any]) -> list[Segment]:
    document = json.loads((root / outputs["transcript_json"]).read_text(encoding="utf-8"))
    return [Segment(s["start"], s["end"], s["text"]) for s in document["segments"]]
