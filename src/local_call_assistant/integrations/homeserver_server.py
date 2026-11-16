"""Server side of homeserver transcription: ``lca transcribe-stdin``.

Run by the client over SSH. It reads one recording from standard input,
checks it against the checksum the client sent, transcribes it and writes a
single JSON document to standard output. It reads no lca configuration and
keeps nothing:

- the audio lives only in an anonymous temporary file: on Linux it has no
  name (checked before any audio is written), so it disappears when the
  process ends for any reason. A kill at the instant of creation can at most
  leave an empty file behind, never recording data;
- a deadline (``--timeout``, set by the client just below its own) ends the
  process. SSH does not stop remote work when the client goes away, so an
  abandoned transcription runs until that deadline; only one runs at a time,
  so retries cannot pile up but report "busy" until it ends;
- everything the libraries print goes to standard error; standard output
  carries only the final JSON document.

The Whisper model files themselves stay in the Hugging Face cache. A model
that is not cached yet is downloaded on first use (pre-download it when
installing the server to keep the first call within its timeout).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from local_call_assistant.config import (
    COMPUTE_TYPES, LANGUAGE_PATTERN, WHISPER_MODEL_PATTERN, TranscriptionConfig,
)
from local_call_assistant.storage.filelock import JobLock
from local_call_assistant.transcription import (
    FasterWhisperBackend, TranscriptionBackend, transcript_to_dict,
)

PROTOCOL = 1
EXIT_OK, EXIT_USAGE, EXIT_MISMATCH, EXIT_BUSY, EXIT_TOO_LARGE, EXIT_FAILED = 0, 2, 3, 4, 5, 6
MAX_AUDIO_BYTES = 4 * 1024 ** 3
CHUNK_BYTES = 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
DEVICES = frozenset({"cpu", "cuda", "auto"})
CACHE_SUBDIR = "lca"
LOCK_NAME = "transcribe.lock"
MAX_TIMEOUT_SECONDS = 24 * 3600

BackendFactory = Callable[[TranscriptionConfig], TranscriptionBackend]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--timeout", type=float, required=True)


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / CACHE_SUBDIR


def run(args: argparse.Namespace) -> int:
    """CLI entry: keep stdout for the result only, then serve one request."""
    result_fd = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # library output goes to stderr
    code, document = serve(args, sys.stdin.buffer)
    if document is not None:
        with os.fdopen(result_fd, "wb") as result:
            result.write(json.dumps(document, ensure_ascii=False).encode("utf-8"))
    else:
        os.close(result_fd)
    return code


def serve(args: argparse.Namespace, stdin: BinaryIO,
          backend_factory: BackendFactory = FasterWhisperBackend) -> tuple[int, dict[str, Any] | None]:
    try:
        config = _validated(args)
    except ValueError:
        return EXIT_USAGE, None
    _arm_deadline(args.timeout)
    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = JobLock(directory / LOCK_NAME)
    if not lock.acquire():
        return EXIT_BUSY, None
    try:
        with tempfile.TemporaryFile(dir=directory) as audio:
            if os.name != "nt" and os.fstat(audio.fileno()).st_nlink != 0:
                return EXIT_FAILED, None  # refuse to write audio into a named file
            status = _receive(stdin, audio, args.sha256)
            if status != EXIT_OK:
                return status, None
            audio.seek(0)  # the decoder reads from the current position
            try:
                transcript = backend_factory(config).transcribe(audio)
            except Exception:  # noqa: BLE001 - reported as a fixed exit code only
                return EXIT_FAILED, None
    finally:
        lock.release()
    return EXIT_OK, {"protocol": PROTOCOL, "sha256": args.sha256,
                     "transcript": transcript_to_dict(transcript)}


def _validated(args: argparse.Namespace) -> TranscriptionConfig:
    if not SHA256_PATTERN.fullmatch(args.sha256 or ""):
        raise ValueError("sha256")
    if not WHISPER_MODEL_PATTERN.fullmatch(args.model or ""):
        raise ValueError("model")
    if args.language is not None and not LANGUAGE_PATTERN.fullmatch(args.language):
        raise ValueError("language")
    if args.device not in DEVICES or args.compute_type not in COMPUTE_TYPES:
        raise ValueError("device or compute type")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout")
    return TranscriptionConfig(enabled=True, model=args.model, device=args.device,
                               compute_type=args.compute_type,
                               cpu_compute_type=args.compute_type, language=args.language)


def _arm_deadline(seconds: float) -> None:
    # Default SIGALRM action terminates the process even while native code runs.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(max(1, math.ceil(seconds)))


def _receive(stdin: BinaryIO, audio: BinaryIO, expected: str) -> int:
    digest, size = hashlib.sha256(), 0
    while chunk := stdin.read(CHUNK_BYTES):
        size += len(chunk)
        if size > MAX_AUDIO_BYTES:
            return EXIT_TOO_LARGE
        digest.update(chunk)
        audio.write(chunk)
    audio.flush()
    return EXIT_OK if digest.hexdigest() == expected else EXIT_MISMATCH
