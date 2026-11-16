"""Transcribe on the homeserver over SSH (optional, off by default).

Secondary to local processing: recording, ingest, recovery and replay never
use it, and it runs only as the transcription stage of ``lca process``.

The spooled audio is streamed over the user's existing SSH setup to
``lca transcribe-stdin`` on the server, which returns the transcript as JSON
and keeps nothing (see homeserver_server). The call is constrained:

- one ssh executable, run directly (never a .cmd/.bat launcher), with options
  that disable forwarding, local commands, a configured RemoteCommand and
  password prompts, then ``--`` before the validated host;
- the remote command line is built only from validated values and quoted
  for the POSIX shell on the server;
- the file sent is the one inside the spool whose checksum was just
  verified on the same open handle;
- the response is size-limited and validated strictly, including the echo of
  the checksum the server received;
- error messages are fixed texts and never include server output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

from local_call_assistant.config import HomeserverConfig, TranscriptionConfig
from local_call_assistant.integrations import homeserver_server as server
from local_call_assistant.processing import StageContext, StageSkipped
from local_call_assistant.transcription import Transcript, transcript_from_dict, write_transcript

SSH_SAFETY_OPTIONS = (
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "ForwardAgent=no",
    "-o", "ForwardX11=no",
    "-o", "ClearAllForwardings=yes",
    "-o", "PermitLocalCommand=no",
    "-o", "RemoteCommand=none",
)
BATCH_SUFFIXES = (".cmd", ".bat")
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
POLL_SECONDS = 0.2
SERVER_DEADLINE_MARGIN_SECONDS = 60.0
HASH_CHUNK_BYTES = 1024 * 1024
SSH_CONNECTION_FAILED = 255
EXIT_MESSAGES = {
    server.EXIT_USAGE: "the homeserver rejected the request",
    server.EXIT_MISMATCH: "the homeserver received different audio than was sent",
    server.EXIT_BUSY: "the homeserver is busy with another transcription",
    server.EXIT_TOO_LARGE: "the recording is too large for the homeserver",
    server.EXIT_FAILED: "transcription failed on the homeserver",
    SSH_CONNECTION_FAILED: "could not connect to the homeserver over SSH",
}


class HomeserverError(RuntimeError):
    pass


def resolve_ssh(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise HomeserverError("the ssh command was not found")
    if found.lower().endswith(BATCH_SUFFIXES):
        raise HomeserverError("ssh resolves to a .cmd/.bat launcher; set ssh_command to ssh itself")
    return found


def remote_timeout(config: HomeserverConfig) -> int:
    # The server stops first, so a timed-out call does not leave it busy.
    margin = min(SERVER_DEADLINE_MARGIN_SECONDS, config.timeout_seconds / 10)
    return max(1, int(config.timeout_seconds - margin))


def build_command(ssh: str, config: HomeserverConfig, transcription: TranscriptionConfig,
                  sha256: str) -> list[str]:
    remote = [config.remote_command, "transcribe-stdin", "--sha256", sha256,
              "--model", transcription.model, "--device", "cpu",
              "--compute-type", transcription.cpu_compute_type,
              "--timeout", str(remote_timeout(config))]
    if transcription.language:
        remote += ["--language", transcription.language]
    return [ssh, *SSH_SAFETY_OPTIONS, "--", config.host, shlex.join(remote)]


def transcribe_remote(config: HomeserverConfig, transcription: TranscriptionConfig,
                      audio_path: Path, sha256: str, resolve=resolve_ssh) -> Transcript:
    command = build_command(resolve(config.ssh_command), config, transcription, sha256)
    with open(audio_path, "rb", buffering=0) as audio:
        # The opened object must be the regular file that passed the spool's
        # containment check, not something swapped in since.
        try:
            same = os.path.samestat(os.fstat(audio.fileno()),
                                    os.stat(audio_path, follow_symlinks=False))
        except OSError:
            same = False
        if not same:
            raise HomeserverError("spooled audio changed before it could be sent")
        if _sha256_of(audio) != sha256:
            raise HomeserverError("spooled audio changed before it could be sent")
        audio.seek(0)  # the child inherits this position
        code, output = _run(command, audio, config.timeout_seconds)
    if code != 0:
        raise HomeserverError(EXIT_MESSAGES.get(code, f"homeserver transcription failed (exit {code})"))
    return _parse(output, sha256)


def homeserver_handler(transcription: TranscriptionConfig, config: HomeserverConfig,
                       resolve=resolve_ssh):
    def handle(context: StageContext) -> dict[str, Any]:
        if not transcription.enabled:
            raise StageSkipped("transcription is disabled in the configuration")
        result = transcribe_remote(config, transcription, context.audio_path,
                                   context.manifest.sha256, resolve)
        return write_transcript(context.spool.root, context.manifest.job_id, result, "homeserver")
    return handle


def _sha256_of(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(HASH_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], audio: BinaryIO, timeout: float) -> tuple[int, bytes]:
    # stdout goes to a temporary file rather than memory; no pipes, so the
    # timeout always holds. Its size is watched while ssh runs, so a server
    # that keeps writing is stopped instead of filling the disk.
    with tempfile.TemporaryFile(prefix="lca-homeserver-") as output:
        try:
            process = subprocess.Popen(command, stdin=audio, stdout=output,
                                       stderr=subprocess.DEVNULL)
        except OSError:
            raise HomeserverError("ssh could not be started") from None
        deadline = time.monotonic() + timeout
        while True:
            try:
                code = process.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                code = None
            too_large = os.fstat(output.fileno()).st_size > MAX_RESPONSE_BYTES
            if code is not None and not too_large:
                break
            if too_large or time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise HomeserverError("the homeserver response was too large" if too_large
                                      else "the homeserver did not finish within the timeout")
        output.seek(0)
        return code, output.read()


def _parse(output: bytes, sha256: str) -> Transcript:
    try:
        document = json.loads(output.decode("utf-8"))
        protocol = document.get("protocol") if isinstance(document, dict) else None
        # type() check: JSON true and 1.0 must not pass as protocol version 1.
        if (type(protocol) is not int or protocol != server.PROTOCOL
                or document.get("sha256") != sha256):
            raise ValueError("unexpected envelope")
        return transcript_from_dict(document.get("transcript"))
    except ValueError:  # includes JSON and Unicode decoding errors
        raise HomeserverError("the homeserver response was not a valid transcript") from None
