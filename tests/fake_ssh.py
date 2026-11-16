"""Stand-in for ssh in tests.

Records its arguments and a digest of what arrived on stdin. In "serve" mode
it parses the remote command line the way a POSIX shell would and runs the
real server code in-process with a fake Whisper backend.
"""

import argparse
import hashlib
import io
import json
import os
import shlex
import sys
import time
from pathlib import Path

SECRET_OUTPUT = "server-secret-quote"

mode = os.environ.get("FAKE_SSH_MODE", "serve")
if mode == "never-reads":
    time.sleep(30)
    sys.exit(0)

argv = sys.argv[1:]
data = sys.stdin.buffer.read()
record = os.environ.get("FAKE_SSH_RECORD")
if record:
    Path(record).write_text(json.dumps({
        "argv": argv, "stdin_sha256": hashlib.sha256(data).hexdigest(), "stdin_bytes": len(data),
    }), encoding="utf-8")

if mode == "garbage":
    print(f"not json {SECRET_OUTPUT}")
    sys.exit(0)
if mode.startswith("exit:"):
    print(SECRET_OUTPUT, file=sys.stderr)
    sys.exit(int(mode.split(":")[1]))
if mode == "huge":
    sys.stdout.buffer.write(b" " * (int(os.environ["FAKE_SSH_HUGE_BYTES"])))
    sys.exit(0)

from local_call_assistant.integrations import homeserver_server  # noqa: E402
from local_call_assistant.transcription import Segment, Transcript  # noqa: E402


class FakeBackend:
    def __init__(self, config) -> None:
        self.config = config

    def transcribe(self, audio) -> Transcript:
        content = audio.read()  # proves the file was rewound
        text = " tampered" if mode == "tamper" else f" heard {len(content)} bytes"
        return Transcript(language=self.config.language, duration=1.0,
                          segments=[Segment(0.0, 1.0, text)], model=self.config.model,
                          device=self.config.device, compute_type=self.config.compute_type)


remote = shlex.split(argv[argv.index("--") + 2])
parser = argparse.ArgumentParser()
parser.add_argument("subcommand")
homeserver_server.add_arguments(parser)
args = parser.parse_args(remote[1:])
code, document = homeserver_server.serve(args, io.BytesIO(data), FakeBackend)
if document is not None:
    if mode == "wrong-sha":
        document["sha256"] = "0" * 64
    sys.stdout.write(json.dumps(document))
sys.exit(code)
