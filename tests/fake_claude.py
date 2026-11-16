"""Stand-in for the claude CLI in tests. Records what it was given, then replies."""

import json
import os
import sys
import time
from pathlib import Path

SECRET_OUTPUT = "stderr-secret-quote"

mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if mode == "never-reads":
    time.sleep(30)  # stalls before reading any of its input
    sys.exit(0)

prompt = sys.stdin.read()
record = os.environ.get("FAKE_CLAUDE_RECORD")
if record:
    Path(record).write_text(json.dumps({
        "argv": sys.argv[1:], "stdin": prompt, "cwd": os.getcwd(),
        "cwd_entries": os.listdir(os.getcwd()),
        "env_keys": sorted(os.environ),
    }), encoding="utf-8")

if mode == "ok":
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                      "result": "## Summary\nShip on Friday.\n\n## Decisions\nShip on Friday."}))
elif mode == "is-error":
    print(json.dumps({"type": "result", "is_error": True, "result": SECRET_OUTPUT}))
elif mode == "exit":
    print(SECRET_OUTPUT, file=sys.stderr)
    sys.exit(3)
elif mode == "garbage":
    print(f"not json {SECRET_OUTPUT}")
elif mode == "slow":
    time.sleep(30)
