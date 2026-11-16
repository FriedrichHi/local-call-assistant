# Homeserver transcription

Optional and secondary. Recording, ingest, recovery, replay and local
transcription never need the homeserver, and it is off by default. When
`[homeserver]` is enabled, the transcription stage of `lca process` runs on
another machine you own instead of this one, for example because this PC has
no usable GPU or should not spend its CPU on it. Analysis and notes still run
here.

## How it works

`lca process` streams the spooled recording over your existing SSH setup to
`lca transcribe-stdin` on the server, which transcribes it and sends back the
transcript as JSON. There is no daemon, no open port and no shared folder.

- **What is sent:** the recording's audio, and nothing else from this machine.
  Only a file inside the spool can be sent, and only after its checksum has been
  verified on the same open file.
- **What is kept on the server:** no recording data. The audio is held in an
  anonymous temporary file that has no name and disappears when the process
  ends, however it ends. The Whisper model files stay in the server's Hugging
  Face cache, and a lock file stays in `~/.cache/lca`.
- **Failures** (no connection, server busy, timeout, a malformed answer) are
  recorded as a failed transcription attempt with a fixed message; the local
  recording is untouched and `lca process` retries it, up to `max_attempts`.
  `lca retry <job> --stage transcription` resets the budget.
- **Timeouts:** the server stops itself shortly before the client's
  `timeout_seconds`. SSH does not stop remote work when the client goes away, so
  an abandoned transcription runs until that deadline. Only one transcription
  runs on the server at a time; others get "busy" and are retried later.
- **SSH:** `ssh` runs with `BatchMode=yes` (no password prompts), and with
  forwarding, local commands and any configured `RemoteCommand` disabled. Key
  login must already work: `ssh <host> true` must succeed without a prompt.

## Set up the server

On the server (Linux, Python 3.11 or newer):

```bash
git clone https://github.com/FriedrichHi/local-call-assistant.git
cd local-call-assistant
python3 -m venv .venv            # or: uv venv .venv
.venv/bin/pip install -e ".[transcription]"
# Pre-download the model so the first call stays within its timeout:
.venv/bin/python -c 'from faster_whisper import WhisperModel; WhisperModel("small", device="cpu", compute_type="int8")'
```

A model that is not cached yet is downloaded from Hugging Face on first use.

## Enable it here

```toml
[transcription]
enabled = true
model = "small"        # a plain model name; it must exist on the server
language = "de"

[homeserver]
enabled = true
host = "homeserver"    # SSH host or alias
remote_command = "/home/you/local-call-assistant/.venv/bin/lca"   # absolute path
timeout_seconds = 3600
```

The server transcribes on its CPU with `cpu_compute_type` (default `int8`).
To go back to local transcription, set `enabled = false` and run
`lca retry <job> --stage transcription` for any failed recordings.
