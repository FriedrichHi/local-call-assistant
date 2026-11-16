# Architecture

```text
capture (existing recorder, or RecordingSession)
  -> spool: audio file + manifest, checksummed, crash-safe
  -> transcription (optional, faster-whisper, local)
  -> analysis (optional, Claude CLI or Hermes, transcript text only)
  -> note (optional, Obsidian draft, never overwrites)
```

## Runtime folder

All data lives outside the repository, in a per-user folder:

| Platform | Default |
|---|---|
| Windows | `%LOCALAPPDATA%\local-call-assistant` |
| macOS | `~/Library/Application Support/local-call-assistant` |
| Linux | `$XDG_DATA_HOME/local-call-assistant` (or `~/.local/share/...`) |

Set `LCA_HOME` or pass `--home` to use another folder. Inside it:
`recordings/` (audio), `manifests/` (one JSON per recording), `transcripts/`,
`analysis/`, `locks/` and the optional `config.toml`.

## Guarantees

- **Audio first.** A recording's manifest exists before its first byte of audio,
  and audio is flushed on every write. Nothing downstream deletes or rewrites
  spooled audio.
- **Crash recovery.** A recording whose process died is sealed as
  `interrupted` with the audio it had (`lca recover`, also run by
  `lca process`). A recording still owned by a live process is left alone: the
  owner holds an OS file lock that disappears when it dies.
- **Checksums.** Every spooled file has a SHA-256 in its manifest, and each
  stage re-checks it before reading the audio.
- **Visible retries.** Every stage attempt is recorded with its time, outcome
  and error. Failed stages retry on the next `lca process`, up to
  `max_attempts`; `lca retry <job> --stage <name>` resets the budget.
- **Ingest is read-only.** `lca ingest` copies a recorder's file, verifies the
  copy and never modifies the original. The same file is not ingested twice.

## Configuration (`config.toml`)

```toml
max_attempts = 3
ingest_min_age_seconds = 60

[transcription]
enabled = true
model = "small"        # tiny, base, small, medium, large-v3 ...
device = "auto"        # auto = try CUDA, fall back to CPU
language = "de"        # omit to auto-detect

[obsidian]
inbox = "C:/path/to/vault/Inbox"

# Analysis: enable at most one of [claude] and [hermes]. Both are off by default.
[claude]               # the Claude Code CLI, signed in with a Claude subscription
enabled = false
command = ["claude"]
# model = "sonnet"     # omit to use the CLI's default model
timeout_seconds = 600

[hermes]               # alternative: a Hermes API server
enabled = false
base_url = "http://127.0.0.1:8642"
api_key_env = "LCA_HERMES_API_KEY"
```

## Privacy

Recording, transcription and note writing run entirely on this machine. Audio
never leaves it. Enabling analysis sends transcript text only:

- **`[claude]`** runs `claude -p` locally, which sends the transcript to
  Anthropic under the signed-in Claude account. The call is isolated: no
  tools, no MCP servers, no user or project settings (so no hooks or plugins
  that could copy the transcript elsewhere), no saved session, and an empty
  temporary working directory. It gets only an allowlist of basic operating
  system variables, so nothing inherited can switch it to API billing or
  another provider, change its endpoint, add headers, route it through a proxy
  or load plugins. The native executable is run directly (npm's `claude.cmd`
  is resolved to the `claude.exe` it wraps; other `.cmd`/`.bat` launchers are
  refused), and the transcript goes in on standard input, never on the command
  line, from a temporary file that is deleted when the call ends.
  `command` may not contain options, and `model` must be a plain name.
- **`[hermes]`** sends it to the Hermes API server and from there to whichever
  model provider Hermes uses. Plain HTTP is refused except to a loopback
  address, redirects are refused, and system proxy settings are ignored. The
  API key is read from the named environment variable, never the config file.

Error messages stored in manifests are fixed texts and never quote the
service's output.

## Not built yet

Own WASAPI capture (fallback if no existing recorder passes phase 0), a tray
interface, the homeserver worker (OPS-59) and Linux/macOS capture (OPS-60).
