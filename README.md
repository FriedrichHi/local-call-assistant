# Local Call Assistant

A local-first call recorder and transcription pipeline for Windows, with Linux
and macOS to follow.

The core guarantee is simple: **record locally first; process whenever possible**.

## What works today

- A crash-safe local spool: every recording gets a manifest and a SHA-256
  checksum, and a recording interrupted by a crash is recovered with its audio.
- Ingest from an existing recorder's output folder, read-only and checksummed.
- Optional local transcription with faster-whisper, GPU or CPU.
- Optional analysis through a Hermes API server (off by default).
- Draft notes written to an Obsidian inbox, never overwriting existing files.
- Retry and recovery state visible per recording.

Live capture comes from an existing recorder for now; choosing it is the
[phase 0 test](docs/phase0-recorder-test.md).

## Quick start

```bash
pip install -e ".[transcription]"
lca ingest-dir "<recorder output folder>" --recursive
lca process
lca status
```

See [architecture](docs/architecture.md) for the configuration file, runtime
folder and guarantees, and [transcription](docs/transcription.md) for GPU setup.

## Development

```bash
pip install -e ".[test]"
pytest
```

## Privacy and consent

This project is intended for lawful, disclosed recording. Users are responsible
for obtaining any required consent and complying with applicable law, contracts,
workplace policy, and platform rules. Recordings, transcripts and notes stay on
your machine unless you enable the Hermes integration.

## License

MIT. See [LICENSE](LICENSE).
