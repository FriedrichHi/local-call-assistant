# Local Call Assistant

A local-first, cross-platform call recorder and transcription pipeline for Windows, Linux, and macOS.

The core guarantee is simple: **record locally first; process whenever possible**.

## Planned capabilities

- Capture microphone and call/system audio without joining meetings as a bot
- Preserve recordings locally before any processing begins
- Run Whisper locally on the desktop when desired
- Fall back to a homeserver worker for asynchronous processing
- Route transcripts to Hermes for analysis using the user's chosen provider
- Create draft Markdown notes for Obsidian
- Optionally index approved notes in OpenViking
- Support voice interactions with agents as a separate input path

## Status

Early planning. The first milestone is a reliable Windows local recorder with a durable job spool. Homeserver processing, Linux/macOS capture, and richer integrations follow after the recording path is verified.

## Privacy and consent

This project is intended for lawful, disclosed recording. Users are responsible for obtaining any required consent and complying with applicable law, contracts, workplace policy, and platform rules.

## License

MIT. See [LICENSE](LICENSE).
