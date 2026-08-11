# Local Call Assistant Implementation Plan

> **For Hermes:** Implement this plan task-by-task with tests and verification after each milestone.

**Goal:** Build a public, local-first, cross-platform call recorder that preserves meeting audio locally first, processes it locally or on a homeserver later, sends transcripts to Hermes for analysis, and creates draft Obsidian notes.

**Architecture:** A platform-neutral job/recording core will sit above platform-specific audio capture backends. Windows is the first full target because it is the immediate user workflow; Linux and macOS backends are designed into the interfaces but can follow after the Windows recording path is reliable. Homeserver processing is asynchronous and optional.

**Tech Stack:** Python service/core, SQLite job manifest, WASAPI on Windows, PipeWire/PulseAudio on Linux, ScreenCaptureKit/Core Audio on macOS, faster-whisper or whisper.cpp, Hermes-compatible HTTP/webhook adapter, Markdown Obsidian output.

---

## Priorities and acceptance criteria

### P0 — Preserve the recording

- Recording begins only after the user explicitly starts it.
- Audio is written to a local durable spool immediately.
- A recording is never deleted because transcription, Hermes, Obsidian, network, or homeserver processing failed.
- Every recording has a manifest, stable job ID, timestamps, source metadata, and checksum.
- Interrupted or failed processing can be resumed without re-recording.

### P1 — Windows local workflow

- Capture microphone and system/call audio on Windows 10/11.
- Support record, pause, resume, stop, and record-only modes.
- Provide a small tray/status interface or local status page.
- Store recordings and transcripts locally by default.
- Process locally with Whisper when enabled.

### P2 — Hermes and Obsidian

- Submit completed transcripts—not raw audio by default—to Hermes.
- Generate a reviewable Markdown note.
- Write notes to a configurable Obsidian inbox/outbox path.
- Keep external task creation disabled until explicitly enabled.

### P3 — Homeserver fallback

- Upload encrypted recordings or transcripts to a private homeserver worker.
- Support automatic retry, resumable jobs, and idempotent processing.
- Allow local, homeserver, and automatic processing modes.

### P4 — Linux and macOS

- Keep the core and job protocol platform-neutral from the first implementation.
- Add Linux PipeWire/PulseAudio capture after Windows P1 is verified.
- Add macOS ScreenCaptureKit/Core Audio capture after Linux interfaces stabilize.
- Provide platform-specific permission diagnostics and packaging later.

### P5 — Voice interactions

- Support microphone dictation and agent voice input as a separate workflow.
- Accept imported phone recordings or voice messages through the homeserver/Hermes path.

---

## Milestone 0: repository and architecture skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/local_call_assistant/`
- Create: `tests/`
- Create: `docs/architecture.md`
- Create: `docs/privacy-and-consent.md`

**Tasks:**
1. Define the package and supported Python versions.
2. Add typed domain models for `Recording`, `Manifest`, `ProcessingJob`, and job states.
3. Define the `AudioCaptureBackend` interface.
4. Define the `TranscriptionBackend`, `AnalysisAdapter`, and `NoteWriter` interfaces.
5. Add unit tests for state transitions and invalid transitions.
6. Add CI for formatting, type checking, and tests.

**Verification:** Run the complete unit-test suite and validate the package in a clean virtual environment.

## Milestone 1: durable local recording spool

**Files:**
- Create: `src/local_call_assistant/storage/manifest_store.py`
- Create: `src/local_call_assistant/storage/spool.py`
- Create: `src/local_call_assistant/recording/session.py`
- Create: `tests/test_spool.py`
- Create: `tests/test_recording_session.py`

**Tasks:**
1. Create deterministic runtime directories outside the repository.
2. Write recording data to a temporary file while recording.
3. Atomically finalize the file on stop.
4. Calculate and store SHA-256 after finalization.
5. Persist a manifest before marking a recording complete.
6. Recover incomplete jobs on startup.
7. Add retention configuration without automatic deletion in the first release.

**Verification:** Kill the recorder during a synthetic recording, restart it, and verify the partial file and manifest are preserved and flagged for recovery.

## Milestone 2: Windows audio capture

**Files:**
- Create: `src/local_call_assistant/capture/windows_wasapi.py`
- Create: `src/local_call_assistant/capture/device_catalog.py`
- Create: `tests/integration/test_windows_capture.py`
- Create: `docs/windows-setup.md`

**Tasks:**
1. Enumerate microphone and render endpoints.
2. Capture microphone and system-output audio using WASAPI loopback.
3. Preserve separate channels when possible and define a fallback mix.
4. Implement record/pause/resume/stop against the durable spool.
5. Add a local test command that records a bounded sample.
6. Document Windows microphone and audio permissions and supported routing.

**Verification:** Record a real test call and an offline audio sample; verify playback, both sides of the conversation, timestamps, checksum, and recovery behavior.

## Milestone 3: local Whisper processing

**Files:**
- Create: `src/local_call_assistant/transcription/faster_whisper_backend.py`
- Create: `src/local_call_assistant/transcription/chunking.py`
- Create: `tests/test_transcription_jobs.py`
- Create: `docs/transcription.md`

**Tasks:**
1. Add configurable model, device, compute type, language, and chunk settings.
2. Process completed recordings asynchronously so recording is never blocked.
3. Store timestamped transcript segments and processing metadata.
4. Support re-transcription without changing the original recording.
5. Add local GPU and CPU fallback modes.

**Verification:** Transcribe a controlled fixture and a recorded test call; compare elapsed time, transcript completeness, and GPU/CPU behavior.

## Milestone 4: Windows tray/status interface

**Files:**
- Create: `src/local_call_assistant/ui/`
- Create: `tests/test_ui_commands.py`
- Create: `docs/user-guide.md`

**Tasks:**
1. Add start/stop/pause/resume commands.
2. Show recording and processing state.
3. Add local, homeserver, automatic, and record-only modes.
4. Make the active recording state visible and easy to stop.
5. Add an error view that links to the preserved recording and manifest.

**Verification:** Exercise all controls during a test recording and confirm that UI failures do not interrupt the recorder.

## Milestone 5: Hermes and Obsidian integration

**Files:**
- Create: `src/local_call_assistant/integrations/hermes.py`
- Create: `src/local_call_assistant/integrations/obsidian.py`
- Create: `src/local_call_assistant/templates/call-note.md`
- Create: `tests/test_hermes_payload.py`
- Create: `tests/test_obsidian_writer.py`

**Tasks:**
1. Define a versioned transcript payload schema.
2. Add authenticated Hermes submission using a configurable endpoint.
3. Ensure raw audio is not sent unless explicitly configured.
4. Render a draft Markdown note with summary, decisions, actions, questions, and transcript.
5. Write atomically to an Obsidian inbox/outbox folder.
6. Make the note writer idempotent by recording job ID and output path.

**Verification:** Run against a local mock Hermes endpoint and a test Obsidian vault; confirm retries do not duplicate notes.

## Milestone 6: homeserver processing layer

**Files:**
- Create: `server/`
- Create: `deploy/homeserver/`
- Create: `tests/test_remote_jobs.py`
- Create: `docs/homeserver.md`

**Tasks:**
1. Add a private upload endpoint with authentication.
2. Support resumable upload or chunked transfer.
3. Add a homeserver Whisper worker.
4. Implement job acknowledgements, retries, and idempotency.
5. Add automatic mode routing and local fallback.
6. Keep raw-audio retention separate from transcript/note retention.

**Verification:** Stop the homeserver during upload and processing; verify local retry, no duplicate jobs, and preserved recordings.

## Milestone 7: Linux and macOS capture backends

**Files:**
- Create: `src/local_call_assistant/capture/linux_pipewire.py`
- Create: `src/local_call_assistant/capture/macos_screencapturekit.py`
- Create: `docs/linux-setup.md`
- Create: `docs/macos-setup.md`

**Tasks:**
1. Implement Linux microphone and monitor-source capture.
2. Implement macOS system-audio and microphone capture through native APIs.
3. Add permission diagnostics.
4. Reuse the same manifest, spool, processing, Hermes, and Obsidian paths.
5. Add platform build and packaging workflows.

**Verification:** Test on representative Linux desktop/audio stacks and Apple Silicon macOS; explicitly test permission denial and recovery.

## Milestone 8: voice interactions and phone input

**Files:**
- Create: `src/local_call_assistant/voice/`
- Modify: `server/`
- Create: `docs/voice-input.md`

**Tasks:**
1. Add microphone dictation mode separate from call recording.
2. Add Hermes voice-message or upload ingestion.
3. Accept imported phone recordings through the same job protocol.
4. Keep phone-call capture platform-specific and out of the first MVP.

**Verification:** Send a voice message through the configured Hermes channel, transcribe it, and verify it reaches the intended agent without mixing it with call archives.

---

## Security, privacy, and governance requirements

- Never commit recordings, transcripts, credentials, tokens, or local databases.
- Default to local recording and no raw-audio upload.
- Require explicit configuration before uploading audio to the homeserver.
- Use authenticated, encrypted transport for remote jobs.
- Keep recording status visible to the user.
- Include consent and lawful-use documentation in the public project.
- Make external writes draft-only until the user enables them.
- Log processing events without logging transcript contents by default.
- Add tests for path traversal, unsafe output paths, oversized uploads, and duplicate job IDs.

## First release scope

The first usable release should stop after Milestone 5:

```text
Windows local recording
→ durable local spool
→ optional local Whisper
→ Hermes transcript analysis
→ Obsidian draft note
```

The homeserver is secondary and should not block the first local recording workflow.
