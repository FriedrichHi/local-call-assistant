# Phase 0: test existing recorders on Windows

Goal: decide which recorder captures calls for the first release. Our own code
(spool, manifest, transcription, notes) then works on that recorder's output
folder, so it does not depend on which recorder wins.

Candidates: **Meetily** (current release, with the previous release kept as a
fallback) and **LocalScribe**. Neither is a tool we wrote; install them only on
your own machine and only from their official GitHub release pages.

## 1. Download and check the installer

1. Download the Windows installer from the official release page:
   - Meetily: <https://github.com/Zackriya-Solutions/meeting-minutes/releases>
   - LocalScribe: <https://github.com/imnotwallace/LocalScribe/releases>
2. Before running it, check it against the digest GitHub publishes:

   ```powershell
   .\scripts\Test-RecorderInstaller.ps1 -Path <installer> -Repository <owner/repo> -Tag <tag>
   ```

   Run it only if `DigestMatches` is `True`. `Authenticode` tells you whether
   the installer is code-signed; an unsigned installer will show a SmartScreen
   warning.
3. Keep the installer file, so the exact version can be reinstalled.

## 2. Two-minute self-test

Call yourself (for example a phone on speaker) and confirm the recording has
both your microphone and the other side's audio. Do not record anyone else
until this works.

## 3. Real call (with consent)

Record one real call of about 30 minutes, only with every participant's
consent, and tell them it is a local-only test.

## 4. Kill test

Start a new test recording, talk for a minute, then end the recorder process
without stopping the recording:

```powershell
Get-Process | Where-Object { $_.ProcessName -match 'meetily|LocalScribe' }
Stop-Process -Id <id> -Force
```

Restart the recorder. Note whether the minute of audio survived, whether the
app shows it as interrupted or recoverable, and what is on disk.

## 5. Results sheet

Copy this for each recorder:

| Check | Result |
|---|---|
| Version and installer digest | |
| Both sides captured in self-test | |
| 30-minute call: complete, no gaps | |
| Transcript accuracy, German | |
| Transcript accuracy, English | |
| Speaker separation | |
| CPU / fan load during the call | |
| Where recordings are stored, and in which format | |
| Kill test: audio kept? | |
| Kill test: app recovered it? | |
| What is missing for your use | |

## 6. Feed the recordings to our pipeline

Point our ingest at the recorder's output folder (the one you noted above).
It copies files and never changes or deletes the originals:

```powershell
lca ingest-dir "<recorder output folder>" --recursive
lca process
lca status
```

**Stop or finish every recording in the recorder before running this.** Files
modified in the last 60 seconds are skipped and a file that changes while it is
being copied is not ingested, but these are heuristics: a recording that is
paused for longer than a minute can still look finished. Until we know how the
chosen recorder signals that a file is complete, ingest only finished
recordings.
