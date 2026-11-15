# Transcription

Install the optional dependency and enable it in `config.toml`:

```bash
pip install -e ".[transcription]"
```

```toml
[transcription]
enabled = true
model = "small"
```

The model is downloaded from Hugging Face on first use and cached locally.
Transcription only reads the spooled audio; transcripts go to `transcripts/`
as JSON (timestamped segments) and plain text.

## CPU and GPU

`device = "auto"` (the default) tries CUDA and falls back to the CPU profile
(`cpu_compute_type = "int8"`). The manifest records which device was used.

For NVIDIA GPU transcription on Windows, faster-whisper needs the CUDA 12
cuBLAS and cuDNN 9 runtime libraries on `PATH`. Without them, CUDA fails with
`Library cublas64_12.dll is not found` and the CPU fallback is used. One way
to get them is NVIDIA's pip packages:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

then add their `bin` folders (under `site-packages/nvidia/`) to `PATH`.

To force a device, set `device = "cpu"` or `device = "cuda"`; a forced device
does not fall back.

## Measured

2026-11-15, Windows 11, `base` model, CPU int8 (cuBLAS not installed):
a 12-second synthetic English clip transcribed correctly in about 25 seconds
including model load.
