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

For NVIDIA GPU transcription on Windows, install the `gpu` extra as well:

```bash
pip install -e ".[transcription,gpu]"
```

It adds NVIDIA's cuBLAS 12 (about 500 MB); CTranslate2 already bundles cuDNN.
lca finds it by itself: while transcribing on the GPU it appends the package's
`bin` folder to its own `PATH` (Windows only; nothing outside the process
changes). An existing CUDA 12 installation on `PATH` works too and takes
priority. Without either, CUDA fails with `Library cublas64_12.dll is not
found` and `device = "auto"` falls back to the CPU.

The model is loaded only while `lca process` transcribes and is released when
it finishes; nothing stays running in the background.

To force a device, set `device = "cpu"` or `device = "cuda"`; a forced device
does not fall back.

## Measured

2026-11-16, Windows 11, RTX 4060 Laptop GPU (8 GB), a 194-second synthetic
English clip, model already downloaded, time for load plus transcription:

| Model | Device | Time | Result |
|---|---|---|---|
| small | CPU int8 | 103 s | correct |
| small | GPU float16 | 21 s | correct |
| large-v3-turbo | GPU float16 | 16 s | repeated whole sentences on this clip |
| large-v3 | GPU float16 | 61 s | correct |
| large-v3 | CPU int8 | 380 s | correct |

With a GPU, `large-v3` is the most accurate model at about a third of real
time. Real German and English calls have not been measured yet. First use of
a model downloads it (small about 0.5 GB, large-v3 about 3 GB). On Windows
without symlink rights, the Hugging Face cache stores copies instead of links
and warns about it; that only costs disk space.
