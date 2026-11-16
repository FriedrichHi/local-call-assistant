import importlib.metadata
import os
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace

import pytest

from local_call_assistant import transcription
from local_call_assistant.config import TranscriptionConfig
from local_call_assistant.transcription import FasterWhisperBackend

ORIGINAL_PATH = os.pathsep.join(["/existing", "/other"])


class FakeDistribution:
    def __init__(self, root: Path, files: list[str]) -> None:
        self.root, self.files = root, [PurePosixPath(f) for f in files]

    def locate_file(self, path: PurePosixPath) -> Path:
        return self.root / path


def _install(monkeypatch: pytest.MonkeyPatch, distribution: FakeDistribution | None) -> None:
    def lookup(name: str) -> FakeDistribution:
        if distribution is None or name != transcription.CUBLAS_DISTRIBUTION:
            raise importlib.metadata.PackageNotFoundError(name)
        return distribution
    monkeypatch.setattr(transcription.importlib.metadata, "distribution", lookup)


@pytest.fixture
def cublas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    root = tmp_path / "site-packages"
    dll = root / transcription.CUBLAS_DLL
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"MZ")
    planted = root / "nvidia" / "planted" / "bin" / "other.dll"  # listed first, never exposed
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"MZ")
    _install(monkeypatch, FakeDistribution(
        root, ["nvidia/planted/bin/other.dll", "nvidia/__init__.py", transcription.CUBLAS_DLL]))
    monkeypatch.setenv("PATH", ORIGINAL_PATH)
    monkeypatch.setattr(transcription, "_is_windows", lambda: True)
    return str(dll.parent.resolve())


def test_only_the_recorded_cublas_folder_is_used(cublas: str) -> None:
    assert transcription.cublas_dir() == cublas


@pytest.mark.parametrize("files", [[], ["nvidia/cublas/bin/other.dll"]])
def test_no_recorded_dll_means_nothing_is_exposed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: list[str]) -> None:
    _install(monkeypatch, FakeDistribution(tmp_path, files))
    assert transcription.cublas_dir() is None


def test_not_installed_means_nothing_is_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, None)
    assert transcription.cublas_dir() is None


def test_path_is_extended_only_during_the_attempt(cublas: str) -> None:
    with transcription.nvidia_libraries_on_path():
        assert os.environ["PATH"].split(os.pathsep) == ["/existing", "/other", cublas]
    assert os.environ["PATH"] == ORIGINAL_PATH


def test_path_is_restored_when_the_attempt_fails(cublas: str) -> None:
    with pytest.raises(RuntimeError):
        with transcription.nvidia_libraries_on_path():
            raise RuntimeError("CUDA failed")
    assert os.environ["PATH"] == ORIGINAL_PATH


def test_other_systems_leave_path_alone(cublas: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription, "_is_windows", lambda: False)
    with transcription.nvidia_libraries_on_path():
        assert os.environ["PATH"] == ORIGINAL_PATH


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """A stand-in faster_whisper module that records the device and PATH it saw."""
    seen: list[tuple[str, str]] = []

    class WhisperModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            self.device = device
            seen.append((f"{device}:load", os.environ["PATH"]))

        def transcribe(self, source, **kwargs):
            def segments():  # decoding is lazy: CUDA work happens while iterating
                seen.append((f"{self.device}:decode", os.environ["PATH"]))
                if self.device == "cuda" and kwargs.get("language") == "fail-gpu":
                    raise RuntimeError("cuBLAS failed")
                yield SimpleNamespace(start=0.0, end=1.0, text=" hi")
            return segments(), SimpleNamespace(language="en", duration=1.0)

    module = ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return seen


@pytest.mark.parametrize("device, gpu_path", [("cpu", False), ("cuda", True)])
def test_only_gpu_attempts_see_the_folder(
        cublas: str, fake_whisper: list, device: str, gpu_path: bool) -> None:
    result = FasterWhisperBackend(TranscriptionConfig(enabled=True, device=device)).transcribe(Path("x.wav"))
    assert [step for step, _ in fake_whisper] == [f"{device}:load", f"{device}:decode"]
    assert all((cublas in path) is gpu_path for _, path in fake_whisper)  # load and decode
    assert result.text == "hi"
    assert os.environ["PATH"] == ORIGINAL_PATH


def test_cpu_fallback_runs_without_the_folder(cublas: str, fake_whisper: list) -> None:
    config = TranscriptionConfig(enabled=True, device="auto", language="fail-gpu")
    FasterWhisperBackend(config).transcribe(Path("x.wav"))  # CUDA fails while decoding
    assert [step for step, _ in fake_whisper] == ["cuda:load", "cuda:decode", "cpu:load", "cpu:decode"]
    assert [cublas in path for _, path in fake_whisper] == [True, True, False, False]
    assert os.environ["PATH"] == ORIGINAL_PATH  # later subprocesses (claude, ssh) see the original
