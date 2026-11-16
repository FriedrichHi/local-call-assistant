import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_call_assistant import transcription
from local_call_assistant.config import TranscriptionConfig
from local_call_assistant.transcription import FasterWhisperBackend


@pytest.fixture
def nvidia_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    root = tmp_path / "site-packages" / "nvidia"
    dirs = [root / "cublas" / "bin", root / "cudnn" / "bin"]
    for directory in dirs:
        directory.mkdir(parents=True)
    (root / "cuda_runtime").mkdir()  # a package without a bin folder is ignored
    monkeypatch.setattr(transcription.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(submodule_search_locations=[str(root)]))
    monkeypatch.setenv("PATH", os.pathsep.join(["C:\\existing", "C:\\other"]))
    return [str(d) for d in dirs]


def test_library_dirs_are_the_bin_folders(nvidia_packages: list[str]) -> None:
    assert transcription.nvidia_library_dirs() == sorted(nvidia_packages)


def test_no_nvidia_packages_means_no_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription.importlib.util, "find_spec", lambda name: None)
    assert transcription.nvidia_library_dirs() == []


def test_windows_appends_once_after_existing_entries(
        nvidia_packages: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription.os, "name", "nt")
    transcription.expose_nvidia_libraries()
    transcription.expose_nvidia_libraries()
    assert os.environ["PATH"].split(os.pathsep) == ["C:\\existing", "C:\\other",
                                                     *sorted(nvidia_packages)]


def test_other_systems_leave_path_alone(
        nvidia_packages: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription.os, "name", "posix")
    transcription.expose_nvidia_libraries()
    assert os.environ["PATH"] == os.pathsep.join(["C:\\existing", "C:\\other"])


@pytest.mark.parametrize("device, exposed", [("cpu", False), ("cuda", True), ("auto", True)])
def test_libraries_are_exposed_only_for_gpu_attempts(
        monkeypatch: pytest.MonkeyPatch, device: str, exposed: bool) -> None:
    calls: list[str] = []
    monkeypatch.setattr(transcription, "expose_nvidia_libraries", lambda: calls.append("x"))

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, *args, **kwargs):
            return [], SimpleNamespace(language="en", duration=0.0)

    import faster_whisper  # noqa: F401 - patched below; installed with the transcription extra
    monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel)
    FasterWhisperBackend(TranscriptionConfig(enabled=True, device=device)).transcribe(Path("x.wav"))
    assert bool(calls) is exposed
