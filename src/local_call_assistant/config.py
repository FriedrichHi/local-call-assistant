"""Configuration: an optional TOML file in the runtime root, with safe defaults.

Everything that sends data off the machine is disabled until configured.
Secrets are never stored in the file: the Hermes key is read from the
environment variable the file names.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

APP_DIR_NAME = "local-call-assistant"
CONFIG_FILE_NAME = "config.toml"
HOME_ENV = "LCA_HOME"
CONFIG_ENV = "LCA_CONFIG"


@dataclass(frozen=True)
class TranscriptionConfig:
    enabled: bool = False
    model: str = "small"
    device: str = "auto"  # "auto" tries CUDA, then falls back to CPU
    compute_type: str = "default"
    cpu_compute_type: str = "int8"
    language: str | None = None


@dataclass(frozen=True)
class HermesConfig:
    enabled: bool = False
    base_url: str = ""
    api_key_env: str = "LCA_HERMES_API_KEY"
    model: str = "hermes-agent"
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class ObsidianConfig:
    inbox: Path | None = None


@dataclass(frozen=True)
class Config:
    runtime_root: Path
    max_attempts: int = 3
    ingest_min_age_seconds: float = 60.0
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    hermes: HermesConfig = field(default_factory=HermesConfig)
    obsidian: ObsidianConfig = field(default_factory=ObsidianConfig)


def default_runtime_root() -> Path:
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def config_path(runtime_root: Path) -> Path:
    override = os.environ.get(CONFIG_ENV)
    return Path(override) if override else runtime_root / CONFIG_FILE_NAME


def load_config(runtime_root: Path | None = None) -> Config:
    root = runtime_root or default_runtime_root()
    path = config_path(root)
    data: dict[str, Any] = {}
    if path.exists():
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    return _from_dict(root, data)


def _from_dict(root: Path, data: dict[str, Any]) -> Config:
    obsidian = data.get("obsidian", {})
    inbox = obsidian.get("inbox")
    max_attempts = int(data.get("max_attempts", 3))
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    return Config(
        runtime_root=root,
        max_attempts=max_attempts,
        ingest_min_age_seconds=float(data.get("ingest_min_age_seconds", 60.0)),
        transcription=TranscriptionConfig(**data.get("transcription", {})),
        hermes=HermesConfig(**data.get("hermes", {})),
        obsidian=ObsidianConfig(inbox=Path(inbox).expanduser() if inbox else None),
    )
