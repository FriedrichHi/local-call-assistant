"""Durable file writes: unique temp file, fsync, rename, then sync the directory."""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path

_UNSUPPORTED_DIRECTORY_FSYNC = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EBADF})


def fsync_directory(directory: Path) -> None:
    """Persist renames and deletions in ``directory`` (POSIX; Windows has no equivalent)."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as error:
        # Some filesystems (certain network or FUSE mounts) cannot fsync a
        # directory; there is nothing stronger to do there.
        if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC:
            raise
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = _stage(path, data)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def create_new_text(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` only if nothing exists there. Never overwrites.

    Returns False when the target already exists. Content is staged in a temp
    file unique to this call, so neither a crash nor a concurrent caller can
    truncate a published file.
    """
    temporary = _stage(path, text.encode("utf-8"))
    try:
        _publish_without_overwrite(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return False
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)
    return True


def _stage(path: Path, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _publish_without_overwrite(source: Path, target: Path) -> None:
    # Windows rename refuses an existing target; POSIX rename would replace it,
    # so POSIX uses link (which refuses) and then drops the unique temp name.
    if os.name == "nt":
        os.rename(source, target)
        return
    os.link(source, target)
    source.unlink()
