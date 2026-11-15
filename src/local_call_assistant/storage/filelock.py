"""Advisory per-job locks.

Whoever records, seals, recovers or processes a job holds its lock for the
whole operation. The operating system releases the lock when the holder
dies, so a lock that cannot be taken means another live process owns the job.
Lock files are never deleted (on POSIX that would let two holders coexist).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_LOCK_BYTES = 1


class JobLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[bytes] | None = None

    def acquire(self) -> bool:
        """Take the lock without blocking. Returns False if another process holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            _lock(handle)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            _unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None


def _lock(handle: IO[bytes]) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_BYTES)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: IO[bytes]) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTES)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
