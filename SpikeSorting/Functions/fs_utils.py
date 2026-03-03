"""Filesystem utilities shared across pipeline helpers."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def safe_rmtree(path: Path, retries: int = 6, wait_s: float = 0.5) -> None:
    """Remove a directory with retry loop to handle transient file locks."""
    for _ in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except (PermissionError, OSError):
            time.sleep(wait_s)
    if path.exists():
        shutil.rmtree(path)


__all__ = [
    "safe_rmtree",
]
