"""Shared test helpers that avoid sandbox-hostile Windows temp ACLs."""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_TMP_ROOT = PROJECT_ROOT / ".tmp_tests"


@contextmanager
def workspace_tempdir(prefix: str = "tmp"):
    """Create a writable temp dir inside the repo workspace.

    Python's tempfile helpers create Windows directories with restrictive ACLs in
    this environment, which prevents subsequent file writes. Creating the folder
    with Path.mkdir() preserves the workspace ACLs that the sandbox can access.
    """

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    path = TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
