"""Minimal import smoke test for split helper modules.

Run from the SpikeSorting root:
    python Functions/smoke_test_imports.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure `Functions.*` is importable when run as `python Functions/smoke_test_imports.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_imports() -> None:
    """Import helper modules to catch circular-import regressions early."""
    modules = [
        "Functions.session_utils",
        "Functions.channel_utils",
        "Functions.params_utils",
        "Functions.fs_utils",
        "Functions.pipeline_utils",  # compatibility shim
    ]
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(f"Failed importing {module_name}: {exc}") from exc


if __name__ == "__main__":
    test_imports()
    print("Import smoke test passed.")
