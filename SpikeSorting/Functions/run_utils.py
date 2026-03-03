"""
Run I/O and logging utilities shared across pipeline scripts.

Shared between: spikeinterface_sc2_tet_pipeline.py, spikeinterface_sc2_mixed_pipeline.py
"""
from __future__ import annotations

import builtins
import sys
from datetime import datetime
from pathlib import Path

from .fs_utils import safe_rmtree


def log_info(message: str) -> None:
    """Emit an informational log line."""
    builtins.print(message, file=sys.stdout)


def log_warn(message: str) -> None:
    """Emit a warning log line."""
    text = str(message)
    if text.startswith("WARNING:"):
        builtins.print(text, file=sys.stderr)
    else:
        builtins.print(f"WARNING: {text}", file=sys.stderr)


class TeeStream:
    """Duplicate writes to multiple streams (stdout/stderr + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return bool(self._streams) and hasattr(self._streams[0], "isatty") and self._streams[0].isatty()


def reserve_run_folder(base_out: Path) -> Path:
    """Reserve a unique run folder path under `sc2_outputs`."""
    out_root = base_out / "sc2_outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = out_root / f"sc2_run_{run_tag}"
    suffix = 1
    while run_folder.exists():
        run_folder = out_root / f"sc2_run_{run_tag}_{suffix:02d}"
        suffix += 1
    return run_folder


def enable_run_logging(log_path: Path) -> object:
    """Mirror stdout/stderr to a per-run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, handle)
    sys.stderr = TeeStream(sys.stderr, handle)
    return handle


def initialize_run_io(base_out: Path, sc2_run: Path, *, export_to_si_gui: bool):
    """Create output dirs, enable run logging, and return resolved run I/O paths."""
    sc2_out = base_out / "sc2_outputs"
    si_gui_out = base_out / "si_gui_exports"
    sc2_out.mkdir(parents=True, exist_ok=True)
    if export_to_si_gui:
        si_gui_out.mkdir(parents=True, exist_ok=True)
    run_log_path = sc2_out / "run_logs" / f"{sc2_run.name}.log"
    run_log_handle = enable_run_logging(run_log_path)
    log_info(f"Run Log: {run_log_path}")
    return sc2_out, si_gui_out, run_log_handle, run_log_path


def disable_run_logging(handle) -> None:
    """Restore stdout/stderr and close the log handle."""
    if isinstance(sys.stdout, TeeStream):
        sys.stdout = sys.stdout._streams[0]
    if isinstance(sys.stderr, TeeStream):
        sys.stderr = sys.stderr._streams[0]
    if handle:
        try:
            handle.flush()
        finally:
            handle.close()


def export_for_si_gui(analyzer, base_folder: Path, label: str) -> Path:
    """Save a Zarr copy that SpikeInterface GUI can open directly."""
    from datetime import datetime

    folder = base_folder / f"si_gui_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing folder if present
    if folder.exists():
        safe_rmtree(folder)

    analyzer.save_as(format="zarr", folder=folder)
    log_info(f"Exported {label} analyzer to SpikeInterface GUI folder {folder}")
    return folder


__all__ = [
    "TeeStream",
    "reserve_run_folder",
    "enable_run_logging",
    "initialize_run_io",
    "disable_run_logging",
    "log_info",
    "log_warn",
    "export_for_si_gui",
]
