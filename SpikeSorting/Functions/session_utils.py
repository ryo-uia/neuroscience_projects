"""Session and stream selection helpers for pipeline entry points.

Note: stream discovery relies on SI helpers that lazy-load heavy dependencies.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from .run_utils import log_info, log_warn
from .si_utils import discover_oe_stream_names, normalize_stream_name


def pick_stream(data_path: Path, preferred: str | None) -> str:
    """Select a neural stream name, honoring explicit override when provided.

    Example:
        stream = pick_stream(data_path, "Record Node 125#Acquisition_Board-100.Rhythm Data")
    """
    names = discover_oe_stream_names(data_path)
    if preferred:
        if preferred in names:
            return preferred
        preferred_norm = normalize_stream_name(preferred)
        normalized_matches = [n for n in names if normalize_stream_name(n) == preferred_norm]
        if len(normalized_matches) == 1:
            resolved = normalized_matches[0]
            log_warn(
                f"STREAM_NAME '{preferred}' did not match exactly; using '{resolved}' via normalized match."
            )
            return resolved
        if len(normalized_matches) > 1:
            raise RuntimeError(
                "STREAM_NAME is ambiguous after normalization. "
                f"Requested: {preferred!r}. Matches: {normalized_matches}"
            )
        raise RuntimeError(
            f"STREAM_NAME '{preferred}' not found. "
            f"Available streams: {names}"
        )
    neural = [n for n in names if "ADC" not in n and "SYNC" not in n]
    if not neural:
        raise RuntimeError("No neural streams found; set STREAM_NAME manually.")
    if len(neural) > 1:
        log_info("Available streams:")
        for idx, name in enumerate(names):
            log_info(f"  [{idx}] {name}")
        default_idx = names.index(neural[0])
        prompt = f"Select stream index [default {default_idx}]: "
        try:
            choice = input(prompt).strip()
        except EOFError:
            choice = ""
        if choice:
            try:
                pick = int(choice)
                if 0 <= pick < len(names):
                    chosen = names[pick]
                else:
                    log_warn(f"Stream index {pick} out of range; using default.")
                    chosen = neural[0]
            except ValueError:
                log_warn("Invalid stream selection; using default.")
                chosen = neural[0]
        else:
            chosen = neural[0]
        log_info(f"Using stream: {chosen}")
        return chosen
    return neural[0]


def choose_config_json(label: str, candidates: list[Path], default_path: Path | None) -> Path | None:
    """Prompt-select a JSON config path from candidates with a default fallback."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    default_idx = 0
    if default_path in candidates:
        default_idx = candidates.index(default_path)
    log_info(f"Available {label} JSON files:")
    for idx, path in enumerate(candidates):
        log_info(f"  [{idx}] {path}")
    try:
        resp = input(f"Select {label} JSON index [default {default_idx}]: ").strip()
    except EOFError:
        return candidates[default_idx]
    if resp == "":
        return candidates[default_idx]
    try:
        idx = int(resp)
    except ValueError:
        return candidates[default_idx]
    if 0 <= idx < len(candidates):
        return candidates[idx]
    return candidates[default_idx]


def resolve_config_sources(
    args,
    *,
    project_root: Path,
    use_config_jsons_default: bool,
    channel_groups_path_default,
    bad_channels_path_default,
):
    """Resolve config JSON paths from defaults/env/CLI prompts."""
    env_groups_path = os.environ.get("SPIKESORT_CHANNEL_GROUPS", None)
    env_bad_path = os.environ.get("SPIKESORT_BAD_CHANNELS", None)
    config_dir = project_root / "config"

    # Start from module-level defaults (inline JSON paths), then override via env/CLI/prompt.
    channel_groups_path = channel_groups_path_default
    bad_channels_path = bad_channels_path_default
    use_config_jsons = bool(use_config_jsons_default) and not bool(getattr(args, "no_config_json", False))

    if (
        use_config_jsons
        and not getattr(args, "channel_groups", None)
        and not env_groups_path
        and not channel_groups_path
    ):
        group_candidates = sorted(config_dir.glob("channel_groups_*.json"))
        if group_candidates:
            channel_groups_path = choose_config_json(
                "channel groups",
                group_candidates,
                group_candidates[0] if len(group_candidates) == 1 else None,
            )

    if (
        use_config_jsons
        and not getattr(args, "bad_channels", None)
        and not env_bad_path
        and not bad_channels_path
    ):
        bad_candidates = sorted(config_dir.glob("bad_channels_*.json"))
        if bad_candidates:
            bad_channels_path = choose_config_json(
                "bad channels",
                bad_candidates,
                bad_candidates[0] if len(bad_candidates) == 1 else None,
            )

    return env_groups_path, env_bad_path, channel_groups_path, bad_channels_path, use_config_jsons


def first_seconds(recording, seconds: int | None):
    """Return recording truncated to the first N seconds (or unchanged for None)."""
    if seconds is None:
        return recording
    try:
        sec_value = float(seconds)
    except Exception as exc:
        raise ValueError("TEST_SECONDS must be None or a positive number of seconds.") from exc
    if not np.isfinite(sec_value) or sec_value <= 0:
        raise ValueError("TEST_SECONDS must be > 0 when set.")
    fs = recording.get_sampling_frequency()
    total = (
        recording.get_num_samples(0)
        if hasattr(recording, "get_num_samples")
        else recording.get_num_frames(0)
    )
    end = min(int(fs * sec_value), total)
    if end <= 0:
        raise ValueError("TEST_SECONDS resulted in an empty recording slice.")
    log_info(f"Recording sliced to first {sec_value:g}s (end_frame={end}).")
    return recording.frame_slice(0, end)


def discover_recording_folders(root: Path) -> List[Path]:
    """Find Open Ephys recording folders by searching for structure.oebin files."""
    if not root.exists():
        return []
    folders = {oebin.parent.resolve() for oebin in root.rglob("structure.oebin")}
    return sorted(folders)


def choose_recording_folder(root: Path, subpath, selection: str | None, index: int) -> Path:
    """Resolve a recording folder from explicit subpath or selection strategy.

    Example:
        data_path = choose_recording_folder(root, None, "prompt", 0)
    """
    root = Path(root)
    root_abs = root.resolve()

    def _display_path(path: Path) -> Path:
        try:
            return path.relative_to(root_abs)
        except Exception as exc:
            log_warn(f"could not relativize session path; displaying absolute path ({exc})")
            return path

    if subpath:
        candidate = Path(subpath)
        if not candidate.is_absolute():
            candidate = root_abs / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Recording path not found: {candidate}")
        return candidate.resolve()

    folders = discover_recording_folders(root_abs)
    if not folders:
        raise RuntimeError(f"No Open Ephys sessions found under {root}")

    def _safe_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception as exc:
            log_warn(f"failed to read session mtime for {path}; using -inf ({exc})")
            return float("-inf")

    if selection is None or selection == "latest":
        return max(folders, key=lambda p: (_safe_mtime(p), str(p)))
    if selection == "earliest":
        return min(folders, key=lambda p: (_safe_mtime(p), str(p)))
    if selection == "index":
        if index < 0 or index >= len(folders):
            log_info("Available sessions:")
            for idx, path in enumerate(folders):
                log_info(f"  [{idx}] {_display_path(path)}")
            raise ValueError(f"SESSION_INDEX {index} out of range")
        return folders[index]
    if selection in ("prompt", "interactive"):
        log_info("Available sessions:")
        for idx, path in enumerate(folders):
            rel = _display_path(path)
            try:
                stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as exc:
                log_warn(f"failed to format mtime for {path}; showing fallback text ({exc})")
                stamp = "mtime unavailable"
            log_info(f"  [{idx}] {rel} (modified {stamp})")
        default_idx = max(
            range(len(folders)),
            key=lambda i: (_safe_mtime(folders[i]), str(folders[i])),
        )
        prompt = f"Select session index [default {default_idx}]: "
        try:
            choice = input(prompt).strip()
        except EOFError:
            choice = ""
        if not choice:
            return folders[default_idx]
        try:
            chosen = int(choice)
        except ValueError:
            raise ValueError(f"Invalid session choice: {choice}")
        if chosen < 0 or chosen >= len(folders):
            raise ValueError(f"Session choice {chosen} out of range (0-{len(folders) - 1})")
        return folders[chosen]
    raise ValueError("SESSION_SELECTION must be 'latest', 'earliest', 'index', or 'prompt'")


__all__ = [
    "pick_stream",
    "choose_config_json",
    "resolve_config_sources",
    "first_seconds",
    "discover_recording_folders",
    "choose_recording_folder",
]
