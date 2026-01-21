import json
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import spikeinterface.preprocessing as spre

from Functions.si_utils import discover_oe_stream_names

# Note: export-time uV scaling (with zero-offset fallback when offsets are missing)
# is handled in the pipelines, not in this helper module.


def pick_stream(data_path: Path, preferred: str | None) -> str:
    if preferred:
        return preferred
    names = discover_oe_stream_names(data_path)
    neural = [n for n in names if "ADC" not in n and "SYNC" not in n]
    if not neural:
        raise RuntimeError("No neural streams found; set STREAM_NAME manually.")
    print("Available streams:")
    for idx, name in enumerate(names):
        print(f"  [{idx}] {name}")
    return neural[0]


def first_seconds(recording, seconds: int | None):
    if seconds is None:
        return recording
    fs = recording.get_sampling_frequency()
    total = (
        recording.get_num_samples(0)
        if hasattr(recording, "get_num_samples")
        else recording.get_num_frames(0)
    )
    end = min(int(fs * seconds), total)
    print(f"Recording sliced to first {seconds}s (end_frame={end}).")
    return recording.frame_slice(0, end)


def safe_channel_slice(recording, keep_ids):
    channel_slice_fn = getattr(recording, "channel_slice", None)
    if callable(channel_slice_fn):
        try:
            return channel_slice_fn(keep_channel_ids=keep_ids)
        except TypeError:
            try:
                return channel_slice_fn(channel_ids=keep_ids)
            except TypeError:
                try:
                    return channel_slice_fn(keep_ids)
                except TypeError:
                    pass
    try:
        from spikeinterface.core import ChannelSliceRecording
    except Exception as exc:
        raise RuntimeError("channel_slice is unavailable and ChannelSliceRecording import failed") from exc
    return ChannelSliceRecording(recording, keep_ids)


def chunk_groups(channel_ids: Iterable, size: int = 4) -> List[List]:
    ids = list(channel_ids)
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def resolve_manual_groups(recording, manual_groups: Sequence[Sequence[int | str]]) -> List[List]:
    """Resolve user-specified channel groups to recording channel_ids."""
    if not manual_groups:
        return []

    channel_ids = list(recording.channel_ids)
    id_set = set(channel_ids)

    try:
        if "channel_name" in recording.get_property_keys():
            names = list(recording.get_property("channel_name"))
        else:
            names = None
    except Exception:
        names = None

    name_to_id = {name: channel_ids[idx] for idx, name in enumerate(names)} if names else {}

    resolved_groups: List[List] = []
    for g_idx, group in enumerate(manual_groups):
        resolved: List = []
        seen = set()
        for item in group:
            candidate = None
            if isinstance(item, (int, np.integer)) and item in id_set:
                candidate = item
            elif isinstance(item, (int, np.integer)) and 0 <= int(item) < len(channel_ids):
                candidate = channel_ids[int(item)]
            elif isinstance(item, str) and item in id_set:
                candidate = item
            elif isinstance(item, str) and item.isdigit():
                as_int = int(item)
                if as_int in id_set:
                    candidate = as_int
                elif 0 <= as_int < len(channel_ids):
                    candidate = channel_ids[as_int]
            elif isinstance(item, str) and item in name_to_id:
                candidate = name_to_id[item]

            if candidate is None:
                print(f"Warning: could not resolve channel '{item}' in CHANNEL_GROUPS[{g_idx}]")
                continue
            if candidate in seen:
                continue
            resolved.append(candidate)
            seen.add(candidate)
        if resolved:
            resolved_groups.append(resolved)
    return resolved_groups


def load_channel_groups_from_path(path: Path | None) -> List[List] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Warning: channel groups file not found: {p}")
        return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, (list, tuple)):
            if not data:
                print(f"Warning: channel groups file is empty: {p}")
                return None
            return [list(g) for g in data]
        print(f"Warning: expected list of lists in {p}")
    except Exception as exc:
        print(f"Warning: failed to read channel groups from {p}: {exc}")
    return None


def load_bad_channels_from_path(path: Path | None) -> list | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Warning: bad-channels file not found: {p}")
        return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, (list, tuple)):
            return list(data)
        print(f"Warning: expected list of channel IDs in {p}")
    except Exception as exc:
        print(f"Warning: failed to read bad channels from {p}: {exc}")
    return None


def resolve_bad_channel_ids(recording, manual_list: Sequence[int | str]) -> np.ndarray:
    if not manual_list:
        return np.array([], dtype=object)

    channel_ids = list(recording.channel_ids)
    id_set = set(channel_ids)

    try:
        if "channel_name" in recording.get_property_keys():
            names = list(recording.get_property("channel_name"))
        else:
            names = None
    except Exception:
        names = None

    name_to_id = {name: channel_ids[idx] for idx, name in enumerate(names)} if names else {}

    resolved = []
    for item in manual_list:
        if isinstance(item, (int, np.integer)):
            if item in id_set:
                resolved.append(item)
            elif 0 <= int(item) < len(channel_ids):
                resolved.append(channel_ids[int(item)])
            else:
                print(f"Warning: could not resolve bad channel '{item}'")
        elif isinstance(item, str):
            if item in id_set:
                resolved.append(item)
            elif item in name_to_id:
                resolved.append(name_to_id[item])
            else:
                print(f"Warning: could not resolve bad channel '{item}'")
        else:
            print(f"Warning: could not resolve bad channel '{item}'")

    if not resolved:
        return np.array([], dtype=object)

    unique = []
    seen = set()
    for value in resolved:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return np.array(unique, dtype=object)


def detect_bad_channel_ids(recording, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.array([], dtype=object)
    try:
        detected = spre.detect_bad_channels(recording)
    except Exception as exc:
        print(f"Auto bad-channel detection failed: {exc}")
        return np.array([], dtype=object)
    if isinstance(detected, tuple):
        detected = detected[0]
    detected_arr = np.asarray(detected)
    if detected_arr.dtype == bool:
        if detected_arr.size == recording.get_num_channels():
            detected_arr = np.asarray(recording.channel_ids, dtype=object)[detected_arr]
        else:
            detected_arr = np.array([], dtype=object)
    return np.array(detected_arr, dtype=object)


def merge_params(defaults: dict, overrides: dict) -> dict:
    """Deep-merge overrides into defaults without mutating the original dict."""
    if not overrides:
        return defaults
    merged = deepcopy(defaults)

    def _apply(target: dict, source: dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _apply(target[key], value)
            else:
                target[key] = value

    _apply(merged, overrides)
    return merged


def discover_recording_folders(root: Path) -> List[Path]:
    if not root.exists():
        return []
    folders = {oebin.parent.resolve() for oebin in root.rglob("structure.oebin")}
    return sorted(folders)


def choose_recording_folder(root: Path, subpath, selection: str, index: int) -> Path:
    root = Path(root)
    if subpath:
        candidate = root / subpath
        if not candidate.exists():
            raise FileNotFoundError(f"Recording path not found: {candidate}")
        return candidate

    folders = discover_recording_folders(root)
    if not folders:
        raise RuntimeError(f"No Open Ephys sessions found under {root}")

    if selection == "latest":
        return folders[-1]
    if selection == "earliest":
        return folders[0]
    if selection == "index":
        if index < 0 or index >= len(folders):
            print("Available sessions:")
            for idx, path in enumerate(folders):
                print(f"  [{idx}] {path.relative_to(root)}")
            raise ValueError(f"SESSION_INDEX {index} out of range")
        return folders[index]
    if selection in ("prompt", "interactive"):
        print("Available sessions:")
        for idx, path in enumerate(folders):
            rel = path.relative_to(root)
            try:
                stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                stamp = "mtime unavailable"
            print(f"  [{idx}] {rel} (modified {stamp})")
        default_idx = len(folders) - 1
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


def safe_rmtree(path: Path, retries: int = 6, wait_s: float = 0.5) -> None:
    for _ in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(wait_s)
    if path.exists():
        shutil.rmtree(path)
