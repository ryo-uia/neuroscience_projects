"""Channel/group/bad-channel helpers for pipeline preprocessing.

Note: heavy SpikeInterface imports are lazy-loaded to keep module import fast.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from os import PathLike
from typing import Iterable, List, Sequence

import numpy as np

from .run_utils import log_info, log_warn


def _load_spre():
    """Lazy-load spikeinterface.preprocessing to keep module import lightweight."""
    try:
        import spikeinterface.preprocessing as spre
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface is required for preprocessing utilities. "
            "Activate the spikeinterface environment before running this pipeline."
        ) from exc
    return spre


def safe_channel_slice(recording, keep_ids):
    """Slice channels with API-compatible fallbacks across SI versions."""
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
                    log_warn(
                        "recording.channel_slice exists but no tested signature matched; "
                        "falling back to ChannelSliceRecording."
                    )
    try:
        from spikeinterface.core import ChannelSliceRecording
    except Exception as exc:
        raise RuntimeError("channel_slice is unavailable and ChannelSliceRecording import failed") from exc
    return ChannelSliceRecording(recording, keep_ids)


def chunk_groups(channel_ids: Iterable, size: int = 4) -> List[List]:
    """Split a channel-id sequence into fixed-size groups."""
    ids = list(channel_ids)
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def _is_empty_sequence(value) -> bool:
    """Return True when value is None or has zero length/size."""
    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:
        try:
            return np.asarray(value).size == 0
        except Exception:
            return False


def _resolve_ch_label_variant(as_int: int, id_set: set, name_to_id: dict):
    """Resolve CH-style label variants (for example CH1/CH01/CH001) before index fallback."""
    for label in (f"CH{as_int}", f"CH{as_int:02d}", f"CH{as_int:03d}"):
        if label in id_set:
            return label
        if label in name_to_id:
            return name_to_id[label]
    return None


def resolve_manual_groups(recording, manual_groups: Sequence[Sequence[int | str]]) -> List[List]:
    """Resolve user-specified channel groups to recording channel_ids.

    Example:
        groups = resolve_manual_groups(rec, [["CH40", "CH38", "CH36", "CH34"], ["CH48", "CH46"]])
    """
    if _is_empty_sequence(manual_groups):
        return []

    channel_ids = list(recording.channel_ids)
    id_set = set(channel_ids)

    try:
        if "channel_name" in recording.get_property_keys():
            names = list(recording.get_property("channel_name"))
        else:
            names = None
    except Exception as exc:
        log_warn(f"failed to read channel_name property; resolving groups from channel_ids only ({exc})")
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
            elif isinstance(item, str):
                match = re.match(r"^[A-Za-z]*?(\d+)$", item)
                if match:
                    as_int = int(match.group(1))
                    # Prefer explicit CH label variants before positional/index fallback.
                    candidate = _resolve_ch_label_variant(as_int, id_set, name_to_id)
                    if candidate is not None:
                        pass
                    elif as_int in id_set:
                        candidate = as_int
                    elif 0 <= as_int < len(channel_ids):
                        candidate = channel_ids[as_int]

            if candidate is None:
                log_warn(f"could not resolve channel '{item}' in CHANNEL_GROUPS[{g_idx}]")
                continue
            if candidate in seen:
                continue
            resolved.append(candidate)
            seen.add(candidate)
        if resolved:
            resolved_groups.append(resolved)
    flat = [ch for grp in resolved_groups for ch in grp]
    if len(flat) != len(set(flat)):
        log_warn("some channels appear in multiple groups.")
    return resolved_groups


def _normalize_json_path(path, label: str) -> Path | None:
    """Normalize/validate user path input for JSON loaders.

    Returns None and logs a warning for unsupported types.
    """
    if not path:
        return None
    if isinstance(path, (str, bytes, PathLike)):
        return Path(path)
    log_warn(
        f"{label} path has invalid type {type(path).__name__}; "
        "expected str/Path-like. Ignoring."
    )
    return None


def load_channel_groups_from_path(path: Path | str | None) -> List[List] | None:
    """Load channel groups from JSON file path; return None on invalid/missing input."""
    p = _normalize_json_path(path, "channel groups")
    if p is None:
        return None
    if not p.exists():
        log_warn(f"channel groups file not found: {p}")
        return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, (list, tuple)):
            if not data:
                log_warn(f"channel groups file is empty: {p}")
                return None
            return [list(g) for g in data]
        log_warn(f"expected list of lists in {p}")
    except Exception as exc:
        log_warn(f"failed to read channel groups from {p}: {exc}")
    return None


def load_bad_channels_from_path(path: Path | str | None) -> list | None:
    """Load bad-channel IDs from JSON file path; return None on invalid/missing input."""
    p = _normalize_json_path(path, "bad-channels")
    if p is None:
        return None
    if not p.exists():
        log_warn(f"bad-channels file not found: {p}")
        return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                log_warn(f"bad-channels file is empty: {p} (using inline BAD_CHANNELS)")
                return None
            return list(data)
        log_warn(f"expected list of channel IDs in {p}")
    except Exception as exc:
        log_warn(f"failed to read bad channels from {p}: {exc}")
    return None


def resolve_bad_channel_ids(recording, manual_list: Sequence[int | str]) -> np.ndarray:
    """Resolve user bad-channel tokens to concrete recording channel IDs."""
    if _is_empty_sequence(manual_list):
        return np.array([], dtype=object)

    channel_ids = list(recording.channel_ids)
    id_set = set(channel_ids)

    try:
        if "channel_name" in recording.get_property_keys():
            names = list(recording.get_property("channel_name"))
        else:
            names = None
    except Exception as exc:
        log_warn(f"failed to read channel_name property; resolving bad channels from channel_ids only ({exc})")
        names = None

    name_to_id = {name: channel_ids[idx] for idx, name in enumerate(names)} if names else {}

    resolved = []
    for item in manual_list:
        if isinstance(item, (int, np.integer)):
            if item in id_set:
                resolved.append(item)
                continue
            if 0 <= int(item) < len(channel_ids):
                resolved.append(channel_ids[int(item)])
                continue
            log_warn(f"could not resolve bad channel '{item}'")
            continue

        if isinstance(item, str):
            if item in id_set:
                resolved.append(item)
                continue
            if item in name_to_id:
                resolved.append(name_to_id[item])
                continue
            match = re.match(r"^[A-Za-z]*?(\d+)$", item)
            if match:
                as_int = int(match.group(1))
                # Prefer explicit CH label variants before positional/index fallback.
                candidate = _resolve_ch_label_variant(as_int, id_set, name_to_id)
                if candidate is not None:
                    resolved.append(candidate)
                    continue
                if as_int in id_set:
                    resolved.append(as_int)
                    continue
                if 0 <= as_int < len(channel_ids):
                    resolved.append(channel_ids[as_int])
                    continue
            log_warn(f"could not resolve bad channel '{item}'")
            continue

        log_warn(f"could not resolve bad channel '{item}'")

    if not resolved:
        return np.array([], dtype=object)

    unique = []
    seen = set()
    for value in resolved:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return np.array(unique, dtype=object)


def detect_bad_channel_ids(recording, enabled: bool, method: str | None = None, **kwargs) -> np.ndarray:
    """Run SI bad-channel detection and normalize output to channel IDs.

    Note: returns channel IDs, not positional indices.

    Example:
        bad_ids = detect_bad_channel_ids(rec, True, method="mad")
    """
    if not enabled:
        return np.array([], dtype=object)
    spre = _load_spre()
    try:
        if method is not None:
            try:
                detected = spre.detect_bad_channels(recording, method=method, **kwargs)
            except TypeError:
                detected = spre.detect_bad_channels(recording, **kwargs)
        else:
            detected = spre.detect_bad_channels(recording, **kwargs)
    except Exception as exc:
        log_warn(f"auto bad-channel detection failed: {exc}")
        return np.array([], dtype=object)
    if isinstance(detected, tuple):
        detected = detected[0]
    detected_arr = np.asarray(detected)
    if detected_arr.dtype == bool:
        if detected_arr.size == recording.get_num_channels():
            detected_arr = np.asarray(recording.channel_ids, dtype=object)[detected_arr]
        else:
            detected_arr = np.array([], dtype=object)
    return np.array(detected_arr, dtype=np.asarray(recording.channel_ids).dtype)


def build_oe_index_map(recording, fallback_map: dict) -> dict:
    """Return channel_id -> numeric OE index when available; fallback to original order."""
    ch_ids = list(recording.channel_ids)
    try:
        prop_keys = recording.get_property_keys()
    except Exception as exc:
        log_warn(f"failed to read property keys; using OE index fallback map ({exc})")
        prop_keys = []

    def _to_int_or_none(value):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            as_float = float(value)
            if as_float.is_integer():
                return int(as_float)
            return None
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"-?\d+", text):
                return int(text)
        return None

    for key in ("oe_channel_index", "oe_channel_indices", "channel_index", "channel_indices", "oe_index"):
        if key in prop_keys:
            try:
                vals = recording.get_property(key)
                if vals is None or len(vals) != len(ch_ids):
                    continue
                mapping = {}
                ok = True
                for ch, value in zip(ch_ids, vals):
                    idx = _to_int_or_none(value)
                    if idx is None:
                        ok = False
                        break
                    mapping[ch] = idx
                    mapping[str(ch)] = idx
                if ok and len(mapping) >= len(ch_ids):
                    return mapping
            except Exception:
                pass

    log_info("OE index map: explicit numeric OE index property not found; using current channel order fallback.")
    mapping = {}
    for idx, ch in enumerate(ch_ids):
        fallback = fallback_map.get(ch, fallback_map.get(str(ch), idx))
        try:
            oe_idx = int(fallback)
        except Exception:
            oe_idx = int(idx)
        mapping[ch] = oe_idx
        mapping[str(ch)] = oe_idx
    return mapping


def filter_groups_with_valid_ids(groups: Iterable[Iterable], valid_ids: Sequence) -> tuple[List[List], List[int]]:
    """Return filtered groups and their original indices (for mapping to group IDs)."""
    valid_set = set(valid_ids)
    filtered: List[List] = []
    indices: List[int] = []
    for idx, group in enumerate(groups):
        subset = [ch for ch in group if ch in valid_set]
        if subset:
            filtered.append(subset)
            indices.append(idx)
    return filtered, indices


def filter_groups_with_indices(groups: Iterable[Iterable], valid_ids: Sequence) -> tuple[List[List], List[int]]:
    """Compatibility alias for filter_groups_with_valid_ids()."""
    return filter_groups_with_valid_ids(groups, valid_ids)


__all__ = [
    "safe_channel_slice",
    "chunk_groups",
    "resolve_manual_groups",
    "load_channel_groups_from_path",
    "load_bad_channels_from_path",
    "resolve_bad_channel_ids",
    "detect_bad_channel_ids",
    "build_oe_index_map",
    "filter_groups_with_valid_ids",
    "filter_groups_with_indices",
]
