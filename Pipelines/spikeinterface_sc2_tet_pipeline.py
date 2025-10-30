"""
Minimal SpikeInterface pipeline to run SpykingCircus2 on tetrode recordings.
"""

from __future__ import annotations

import re
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from spikeinterface.core import ChannelSliceRecording, create_sorting_analyzer
from spikeinterface.exporters import export_to_phy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Functions.si_utils import (
    discover_oe_stream_names,
    ensure_geom_and_units,
    set_group_property,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_SECONDS = 300  # set to None for full recording
ROOT_DIR = Path(r"C:/Users/ryoi/Documents/SpikeSorting/recordings")
SESSION_SUBPATH = None  # provide relative Open Ephys path to skip auto-discovery
SESSION_SELECTION = "latest"  # "latest", "earliest", or "index"
SESSION_INDEX = 0  # used only when SESSION_SELECTION == "index"

BASE_OUT = Path(r"C:/Users/ryoi/Documents/SpikeSorting")
SC2_OUT = BASE_OUT / "sc2_outputs"

STREAM_NAME = None  # e.g. "Record Node 125#Acquisition_Board-100.Rhythm Data"

CHANNELS_PER_TETRODE = 4
BAD_CHANNELS = [17,18,19]  # drop by index (0-based) or channel name (e.g. "CH61")
AUTO_BAD_CHANNELS = False  # run SpikeInterface's detect_bad_channels

ATTACH_GEOMETRY = True
LINEARIZE_TRACEVIEW = True
TRACEVIEW_CONTACT_SPACING_UM = 20.0
TRACEVIEW_GROUP_SPACING_UM = 200.0

# Optional narrow-band cleanup before handing data to SC2
APPLY_NOTCH = False
NOTCH_FREQUENCIES = [50, 100, 150]
NOTCH_Q = 30

# Optional overrides for SpykingCircus2 parameters (keys mirror default params structure)
SC2_PARAM_OVERRIDES: dict = {
    # Example tweaks:
    # "filtering": {"freq_min": 300, "freq_max": 6000},
    # "detection": {"method_kwargs": {"detect_threshold": 4}},
}

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


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


def chunk_groups(channel_ids: Iterable, size: int = CHANNELS_PER_TETRODE) -> List[List]:
    ids = list(channel_ids)
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def filter_groups_by_channels(groups: Iterable[Iterable], valid_ids: Sequence) -> List[List]:
    valid_set = set(valid_ids)
    filtered: List[List] = []
    for group in groups:
        subset = [ch for ch in group if ch in valid_set]
        if subset:
            filtered.append(subset)
    return filtered


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
        return np.array([], dtype=int)
    try:
        detected = spre.detect_bad_channels(recording)
    except Exception as exc:
        print(f"Auto bad-channel detection failed: {exc}")
        return np.array([], dtype=int)
    if isinstance(detected, tuple):
        detected = detected[0]
    return np.array(detected, dtype=int)


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
    raise ValueError("SESSION_SELECTION must be 'latest', 'earliest', or 'index'")


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


def preprocess_for_sc2(recording):
    if APPLY_NOTCH and NOTCH_FREQUENCIES:
        rec = recording
        for freq in NOTCH_FREQUENCIES:
            rec = spre.notch_filter(rec, freq=freq, q=NOTCH_Q)
        print(f"Preprocessing: applied notch filters at {NOTCH_FREQUENCIES} Hz (Q={NOTCH_Q}).")
        return rec
    print("Preprocessing skipped: using raw recording (SpykingCircus2 handles filtering internally).")
    return recording


def build_analyzer(sorting, recording, base_folder: Path, label: str):
    folder = base_folder / f"analyzer_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, folder=folder, overwrite=True)
    analyzer.compute({"random_spikes": {"max_spikes_per_unit": 500, "seed": 42}}, verbose=True)
    analyzer.compute("waveforms", ms_before=1.5, ms_after=2.5, dtype="float32", verbose=True, n_jobs=4, chunk_duration="2s")
    analyzer.compute("templates")
    analyzer.compute("principal_components")
    print(f"Analyzer computed for {label} at {folder}")
    return analyzer


def export_for_phy(analyzer, base_folder: Path, label: str, groups, original_index_map: dict):
    folder = base_folder / f"phy_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_to_phy(analyzer, output_folder=folder, remove_if_exists=True)
    params_path = folder / "params.py"

    channel_ids_rec = list(analyzer.recording.channel_ids)
    def lookup_index(ch, fallback):
        if ch in original_index_map:
            return int(original_index_map[ch])
        ch_str = str(ch)
        if ch_str in original_index_map:
            return int(original_index_map[ch_str])
        return int(fallback)

    channel_indices = np.array(
        [lookup_index(ch, idx) for idx, ch in enumerate(channel_ids_rec)],
        dtype=np.int32,
    )
    channel_map = np.arange(len(channel_ids_rec), dtype=np.int32)

    channel_groups_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    channel_shanks_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    text = None

    if params_path.exists():
        text = params_path.read_text()
        if "import numpy as np" not in text:
            text = "import numpy as np\n" + text

        pattern_ids = re.compile(r"channel_ids\s*=.*")
        replacement_ids = f"channel_ids = np.array({channel_indices.tolist()}, dtype=np.int32)"
        text, count_ids = pattern_ids.subn(replacement_ids, text, count=1)
        if not count_ids:
            text += f"\n{replacement_ids}\n"
        try:
            np.save(folder / "channel_ids.npy", channel_indices.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_ids.npy: {exc}")
        pattern_map = re.compile(r"channel_map\s*=.*")
        replacement_map = f"channel_map = np.array({channel_map.tolist()}, dtype=np.int32)"
        text, count_map = pattern_map.subn(replacement_map, text, count=1)
        if not count_map:
            text += f"\n{replacement_map}\n"
        try:
            np.save(folder / "channel_map.npy", channel_map.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_map.npy: {exc}")

    for idx, ch in enumerate(channel_ids_rec):
        orig_idx = lookup_index(ch, idx)
        channel_groups_out[idx] = orig_idx // CHANNELS_PER_TETRODE if CHANNELS_PER_TETRODE else 0
    channel_shanks_out = channel_groups_out.copy()

    if LINEARIZE_TRACEVIEW:
        contact_spacing = float(TRACEVIEW_CONTACT_SPACING_UM)
        group_spacing = float(TRACEVIEW_GROUP_SPACING_UM)
        positions = np.zeros((len(channel_ids_rec), 2), dtype=np.float32)

        for idx, ch in enumerate(channel_ids_rec):
            orig_idx = lookup_index(ch, idx)
            tetrode_idx = orig_idx // CHANNELS_PER_TETRODE if CHANNELS_PER_TETRODE else 0
            slot = orig_idx % CHANNELS_PER_TETRODE if CHANNELS_PER_TETRODE else 0
            col = slot % 2
            row = slot // 2
            positions[idx, 0] = col * contact_spacing
            positions[idx, 1] = tetrode_idx * group_spacing + row * contact_spacing + (slot * 1e-2)

        for name in ("channel_positions.npy", "channel_locations.npy"):
            try:
                np.save(folder / name, positions)
            except Exception as exc:
                print(f"Warning: could not overwrite {name}: {exc}")

        if channel_groups_out.size:
            channel_groups_out[channel_groups_out < 0] = 0
        if channel_shanks_out.size:
            channel_shanks_out[channel_shanks_out < 0] = 0

    if params_path.exists() and text is not None:
        pattern_groups = re.compile(r"channel_groups\s*=.*")
        replacement_groups = f"channel_groups = np.array({channel_groups_out.tolist()}, dtype=np.int32)"
        text, count_groups = pattern_groups.subn(replacement_groups, text, count=1)
        if not count_groups:
            text += f"\nchannel_groups = np.array({channel_groups_out.tolist()}, dtype=np.int32)\n"
        try:
            np.save(folder / "channel_groups.npy", channel_groups_out.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_groups.npy: {exc}")
        try:
            np.save(folder / "channel_shanks.npy", channel_shanks_out.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_shanks.npy: {exc}")
        params_path.write_text(text)
    print(f"Exported {label} to Phy folder {folder}")


# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------


def main():
    SC2_OUT.mkdir(parents=True, exist_ok=True)

    data_path = choose_recording_folder(ROOT_DIR, SESSION_SUBPATH, SESSION_SELECTION, SESSION_INDEX)
    print(f"Recording root: {ROOT_DIR}")
    print(f"SC2 output: {SC2_OUT}")
    print(f"Using Open Ephys folder: {data_path}")

    stream = pick_stream(data_path, STREAM_NAME)
    print(f"Using stream: {stream}")

    recording = se.read_openephys(data_path, stream_name=stream)
    print(recording)
    print(
        "Segments:", recording.get_num_segments(),
        "| Fs:", recording.get_sampling_frequency(),
        "| Channels:", recording.get_num_channels(),
    )

    if recording.get_num_segments() > 1:
        recording = recording.select_segments([0])
        print("Selected segment 0 (single segment).")

    recording = first_seconds(recording, TEST_SECONDS)
    original_channel_order = list(recording.channel_ids)
    original_index_map = {}
    for idx, ch in enumerate(original_channel_order):
        original_index_map[ch] = idx
        original_index_map[str(ch)] = idx
    channel_order = original_channel_order.copy()

    base_groups = chunk_groups(original_channel_order, CHANNELS_PER_TETRODE)
    manual_bad = resolve_bad_channel_ids(recording, BAD_CHANNELS)
    auto_bad = detect_bad_channel_ids(recording, AUTO_BAD_CHANNELS)

    bad_arrays = [arr for arr in (manual_bad, auto_bad) if arr.size]
    if bad_arrays:
        bad_ids = np.unique(np.concatenate(bad_arrays))
        keep_ids = [ch for ch in channel_order if ch not in bad_ids]
        channel_slice_fn = getattr(recording, "channel_slice", None)
        if callable(channel_slice_fn):
            recording = channel_slice_fn(keep_channel_ids=keep_ids)
        else:
            recording = ChannelSliceRecording(recording, keep_ids)
        channel_order = keep_ids
        print(f"Removed {len(bad_ids)} bad channels: {bad_ids.tolist()}")
    else:
        channel_order = original_channel_order.copy()

    groups = filter_groups_by_channels(base_groups, channel_order)
    print(f"Tetrodes: {len(groups)}; first group: {groups[0] if groups else 'n/a'}")

    tetrode_offsets = None
    tetrodes_per_row = max(1, int(np.ceil(np.sqrt(len(groups))))) if groups else 1
    if groups:
        dx = 150.0
        dy = 150.0
        num_rows = int(np.ceil(len(groups) / tetrodes_per_row))
        tetrode_offsets = []
        for idx in range(len(groups)):
            row = idx // tetrodes_per_row
            col = idx % tetrodes_per_row
            x = col * dx
            y = (num_rows - 1 - row) * dy
            tetrode_offsets.append((x, y))

    if ATTACH_GEOMETRY and groups:
        recording = ensure_geom_and_units(
            recording,
            groups,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
        )
        print(f"Geometry attached to recording (tetrodes_per_row={tetrodes_per_row}).")

    rec_sc2 = preprocess_for_sc2(recording)

    if ATTACH_GEOMETRY and groups:
        rec_sc2 = ensure_geom_and_units(
            rec_sc2,
            groups,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
        )

    sc2_run = SC2_OUT / f"sc2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe_rmtree(sc2_run)

    set_group_property(rec_sc2, groups)
    sc2_params = merge_params(ss.Spykingcircus2Sorter.default_params(), SC2_PARAM_OVERRIDES)
    if SC2_PARAM_OVERRIDES:
        print(f"Applying SC2 overrides: {SC2_PARAM_OVERRIDES}")
    sorting_sc2 = ss.run_sorter(
        "spykingcircus2",
        rec_sc2,
        folder=sc2_run,
        verbose=True,
        **sc2_params,
    )
    print(f"SC2 units: {sorting_sc2.get_num_units()} | output: {sc2_run}")

    analyzer_sc2 = build_analyzer(sorting_sc2, rec_sc2, SC2_OUT, "sc2")

    export_for_phy(analyzer_sc2, SC2_OUT, "sc2", groups, original_index_map)

    print("SC2 pipeline complete.")


if __name__ == "__main__":
    main()
