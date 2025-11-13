import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.core import (
    ChannelSliceRecording,
    NumpySorting,
    aggregate_units,
    create_sorting_analyzer,
    load_sorting_analyzer,
)
from spikeinterface.exporters import export_to_phy
from spikeinterface.sorters.utils.misc import SpikeSortingError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Functions.si_utils import (
    discover_oe_stream_names,
    ensure_geom_and_units,
    reorder_recording_by_locations,
    set_group_property,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_SECONDS = 300  # set to None for full recording
ROOT_DIR = Path(r"C:/Users/ryoi/Documents/Code/SpikeSorting/recordings")
SESSION_SUBPATH = None  # set to relative path if you want a specific session
SESSION_SELECTION = "latest"  # "latest", "earliest", or "index"
SESSION_INDEX = 0  # used when SESSION_SELECTION == "index"

BASE_OUT = Path(r"C:/Users/ryoi/Documents/Code/SpikeSorting")
KS4_OUT = BASE_OUT / "ks4_outputs"
SC2_OUT = BASE_OUT / "sc2_outputs"

STREAM_NAME = None  # e.g. "Record Node 125#Acquisition_Board-100.Rhythm Data"

CHANNELS_PER_TETRODE = 4
BAD_CHANNELS = []  # channels to drop; use indices (e.g. 17) or names (e.g. "CH61")
AUTO_BAD_CHANNELS = False  # run SpikeInterface's detect_bad_channels

ATTACH_GEOMETRY = True
RUN_KS4 = True
RUN_SC2 = True
KS4_TORCH_DEVICE = "cpu"  # switch to "cuda" when GPU available

REORDER_FOR_PHY = False  # True -> sort channels by geometry before preprocessing
LINEARIZE_TRACEVIEW = True
TRACEVIEW_CONTACT_SPACING_UM = 20.0
TRACEVIEW_GROUP_SPACING_UM = 200.0

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def set_seed():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    random.seed(0)
    np.random.seed(0)


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


def chunk_tetrode_groups(channel_ids, size=CHANNELS_PER_TETRODE):
    ids = list(channel_ids)
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def filter_groups_by_channels(groups, valid_ids):
    valid_ids = set(valid_ids)
    filtered = []
    for group in groups:
        subset = [ch for ch in group if ch in valid_ids]
        if subset:
            filtered.append(subset)
    return filtered


def resolve_bad_channel_ids(recording, manual_list):
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


def detect_bad_channel_ids(recording, enabled):
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


def discover_recording_folders(root: Path):
    if not root.exists():
        return []
    folders = {oebin.parent.resolve() for oebin in root.rglob("structure.oebin")}
    return sorted(folders)


def choose_recording_folder(root, subpath, selection, index):
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


def preprocess_for_sorters(recording, groups):
    rec_bp = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    for f0 in (50, 100, 150):
        rec_bp = spre.notch_filter(rec_bp, freq=f0, q=30)
    if groups:
        rec_ks4 = spre.common_reference(rec_bp, reference="global", operator="median", groups=groups)
        print("Preprocessing ready: bandpass + notch; KS4 uses median CAR, SC2 keeps bandpass only.")
    else:
        rec_ks4 = rec_bp
        print("Preprocessing ready: bandpass + notch; no CAR applied.")
    rec_sc2 = rec_bp
    return rec_ks4, rec_sc2


def safe_rmtree(path: Path, retries: int = 6, wait_s: float = 0.5):
    for _ in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(wait_s)
    if path.exists():
        shutil.rmtree(path)


def cache_recording(folder: Path, recording, label: str):
    cache_dir = folder / f"cached_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe_rmtree(cache_dir)
    recording.save(folder=cache_dir, format="binary", dtype="float32", chunk_duration="1s", overwrite=True)
    print(f"Cached '{label}' to {cache_dir}")


def run_sorter(name, recording, folder, params, grouping_property=None):
    safe_rmtree(folder)
    base_folder = Path(folder)
    if grouping_property:
        working_folder = base_folder
        working_folder.mkdir(parents=True, exist_ok=True)
        split_recordings = recording.split_by(grouping_property)
        mapping_dtype = np.asarray(recording.get_property(grouping_property)).dtype
        sorting_list = []
        unit_groups = []
        for group_value, group_recording in split_recordings.items():
            subfolder = working_folder / str(group_value)
            try:
                sorting = ss.run_sorter(
                    name,
                    group_recording,
                    folder=subfolder,
                    verbose=True,
                    **params,
                )
            except SpikeSortingError as exc:
                print(f"{name} failed for group {group_value}: {exc}")
                print("Continuing with empty sorting for this group.")
                empty_times = [np.array([], dtype=np.int64) for _ in range(group_recording.get_num_segments())]
                empty_labels = [np.array([], dtype=np.int64) for _ in range(group_recording.get_num_segments())]
                sorting = NumpySorting.from_samples_and_labels(
                    empty_times,
                    empty_labels,
                    group_recording.get_sampling_frequency(),
                    unit_ids=np.array([], dtype=np.int64),
                )
            sorting_list.append(sorting)
            if sorting.get_num_units():
                unit_groups.extend([group_value] * sorting.get_num_units())
        aggregate_sorting = aggregate_units(sorting_list)
        unit_groups_array = np.array(unit_groups, dtype=mapping_dtype)
        if aggregate_sorting.get_num_units() == unit_groups_array.size:
            aggregate_sorting.set_property(key=grouping_property, values=unit_groups_array)
        else:
            print(
                "Warning: could not align unit groups with aggregated units "
                f"({aggregate_sorting.get_num_units()} units vs {unit_groups_array.size} labels)."
            )
        aggregate_sorting.register_recording(recording)
        return aggregate_sorting
    return ss.run_sorter(name, recording, folder=base_folder, verbose=True, **params)


def build_analyzer(sorting, recording, base_folder, label):
    folder = base_folder / f"analyzer_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, folder=folder, overwrite=True)
    analyzer.compute({"random_spikes": {"max_spikes_per_unit": 500, "seed": 42}}, verbose=True)
    analyzer.compute("waveforms", ms_before=1.5, ms_after=2.5, dtype="float32", verbose=True, n_jobs=4, chunk_duration="2s")
    analyzer.compute("templates")
    analyzer.compute("principal_components")
    print(f"Analyzer computed for {label} at {folder}")
    return analyzer


def load_latest_analyzer(base_folder: Path, label: str):
    pattern = f"analyzer_{label}_*"
    folders = sorted(base_folder.glob(pattern))
    for folder in reversed(folders):
        try:
            return load_sorting_analyzer(folder)
        except Exception:
            continue
    return None


def export_for_phy(analyzer, base_folder, label, groups, original_index_map):
    folder = base_folder / f"phy_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_to_phy(analyzer, output_folder=folder, remove_if_exists=True)
    params_path = folder / "params.py"

    channel_ids_rec = list(analyzer.recording.channel_ids)
    channel_indices = np.array(
        [int(original_index_map.get(ch, idx)) for idx, ch in enumerate(channel_ids_rec)],
        dtype=np.int32,
    )

    channel_groups_out = None
    channel_shanks_out = None

    if params_path.exists():
        text = params_path.read_text()
        if "import numpy as np" not in text:
            text = "import numpy as np\n" + text

        pattern_ids = re.compile(r"channel_ids\s*=.*")
        replacement_ids = f"channel_ids = np.array({channel_indices.tolist()}, dtype=np.int32)"
        text, _ = pattern_ids.subn(replacement_ids, text, count=1)

        group_map = {}
        for group_idx, group in enumerate(groups):
            for ch in group:
                group_map[ch] = group_idx

        n_channels = len(channel_ids_rec)
        channel_groups_out = np.full(n_channels, -1, dtype=np.int32)
        channel_shanks_out = np.full(n_channels, -1, dtype=np.int32)

        if LINEARIZE_TRACEVIEW:
            contact_spacing = float(TRACEVIEW_CONTACT_SPACING_UM)
            group_spacing = float(TRACEVIEW_GROUP_SPACING_UM)
            positions = np.zeros((n_channels, 2), dtype=np.float32)

            for idx, ch in enumerate(channel_ids_rec):
                orig_idx = int(original_index_map.get(ch, idx))
                tetrode_idx = orig_idx // CHANNELS_PER_TETRODE if CHANNELS_PER_TETRODE else 0
                slot = orig_idx % CHANNELS_PER_TETRODE if CHANNELS_PER_TETRODE else 0
                col = slot % 2
                row = slot // 2

                positions[idx, 0] = col * contact_spacing
                positions[idx, 1] = tetrode_idx * group_spacing + row * contact_spacing

                group_idx = group_map.get(ch, tetrode_idx)
                channel_groups_out[idx] = group_idx
                channel_shanks_out[idx] = group_idx

            for name in ("channel_positions.npy", "channel_locations.npy"):
                try:
                    np.save(folder / name, positions)
                except Exception as exc:
                    print(f"Warning: could not overwrite {name}: {exc}")
        else:
            for idx, ch in enumerate(channel_ids_rec):
                group_idx = group_map.get(ch, 0)
                channel_groups_out[idx] = group_idx
                channel_shanks_out[idx] = group_idx

        if channel_groups_out.size:
            channel_groups_out[channel_groups_out < 0] = 0
        if channel_shanks_out.size:
            channel_shanks_out[channel_shanks_out < 0] = 0

        pattern_groups = re.compile(r"channel_groups\s*=.*")
        replacement_groups = f"channel_groups = np.array({channel_groups_out.tolist()}, dtype=np.int32)"
        text, count_groups = pattern_groups.subn(replacement_groups, text, count=1)
        if not count_groups:
            text += f"\nchannel_groups = np.array({channel_groups_out.tolist()}, dtype=np.int32)\n"
        params_path.write_text(text)

        try:
            np.save(folder / "channel_groups.npy", channel_groups_out.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_groups.npy: {exc}")
        try:
            np.save(folder / "channel_shanks.npy", channel_shanks_out.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_shanks.npy: {exc}")

    print(f"Exported {label} to Phy folder {folder}")


def summarize_agreement(sorting_a, sorting_b):
    if sorting_a is None or sorting_b is None:
        return
    comp = compare_two_sorters(sorting_a, sorting_b)
    scores = comp.agreement_scores
    if scores.empty:
        print("Agreement: no matched units.")
        return
    values = scores.to_numpy().flatten()
    values = values[~np.isnan(values)]
    print(f"Agreement: {values.size} scores, mean={float(np.mean(values)):.3f}")


def prepare_tetrode_recording(recording, groups):
    groups = [list(group) for group in groups]
    print(f"Tetrodes: {len(groups)}; first group: {groups[0] if groups else 'n/a'}")

    offsets = None
    tetrodes_per_row = max(1, int(np.ceil(np.sqrt(len(groups))))) if groups else 1
    if groups:
        dx = 150.0
        dy = 150.0
        num_rows = int(np.ceil(len(groups) / tetrodes_per_row))
        offsets = []
        for idx in range(len(groups)):
            row = idx // tetrodes_per_row
            col = idx % tetrodes_per_row
            x = col * dx
            y = (num_rows - 1 - row) * dy
            offsets.append((x, y))

    if ATTACH_GEOMETRY:
        recording = ensure_geom_and_units(recording, groups, tetrodes_per_row, tetrode_offsets=offsets)
        print(f"Geometry attached to recording (tetrodes_per_row={tetrodes_per_row}).")

    return recording, groups, offsets, tetrodes_per_row


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def main():
    set_seed()
    KS4_OUT.mkdir(parents=True, exist_ok=True)
    SC2_OUT.mkdir(parents=True, exist_ok=True)

    data_path = choose_recording_folder(ROOT_DIR, SESSION_SUBPATH, SESSION_SELECTION, SESSION_INDEX)
    print(f"Recording root: {ROOT_DIR}")
    print(f"KS4 output: {KS4_OUT}")
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
    original_index_map = {ch: idx for idx, ch in enumerate(original_channel_order)}
    channel_order = original_channel_order.copy()

    base_groups = chunk_tetrode_groups(original_channel_order, CHANNELS_PER_TETRODE)
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

    recording, groups, offsets, tetrodes_per_row = prepare_tetrode_recording(recording, groups)

    if REORDER_FOR_PHY:
        recording = reorder_recording_by_locations(recording)
        print("Channels reordered by geometry before preprocessing.")

    rec_ks4, rec_sc2 = preprocess_for_sorters(recording, groups)

    if ATTACH_GEOMETRY and groups:
        rec_ks4 = ensure_geom_and_units(rec_ks4, groups, tetrodes_per_row, tetrode_offsets=offsets)
        rec_sc2 = ensure_geom_and_units(rec_sc2, groups, tetrodes_per_row, tetrode_offsets=offsets)

    print("Branch locations:", rec_ks4.get_channel_locations().shape, rec_sc2.get_channel_locations().shape)

    cache_recording(KS4_OUT, rec_ks4, "ks4")
    cache_recording(SC2_OUT, rec_sc2, "sc2")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    sorting_ks4 = sorting_sc2 = None

    if RUN_KS4 and groups:
        set_group_property(rec_ks4, groups)
        ks4_params = ss.Kilosort4Sorter.default_params()
        ks4_params.update(
            {
                "torch_device": KS4_TORCH_DEVICE,
                "do_CAR": False,
                "progress_bar": True,
                "bad_channels": (BAD_CHANNELS or None),
            }
        )
        ks4_run = KS4_OUT / f"ks4_run_{tag}"
        sorting_ks4 = run_sorter("kilosort4", rec_ks4, ks4_run, ks4_params, grouping_property="group")
        print(f"KS4 units: {sorting_ks4.get_num_units()} | output: {ks4_run}")

    if RUN_SC2 and groups:
        set_group_property(rec_sc2, groups)
        sc2_params = ss.Spykingcircus2Sorter.default_params()
        sc2_run = SC2_OUT / f"sc2_run_{tag}"
        sorting_sc2 = run_sorter("spykingcircus2", rec_sc2, sc2_run, sc2_params, grouping_property="group")
        print(f"SC2 units: {sorting_sc2.get_num_units()} | output: {sc2_run}")

    analyzer_ks4 = analyzer_sc2 = None
    if RUN_KS4 and sorting_ks4 is not None:
        analyzer_ks4 = build_analyzer(sorting_ks4, rec_ks4, KS4_OUT, "ks4")
    if RUN_SC2 and sorting_sc2 is not None:
        analyzer_sc2 = build_analyzer(sorting_sc2, rec_sc2, SC2_OUT, "sc2")

    if analyzer_ks4 is None:
        analyzer_ks4 = load_latest_analyzer(KS4_OUT, "ks4")
    if analyzer_sc2 is None:
        analyzer_sc2 = load_latest_analyzer(SC2_OUT, "sc2")

    summarize_agreement(sorting_ks4, sorting_sc2)

    if analyzer_ks4 is not None:
        export_for_phy(analyzer_ks4, KS4_OUT, "ks4", groups, original_index_map)
    if analyzer_sc2 is not None:
        export_for_phy(analyzer_sc2, SC2_OUT, "sc2", groups, original_index_map)

    print("All done - ready for Phy curation.")


if __name__ == "__main__":
    main()
