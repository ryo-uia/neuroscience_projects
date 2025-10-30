from __future__ import annotations

import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.core import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy

HELPERS_DIR = Path(__file__).resolve().parent.parent / "Functions"
if HELPERS_DIR.exists():
    sys.path.insert(0, str(HELPERS_DIR))

from si_utils import (
    discover_oe_stream_names,
    ensure_geom_and_units,
    reorder_recording_by_locations,
    set_group_property,
)


# --- User configuration ----------------------------------------------------

TEST_SECONDS: Optional[int] = 300  # None to use full recording

ROOT_DIR = Path(r"C:/Users/ryoi/Documents/SpikeSorting/recordings")
SESSION_SUBPATH = Path(r"2025-10-01_15-53-19/Record Node 125/experiment1/recording1")

BASE_OUT = Path(r"C:/Users/ryoi/Documents/SpikeSorting")
KS4_OUT = BASE_OUT / "ks4_outputs"
SC2_OUT = BASE_OUT / "sc2_outputs"

STREAM_NAME: Optional[str] = None  # e.g. "Record Node 125#Acquisition_Board-100.Rhythm Data"

# Channel configuration
BAD_CHANNELS: Sequence[str] = []  # e.g. ["CH12", "CH59"]
CHANNELS_PER_TETRODE = 4

# Geometry flags
ATTACH_GEOMETRY = True

# Sorter toggles
RUN_KS4 = True
RUN_SC2 = True

# Sorter parameters
KS4_TORCH_DEVICE = "cpu"  # switch to "cuda" when GPU is available
LINEARIZE_TRACEVIEW = True
TRACEVIEW_CONTACT_SPACING_UM = 20.0
TRACEVIEW_GROUP_SPACING_UM = 200.0


# --- Helpers ----------------------------------------------------------------

def configure_environment() -> None:
    """Set deterministic environment seeds and default thread counts."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    random.seed(0)
    np.random.seed(0)


def ensure_output_dirs() -> None:
    for folder in (KS4_OUT, SC2_OUT):
        folder.mkdir(parents=True, exist_ok=True)


def resolve_stream_name(data_path: Path, preferred: Optional[str]) -> str:
    if preferred:
        return preferred
    names = discover_streams(data_path)
    neural = [n for n in names if "ADC" not in n and "SYNC" not in n]
    if not neural:
        raise RuntimeError("Could not determine a neural Open Ephys stream. Set STREAM_NAME explicitly.")
    print("Available streams:")
    for idx, name in enumerate(names):
        print(f"  [{idx}] {name}")
    return neural[0]


def discover_streams(folder: Path) -> List[str]:
    try:
        names = discover_oe_stream_names(folder)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        raise RuntimeError(
            f"Stream discovery failed for {folder}. Set STREAM_NAME manually."
        ) from exc
    if not names:
        raise RuntimeError("Open Ephys extractor returned no stream names.")
    return names


def slice_recording(recording, seconds: int):
    if seconds is None:
        return recording
    fs = recording.get_sampling_frequency()
    total_frames = recording.get_num_samples(0) if hasattr(recording, "get_num_samples") else recording.get_num_frames(0)
    end_frame = min(int(fs * seconds), total_frames)
    print(f"Recording sliced to first {seconds}s (end_frame={end_frame}).")
    return recording.frame_slice(0, end_frame)


def group_channels(channel_ids: Sequence, group_size: int = CHANNELS_PER_TETRODE) -> List[List]:
    if len(channel_ids) % group_size != 0:
        raise ValueError("Channel count not divisible by tetrode size; update CHANNELS_PER_TETRODE or drop channels.")
    channel_list = list(channel_ids)
    return [channel_list[i : i + group_size] for i in range(0, len(channel_list), group_size)]


def drop_bad_channels(recording, groups: List[List], bad_channels: Sequence[str]):
    if not bad_channels:
        return recording, groups
    keep_groups = [[ch for ch in group if ch not in bad_channels] for group in groups]
    keep_groups = [group for group in keep_groups if group]
    keep_ids = [ch for group in keep_groups for ch in group]
    rec = recording.channel_slice(keep_channel_ids=keep_ids)
    print(f"Removed {len(bad_channels)} channels; new channel count: {rec.get_num_channels()}")
    return rec, keep_groups


def preprocessing_branches(recording, groups: Sequence[Sequence]):
    rec_bp = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    for f0 in (50, 100, 150):
        rec_bp = spre.notch_filter(rec_bp, freq=f0, q=30)
    # KS4 expects per-tetrode CAR; use global reference with per-group median.
    rec_ks4 = spre.common_reference(
        rec_bp,
        reference="global",
        operator="median",
        groups=groups,
    )
    rec_sc2 = rec_bp
    print("Preprocessing prepared: bandpass + notch; KS4 with median CAR, SC2 without external CAR.")
    return rec_ks4, rec_sc2


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


def cache_branch(folder: Path, recording, name: str, clean: bool = False) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = folder / f"cached_{name}_{ts}"
    if clean:
        safe_rmtree(cache_dir)
    recording.save(
        folder=cache_dir,
        format="binary",
        dtype="float32",
        chunk_duration="1s",
        overwrite=True,
    )
    print(f"Cached branch '{name}' to {cache_dir}")
    return cache_dir


def run_sorter_for_tetrodes(recording, sorter_name: str, output_folder: Path, sorter_params: dict):
    if output_folder.exists():
        safe_rmtree(output_folder)
    return si.run_sorter(
        sorter_name,
        recording,
        folder=output_folder,
        verbose=True,
        **sorter_params,
    )


def compute_analyzer(sorting, recording, base_folder: Path, label: str):
    analyzer_folder = base_folder / f"analyzer_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        folder=analyzer_folder,
        overwrite=True,
    )
    analyzer.compute({"random_spikes": {"max_spikes_per_unit": 500, "seed": 42}}, verbose=True)
    analyzer.compute("waveforms", ms_before=1.5, ms_after=2.5, dtype="float32", verbose=True, n_jobs=4, chunk_duration="2s")
    analyzer.compute("templates")
    analyzer.compute("principal_components")
    print(f"Analyzer computed for {label} at {analyzer_folder}")
    return analyzer


def export_phy(analyzer, folder: Path, label: str, original_index_map: Optional[dict] = None):
    phy_folder = folder / f"phy_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_to_phy(analyzer, output_folder=phy_folder, remove_if_exists=True)
    params_path = phy_folder / "params.py"

    channel_ids_rec = list(analyzer.recording.channel_ids)
    channel_indices = np.array(
        [int(original_index_map.get(ch, idx)) if original_index_map else idx for idx, ch in enumerate(channel_ids_rec)],
        dtype=np.int32,
    )

    channel_groups_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    channel_shanks_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    text = None

    if params_path.exists():
        text = params_path.read_text()
        if "import numpy as np" not in text:
            text = "import numpy as np\n" + text

        pattern_ids = re.compile(r"channel_ids\s*=.*")
        replacement_ids = f"channel_ids = np.array({channel_indices.tolist()}, dtype=np.int32)"
        text, _ = pattern_ids.subn(replacement_ids, text, count=1)

        try:
            groups_prop = analyzer.recording.get_property("group")
        except Exception:
            groups_prop = None
        if groups_prop is not None:
            channel_groups_out = np.asarray(groups_prop, dtype=np.int32)
            if channel_groups_out.size != len(channel_ids_rec):
                channel_groups_out = np.resize(channel_groups_out, len(channel_ids_rec))
        channel_shanks_out = channel_groups_out.copy()

    if LINEARIZE_TRACEVIEW:
        contact_spacing = float(TRACEVIEW_CONTACT_SPACING_UM)
        group_spacing = float(TRACEVIEW_GROUP_SPACING_UM)
        positions = np.zeros((len(channel_ids_rec), 2), dtype=np.float32)

        for idx, ch in enumerate(channel_ids_rec):
            if original_index_map and ch in original_index_map:
                orig_idx = int(original_index_map[ch])
            else:
                orig_idx = idx
            tetrode_idx = orig_idx // CHANNELS_PER_TETRODE if CHANNELS_PER_TETRODE else 0
            slot = orig_idx % CHANNELS_PER_TETRODE if CHANNELS_PER_TETRODE else 0
            col = slot % 2
            row = slot // 2
            positions[idx, 0] = col * contact_spacing
            positions[idx, 1] = tetrode_idx * group_spacing + row * contact_spacing

        for name in ("channel_positions.npy", "channel_locations.npy"):
            try:
                np.save(phy_folder / name, positions)
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
            np.save(phy_folder / "channel_groups.npy", channel_groups_out.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_groups.npy: {exc}")
        try:
            np.save(phy_folder / "channel_shanks.npy", channel_shanks_out.astype(np.int32))
        except Exception as exc:
            print(f"Warning: could not overwrite channel_shanks.npy: {exc}")
        params_path.write_text(text)
    print(f"Exported {label} to Phy folder {phy_folder}")


def compare_sortings_optional(sorting_a, sorting_b) -> None:
    if sorting_a is None or sorting_b is None:
        return
    comp = compare_two_sorters(sorting_a, sorting_b)
    agreement_matrix = comp.agreement_scores
    if agreement_matrix.empty:
        print("Agreement: no matched units or empty results.")
        return
    values = agreement_matrix.to_numpy().flatten()
    values = values[~np.isnan(values)]
    if values.size == 0:
        print("Agreement: no matched units or empty results.")
        return
    mean_score = float(np.mean(values))
    print(f"Agreement: {values.size} unit-pair scores, mean score={mean_score:.3f}")


# --- Main pipeline ----------------------------------------------------------

def main() -> None:
    configure_environment()
    ensure_output_dirs()

    data_path = ROOT_DIR / SESSION_SUBPATH
    print(f"Recording root: {ROOT_DIR}")
    print(f"KS4 output: {KS4_OUT}")
    print(f"SC2 output: {SC2_OUT}")
    print(f"Using Open Ephys folder: {data_path}")

    stream_name = resolve_stream_name(data_path, STREAM_NAME)
    print(f"Using stream: {stream_name}")

    recording = se.read_openephys(data_path, stream_name=stream_name)
    print(recording)
    print(
        "Segments:", recording.get_num_segments(),
        "| Fs:", recording.get_sampling_frequency(),
        "| Channels:", recording.get_num_channels(),
    )

    if recording.get_num_segments() > 1:
        recording = recording.select_segments([0])
        print("Selected segment 0 (single-segment).")

    recording = slice_recording(recording, TEST_SECONDS)

    original_channel_order = list(recording.channel_ids)
    original_index_map = {ch: idx for idx, ch in enumerate(original_channel_order)}

    groups = group_channels(recording.channel_ids, CHANNELS_PER_TETRODE)
    recording, groups = drop_bad_channels(recording, groups, BAD_CHANNELS)
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

    if ATTACH_GEOMETRY:
        recording = ensure_geom_and_units(
            recording,
            groups,
            tetrodes_per_row=tetrodes_per_row,
            scale_to_uV=True,
            tetrode_offsets=tetrode_offsets,
        )

    # Reorder once at the raw stage (most base extractors support channel_slice).
    # This keeps downstream wrappers simple and avoids warnings on views that don't expose channel_slice.
    rec_ks4, rec_sc2 = preprocessing_branches(recording, groups)

    if ATTACH_GEOMETRY:
        rec_ks4 = ensure_geom_and_units(
            rec_ks4,
            groups,
            tetrodes_per_row=tetrodes_per_row,
            scale_to_uV=True,
            tetrode_offsets=tetrode_offsets,
        )
        rec_sc2 = ensure_geom_and_units(
            rec_sc2,
            groups,
            tetrodes_per_row=tetrodes_per_row,
            scale_to_uV=True,
            tetrode_offsets=tetrode_offsets,
        )

    print(
        "Branch locations:",
        rec_ks4.get_channel_locations().shape,
        rec_sc2.get_channel_locations().shape,
    )

    cache_branch(KS4_OUT, rec_ks4, "ks4")
    cache_branch(SC2_OUT, rec_sc2, "sc2")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    sorting_ks4 = None
    sorting_sc2 = None

    if RUN_KS4:
        set_group_property(rec_ks4, groups)
        ks4_params = ss.Kilosort4Sorter.default_params()
        ks4_params.update({
            "torch_device": KS4_TORCH_DEVICE,
            "do_CAR": False,
            "progress_bar": True,
            "bad_channels": (BAD_CHANNELS or None),
        })
        ks4_run = KS4_OUT / f"ks4_run_{tag}"
        sorting_ks4 = run_sorter_for_tetrodes(rec_ks4, "kilosort4", ks4_run, ks4_params)
        print(f"KS4 units: {sorting_ks4.get_num_units()} | output: {ks4_run}")

    if RUN_SC2:
        set_group_property(rec_sc2, groups)
        sc2_params = ss.Spykingcircus2Sorter.default_params()
        sc2_run = SC2_OUT / f"sc2_run_{tag}"
        sorting_sc2 = run_sorter_for_tetrodes(rec_sc2, "spykingcircus2", sc2_run, sc2_params)
        print(f"SC2 units: {sorting_sc2.get_num_units()} | output: {sc2_run}")

    analyzer_ks4 = analyzer_sc2 = None
    if RUN_KS4 and sorting_ks4 is not None:
        analyzer_ks4 = compute_analyzer(sorting_ks4, rec_ks4, KS4_OUT, "ks4")
    if RUN_SC2 and sorting_sc2 is not None:
        analyzer_sc2 = compute_analyzer(sorting_sc2, rec_sc2, SC2_OUT, "sc2")

    compare_sortings_optional(sorting_ks4, sorting_sc2)

    if analyzer_ks4 is not None:
        export_phy(analyzer_ks4, KS4_OUT, "ks4", original_index_map)
    if analyzer_sc2 is not None:
        export_phy(analyzer_sc2, SC2_OUT, "sc2", original_index_map)

    print("All done - ready for Phy curation.")


if __name__ == "__main__":
    main()


