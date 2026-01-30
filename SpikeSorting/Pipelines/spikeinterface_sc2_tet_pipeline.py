"""
Minimal SpikeInterface pipeline to run SpykingCircus2 on tetrode recordings.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from spikeinterface.core import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy
from probeinterface import Probe

try:
    import spikeinterface.curation as sc
except Exception:
    sc = None
try:
    from spikeinterface.qualitymetrics import compute_quality_metrics
except Exception:
    compute_quality_metrics = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Functions.si_utils import ensure_geom_and_units, ensure_probe_attached, set_group_property
from Functions.pipeline_utils import (
    choose_recording_folder,
    chunk_groups,
    detect_bad_channel_ids,
    first_seconds,
    load_bad_channels_from_path,
    load_channel_groups_from_path,
    merge_params,
    pick_stream,
    resolve_bad_channel_ids,
    resolve_manual_groups,
    safe_channel_slice,
    safe_rmtree,
)

# ---------------------------------------------------------------------
# Config selection helpers (optional prompts)
# ---------------------------------------------------------------------

def _choose_config_json(label: str, candidates: list[Path], default_path: Path | None) -> Path | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    default_idx = 0
    if default_path in candidates:
        default_idx = candidates.index(default_path)
    print(f"Available {label} JSON files:")
    for idx, path in enumerate(candidates):
        print(f"  [{idx}] {path}")
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

# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------

# Session/data selection
TEST_SECONDS = 600  # set to None for full recording
DEFAULT_ROOT_DIR = PROJECT_ROOT / "recordings"
SESSION_SUBPATH = None  # provide relative Open Ephys path to skip auto-discovery
SESSION_SELECTION = "prompt"  # always prompt for session selection
STREAM_NAME = None  # e.g. "Record Node 125#Acquisition_Board-100.Rhythm Data"

# Output folders
DEFAULT_BASE_OUT = PROJECT_ROOT
# Initialized in main() after parsing CLI/env
SC2_OUT = None
SI_GUI_OUT = None

# Export controls
EXPORT_TO_PHY = True
EXPORT_TO_SI_GUI = False
EXPORT_PHY_EXTRACT_WAVEFORMS = False  # run `phy extract-waveforms` to precompute Phy waveforms (UI speed only; requires Phy CLI)
SIMPLE_PHY_EXPORT = None  # auto: follows SORT_BY_GROUP; set True/False to override
EXPORT_PHY_CHANNEL_IDS_MODE = "oe_index"  # "oe_index"=OE numeric IDs (0-based, gaps after removals), "oe_label"=CH## labels, "as_exported"=compact 0..N-1
MATERIALIZE_EXPORT = False  # save rec_export to disk before analyzer/export (faster reuse; uses extra space)

# Analyzer/QC
COMPUTE_QC_METRICS = True  # compute QC metrics on analyzer output
QC_METRIC_NAMES = [
    "firing_rate",
    "presence_ratio",
    "isi_violation",
    "snr",
    "amplitude_cutoff",
]
QC_PC_METRICS = {
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    "nn_isolation",
    "nn_noise_overlap",
    "silhouette",
}

# Channel grouping (tetrodes)
CHANNELS_PER_TETRODE = 4
# Optional explicit channel grouping. If set, this fixed map is used instead of chunking by order
# (prevents shifts after bad-channel removal).
CHANNEL_GROUPS: list[list[int | str]] = [
    ["CH40", "CH38", "CH36", "CH34"],  # TT1A-D
    ["CH48", "CH46", "CH44", "CH42"],  # TT2A-D
    ["CH56", "CH54", "CH52", "CH50"],  # TT3A-D
    ["CH58", "CH64", "CH62", "CH60"],  # TT4A-D
    ["CH63", "CH61", "CH59", "CH57"],  # TT5A-D
    ["CH55", "CH53", "CH51", "CH49"],  # TT6A-D
    ["CH47", "CH45", "CH43", "CH41"],  # TT7A-D
    ["CH39", "CH37", "CH35", "CH33"],  # TT8A-D
    ["CH25", "CH27", "CH29", "CH31"],  # TT9A-D
    ["CH17", "CH19", "CH21", "CH23"],  # TT10A-D
    ["CH9", "CH11", "CH13", "CH15"],   # TT11A-D
    ["CH1", "CH3", "CH5", "CH7"],      # TT12A-D
    ["CH4", "CH6", "CH8", "CH2"],      # TT13A-D
    ["CH10", "CH12", "CH14", "CH16"],  # TT14A-D
    ["CH18", "CH20", "CH22", "CH24"],  # TT15A-D
    ["CH26", "CH28", "CH30", "CH32"],  # TT16A-D
]
CHANNEL_GROUPS_PATH = None  # optional JSON file path or env SPIKESORT_CHANNEL_GROUPS
STRICT_GROUPS = True  # error out if no valid groups can be resolved (recommended for tetrodes)
SORT_BY_GROUP = False  # when True, run SC2 separately per tetrode group (one sorter/export per group)
# Bundle sorting option: keep one group with grid geometry (similar to .prb layout).
BUNDLE_GROUPING_MODE = "tetrode"  # "tetrode" (default) or "single_grid"
# When "tetrode", the bundle grid settings below are ignored.
BUNDLE_GRID_COLS = 4
BUNDLE_GRID_DX_UM = 10.0  # bundle grid x-spacing (synthetic; mirrors .prb layout when single_grid)
BUNDLE_GRID_DY_UM = 200.0  # bundle grid y-spacing (synthetic; mirrors .prb layout when single_grid)

# Bad channels (use IDs to avoid positional mismatch across sessions or after slicing)
#BAD_CHANNELS = ["CH7", "CH2", "CH26", "CH4", "CH42", "CH50", "CH51", "CH52", "CH54", "CH56", "CH6", "CH8"] #Tatsu
BAD_CHANNELS = ["CH58", "CH64", "CH62", "CH60", "CH63", "CH61", "CH59", "CH57", "CH47", "CH45", "CH43", "CH41"] #Fuyu
BAD_CHANNELS_PATH = None  # optional JSON file path or env SPIKESORT_BAD_CHANNELS
AUTO_BAD_CHANNELS = False  # auto-detect bad channels (merged with manual list; can be aggressive)
AUTO_BAD_CHANNELS_METHOD = "std"  # "std" or "mad" are typical for tetrodes
AUTO_BAD_CHANNELS_KWARGS = {}  # extra args passed to detect_bad_channels

# Geometry/traceview display
ATTACH_GEOMETRY = True
LINEARIZE_TRACEVIEW = True
TRACEVIEW_CONTACT_SPACING_UM = 20.0
TRACEVIEW_GROUP_SPACING_UM = 200.0
# Synthetic within‑tetrode pitch for the 2x2 layout (geometry only).
TETRODE_PITCH_UM = 20.0
# Synthetic spacing between tetrodes in the bundle layout (only affects geometry/whitening/visualization).
TETRODE_SPACING_DX_UM = 300.0
TETRODE_SPACING_DY_UM = 300.0

# Optional SI preprocessing: bandpass (+ optional notch/CAR/whitening).
# When enabled, SC2 filtering/CMR are disabled, but SC2 still whitens internally.
# OE raw data are typically int16; SI scaling/filtering/whitening outputs float32.
USE_SI_PREPROCESS = True
SI_BP_MIN_HZ = 300
SI_BP_MAX_HZ = 6000
SI_BP_FTYPE = "bessel"
SI_BP_ORDER = 2
SI_BP_MARGIN_MS = 10
MATERIALIZE_SI_PREPROCESS = False  # save preprocessed rec_sc2 to disk (faster reuse; uses extra space)

SI_APPLY_WHITEN = False  # only applies when USE_SI_PREPROCESS=True; SC2 whitening still runs (double-whitening)
SI_WHITEN_MODE = "local"  # "global" or "local"
SI_WHITEN_RADIUS_UM = 100.0  # used when SI_WHITEN_MODE == "local"
# Tetrodes: prefer local whitening if geometry is attached; use global if locations are missing.
# Note: the current SpikeInterface SC2 wrapper always applies SC2 whitening; there is no
# reliable disable flag in this version.

# Optional SI common reference before SC2. SC2 preprocessing (when enabled) does its own bandpass/CMR/whitening.
SI_APPLY_CAR = False
# "tetrode" -> per-tetrode CAR; "global" -> CAR over all channels
CAR_MODE = "global"
CAR_OPERATOR = "median"  # "median" (robust) or "average"
# Tetrodes: per-tetrode median is common but can induce bipolar spikes with small/imbalanced groups; global is safer.

# Optional notch filtering (applies after SI bandpass when USE_SI_PREPROCESS=True,
# or before SC2 preprocessing when USE_SI_PREPROCESS=False).
APPLY_NOTCH = False
NOTCH_FREQUENCIES = [50, 100, 150]
NOTCH_Q = 30

# Optional bandpass copy for Phy/export (useful when SC2 preprocessing is enabled so Phy doesn't read raw).
# Common practice: Phy export is scaled + optionally bandpassed, but not whitened.
EXPORT_BANDPASS_FOR_PHY = True  # set True for bandpassed export; False exports raw for comparison
EXPORT_BP_MIN_HZ = SI_BP_MIN_HZ
EXPORT_BP_MAX_HZ = SI_BP_MAX_HZ
EXPORT_BP_FTYPE = SI_BP_FTYPE
EXPORT_BP_ORDER = SI_BP_ORDER
EXPORT_BP_MARGIN_MS = SI_BP_MARGIN_MS
EXPORT_SCALE_TO_UV = True  # scale export recording to microvolts if gain info is present
ANALYZER_FROM_SORTER = True  # use rec_sc2 for analyzer/QC (more faithful to sorter input)
SAVE_ANALYZER = False  # persist analyzer to disk (binary_folder); set True for full runs
REMOVE_REDUNDANT_UNITS = False  # optional post-sort curation step (remove near-duplicate units)
REDUNDANT_THRESHOLD = 0.95  # higher = stricter duplicate detection
REDUNDANT_STRATEGY = "minimum_shift"  # curation strategy for duplicates

# Optional overrides for SpykingCircus2 parameters (keys mirror default params structure)
# Add tweaks here if you need to change SC2 defaults (filtering, thresholds, etc.).
SC2_PARAM_OVERRIDES = {
     # Example: disable SC2 filtering when using SI preprocessing
    # "preprocessing": {
    #     "apply": False,
    # },
    # Example: adjust detection threshold
    # "detection": {
    #     "peak_sign": "both",
    #     "relative_threshold": 5.0,
    # },

    # Tetrodes: disable motion correction (intended for dense probes).
    "apply_motion_correction": False,
    # Match native SC2 waveform window (N_t = 3 ms total).
    "general": {
        "ms_before": 1.0,
        "ms_after": 2.0,
    },
    # Match native SC2 detection threshold and peak sign.
    "detection": {
        "method": "matched_filtering",
        "method_kwargs": {
            "peak_sign": "neg",
            "detect_threshold": 6,
        },
    },
}

# Optional: print stack traces for warnings (debug-only).
DEBUG_WARN_TRACE = False


def enable_warning_trace(limit: int = 12) -> None:
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        traceback.print_stack(limit=limit)
        print(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = _showwarning
    warnings.simplefilter("always")


# ---------------------------------------------------------------------
# Pipeline helpers from Functions.pipeline_utils and Functions.si_utils
# ---------------------------------------------------------------------

def attach_bundle_grid_geom(recording, ncols: int, dx_um: float, dy_um: float):
    """Attach a simple grid geometry for bundle sorting (one group, N channels)."""
    n_ch = recording.get_num_channels()
    positions = np.zeros((n_ch, 2), dtype=float)
    for idx in range(n_ch):
        row = idx // ncols
        col = idx % ncols
        positions[idx] = (col * dx_um, row * dy_um)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7})
    probe.set_device_channel_indices(np.arange(n_ch))
    return recording.set_probe(probe, in_place=False)


def preprocess_for_sc2(recording, groups=None):
    rec = recording
    groups = groups or []

    # Optional SI-driven path: bandpass -> optional notch -> optional CAR -> whitening.
    if USE_SI_PREPROCESS:
        rec = spre.bandpass_filter(
            rec,
            freq_min=SI_BP_MIN_HZ,
            freq_max=SI_BP_MAX_HZ,
            ftype=SI_BP_FTYPE,
            filter_order=SI_BP_ORDER,
            margin_ms=SI_BP_MARGIN_MS,
        )
        if APPLY_NOTCH and NOTCH_FREQUENCIES:
            for freq in NOTCH_FREQUENCIES:
                rec = spre.notch_filter(rec, freq=freq, q=NOTCH_Q)
            print(f"Preprocessing (SI): bandpass {SI_BP_MIN_HZ}-{SI_BP_MAX_HZ} Hz + notch {NOTCH_FREQUENCIES} Hz.")
        else:
            print(f"Preprocessing (SI): bandpass {SI_BP_MIN_HZ}-{SI_BP_MAX_HZ} Hz (no notch).")

        if SI_APPLY_CAR:
            mode = (CAR_MODE or "tetrode").lower()
            operator = (CAR_OPERATOR or "median").lower()
            if mode == "tetrode" and groups:
                rec = spre.common_reference(rec, reference="global", operator=operator, groups=groups)
                print(f"Preprocessing (SI): applied per-tetrode CAR ({operator}) on {len(groups)} tetrodes.")
            else:
                rec = spre.common_reference(rec, reference="global", operator=operator)
                print(f"Preprocessing (SI): applied global CAR ({operator}).")
        else:
            print("Preprocessing (SI): CAR disabled.")

        if SI_APPLY_WHITEN:
            whiten_mode = SI_WHITEN_MODE
            if whiten_mode == "local":
                try:
                    rec.get_channel_locations()
                except Exception:
                    print("Preprocessing (SI): local whitening requested but channel locations missing; falling back to global.")
                    whiten_mode = "global"
            whiten_kwargs = dict(dtype="float32", mode=whiten_mode)
            if whiten_mode == "local":
                whiten_kwargs["radius_um"] = SI_WHITEN_RADIUS_UM
            rec = spre.whiten(rec, **whiten_kwargs)
            print(f"Preprocessing (SI): whitening applied (mode={whiten_mode}).")
        else:
            print("Preprocessing (SI): whitening skipped (SI_APPLY_WHITEN=False).")
        return rec

    # Default path: optional notch only; SC2 will handle bandpass/CMR/whitening.
    if APPLY_NOTCH and NOTCH_FREQUENCIES:
        rec = recording
        for freq in NOTCH_FREQUENCIES:
            rec = spre.notch_filter(rec, freq=freq, q=NOTCH_Q)
        print(f"Preprocessing: applied notch filters at {NOTCH_FREQUENCIES} Hz (Q={NOTCH_Q}).")
        return rec

    print(
        "Passing recording to SC2: SC2 will run its preprocessing (bandpass + CMR + whitening)"
        " unless disabled via SC2_PARAM_OVERRIDES."
    )
    return rec


def _attach_probe_from_group_layout(recording, group_prop, pitch=20.0, dx=150.0, dy=150.0):
    group_prop = np.asarray(group_prop)
    if group_prop.size != recording.get_num_channels():
        raise ValueError("group property length mismatch")
    unique_groups = sorted({int(g) for g in group_prop.tolist()})
    if not unique_groups:
        raise ValueError("no group ids available")
    tetrodes_per_row = max(1, int(np.ceil(np.sqrt(len(unique_groups)))))
    positions = np.zeros((group_prop.size, 2), dtype=float)
    base = np.array([[0.0, 0.0], [pitch, 0.0], [0.0, pitch], [pitch, pitch]], dtype=float)
    for idx, group_id in enumerate(unique_groups):
        row = idx // tetrodes_per_row
        col = idx % tetrodes_per_row
        offset = np.array([col * dx, row * dy], dtype=float)
        ch_idx = np.where(group_prop == group_id)[0]
        for slot, ch_i in enumerate(ch_idx):
            positions[ch_i] = offset + base[slot % 4]
    from probeinterface import Probe

    probe = Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7})
    probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
    return recording.set_probe(probe, in_place=False)


def ensure_probe_for_analyzer(recording):
    try:
        if recording.get_probe() is not None:
            return recording
    except Exception:
        pass
    try:
        recording = ensure_probe_attached(recording)
        if recording.get_probe() is not None:
            return recording
    except Exception:
        pass
    try:
        group_prop = recording.get_property("group")
    except Exception:
        group_prop = None
    if group_prop is not None:
        try:
            return _attach_probe_from_group_layout(recording, group_prop)
        except Exception:
            pass
    return recording


def build_analyzer(
    sorting,
    recording,
    base_folder: Path,
    label: str,
    wf_ms_before: float | None = None,
    wf_ms_after: float | None = None,
):
    folder = base_folder / f"analyzer_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    recording = ensure_probe_for_analyzer(recording)
    try:
        has_probe = recording.get_probe() is not None
    except Exception:
        has_probe = False
    if has_probe:
        print("Analyzer recording probe attached.")
    else:
        print("Warning: analyzer recording has no probe attached.")
    analyzer_format = "binary_folder" if SAVE_ANALYZER else "memory"
    if SAVE_ANALYZER:
        folder.mkdir(parents=True, exist_ok=True)
    analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format=analyzer_format,
        folder=folder,
        overwrite=True,
    )
    analyzer.compute("random_spikes")
    wf_kwargs = {}
    if wf_ms_before is not None:
        wf_kwargs["ms_before"] = wf_ms_before
    if wf_ms_after is not None:
        wf_kwargs["ms_after"] = wf_ms_after
    analyzer.compute("waveforms", **wf_kwargs) if wf_kwargs else analyzer.compute("waveforms")
    analyzer.compute("templates")
    analyzer.compute("principal_components")
    if COMPUTE_QC_METRICS:
        qc_done = False
        try:
            try:
                analyzer.compute("noise_levels")
            except Exception as exc:
                print(f"Warning: noise_levels failed: {exc}")
            skip_pc = not any(name in QC_PC_METRICS for name in QC_METRIC_NAMES)
            try:
                analyzer.compute(
                    "quality_metrics",
                    metric_names=QC_METRIC_NAMES,
                    skip_pc_metrics=skip_pc,
                )
            except TypeError:
                analyzer.compute("quality_metrics", metric_names=QC_METRIC_NAMES)
            try:
                qm = analyzer.get_extension("quality_metrics").get_data()
            except Exception:
                qm = None
            if qm is not None:
                print("QC metrics computed via analyzer extension.")
                out_path = Path(analyzer.folder) / "qc_metrics.csv" if analyzer.folder else (folder / "qc_metrics.csv")
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    qm.to_csv(out_path)
                    print(f"QC metrics saved: {out_path}")
                except Exception as exc:
                    print(f"Warning: QC metrics save failed: {exc}")
                qc_done = True
            else:
                print("QC metrics computed (extension) but no data returned.")
        except Exception:
            pass
        if not qc_done:
            if compute_quality_metrics is None:
                print("Warning: QC metrics unavailable; skipping quality metrics.")
            else:
                try:
                    try:
                        analyzer.compute("noise_levels")
                    except Exception:
                        pass
                    qm = compute_quality_metrics(analyzer, metric_names=QC_METRIC_NAMES)
                    print("QC metrics computed via compute_quality_metrics fallback.")
                    out_path = Path(analyzer.folder) / "qc_metrics.csv" if analyzer.folder else (folder / "qc_metrics.csv")
                    try:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        qm.to_csv(out_path)
                        print(f"QC metrics saved: {out_path}")
                    except Exception as exc:
                        print(f"Warning: QC metrics save failed: {exc}")
                except Exception as exc:
                    print(f"Warning: QC metrics failed: {exc}")
    print(f"Analyzer computed for {label} at {folder}")
    return analyzer


def maybe_remove_redundant_units(analyzer, label: str):
    if not REMOVE_REDUNDANT_UNITS:
        return analyzer
    if sc is None:
        print("Warning: spikeinterface.curation unavailable; skipping redundant-unit removal.")
        return analyzer
    try:
        clean_sorting = sc.remove_redundant_units(
            analyzer,
            duplicate_threshold=REDUNDANT_THRESHOLD,
            remove_strategy=REDUNDANT_STRATEGY,
        )
        analyzer = analyzer.select_units(clean_sorting.unit_ids)
        print(f"Redundant units removed for {label} (threshold={REDUNDANT_THRESHOLD}).")
    except Exception as exc:
        print(f"Warning: redundant-unit removal failed for {label}: {exc}")
    return analyzer


def export_for_phy(analyzer, base_folder: Path, label: str, groups, original_index_map: dict, group_ids=None):
    folder = base_folder / f"phy_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # copy_binary=True so Phy extracts snippets from the exported binary (scaled/bandpassed),
    # instead of falling back to the raw recording on disk.
    export_kwargs = dict(output_folder=folder, remove_if_exists=True, verbose=False, copy_binary=True)
    export_to_phy(analyzer, **export_kwargs)
    simple_flag = SORT_BY_GROUP if SIMPLE_PHY_EXPORT is None else SIMPLE_PHY_EXPORT
    simple_export = simple_flag and groups is not None and len(groups) <= 1
    if simple_export and EXPORT_PHY_CHANNEL_IDS_MODE == "as_exported":
        return folder, None
    params_path = folder / "params.py"

    cache_dir = folder / ".phy"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    channel_ids_rec = list(analyzer.recording.channel_ids)
    group_unique = None
    try:
        group_prop_check = analyzer.recording.get_property("group")
        if group_prop_check is None:
            print("Warning: analyzer has no 'group' property.")
        else:
            group_unique = np.unique(group_prop_check)
    except Exception as exc:
        print(f"Warning: could not read 'group' property: {exc}")
    def lookup_index(ch, fallback):
        if ch in original_index_map:
            return int(original_index_map[ch])
        ch_str = str(ch)
        if ch_str in original_index_map:
            return int(original_index_map[ch_str])
        return int(fallback)

    if EXPORT_PHY_CHANNEL_IDS_MODE not in ("oe_index", "oe_label", "as_exported"):
        print(
            f"Warning: EXPORT_PHY_CHANNEL_IDS_MODE={EXPORT_PHY_CHANNEL_IDS_MODE!r} is invalid; "
            "falling back to 'oe_index'."
        )
        EXPORT_PHY_CHANNEL_IDS_MODE_LOCAL = "oe_index"
    else:
        EXPORT_PHY_CHANNEL_IDS_MODE_LOCAL = EXPORT_PHY_CHANNEL_IDS_MODE

    if EXPORT_PHY_CHANNEL_IDS_MODE_LOCAL == "oe_label":
        labels = None
        try:
            if "channel_name" in analyzer.recording.get_property_keys():
                labels = list(analyzer.recording.get_property("channel_name"))
        except Exception:
            labels = None
        if not labels:
            labels = [str(ch) for ch in channel_ids_rec]
        channel_ids_out = np.array(labels, dtype=str)
        channel_ids_text = f"channel_ids = np.array({labels!r}, dtype=str)"
    else:
    # Phy uses channel_map for compact 0..N-1 ordering in the exported binary (shifts after removals),
    # and channel_ids for labels; here "OE indices" means the index in the original recording.channel_ids
    # order (after any slicing), not necessarily the physical hardware channel number.
        channel_ids_out = np.array(
            [lookup_index(ch, idx) for idx, ch in enumerate(channel_ids_rec)],
            dtype=np.int32,
        )
        channel_ids_text = f"channel_ids = np.array({channel_ids_out.tolist()}, dtype=np.int32)"
    channel_map = np.arange(len(channel_ids_rec), dtype=np.int32)

    # Build channel->group and slot lookup from the provided groups
    ch_to_group = {}
    ch_to_slot = {}
    group_ids = list(group_ids) if group_ids is not None else list(range(len(groups)))
    if groups:
        for gi, grp in enumerate(groups):
            group_id = group_ids[gi] if gi < len(group_ids) else gi
            for si, ch in enumerate(grp):
                ch_to_group[ch] = group_id
                ch_to_group[str(ch)] = group_id
                ch_to_slot[ch] = si
                ch_to_slot[str(ch)] = si

    channel_groups_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    channel_shanks_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    text = None

    if params_path.exists():
        text = params_path.read_text()
        if "import numpy as np" not in text:
            text = "import numpy as np\n" + text

        pattern_ids = re.compile(r"channel_ids\s*=.*")
        text, count_ids = pattern_ids.subn(channel_ids_text, text, count=1)
        if not count_ids:
            text += f"\n{channel_ids_text}\n"
        try:
            np.save(folder / "channel_ids.npy", channel_ids_out)
        except Exception as exc:
            print(f"Warning: could not overwrite channel_ids.npy: {exc}")
        if simple_export:
            params_path.write_text(text)
            return folder, group_unique
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
        grp = ch_to_group.get(ch, ch_to_group.get(str(ch), 0))
        channel_groups_out[idx] = int(grp)
    channel_shanks_out = channel_groups_out.copy()

    if LINEARIZE_TRACEVIEW:
        contact_spacing = float(TRACEVIEW_CONTACT_SPACING_UM)
        group_spacing = float(TRACEVIEW_GROUP_SPACING_UM)
        positions = np.zeros((len(channel_ids_rec), 2), dtype=np.float32)

        for idx, ch in enumerate(channel_ids_rec):
            tetrode_idx = ch_to_group.get(ch, ch_to_group.get(str(ch), 0))
            slot = ch_to_slot.get(ch, ch_to_slot.get(str(ch), 0))
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

    # Rebuild cluster-level channel group assignments for Phy.
    try:
        template_ind_path = folder / "template_ind.npy"
        if not template_ind_path.exists():
            print("template_ind.npy not found; skipping cluster channel group recompute.")
            return folder, group_unique
        spike_clusters = np.load(folder / "spike_clusters.npy")
        spike_templates = np.load(folder / "spike_templates.npy")
        template_ind = np.load(template_ind_path)
        templates = np.load(folder / "templates.npy")

        peak_local = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
        peak_channels = template_ind[np.arange(template_ind.shape[0]), peak_local]

        n_clusters = int(spike_clusters.max()) + 1 if spike_clusters.size else 0
        cluster_channel_groups = np.zeros(n_clusters, dtype=int)
        for cluster_id in np.unique(spike_clusters):
            mask = spike_clusters == cluster_id
            template_counts = np.bincount(spike_templates[mask], minlength=peak_channels.shape[0])
            best_template = int(np.argmax(template_counts))
            peak_channel = int(peak_channels[best_template])
            if 0 <= peak_channel < channel_groups_out.size:
                cluster_channel_groups[cluster_id] = int(channel_groups_out[peak_channel])

        cluster_file = folder / "cluster_channel_group.tsv"
        with cluster_file.open("w", encoding="utf-8") as f:
            f.write("cluster_id\tchannel_group\n")
            for cid, group_val in enumerate(cluster_channel_groups):
                f.write(f"{cid}\t{int(group_val)}\n")
    except Exception as exc:
        print(f"Warning: could not recompute cluster channel groups: {exc}")

    return folder, group_unique


def warn_unknown_sc2_overrides(defaults: dict, overrides: dict, prefix: str = "") -> None:
    """Warn when SC2 overrides include keys not present in default params."""
    if not isinstance(overrides, dict):
        return
    for key, value in overrides.items():
        path = f"{prefix}{key}"
        if key not in defaults:
            print(f"Warning: SC2 override '{path}' not in default params; it may be ignored.")
            continue
        default_val = defaults.get(key)
        if isinstance(value, dict):
            if isinstance(default_val, dict):
                warn_unknown_sc2_overrides(default_val, value, prefix=f"{path}.")
            else:
                print(f"Warning: SC2 override '{path}' is a dict but default is not; it may be ignored.")


def export_for_si_gui(analyzer, base_folder: Path, label: str):
    """Save a Zarr copy that SpikeInterface GUI can open directly."""
    folder = base_folder / f"si_gui_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe_rmtree(folder)
    analyzer.save_as(format="zarr", folder=folder)
    print(f"Exported {label} analyzer to SpikeInterface GUI folder {folder}")
    return folder


# ---------------------------------------------------------------------
# Main pipeline flow
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SpykingCircus2 tetrode pipeline")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(os.environ.get("SPIKESORT_ROOT_DIR", DEFAULT_ROOT_DIR)),
        help="Root folder containing recordings (default: %(default)s or env SPIKESORT_ROOT_DIR)",
    )
    parser.add_argument(
        "--base-out",
        type=Path,
        default=Path(os.environ.get("SPIKESORT_BASE_OUT", DEFAULT_BASE_OUT)),
        help="Base output folder (default: %(default)s or env SPIKESORT_BASE_OUT)",
    )
    parser.add_argument(
        "--channel-groups",
        type=Path,
        default=None,
        help="Path to JSON file containing channel groups (list of lists). Overrides CHANNEL_GROUPS/env.",
    )
    parser.add_argument(
        "--bad-channels",
        type=Path,
        default=None,
        help="Path to JSON file containing bad channels (list). Overrides BAD_CHANNELS/env.",
    )
    args = parser.parse_args()

    if DEBUG_WARN_TRACE:
        enable_warning_trace()

    env_groups_path = os.environ.get("SPIKESORT_CHANNEL_GROUPS", None)
    env_bad_path = os.environ.get("SPIKESORT_BAD_CHANNELS", None)
    config_dir = PROJECT_ROOT / "config"
    if not args.channel_groups and not env_groups_path:
        group_candidates = sorted(config_dir.glob("channel_groups_*.json"))
        if group_candidates:
            CHANNEL_GROUPS_PATH = _choose_config_json(
                "channel groups",
                group_candidates,
                group_candidates[0] if len(group_candidates) == 1 else None,
            )
    if not args.bad_channels and not env_bad_path:
        bad_candidates = sorted(config_dir.glob("bad_channels_*.json"))
        if bad_candidates:
            BAD_CHANNELS_PATH = _choose_config_json(
                "bad channels",
                bad_candidates,
                bad_candidates[0] if len(bad_candidates) == 1 else None,
            )

    root_dir = args.root_dir
    base_out = args.base_out

    global SC2_OUT, SI_GUI_OUT
    SC2_OUT = base_out / "sc2_outputs"
    SI_GUI_OUT = base_out / "si_gui_exports"

    SC2_OUT.mkdir(parents=True, exist_ok=True)
    if EXPORT_TO_SI_GUI:
        SI_GUI_OUT.mkdir(parents=True, exist_ok=True)

    print(
        "Config summary: USE_SI_PREPROCESS=",
        USE_SI_PREPROCESS,
        "SI_APPLY_WHITEN=",
        SI_APPLY_WHITEN,
        "SI_APPLY_CAR=",
        SI_APPLY_CAR,
        "CAR_MODE=",
        CAR_MODE,
    )
    print(
        "Config summary: ATTACH_GEOMETRY=",
        ATTACH_GEOMETRY,
    )
    print(
        "Config summary: EXPORT_SCALE_TO_UV=",
        EXPORT_SCALE_TO_UV,
        "EXPORT_BANDPASS_FOR_PHY=",
        EXPORT_BANDPASS_FOR_PHY,
        f"EXPORT_BP={EXPORT_BP_MIN_HZ}-{EXPORT_BP_MAX_HZ}",
    )
    print(
        "Config summary: AUTO_BAD_CHANNELS=",
        AUTO_BAD_CHANNELS,
        "STRICT_GROUPS=",
        STRICT_GROUPS,
        "SORT_BY_GROUP=",
        SORT_BY_GROUP,
        "TEST_SECONDS=",
        TEST_SECONDS,
        "STREAM_NAME=",
        STREAM_NAME or "auto",
    )

    data_path = choose_recording_folder(root_dir, SESSION_SUBPATH, SESSION_SELECTION, 0)
    print(f"Recording root: {root_dir}")
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

    # Resolve manual group configuration (inline, env, or CLI file).
    manual_groups = CHANNEL_GROUPS
    groups_source = "inline CHANNEL_GROUPS"

    env_loaded = load_channel_groups_from_path(env_groups_path) if env_groups_path else None
    if env_loaded:
        manual_groups = env_loaded
        groups_source = f"env file {env_groups_path}"

    config_loaded = load_channel_groups_from_path(CHANNEL_GROUPS_PATH) if CHANNEL_GROUPS_PATH else None
    if config_loaded:
        manual_groups = config_loaded
        groups_source = f"config CHANNEL_GROUPS_PATH {CHANNEL_GROUPS_PATH}"

    cli_loaded = load_channel_groups_from_path(args.channel_groups) if args.channel_groups else None
    if cli_loaded:
        manual_groups = cli_loaded
        groups_source = f"CLI file {args.channel_groups}"

    base_groups = []
    if manual_groups:
        base_groups = resolve_manual_groups(recording, manual_groups)
        if not base_groups and groups_source != "inline CHANNEL_GROUPS":
            print(f"Warning: no channel groups resolved from {groups_source}; falling back to inline CHANNEL_GROUPS.")
            manual_groups = CHANNEL_GROUPS
            groups_source = "inline CHANNEL_GROUPS (fallback)"
            base_groups = resolve_manual_groups(recording, manual_groups)
        if not base_groups:
            print("Warning: no channel groups resolved; falling back to chunked groups.")
            manual_groups = []
    if not base_groups:
        if STRICT_GROUPS:
            raise RuntimeError(
                "No valid channel groups resolved. Provide CHANNEL_GROUPS or --channel-groups "
                "and ensure channel IDs match the recording."
            )
        base_groups = chunk_groups(original_channel_order, CHANNELS_PER_TETRODE)

    # Resolve bad channels (inline, env, or CLI file).
    bad_channels = BAD_CHANNELS
    bad_source = "inline BAD_CHANNELS"
    env_bad_loaded = load_bad_channels_from_path(env_bad_path) if env_bad_path else None
    if env_bad_loaded is not None:
        bad_channels = env_bad_loaded
        bad_source = f"env file {env_bad_path}"
    config_bad_loaded = load_bad_channels_from_path(BAD_CHANNELS_PATH) if BAD_CHANNELS_PATH else None
    if config_bad_loaded is not None:
        bad_channels = config_bad_loaded
        bad_source = f"config BAD_CHANNELS_PATH {BAD_CHANNELS_PATH}"
    cli_bad_loaded = load_bad_channels_from_path(args.bad_channels) if args.bad_channels else None
    if cli_bad_loaded is not None:
        bad_channels = cli_bad_loaded
        bad_source = f"CLI file {args.bad_channels}"
    print(f"Bad channels source: {bad_source}")

    manual_bad = resolve_bad_channel_ids(recording, bad_channels)
    auto_bad = detect_bad_channel_ids(
        recording,
        AUTO_BAD_CHANNELS,
        method=AUTO_BAD_CHANNELS_METHOD,
        **AUTO_BAD_CHANNELS_KWARGS,
    )
    # Sort for readability
    auto_bad_sorted = np.sort(auto_bad)
    if AUTO_BAD_CHANNELS:
        print(f"Auto-detected bad channels ({auto_bad.size}): {auto_bad_sorted.tolist()}")
    else:
        if auto_bad.size:
            print(f"Note: AUTO_BAD_CHANNELS disabled, ignoring detected list: {auto_bad_sorted.tolist()}")

    # Normalize bad channel IDs to the same dtype as recording.channel_ids before slicing.
    channel_dtype = np.asarray(channel_order).dtype if channel_order else object
    bad_arrays = [np.array(arr, dtype=channel_dtype) for arr in (manual_bad, auto_bad) if arr.size]
    if bad_arrays:
        bad_ids = np.unique(np.concatenate(bad_arrays))
        # Report indices in original order for reference
        bad_indices = sorted([original_index_map[str(ch)] for ch in bad_ids if str(ch) in original_index_map])
        print(f"Removed {len(bad_ids)} bad channels: {bad_ids.tolist()}")
        print(f"Removed indices in original order: {bad_indices}")
        keep_ids = [ch for ch in channel_order if ch not in bad_ids]
        recording = safe_channel_slice(recording, keep_ids)
        channel_order = keep_ids
    else:
        channel_order = original_channel_order.copy()

    # Preserve original tetrode grid positions even if some tetrodes are fully removed.
    base_tetrode_count = len(base_groups)
    filtered_groups = []
    filtered_indices = []
    for idx, grp in enumerate(base_groups):
        subset = [ch for ch in grp if ch in channel_order]
        if subset:
            filtered_groups.append(subset)
            filtered_indices.append(idx)
    groups = filtered_groups
    # Preserve original tetrode indices for group labels after bad-channel removal (non-contiguous IDs are expected).
    group_ids = filtered_indices.copy() if filtered_indices else list(range(len(groups)))
    grouped_ids = {ch for grp in groups for ch in grp}
    if grouped_ids:
        # Enforce that only grouped channels remain (avoids ungrouped channels sitting at [0, 0]).
        keep_ids = [ch for ch in channel_order if ch in grouped_ids]
        if len(keep_ids) != len(channel_order):
            missing = [ch for ch in channel_order if ch not in grouped_ids]
            print(
                f"Warning: {len(missing)} channels not present in groups; removing from recording "
                f"(first: {missing[:5]})"
            )
            recording = safe_channel_slice(recording, keep_ids)
            channel_order = keep_ids
    small_groups = [gid for gid, grp in zip(group_ids, groups) if len(grp) < 3]
    if small_groups:
        print(f"Warning: tetrodes with <3 channels: {small_groups}")
    if manual_groups:
        print(f"Tetrodes kept: {len(groups)} (source: {groups_source}); first group: {groups[0] if groups else 'n/a'}")
    else:
        print(f"Tetrodes kept: {len(groups)} (fallback chunking); first group: {groups[0] if groups else 'n/a'}")

    bundle_grid = False
    if not SORT_BY_GROUP and (BUNDLE_GROUPING_MODE or "").lower() in ("single_grid", "grid", "single"):
        bundle_grid = True
        groups = [channel_order.copy()]
        group_ids = [0]
        print(f"Bundle grouping mode: single grid ({len(groups[0])} channels).")

    tetrode_offsets = None
    tetrodes_per_row = max(1, int(np.ceil(np.sqrt(base_tetrode_count)))) if base_tetrode_count else 1
    if groups and not bundle_grid:
        dx = TETRODE_SPACING_DX_UM
        dy = TETRODE_SPACING_DY_UM
        num_rows = int(np.ceil(base_tetrode_count / tetrodes_per_row))
        tetrode_offsets = []
        for orig_idx in filtered_indices:
            row = orig_idx // tetrodes_per_row
            col = orig_idx % tetrodes_per_row
            x = col * dx
            y = (num_rows - 1 - row) * dy
            tetrode_offsets.append((x, y))

    if ATTACH_GEOMETRY and groups:
        if bundle_grid:
            recording = attach_bundle_grid_geom(recording, BUNDLE_GRID_COLS, BUNDLE_GRID_DX_UM, BUNDLE_GRID_DY_UM)
            print(f"Geometry attached to recording (bundle grid {BUNDLE_GRID_COLS} cols).")
        else:
            recording = ensure_geom_and_units(
                recording,
                groups,
                pitch=TETRODE_PITCH_UM,
                tetrodes_per_row=tetrodes_per_row,
                tetrode_offsets=tetrode_offsets,
                scale_to_uv=False,
            )
            print(f"Geometry attached to recording (tetrodes_per_row={tetrodes_per_row}).")

    # Optional SI common median reference before SC2 (per tetrode or global) when not using SI preprocessing
    if not USE_SI_PREPROCESS and SI_APPLY_CAR:
        try:
            mode = (CAR_MODE or "tetrode").lower()
            operator = (CAR_OPERATOR or "median").lower()
            if mode == "tetrode" and groups:
                # SpikeInterface uses reference="global" with groups to mean per-group CAR.
                print("WARNING: per-tetrode CAR can attenuate spikes; use only if verified on your data.")
                recording = spre.common_reference(
                    recording, reference="global", operator=operator, groups=groups
                )
                print(f"Applied SI per-tetrode CAR ({operator}) on {len(groups)} tetrodes.")
            else:
                recording = spre.common_reference(recording, reference="global", operator=operator)
                print(f"Applied SI global CAR ({operator}).")
        except Exception as exc:
            print(f"Warning: CAR failed (continuing without CAR): {exc}")

    rec_sc2 = preprocess_for_sc2(recording, groups=groups)

    if ATTACH_GEOMETRY and groups:
        # Re-attach geometry after preprocessing in case properties/probe were dropped.
        if bundle_grid:
            rec_sc2 = attach_bundle_grid_geom(rec_sc2, BUNDLE_GRID_COLS, BUNDLE_GRID_DX_UM, BUNDLE_GRID_DY_UM)
        else:
            rec_sc2 = ensure_geom_and_units(
                rec_sc2,
                groups,
                pitch=TETRODE_PITCH_UM,
                tetrodes_per_row=tetrodes_per_row,
                tetrode_offsets=tetrode_offsets,
                scale_to_uv=False,
            )
    if ATTACH_GEOMETRY and groups:
        try:
            rec_sc2 = ensure_probe_attached(rec_sc2)
        except Exception as exc:
            print(f"Warning: failed to attach probe to rec_sc2: {exc}")
    if MATERIALIZE_SI_PREPROCESS:
        try:
            preproc_folder = SC2_OUT / "rec_sc2_preprocessed"
            rec_sc2 = rec_sc2.save(folder=preproc_folder, format="binary_folder", overwrite=True)
            print(f"Materialized rec_sc2 at {preproc_folder}")
            if ATTACH_GEOMETRY and groups:
                rec_sc2 = ensure_geom_and_units(
                    rec_sc2,
                    groups,
                    pitch=TETRODE_PITCH_UM,
                    tetrodes_per_row=tetrodes_per_row,
                    tetrode_offsets=tetrode_offsets,
                    scale_to_uv=False,
                )
        except Exception as exc:
            print(f"Warning: failed to materialize rec_sc2: {exc}")

    def _force_dummy_probe(rec, label):
        try:
            if rec.get_probe() is None:
                rec = rec.set_dummy_probe_from_locations()
                print(f"Attached dummy probe to {label} (sorter input).")
        except Exception:
            try:
                rec = rec.set_dummy_probe_from_locations()
                print(f"Attached dummy probe to {label} (sorter input).")
            except Exception as exc:
                print(f"Warning: failed to attach dummy probe to {label}: {exc}")
        return rec

    if ATTACH_GEOMETRY:
        # Note: SC2 can still warn about missing probes in internal snippet wrappers; this
        # re-attaches a dummy probe so geometry is visible to those internal steps.
        rec_sc2 = _force_dummy_probe(rec_sc2, "rec_sc2")

    # Optionally use a scaled, bandpassed (non-whitened) copy for Phy/export so snippets are not raw.
    rec_export = recording
    export_scale = EXPORT_SCALE_TO_UV
    if export_scale:
        gains = None
        offsets = None
        try:
            gains = rec_export.get_channel_gains()
        except Exception:
            gains = None
        try:
            offsets = rec_export.get_channel_offsets()
        except Exception:
            offsets = None
        if gains is None:
            print("Export path: channel gains not found; export left in native units.")
        else:
            n_ch = rec_export.get_num_channels()
            gains_arr = np.asarray(gains)
            if gains_arr.ndim == 0:
                gains_ok = True
            else:
                gains_ok = gains_arr.size == n_ch
            if not gains_ok:
                print(
                    f"Export path: channel gains length {gains_arr.size} != n_channels {n_ch}; "
                    "export left in native units."
                )
            else:
                if offsets is None:
                    offsets_arr = np.zeros_like(gains_arr, dtype="float32")
                    print("Export path: channel offsets missing; using zeros.")
                else:
                    offsets_arr = np.asarray(offsets)
                    if offsets_arr.ndim != 0 and offsets_arr.size != n_ch:
                        offsets_arr = np.zeros_like(gains_arr, dtype="float32")
                        print("Export path: channel offsets length mismatch; using zeros.")
                rec_export = spre.scale(rec_export, gain=gains_arr, offset=offsets_arr, dtype="float32")
                print("Export path: scaled recording to microvolts for Phy/GUI export.")
    if EXPORT_BANDPASS_FOR_PHY:
        rec_export = spre.bandpass_filter(
            rec_export,
            freq_min=EXPORT_BP_MIN_HZ,
            freq_max=EXPORT_BP_MAX_HZ,
            ftype=EXPORT_BP_FTYPE,
            filter_order=EXPORT_BP_ORDER,
            margin_ms=EXPORT_BP_MARGIN_MS,
        )
        print(f"Export path: bandpass {EXPORT_BP_MIN_HZ}-{EXPORT_BP_MAX_HZ} Hz for Phy/GUI export.")
    if ATTACH_GEOMETRY and groups:
        # Re-attach geometry in case scaling/bandpass dropped probe metadata.
        if bundle_grid:
            rec_export = attach_bundle_grid_geom(rec_export, BUNDLE_GRID_COLS, BUNDLE_GRID_DX_UM, BUNDLE_GRID_DY_UM)
        else:
            rec_export = ensure_geom_and_units(
                rec_export,
                groups,
                pitch=TETRODE_PITCH_UM,
                tetrodes_per_row=tetrodes_per_row,
                tetrode_offsets=tetrode_offsets,
                scale_to_uv=False,
            )
    if MATERIALIZE_EXPORT:
        try:
            export_folder = SC2_OUT / "rec_export_prepared"
            rec_export = rec_export.save(folder=export_folder, format="binary_folder", overwrite=True)
            print(f"Materialized rec_export at {export_folder}")
            if ATTACH_GEOMETRY and groups:
                if bundle_grid:
                    rec_export = attach_bundle_grid_geom(rec_export, BUNDLE_GRID_COLS, BUNDLE_GRID_DX_UM, BUNDLE_GRID_DY_UM)
                else:
                    rec_export = ensure_geom_and_units(
                        rec_export,
                        groups,
                        pitch=TETRODE_PITCH_UM,
                        tetrodes_per_row=tetrodes_per_row,
                        tetrode_offsets=tetrode_offsets,
                        scale_to_uv=False,
                    )
        except Exception as exc:
            print(f"Warning: failed to materialize rec_export: {exc}")
    # Ensure group labels are on the export recording for downstream tools
    set_group_property(rec_export, groups, group_ids)
    if ANALYZER_FROM_SORTER:
        rec_analyzer = rec_sc2
        # Optionally scale to uV for QC interpretability (does not affect sorter output)
        if EXPORT_SCALE_TO_UV:
            try:
                gains = rec_analyzer.get_channel_gains()
            except Exception:
                gains = None
            try:
                offsets = rec_analyzer.get_channel_offsets()
            except Exception:
                offsets = None
            if gains is not None:
                gains_arr = np.asarray(gains)
                n_ch = rec_analyzer.get_num_channels()
                if gains_arr.ndim == 0 or gains_arr.size == n_ch:
                    offsets_arr = np.zeros_like(gains_arr, dtype="float32")
                    if offsets is not None:
                        off_arr = np.asarray(offsets)
                        if off_arr.ndim == 0 or off_arr.size == n_ch:
                            offsets_arr = off_arr
                    rec_analyzer = spre.scale(rec_analyzer, gain=gains_arr, offset=offsets_arr, dtype="float32")
                    print("Analyzer path: scaled rec_sc2 to microvolts for QC.")
        if ATTACH_GEOMETRY and groups:
            if bundle_grid:
                rec_analyzer = attach_bundle_grid_geom(rec_analyzer, BUNDLE_GRID_COLS, BUNDLE_GRID_DX_UM, BUNDLE_GRID_DY_UM)
            else:
                rec_analyzer = ensure_geom_and_units(
                    rec_analyzer,
                    groups,
                    pitch=TETRODE_PITCH_UM,
                    tetrodes_per_row=tetrodes_per_row,
                    tetrode_offsets=tetrode_offsets,
                    scale_to_uv=False,
                )
        if MATERIALIZE_EXPORT:
            try:
                analyzer_folder = SC2_OUT / "rec_analyzer_prepared"
                rec_analyzer = rec_analyzer.save(
                    folder=analyzer_folder,
                    format="binary_folder",
                    overwrite=True,
                )
                print(f"Materialized rec_analyzer at {analyzer_folder}")
            except Exception as exc:
                print(f"Warning: failed to materialize rec_analyzer: {exc}")
    else:
        rec_analyzer = rec_export

    sc2_run = SC2_OUT / f"sc2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe_rmtree(sc2_run)

    set_group_property(rec_sc2, groups, group_ids)  # stores tetrode index as 'group' property for downstream tools
    sc2_defaults = ss.Spykingcircus2Sorter.default_params()
    warn_unknown_sc2_overrides(sc2_defaults, SC2_PARAM_OVERRIDES)
    sc2_params = merge_params(sc2_defaults, SC2_PARAM_OVERRIDES)
    if USE_SI_PREPROCESS:
        sc2_params["apply_preprocessing"] = False
        if "preprocessing" in sc2_params:
            sc2_params["preprocessing"]["apply"] = False
    if SI_APPLY_CAR and not USE_SI_PREPROCESS and sc2_params.get("apply_preprocessing", True):
        print(
            "WARNING: SI CAR is enabled while SC2 preprocessing is on; "
            "SC2 will also apply global CMR (double reference)."
        )
    # Whitening guard: SC2 whitening cannot be disabled in the current SI wrapper,
    # so enabling SI whitening will double-whiten the data.
    if SI_APPLY_WHITEN:
        print("WARNING: SI whitening is enabled; SC2 will also whiten (double-whitening).")
    if SC2_PARAM_OVERRIDES:
        print(f"Applying SC2 overrides: {SC2_PARAM_OVERRIDES}")
    print(
        "SC2 params -> apply_preprocessing:",
        sc2_params.get("apply_preprocessing"),
        "| whitening: wrapper-controlled (applied internally by SC2)",
    )
    if isinstance(sc2_params.get("whitening"), dict):
        print("SC2 params -> whitening:", sc2_params.get("whitening"))
    analyzer_ms_before = None
    analyzer_ms_after = None
    try:
        general = sc2_params.get("general", {}) if isinstance(sc2_params, dict) else {}
        analyzer_ms_before = general.get("ms_before")
        analyzer_ms_after = general.get("ms_after")
    except Exception:
        pass
    if analyzer_ms_before is not None or analyzer_ms_after is not None:
        print(
            "Analyzer waveform window:",
            f"{analyzer_ms_before} ms before / {analyzer_ms_after} ms after",
        )
    phy_folders = []
    si_gui_folders = []
    try:
        if SORT_BY_GROUP:
            print("Sorting per tetrode group (split_by='group').")
            rec_sc2_dict = {gid: _force_dummy_probe(rec, f"rec_sc2[{gid}]") for gid, rec in rec_sc2.split_by("group").items()}
            rec_export_dict = rec_export.split_by("group")

            def _gid_key(value):
                try:
                    return int(value)
                except Exception:
                    return value

            def _get_group_rec(rec_dict, key, fallback):
                if rec_dict is None:
                    return fallback
                if key in rec_dict:
                    return rec_dict[key]
                key_int = _gid_key(key)
                if key_int in rec_dict:
                    return rec_dict[key_int]
                key_str = str(key)
                if key_str in rec_dict:
                    return rec_dict[key_str]
                return fallback

            if group_ids is not None:
                group_map = {_gid_key(gid): grp for gid, grp in zip(group_ids, groups)}
            else:
                group_map = {_gid_key(idx): grp for idx, grp in enumerate(groups)}
            sortings_sc2 = ss.run_sorter(
                "spykingcircus2",
                rec_sc2_dict,
                folder=sc2_run,
                remove_existing_folder=True,
                verbose=True,
                **sc2_params,
            )
            for gid, sorting_sc2 in sortings_sc2.items():
                gid_key = _gid_key(gid)
                rec_exp = _get_group_rec(rec_export_dict, gid_key, rec_export)
                rec_an = rec_exp if not ANALYZER_FROM_SORTER else _get_group_rec(rec_sc2_dict, gid_key, rec_exp)
                analyzer_sc2 = build_analyzer(
                    sorting_sc2,
                    rec_an,
                    SC2_OUT,
                    f"sc2_g{gid}",
                    wf_ms_before=analyzer_ms_before,
                    wf_ms_after=analyzer_ms_after,
                )
                analyzer_sc2 = maybe_remove_redundant_units(analyzer_sc2, f"sc2_g{gid}")
                if EXPORT_TO_PHY:
                    grp_channels = group_map.get(gid_key, list(rec_exp.channel_ids))
                    phy_folder, _ = export_for_phy(
                        analyzer_sc2,
                        SC2_OUT,
                        f"sc2_g{gid}",
                        [grp_channels],
                        original_index_map,
                        [gid],
                    )
                    phy_folders.append(phy_folder)
                if EXPORT_TO_SI_GUI:
                    si_gui_folders.append(export_for_si_gui(analyzer_sc2, SI_GUI_OUT, f"sc2_g{gid}"))
            print(f"SC2 per-group runs complete | output: {sc2_run}")
        else:
            sorting_sc2 = ss.run_sorter(
                "spykingcircus2",
                rec_sc2,
                folder=sc2_run,
                remove_existing_folder=True,
                verbose=True,
                **sc2_params,
            )
            print(f"SC2 units: {sorting_sc2.get_num_units()} | output: {sc2_run}")
    except Exception as exc:
        print(f"Error: SpykingCircus2 sorter failed: {exc}")
        print(f"Attempting to remove partial SC2 output folder: {sc2_run}")
        try:
            safe_rmtree(sc2_run)
        except Exception as exc2:
            print(f"Warning: failed to remove partial output folder: {exc2}")
        raise

    if not SORT_BY_GROUP:
        analyzer_sc2 = build_analyzer(
            sorting_sc2,
            rec_analyzer,
            SC2_OUT,
            "sc2",
            wf_ms_before=analyzer_ms_before,
            wf_ms_after=analyzer_ms_after,
        )
        analyzer_sc2 = maybe_remove_redundant_units(analyzer_sc2, "sc2")

        if EXPORT_TO_PHY:
            phy_folder, _ = export_for_phy(analyzer_sc2, SC2_OUT, "sc2", groups, original_index_map, group_ids)
            phy_folders.append(phy_folder)
        if EXPORT_TO_SI_GUI:
            si_gui_folders.append(export_for_si_gui(analyzer_sc2, SI_GUI_OUT, "sc2"))

    print("SC2 pipeline complete.")
    if phy_folders:
        if SORT_BY_GROUP:
            print("Summary: per-group exports written (see each group's QC/units within its folder).")
        for folder in phy_folders:
            params_py = folder / "params.py"
            if EXPORT_PHY_EXTRACT_WAVEFORMS:
                try:
                    subprocess.run(["phy", "extract-waveforms", str(params_py)], check=True)
                    print(f"Phy: extracted waveforms for export ({folder.name}).")
                except FileNotFoundError:
                    print("Warning: 'phy' command not found; skipping extract-waveforms.")
                except subprocess.CalledProcessError as exc:
                    print(f"Warning: phy extract-waveforms failed: {exc}")
            if not SORT_BY_GROUP:
                try:
                    n_units = len(sorting_sc2.get_unit_ids())
                except Exception:
                    n_units = "unknown"
                print(
                    "Summary:",
                    f"units={n_units},",
                    f"qc_metrics={Path(analyzer_sc2.folder) / 'qc_metrics.csv' if getattr(analyzer_sc2, 'folder', None) else 'n/a'}",
                )
            else:
                print("Summary: per-group Phy exports written.")
    if si_gui_folders:
        for folder in si_gui_folders:
            print(f"Run: python -m spikeinterface_gui \"{folder}\"")

if __name__ == "__main__":
    main()
