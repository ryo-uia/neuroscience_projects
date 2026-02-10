"""
Minimal SpikeInterface pipeline to run SpykingCircus2 on tetrode/stereotrode recordings.
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from spikeinterface.core import ChannelSparsity, create_sorting_analyzer
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (Pipelines/..)
sys.path.insert(0, str(PROJECT_ROOT))

from Functions.si_utils import ensure_geom_and_units, ensure_probe_attached, set_group_property
from Functions.pipeline_utils import (
    choose_config_json,
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
# User configuration
# ---------------------------------------------------------------------

# Workflow notes
# - Primary workflow: run this pipeline, then curate units in Phy.
# - Optional assistant workflow: use SpikeAgent after Phy curation.
# Recommended mixed baseline (starting point)
# - USE_SI_PREPROCESS=True, SI_BP_MIN_HZ=300, SI_BP_MAX_HZ=6000
# - SI_APPLY_CAR=False unless you confirm broad shared noise across groups.
# - SI_APPLY_WHITEN=False (SC2 applies whitening internally).
# - Keep mixed CHANNEL_GROUPS explicit; avoid chunk-based auto-grouping when possible.
# - SC2 detection threshold: keep detect_threshold around 5-6; increase if false positives are high.
# - EXPORT_PHY_CHANNEL_IDS_MODE="oe_label" for easiest channel traceability after curation.
# - Runtime workflow: tune with TEST_SECONDS=300-600, then run TEST_SECONDS=None for final exports.
# - Validation workflow: run mixed bundle mode first, then cross-check a few sessions/groups with stricter grouping.
# - Suggested config order: BAD_CHANNELS/CHANNEL_GROUPS -> preprocessing flags -> export options.
# SpikeAgent launch (Windows PowerShell)
#   1) cd C:\Users\ryoi\Documents\Code\SpikeSorting\docker\spikeagent
#   2) .\run_spikeagent_cpu.ps1 -DataPath "C:\Users\ryoi\Documents\Code\SpikeSorting\recordings" -ResultsPath "C:\Users\ryoi\Documents\Code\SpikeSorting\sc2_outputs"
#   3) Open http://127.0.0.1:8501
# - Stop with Ctrl+C in that terminal; rerun steps 1-3 next session.

# Session/data selection
TEST_SECONDS = 600  # None=full recording (use 300-600 for quick QC runs)
DEFAULT_ROOT_DIR = PROJECT_ROOT / "recordings"  # recordings root
SESSION_SUBPATH = None  # Optional: relative Open Ephys path to skip auto-discovery
SESSION_SELECTION = "prompt"  # session selection strategy ("prompt" recommended)
STREAM_NAME = None  # None=auto-detect; set explicit stream name to override

# Output folders
DEFAULT_BASE_OUT = PROJECT_ROOT  # base output folder for sc2_outputs / si_gui_exports
# Initialized in main() after parsing CLI/env
SC2_OUT = None  # SC2 output folder (set in main)
SI_GUI_OUT = None  # SI GUI output folder (set in main)

# Export controls
EXPORT_TO_PHY = True  # True=write Phy export after sorting; False=skip
EXPORT_TO_SI_GUI = False  # True=write SpikeInterface GUI export; False=skip
EXPORT_PHY_EXTRACT_WAVEFORMS = False  # True=run `phy extract-waveforms` to precompute waveforms for faster UI; False=skip (Phy computes on open)
# Requires Phy CLI on PATH; otherwise it warns/skips.
# None=auto: when exporting a single group, keep as-exported contiguous IDs and skip channel_id rewrites;
# otherwise rewrite channel_ids for stable mapping.
SIMPLE_PHY_EXPORT = None
EXPORT_PHY_CHANNEL_IDS_MODE = "oe_label"  # labels in Phy export metadata: oe_index (numeric), oe_label (CH##, recommended), as_exported (compact)
# Why this matters: Phy shows contiguous export indices only; channel_ids in params.py is the
# stable mapping back to Open Ephys channels for downstream analysis and reporting.
MATERIALIZE_EXPORT = False  # True=save rec_export to disk before analyzer/export (faster reuse; uses extra space); False=in-memory

# Analyzer/QC
COMPUTE_QC_METRICS = True  # True=compute QC metrics on analyzer output; False=skip (PC metrics can add noticeable runtime)
QC_METRIC_NAMES = [  # QC metrics that do not require PCs
    "firing_rate",
    "presence_ratio",
    "isi_violation",
    "snr",
    "amplitude_cutoff",
]
QC_PC_METRICS = {  # QC metrics that require PCs
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    "nn_isolation",
    "nn_noise_overlap",
    "silhouette",
}

# Channel grouping (tetrodes/stereotrodes)
CHANNELS_PER_TETRODE = 4  # used for auto-grouping when no explicit groups
# Config JSON discovery/prompting
USE_CONFIG_JSONS = False  # True=prompt for config/*.json channel groups + bad channels; False=skip (enable only for interactive JSON selection)
# Optional explicit channel grouping. If set, this fixed map is used instead of chunking by order
# (prevents shifts after bad-channel removal).
# Mixed recordings can include stereotrode groups (2 channels).
# Every kept channel should appear exactly once across groups.
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
CHANNEL_GROUPS_PATH = None  # Optional: JSON file path or env SPIKESORT_CHANNEL_GROUPS (None=ignore)
STRICT_GROUPS = True  # True=error if no valid groups; False=chunk by order
# Optional: bundle layout (geometry only). Does NOT change grouping/sorting mode.
BUNDLE_GROUPING_MODE = "tetrode"  # "tetrode" (default) or "single_grid"
# When "tetrode", the bundle grid settings below are ignored.
BUNDLE_GRID_COLS = 4  # columns for single-grid bundle layout
BUNDLE_GRID_DX_UM = 10.0  # bundle grid x-spacing (synthetic; mirrors .prb layout when single_grid)
BUNDLE_GRID_DY_UM = 200.0  # bundle grid y-spacing (synthetic; mirrors .prb layout when single_grid)
#
# Sorting vs grouping (mixed):
# - Mixed pipeline runs a single SC2 bundle by design.
# - CHANNEL_GROUPS still define tetrodes/stereotrodes for geometry/CAR/labels.
# - BUNDLE_GROUPING_MODE only affects geometry when bundling (single_grid is layout-only).

# Bad channels (use OE channel IDs like CH## to avoid positional mismatch across sessions or after slicing; set [] for none)
BAD_CHANNELS = []
BAD_CHANNELS_PATH = None  # Optional: JSON file path or env SPIKESORT_BAD_CHANNELS (None=ignore)
AUTO_BAD_CHANNELS = False  # True=auto-detect bad channels (merged with manual list); False=skip
AUTO_BAD_CHANNELS_METHOD = "std"  # only used when AUTO_BAD_CHANNELS=True; "std" or "mad" are typical for tetrodes
AUTO_BAD_CHANNELS_KWARGS = {}  # only used when AUTO_BAD_CHANNELS=True; extra args passed to detect_bad_channels ({}=defaults)

# Geometry/traceview display
ATTACH_GEOMETRY = True  # True=attach probe geometry to recording; False=skip
# Note: some internal SI/SC2 wrapper objects may still emit "no Probe attached; creating dummy"
# warnings even when the main recording has probe geometry attached.
LINEARIZE_TRACEVIEW = True  # True=flatten groups for traceview display; False=keep group layout
TRACEVIEW_CONTACT_SPACING_UM = 20.0  # spacing between contacts in traceview
TRACEVIEW_GROUP_SPACING_UM = 200.0  # spacing between groups in traceview
LABEL_STEREOTRODES_AS_MUA = False  # True=pre-label stereotrode groups as MUA; False=label in Phy
STEREOTRODE_MUA_LABEL = "mua"  # label used when LABEL_STEREOTRODES_AS_MUA=True
DEFAULT_CLUSTER_LABEL = "unsorted"  # default cluster label for other groups
# Synthetic within-tetrode pitch for the 2x2/2x1 layout (geometry only).
TETRODE_PITCH_UM = 20.0  # within-tetrode spacing (2x2)
# Synthetic spacing between tetrodes in the bundle layout (only affects geometry/whitening/visualization).
TETRODE_SPACING_DX_UM = 300.0  # spacing between tetrodes (x)
TETRODE_SPACING_DY_UM = 300.0  # spacing between tetrodes (y)

# Optional SI preprocessing: bandpass (+ optional notch/CAR/whitening).
# When enabled, SC2 filtering/CMR are disabled, but SC2 still whitens internally.
# OE raw data are typically int16; SI scaling/filtering/whitening outputs float32.
USE_SI_PREPROCESS = False  # True=apply SI preprocessing before SC2; False=let SC2 handle filtering
SI_BP_MIN_HZ = 300  # SI bandpass low cut (Hz)
SI_BP_MAX_HZ = 6000  # SI bandpass high cut (Hz)
SI_BP_FTYPE = "bessel"  # SI bandpass filter type
SI_BP_ORDER = 2  # SI bandpass filter order
SI_BP_MARGIN_MS = 10  # SI bandpass margin (ms)
MATERIALIZE_SI_PREPROCESS = False  # True=save preprocessed rec_sc2 to disk (faster reuse; uses extra space); False=in-memory

SI_APPLY_WHITEN = False  # True=apply SI whitening when USE_SI_PREPROCESS=True (double-whitening); False=skip
SI_WHITEN_MODE = "local"  # "global" or "local"
SI_WHITEN_RADIUS_UM = 100.0  # used when SI_WHITEN_MODE == "local"

# Optional SI common reference before SC2. SC2 preprocessing (when enabled) does its own bandpass/CMR/whitening.
SI_APPLY_CAR = False  # True=enable common average reference; False=skip
# "tetrode" -> per-tetrode CAR; "global" -> CAR over all channels
CAR_MODE = "global"  # CAR scope: global or tetrode
CAR_OPERATOR = "median"  # "median" (robust) or "average"
# Tetrodes: per-group median is common but can induce bipolar spikes with small/imbalanced groups; global is safer.

# Optional notch filtering (applies after SI bandpass when USE_SI_PREPROCESS=True,
# or before SC2 preprocessing when USE_SI_PREPROCESS=False).
APPLY_NOTCH = False  # True=enable notch filter; False=skip
NOTCH_FREQUENCIES = [50, 100, 150]  # notch frequencies (Hz)
NOTCH_Q = 30  # notch filter Q

# Optional bandpass copy for Phy/export (useful when SC2 preprocessing is enabled so Phy doesn't read raw).
# Common practice: Phy export is scaled + optionally bandpassed, but not whitened.
EXPORT_BANDPASS_FOR_PHY = True  # True=bandpass export; False=export raw for comparison
EXPORT_BP_MIN_HZ = SI_BP_MIN_HZ  # export bandpass low cut (Hz)
EXPORT_BP_MAX_HZ = SI_BP_MAX_HZ  # export bandpass high cut (Hz)
EXPORT_BP_FTYPE = SI_BP_FTYPE  # export bandpass filter type
EXPORT_BP_ORDER = SI_BP_ORDER  # export bandpass filter order
EXPORT_BP_MARGIN_MS = SI_BP_MARGIN_MS  # export bandpass margin (ms)
EXPORT_SCALE_TO_UV = True  # True=scale export recording to microvolts if gain info is present; False=leave native units
# Phy export sparsity: top-N channels per unit in exported templates/features.
# Set to None or <=0 for dense export (all channels).
EXPORT_SPARSE_CHANNELS = 4
ANALYZER_FROM_SORTER = True  # True=use rec_sc2 for analyzer/QC; False=use rec_export
SAVE_ANALYZER = False  # True=persist analyzer to disk (binary_folder); False=skip
REMOVE_REDUNDANT_UNITS = False  # True=remove near-duplicate units post-sort; False=skip
REDUNDANT_THRESHOLD = 0.95  # higher = stricter duplicate detection
REDUNDANT_STRATEGY = "minimum_shift"  # curation strategy for duplicates

# Optional overrides for SpykingCircus2 parameters (keys mirror default params structure)
# Add tweaks here if you need to change SC2 defaults (filtering, thresholds, etc.).
SC2_PARAM_OVERRIDES: dict = {  # {}=use SC2 defaults
    # Tetrodes/stereotrodes: disable motion correction (intended for dense probes).
    "apply_motion_correction": False,
    # Match native SC2 waveform window (N_t = 3 ms total).
    "general": {
        "ms_before": 1.0,
        "ms_after": 2.0,
    },
    # SC2 internal filtering (only used when SC2 preprocessing is enabled).
    "filtering": {
        "freq_min": SI_BP_MIN_HZ,
        "freq_max": SI_BP_MAX_HZ,
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
DEBUG_WARN_TRACE = False  # print stack traces for warnings (debug)
DEBUG_GEOMETRY_ATTACH = False  # log each geometry attach/reattach (debug)
# ---------------------------------------------------------------------
# Pipeline helpers from Functions.pipeline_utils and Functions.si_utils.
# ---------------------------------------------------------------------


def filter_groups_with_indices(groups: Iterable[Iterable], valid_ids: Sequence) -> tuple[List[List], List[int]]:
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


def attach_geometry_if_needed(
    recording,
    groups,
    *,
    bundle_grid: bool,
    tetrodes_per_row: int,
    tetrode_offsets,
    scale_to_uv: bool = False,
    label: str | None = None,
):
    """Attach probe geometry if enabled, preserving existing scale when requested."""
    if not ATTACH_GEOMETRY or not groups:
        return recording
    if bundle_grid:
        out = attach_bundle_grid_geom(recording, BUNDLE_GRID_COLS, BUNDLE_GRID_DX_UM, BUNDLE_GRID_DY_UM)
    else:
        out = ensure_geom_and_units(
            recording,
            groups,
            pitch=TETRODE_PITCH_UM,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
            scale_to_uv=scale_to_uv,
        )
    if DEBUG_GEOMETRY_ATTACH:
        label_text = f" ({label})" if label else ""
        print(f"Geometry attach{label_text}: bundle_grid={bundle_grid}, groups={len(groups)}.")
    return out


def enable_warning_trace(limit: int = 12) -> None:
    """Install a warning hook that prints a short traceback (debug-only)."""
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        traceback.print_stack(limit=limit)
        print(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = _showwarning
    warnings.simplefilter("always")


def preprocess_for_sc2(recording, groups=None):
    """Apply optional SI preprocessing; otherwise leave SC2 to preprocess."""
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
            mode = (CAR_MODE or "group").lower()
            operator = (CAR_OPERATOR or "median").lower()
            if mode in ("group", "tetrode") and groups:
                rec = spre.common_reference(rec, reference="global", operator=operator, groups=groups)
                print(f"Preprocessing (SI): applied per-group CAR ({operator}) on {len(groups)} groups.")
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


def build_analyzer(
    sorting,
    recording,
    base_folder: Path,
    label: str,
    wf_ms_before: float | None = None,
    wf_ms_after: float | None = None,
):
    """Create a SortingAnalyzer, compute waveforms/PCs/QC, and optionally save outputs."""
    folder = base_folder / f"analyzer_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    qc_metrics_path = None
    try:
        recording = ensure_probe_attached(recording)
        print("Analyzer recording probe attached.")
    except Exception as exc:
        print(f"WARNING: failed to attach probe for analyzer: {exc}")
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
    analyzer.compute({"random_spikes": {"max_spikes_per_unit": 500, "seed": 42}}, verbose=True)
    wf_kwargs = {}
    if wf_ms_before is not None:
        wf_kwargs["ms_before"] = wf_ms_before
    if wf_ms_after is not None:
        wf_kwargs["ms_after"] = wf_ms_after
    analyzer.compute(
        "waveforms",
        **wf_kwargs,
        dtype="float32",
        verbose=True,
        n_jobs=4,
        chunk_duration="2s",
    )
    analyzer.compute("templates")
    analyzer.compute("principal_components")
    if COMPUTE_QC_METRICS:
        qc_done = False
        try:
            try:
                analyzer.compute("noise_levels")
            except Exception as exc:
                print(f"WARNING: noise_levels failed: {exc}")
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
                    qc_metrics_path = out_path
                except Exception as exc:
                    print(f"WARNING: QC metrics save failed: {exc}")
                qc_done = True
            else:
                print("QC metrics computed (extension) but no data returned.")
        except Exception:
            pass
        if not qc_done:
            if compute_quality_metrics is None:
                print("WARNING: QC metrics unavailable; skipping quality metrics.")
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
                        qc_metrics_path = out_path
                    except Exception as exc:
                        print(f"WARNING: QC metrics save failed: {exc}")
                except Exception as exc:
                    print(f"WARNING: QC metrics failed: {exc}")
    if qc_metrics_path is not None:
        setattr(analyzer, "_qc_metrics_path", str(qc_metrics_path))
    print(f"Analyzer computed for {label} at {folder}")
    return analyzer


def maybe_remove_redundant_units(analyzer, label: str):
    """Optionally remove redundant units via spikeinterface.curation."""
    if not REMOVE_REDUNDANT_UNITS:
        return analyzer
    if sc is None:
        print("WARNING: spikeinterface.curation unavailable; skipping redundant-unit removal.")
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
        print(f"WARNING: redundant-unit removal failed for {label}: {exc}")
    return analyzer


def build_phy_export_sparsity(analyzer):
    """Build optional Phy export sparsity from config (best channels per unit)."""
    if EXPORT_SPARSE_CHANNELS is None:
        return None

    try:
        n_sparse = int(EXPORT_SPARSE_CHANNELS)
    except Exception:
        print(
            f"WARNING: EXPORT_SPARSE_CHANNELS={EXPORT_SPARSE_CHANNELS!r} is invalid; "
            "using dense Phy export."
        )
        return None

    if n_sparse <= 0:
        print("Phy export: dense mode (EXPORT_SPARSE_CHANNELS<=0).")
        return None

    try:
        n_channels = int(analyzer.recording.get_num_channels())
    except Exception:
        n_channels = n_sparse
    n_use = max(1, min(n_sparse, n_channels))
    if n_use < n_sparse:
        print(
            f"Phy export: clamped EXPORT_SPARSE_CHANNELS from {n_sparse} to {n_use} "
            f"(recording has {n_channels} channels)."
        )

    try:
        sparsity = ChannelSparsity.from_best_channels(analyzer, num_channels=n_use, peak_sign="neg")
        print(f"Phy export sparsity: top {n_use} channel(s) per unit.")
        return sparsity
    except Exception as exc:
        print(f"WARNING: failed to build Phy export sparsity ({exc}); using dense export.")
        return None


def export_for_phy(
    analyzer,
    base_folder: Path,
    label: str,
    groups,
    original_index_map: dict,
    group_ids=None,
    group_sizes_by_id=None,
    recording_override=None,
):
    """Export Phy folder and (when requested) rewrite channel_ids in params.py."""
    folder = base_folder / f"phy_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_sparsity = build_phy_export_sparsity(analyzer)
    # copy_binary=True so Phy extracts snippets from the exported binary, not the raw recording.
    export_kwargs = dict(
        output_folder=folder,
        remove_if_exists=True,
        copy_binary=True,
        sparsity=export_sparsity,
    )
    export_recording = analyzer.recording
    export_source = "analyzer.recording"
    if recording_override is not None:
        try:
            export_to_phy(analyzer, recording=recording_override, **export_kwargs)
            export_recording = recording_override
            export_source = "recording_override"
        except TypeError as exc:
            # Back-compat for older SI versions: export_to_phy(recording, sorting, ...)
            try:
                export_to_phy(recording_override, analyzer.sorting, **export_kwargs)
                export_recording = recording_override
                export_source = "recording_override (legacy)"
            except Exception:
                print(f"WARNING: export with recording_override failed; falling back to analyzer ({exc})")
                export_to_phy(analyzer, **export_kwargs)
                export_source = "analyzer.recording (fallback)"
        except Exception as exc:
            print(f"WARNING: export with recording_override failed; falling back to analyzer ({exc})")
            export_to_phy(analyzer, **export_kwargs)
            export_source = "analyzer.recording (fallback)"
    else:
        export_to_phy(analyzer, **export_kwargs)
    print(
        f"Phy export source: {export_source} | scaled_to_uV={EXPORT_SCALE_TO_UV} | "
        f"bandpass_for_phy={EXPORT_BANDPASS_FOR_PHY}"
    )
    if EXPORT_BANDPASS_FOR_PHY and export_source != "recording_override":
        print(
            "WARNING: EXPORT_BANDPASS_FOR_PHY=True but Phy export used analyzer.recording; "
            "bandpass export may be bypassed."
        )
    if SIMPLE_PHY_EXPORT is None:
        simple_flag = groups is not None and len(groups) <= 1
    else:
        simple_flag = SIMPLE_PHY_EXPORT
    simple_export = bool(simple_flag) and groups is not None and len(groups) <= 1
    if simple_export and EXPORT_PHY_CHANNEL_IDS_MODE == "as_exported":
        return folder, None
    params_path = folder / "params.py"

    cache_dir = folder / ".phy"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    channel_ids_rec = list(export_recording.channel_ids)
    group_unique = None
    try:
        group_prop_check = export_recording.get_property("group")
        if group_prop_check is None:
            print("WARNING: export recording has no 'group' property.")
        else:
            group_unique = np.unique(group_prop_check)
    except Exception as exc:
        print(f"WARNING: could not read 'group' property: {exc}")
    def lookup_index(ch, fallback):
        if ch in original_index_map:
            return int(original_index_map[ch])
        ch_str = str(ch)
        if ch_str in original_index_map:
            return int(original_index_map[ch_str])
        return int(fallback)

    if EXPORT_PHY_CHANNEL_IDS_MODE not in ("oe_index", "oe_label", "as_exported"):
        print(
            f"WARNING: EXPORT_PHY_CHANNEL_IDS_MODE={EXPORT_PHY_CHANNEL_IDS_MODE!r} is invalid; "
            "falling back to 'oe_index'."
        )
        export_ids_mode = "oe_index"
    else:
        export_ids_mode = EXPORT_PHY_CHANNEL_IDS_MODE

    if export_ids_mode == "oe_label":
        labels = None
        try:
            if "channel_name" in export_recording.get_property_keys():
                labels = list(export_recording.get_property("channel_name"))
        except Exception:
            labels = None
        if not labels:
            labels = [str(ch) for ch in channel_ids_rec]
        channel_ids_out = np.array(labels, dtype=str)
        channel_ids_text = f"channel_ids = np.array({labels!r}, dtype=str)"
    elif export_ids_mode == "as_exported":
        channel_ids_out = np.array(channel_ids_rec, dtype=object)
        channel_ids_text = None
    else:
        # Phy uses channel_map for compact 0..N-1 ordering in the exported binary (shifts after removals),
        # and channel_ids for labels; we keep channel_ids as original OE indices so gaps indicate dropped channels.
        channel_ids_out = np.array(
            [lookup_index(ch, idx) for idx, ch in enumerate(channel_ids_rec)],
            dtype=np.int32,
        )
        channel_ids_text = f"channel_ids = np.array({channel_ids_out.tolist()}, dtype=np.int32)"
    channel_map = np.arange(len(channel_ids_rec), dtype=np.int32)

    group_lookup = {}
    slot_lookup = {}
    group_sizes = {}
    group_ids = list(group_ids) if group_ids is not None else list(range(len(groups)))
    for g_idx, group in enumerate(groups):
        group_id = group_ids[g_idx] if g_idx < len(group_ids) else g_idx
        group_sizes[group_id] = len(group)
        for slot, ch in enumerate(group):
            group_lookup[ch] = group_id
            group_lookup[str(ch)] = group_id
            slot_lookup[ch] = (slot, len(group))
            slot_lookup[str(ch)] = (slot, len(group))

    channel_groups_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    channel_shanks_out = np.zeros(len(channel_ids_rec), dtype=np.int32)
    text = None

    if params_path.exists():
        text = params_path.read_text()
        if "import numpy as np" not in text:
            text = "import numpy as np\n" + text

        if channel_ids_text is not None:
            pattern_ids = re.compile(r"channel_ids\s*=.*")
            text, count_ids = pattern_ids.subn(channel_ids_text, text, count=1)
            if not count_ids:
                text += f"\n{channel_ids_text}\n"
            try:
                np.save(folder / "channel_ids.npy", channel_ids_out)
            except Exception as exc:
                print(f"WARNING: could not overwrite channel_ids.npy: {exc}")
        pattern_map = re.compile(r"channel_map\s*=.*")
        replacement_map = f"channel_map = np.array({channel_map.tolist()}, dtype=np.int32)"
        text, count_map = pattern_map.subn(replacement_map, text, count=1)
        if not count_map:
            text += f"\n{replacement_map}\n"
        try:
            np.save(folder / "channel_map.npy", channel_map.astype(np.int32))
        except Exception as exc:
            print(f"WARNING: could not overwrite channel_map.npy: {exc}")

    def lookup_group(ch, fallback=0):
        if ch in group_lookup:
            return int(group_lookup[ch])
        ch_str = str(ch)
        if ch_str in group_lookup:
            return int(group_lookup[ch_str])
        return int(fallback)

    def lookup_slot(ch, group_id: int):
        if ch in slot_lookup:
            return slot_lookup[ch]
        ch_str = str(ch)
        if ch_str in slot_lookup:
            return slot_lookup[ch_str]
        size = group_sizes.get(group_id, 1)
        return 0, size

    for idx, ch in enumerate(channel_ids_rec):
        group_id = lookup_group(ch, 0)
        channel_groups_out[idx] = group_id
        channel_shanks_out[idx] = group_id

    if LINEARIZE_TRACEVIEW:
        contact_spacing = float(TRACEVIEW_CONTACT_SPACING_UM)
        group_spacing = float(TRACEVIEW_GROUP_SPACING_UM)
        positions = np.zeros((len(channel_ids_rec), 2), dtype=np.float32)

        for idx, ch in enumerate(channel_ids_rec):
            group_id = lookup_group(ch, 0)
            slot, gsize = lookup_slot(ch, group_id)
            col = slot if gsize == 2 else slot % 2
            row = 0 if gsize == 2 else slot // 2
            positions[idx, 0] = col * contact_spacing
            positions[idx, 1] = group_id * group_spacing + row * contact_spacing + (slot * 1e-2)

        for name in ("channel_positions.npy", "channel_locations.npy"):
            try:
                np.save(folder / name, positions)
            except Exception as exc:
                print(f"WARNING: could not overwrite {name}: {exc}")

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
            print(f"WARNING: could not overwrite channel_groups.npy: {exc}")
        try:
            np.save(folder / "channel_shanks.npy", channel_shanks_out.astype(np.int32))
        except Exception as exc:
            print(f"WARNING: could not overwrite channel_shanks.npy: {exc}")
        params_path.write_text(text)

    # Rebuild cluster-level channel group assignments for Phy.
    cluster_channel_groups = None
    try:
        spike_clusters = np.load(folder / "spike_clusters.npy")
        spike_templates = np.load(folder / "spike_templates.npy")
        template_ind_path = folder / "template_ind.npy"
        template_ind = np.load(template_ind_path) if template_ind_path.exists() else None
        templates = np.load(folder / "templates.npy")

        peak_local = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
        if template_ind is not None and template_ind.shape[0] == peak_local.shape[0]:
            peak_channels = template_ind[np.arange(template_ind.shape[0]), peak_local]
        else:
            if template_ind is not None:
                print("WARNING: template_ind shape mismatch; using peak_local indices.")
            peak_channels = peak_local

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
        print(f"WARNING: could not recompute cluster channel groups: {exc}")

    # Optionally pre-label stereotrode clusters as MUA for Phy curation.
    try:
        if LABEL_STEREOTRODES_AS_MUA and cluster_channel_groups is not None:
            if group_sizes_by_id:
                stereotrode_groups = {
                    (group_ids[idx] if idx < len(group_ids) else idx)
                    for idx, g in enumerate(groups)
                    if group_sizes_by_id.get(
                        (group_ids[idx] if idx < len(group_ids) else idx), len(g)
                    )
                    <= 2
                }
            else:
                stereotrode_groups = {
                    (group_ids[idx] if idx < len(group_ids) else idx)
                    for idx, g in enumerate(groups)
                    if len(g) <= 2
                }
            cluster_group_file = folder / "cluster_group.tsv"
            with cluster_group_file.open("w", encoding="utf-8") as f:
                f.write("cluster_id\tgroup\n")
                for cid, group_val in enumerate(cluster_channel_groups):
                    label_out = STEREOTRODE_MUA_LABEL if group_val in stereotrode_groups else DEFAULT_CLUSTER_LABEL
                    f.write(f"{cid}\t{label_out}\n")
            print(
                f"Labeled stereotrode clusters as '{STEREOTRODE_MUA_LABEL}' in cluster_group.tsv "
                f"({len(stereotrode_groups)} stereotrode groups)."
            )
    except Exception as exc:
        print(f"WARNING: could not label stereotrode clusters as MUA: {exc}")

    return folder, group_unique


def warn_unknown_sc2_overrides(defaults: dict, overrides: dict, prefix: str = "") -> None:
    """Warn when SC2 overrides include keys not present in default params."""
    if not isinstance(overrides, dict):
        return
    for key, value in overrides.items():
        path = f"{prefix}{key}"
        if key not in defaults:
            print(f"WARNING: SC2 override '{path}' not in default params; it may be ignored.")
            continue
        default_val = defaults.get(key)
        if isinstance(value, dict):
            if isinstance(default_val, dict):
                warn_unknown_sc2_overrides(default_val, value, prefix=f"{path}.")
            else:
                print(f"WARNING: SC2 override '{path}' is a dict but default is not; it may be ignored.")


def export_for_si_gui(analyzer, base_folder: Path, label: str):
    """Save a Zarr copy that SpikeInterface GUI can open directly."""
    folder = base_folder / f"si_gui_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe_rmtree(folder)
    analyzer.save_as(format="zarr", folder=folder)
    print(f"Exported {label} analyzer to SpikeInterface GUI folder {folder}")
    return folder


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
    """Reserve a unique run folder path under sc2_outputs for this invocation."""
    out_root = base_out / "sc2_outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = out_root / f"sc2_run_{run_tag}"
    suffix = 1
    while run_folder.exists():
        run_folder = out_root / f"sc2_run_{run_tag}_{suffix:02d}"
        suffix += 1
    return run_folder


def enable_run_logging(log_path: Path):
    """Mirror stdout/stderr to a per-run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, handle)
    sys.stderr = TeeStream(sys.stderr, handle)
    return handle


def disable_run_logging(handle):
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


# ---------------------------------------------------------------------
# Main pipeline flow
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SpykingCircus2 tetrode/stereotrode pipeline")
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
    parser.add_argument(
        "--no-config-json",
        action="store_true",
        help="Skip auto-prompting for config/*.json channel group/bad-channel files.",
    )
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Resolve config/session/groups/bad channels and exit before geometry/preprocessing/sorting/export.",
    )
    args = parser.parse_args()

    if DEBUG_WARN_TRACE:
        enable_warning_trace()

    # Resolve config JSON prompts and env/CLI overrides.
    env_groups_path = os.environ.get("SPIKESORT_CHANNEL_GROUPS", None)
    env_bad_path = os.environ.get("SPIKESORT_BAD_CHANNELS", None)
    config_dir = PROJECT_ROOT / "config"
    # Start from module-level defaults (inline JSON paths), then override via env/CLI/prompt.
    channel_groups_path = CHANNEL_GROUPS_PATH
    bad_channels_path = BAD_CHANNELS_PATH
    use_config_jsons = USE_CONFIG_JSONS and not args.no_config_json
    if use_config_jsons and not args.channel_groups and not env_groups_path and not channel_groups_path:
        group_candidates = sorted(config_dir.glob("channel_groups_*.json"))
        if group_candidates:
            channel_groups_path = choose_config_json(
                "channel groups",
                group_candidates,
                group_candidates[0] if len(group_candidates) == 1 else None,
            )
    if use_config_jsons and not args.bad_channels and not env_bad_path and not bad_channels_path:
        bad_candidates = sorted(config_dir.glob("bad_channels_*.json"))
        if bad_candidates:
            bad_channels_path = choose_config_json(
                "bad channels",
                bad_candidates,
                bad_candidates[0] if len(bad_candidates) == 1 else None,
            )

    root_dir = args.root_dir
    base_out = args.base_out
    sc2_run = reserve_run_folder(base_out)

    # Output folders (shared across runs).
    global SC2_OUT, SI_GUI_OUT
    SC2_OUT = base_out / "sc2_outputs"
    SI_GUI_OUT = base_out / "si_gui_exports"

    SC2_OUT.mkdir(parents=True, exist_ok=True)
    if EXPORT_TO_SI_GUI:
        SI_GUI_OUT.mkdir(parents=True, exist_ok=True)
    run_log_path = SC2_OUT / "run_logs" / f"{sc2_run.name}.log"
    _run_log_handle = enable_run_logging(run_log_path)
    atexit.register(disable_run_logging, _run_log_handle)
    print(f"Run log: {run_log_path}")

    # Echo config for reproducibility.
    print(
        "Config (preprocess): USE_SI_PREPROCESS=",
        USE_SI_PREPROCESS,
        "SI_APPLY_WHITEN=",
        SI_APPLY_WHITEN,
        "SI_APPLY_CAR=",
        SI_APPLY_CAR,
        "CAR_MODE=",
        CAR_MODE,
    )
    print(
        "Config (geometry): ATTACH_GEOMETRY=",
        ATTACH_GEOMETRY,
    )
    print(
        "Config (export): EXPORT_SCALE_TO_UV=",
        EXPORT_SCALE_TO_UV,
        "EXPORT_BANDPASS_FOR_PHY=",
        EXPORT_BANDPASS_FOR_PHY,
        f"EXPORT_BP={EXPORT_BP_MIN_HZ}-{EXPORT_BP_MAX_HZ}",
    )
    print(
        "Config (session/groups): AUTO_BAD_CHANNELS=",
        AUTO_BAD_CHANNELS,
        "STRICT_GROUPS=",
        STRICT_GROUPS,
        "USE_CONFIG_JSONS=",
        use_config_jsons,
        "TEST_SECONDS=",
        TEST_SECONDS,
        "STREAM_NAME=",
        STREAM_NAME or "auto",
    )

    print("---- Session ----")
    data_path = choose_recording_folder(root_dir, SESSION_SUBPATH, SESSION_SELECTION, 0)
    print(f"Recording root: {root_dir}")
    print(f"SC2 output: {SC2_OUT}")
    print(f"Using Open Ephys folder: {data_path}")

    print("---- Recording ----")
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

    # Optionally truncate for quick tests/debugging.
    recording = first_seconds(recording, TEST_SECONDS)
    original_channel_order = list(recording.channel_ids)
    original_index_map = {}
    for idx, ch in enumerate(original_channel_order):
        original_index_map[ch] = idx
        original_index_map[str(ch)] = idx
    channel_order = original_channel_order.copy()

    print("---- Channel groups ----")
    # Resolve manual group configuration (inline, env, or CLI file).
    manual_groups = CHANNEL_GROUPS
    groups_source = "inline CHANNEL_GROUPS"

    # Priority: CLI > env > config > inline.
    config_loaded = load_channel_groups_from_path(channel_groups_path) if channel_groups_path else None
    if config_loaded:
        manual_groups = config_loaded
        groups_source = f"config CHANNEL_GROUPS_PATH {channel_groups_path}"

    env_loaded = load_channel_groups_from_path(env_groups_path) if env_groups_path else None
    if env_loaded:
        manual_groups = env_loaded
        groups_source = f"env file {env_groups_path}"

    cli_loaded = load_channel_groups_from_path(args.channel_groups) if args.channel_groups else None
    if cli_loaded:
        manual_groups = cli_loaded
        groups_source = f"CLI file {args.channel_groups}"

    base_groups = []
    if manual_groups:
        base_groups = resolve_manual_groups(recording, manual_groups)
        if not base_groups and groups_source != "inline CHANNEL_GROUPS":
            print(f"WARNING: no channel groups resolved from {groups_source}; falling back to inline CHANNEL_GROUPS.")
            manual_groups = CHANNEL_GROUPS
            groups_source = "inline CHANNEL_GROUPS (fallback)"
            base_groups = resolve_manual_groups(recording, manual_groups)
        if not base_groups:
            print("WARNING: no channel groups resolved; falling back to chunked groups.")
            manual_groups = []
    if not base_groups:
        if STRICT_GROUPS:
            raise RuntimeError(
                "No valid channel groups resolved. Provide CHANNEL_GROUPS/--channel-groups "
                "and ensure channel IDs match the recording."
            )
        base_groups = chunk_groups(original_channel_order, CHANNELS_PER_TETRODE)

    print("---- Bad channels ----")
    # Resolve bad channels (inline, env, or CLI file).
    bad_channels = BAD_CHANNELS
    bad_source = "inline BAD_CHANNELS"
    config_bad_loaded = load_bad_channels_from_path(bad_channels_path) if bad_channels_path else None
    if config_bad_loaded is not None:
        bad_channels = config_bad_loaded
        bad_source = f"config BAD_CHANNELS_PATH {bad_channels_path}"
    env_bad_loaded = load_bad_channels_from_path(env_bad_path) if env_bad_path else None
    if env_bad_loaded is not None:
        bad_channels = env_bad_loaded
        bad_source = f"env file {env_bad_path}"
    cli_bad_loaded = load_bad_channels_from_path(args.bad_channels) if args.bad_channels else None
    if cli_bad_loaded is not None:
        bad_channels = cli_bad_loaded
        bad_source = f"CLI file {args.bad_channels}"

    # Combine manual + auto bad channels, then slice recording.
    manual_bad = resolve_bad_channel_ids(recording, bad_channels)
    auto_bad = detect_bad_channel_ids(
        recording,
        AUTO_BAD_CHANNELS,
        method=AUTO_BAD_CHANNELS_METHOD,
        **AUTO_BAD_CHANNELS_KWARGS,
    )
    auto_bad_sorted = np.sort(auto_bad)
    print(f"Bad channels source: {bad_source}")
    if AUTO_BAD_CHANNELS:
        print(f"Auto-detected bad channels ({auto_bad.size}): {auto_bad_sorted.tolist()}")
    else:
        if auto_bad.size:
            print(f"Note: AUTO_BAD_CHANNELS disabled, ignoring detected list: {auto_bad_sorted.tolist()}")

    # Normalize bad channel IDs to the same dtype as the recording channel_ids.
    channel_dtype = np.asarray(channel_order).dtype if channel_order else object
    bad_arrays = [np.array(arr, dtype=channel_dtype) for arr in (manual_bad, auto_bad) if arr.size]
    if bad_arrays:
        bad_ids = np.unique(np.concatenate(bad_arrays))
        keep_ids = [ch for ch in channel_order if ch not in bad_ids]
        recording = safe_channel_slice(recording, keep_ids)
        channel_order = keep_ids
        print(f"Removed {len(bad_ids)} bad channels: {bad_ids.tolist()}")
    else:
        channel_order = original_channel_order.copy()

    # Build groups after bad-channel removal and drop ungrouped channels.
    groups, filtered_indices = filter_groups_with_indices(base_groups, channel_order)
    group_sizes_by_id = {idx: len(group) for idx, group in enumerate(base_groups)}
    # Preserve original group indices for labels after bad-channel removal (non-contiguous IDs are expected).
    group_ids = filtered_indices.copy() if filtered_indices else list(range(len(groups)))
    grouped_ids = {ch for grp in groups for ch in grp}
    if grouped_ids:
        # Enforce that only grouped channels remain (avoids ungrouped channels sitting at [0, 0]).
        keep_ids = [ch for ch in channel_order if ch in grouped_ids]
        if len(keep_ids) != len(channel_order):
            missing = [ch for ch in channel_order if ch not in grouped_ids]
            print(
                f"WARNING: {len(missing)} channels not present in groups; removing from recording "
                f"(first: {missing[:5]})"
            )
            recording = safe_channel_slice(recording, keep_ids)
            channel_order = keep_ids
    for idx, group in enumerate(groups):
        if len(group) <= 2:
            orig_idx = filtered_indices[idx] if idx < len(filtered_indices) else idx
            orig_size = group_sizes_by_id.get(orig_idx, len(group))
            if orig_size >= 4:
                print(
                    f"WARNING: tetrode {orig_idx} has {len(group)} channel(s) after bad-channel removal (degraded). "
                    "Keeping group for sorting."
                )
            else:
                print(
                    f"Note: stereotrode {orig_idx} has {len(group)} channel(s) after bad-channel removal. "
                    "Keeping group for sorting."
                )
    if manual_groups:
        print(f"Groups: {len(groups)} (source: {groups_source}); first group: {groups[0] if groups else 'n/a'}")
    else:
        print(f"Groups: {len(groups)} (fallback chunking); first group: {groups[0] if groups else 'n/a'}")

    # Optional single-grid bundle mode (treat all channels as one group).
    bundle_grid = False
    if (BUNDLE_GROUPING_MODE or "").lower() in ("single_grid", "grid", "single"):
        bundle_grid = True
        groups = [channel_order.copy()]
        group_ids = [0]
        print(f"Bundle grouping mode: single grid ({len(groups[0])} channels).")
    if args.dry_run_config:
        print("Dry-run complete: configuration resolved.")
        print(f"Planned SC2 run folder: {sc2_run}")
        print("Exiting before geometry/preprocessing/sorting/export.")
        return

    tetrode_offsets = None
    base_group_count = len(base_groups)
    tetrodes_per_row = max(1, int(np.ceil(np.sqrt(base_group_count)))) if base_group_count else 1
    if groups and not bundle_grid:
        dx = TETRODE_SPACING_DX_UM
        dy = TETRODE_SPACING_DY_UM
        num_rows = int(np.ceil(base_group_count / tetrodes_per_row))
        tetrode_offsets = []
        for orig_idx in filtered_indices:
            row = orig_idx // tetrodes_per_row
            col = orig_idx % tetrodes_per_row
            x = col * dx
            y = (num_rows - 1 - row) * dy
            tetrode_offsets.append((x, y))

    print("---- Geometry ----")
    # Attach probe geometry for downstream layout-aware steps.
    if ATTACH_GEOMETRY and groups:
        recording = attach_geometry_if_needed(
            recording,
            groups,
            bundle_grid=bundle_grid,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
            scale_to_uv=False,
            label="recording",
        )
        if bundle_grid:
            print(f"Geometry attached to recording (bundle grid {BUNDLE_GRID_COLS} cols).")
        else:
            print(f"Geometry attached to recording (tetrodes_per_row={tetrodes_per_row}).")

    # Optional SI common median reference before SC2 (per group or global) when not using SI preprocessing.
    if not USE_SI_PREPROCESS and SI_APPLY_CAR:
        try:
            mode = (CAR_MODE or "group").lower()
            operator = (CAR_OPERATOR or "median").lower()
            if mode in ("group", "tetrode") and groups:
                # SpikeInterface uses reference="global" with groups to mean per-group CAR.
                recording = spre.common_reference(
                    recording, reference="global", operator=operator, groups=groups
                )
                print(f"Applied SI per-group CAR ({operator}) on {len(groups)} groups.")
            else:
                recording = spre.common_reference(recording, reference="global", operator=operator)
                print(f"Applied SI global CAR ({operator}).")
        except Exception as exc:
            print(f"WARNING: CAR failed (continuing without CAR): {exc}")

    print("---- Preprocessing ----")
    # Prepare the recording that will be passed into SC2.
    rec_sc2 = preprocess_for_sc2(recording, groups=groups)

    if ATTACH_GEOMETRY and groups:
        rec_sc2 = attach_geometry_if_needed(
            rec_sc2,
            groups,
            bundle_grid=bundle_grid,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
            scale_to_uv=False,
            label="rec_sc2",
        )
    if ATTACH_GEOMETRY and groups:
        try:
            # Note: SC2 can still warn about missing probes in internal snippet wrappers;
            # this keeps the sorter input probe-attached, but some temporary internals may
            # still emit non-fatal dummy-probe warnings.
            rec_sc2 = ensure_probe_attached(rec_sc2)
        except Exception as exc:
            print(f"WARNING: failed to attach probe to rec_sc2: {exc}")
    if MATERIALIZE_SI_PREPROCESS:
        try:
            preproc_folder = SC2_OUT / "rec_sc2_preprocessed"
            rec_sc2 = rec_sc2.save(folder=preproc_folder, format="binary_folder", overwrite=True)
            print(f"Materialized rec_sc2 at {preproc_folder}")
            if ATTACH_GEOMETRY and groups:
                rec_sc2 = attach_geometry_if_needed(
                    rec_sc2,
                    groups,
                    bundle_grid=bundle_grid,
                    tetrodes_per_row=tetrodes_per_row,
                    tetrode_offsets=tetrode_offsets,
                    scale_to_uv=False,
                    label="rec_sc2 (materialized)",
                )
        except Exception as exc:
            print(f"WARNING: failed to materialize rec_sc2: {exc}")

    print("---- Export/Analyzer prep ----")
    # Prepare export/analyzer recordings (do not affect sorter input).
    # Optionally use a scaled, bandpassed copy for Phy/export so snippets are not raw.
    # SC2/SI internal snippet wrappers can still emit dummy-probe warnings; treat as non-fatal
    # when channel mapping and exported geometry are confirmed correct.
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
        try:
            rec_export = attach_geometry_if_needed(
                rec_export,
                groups,
                bundle_grid=bundle_grid,
                tetrodes_per_row=tetrodes_per_row,
                tetrode_offsets=tetrode_offsets,
                scale_to_uv=False,
                label="rec_export",
            )
        except Exception as exc:
            print(f"WARNING: failed to reattach geometry on export path: {exc}")
    if MATERIALIZE_EXPORT:
        try:
            export_folder = SC2_OUT / "rec_export_prepared"
            rec_export = rec_export.save(folder=export_folder, format="binary_folder", overwrite=True)
            print(f"Materialized rec_export at {export_folder}")
            if ATTACH_GEOMETRY and groups:
                rec_export = attach_geometry_if_needed(
                    rec_export,
                    groups,
                    bundle_grid=bundle_grid,
                    tetrodes_per_row=tetrodes_per_row,
                    tetrode_offsets=tetrode_offsets,
                    scale_to_uv=False,
                    label="rec_export (materialized)",
                )
        except Exception as exc:
            print(f"WARNING: failed to materialize rec_export: {exc}")
    set_group_property(rec_export, groups, group_ids)

    if ANALYZER_FROM_SORTER:
        rec_analyzer = rec_sc2
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
                    rec_analyzer = spre.scale(
                        rec_analyzer,
                        gain=gains_arr,
                        offset=offsets_arr,
                        dtype="float32",
                    )
                    print("Analyzer path: scaled rec_sc2 to microvolts for QC.")
        if ATTACH_GEOMETRY and groups:
            try:
                rec_analyzer = attach_geometry_if_needed(
                    rec_analyzer,
                    groups,
                    bundle_grid=bundle_grid,
                    tetrodes_per_row=tetrodes_per_row,
                    tetrode_offsets=tetrode_offsets,
                    scale_to_uv=False,
                    label="rec_analyzer",
                )
            except Exception as exc:
                print(f"WARNING: failed to reattach geometry on analyzer path: {exc}")
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
                print(f"WARNING: failed to materialize rec_analyzer: {exc}")
    else:
        rec_analyzer = rec_export

    print("---- Sorting ----")
    # Run SC2 on the mixed bundle (single run).
    # Uses the pre-reserved run folder created at startup.

    set_group_property(rec_sc2, groups, group_ids)
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
    try:
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
            print(f"WARNING: failed to remove partial output folder: {exc2}")
        raise

    print("---- Postprocessing/Export ----")
    # Build analyzer, QC, and exports from sorter output.
    wf_ms_before = sc2_params.get("general", {}).get("ms_before")
    wf_ms_after = sc2_params.get("general", {}).get("ms_after")
    if wf_ms_before is not None or wf_ms_after is not None:
        print(
            "Analyzer waveform window:",
            f"{wf_ms_before} ms before / {wf_ms_after} ms after",
        )
    analyzer_sc2 = build_analyzer(
        sorting_sc2,
        rec_analyzer,
        SC2_OUT,
        "sc2",
        wf_ms_before=wf_ms_before,
        wf_ms_after=wf_ms_after,
    )
    analyzer_sc2 = maybe_remove_redundant_units(analyzer_sc2, "sc2")

    phy_folder = None
    si_gui_folder = None
    group_vals = None
    if EXPORT_TO_PHY:
        phy_folder, group_vals = export_for_phy(
            analyzer_sc2,
            SC2_OUT,
            "sc2",
            groups,
            original_index_map,
            group_ids,
            group_sizes_by_id,
            recording_override=rec_export,
        )
    if EXPORT_TO_SI_GUI:
        si_gui_folder = export_for_si_gui(analyzer_sc2, SI_GUI_OUT, "sc2")

    print("---- Summary ----")
    print("SC2 pipeline complete.")
    try:
        n_units = len(sorting_sc2.get_unit_ids())
    except Exception:
        n_units = "unknown"
    qc_metrics_path = getattr(analyzer_sc2, "_qc_metrics_path", None)
    if qc_metrics_path is None and getattr(analyzer_sc2, "folder", None):
        qc_metrics_path = str(Path(analyzer_sc2.folder) / "qc_metrics.csv")
    print("Summary:", f"units={n_units},", f"qc_metrics={qc_metrics_path or 'n/a'}")
    if phy_folder:
        print(f"Run: phy template-gui \"{phy_folder / 'params.py'}\"")
        params_py = phy_folder / "params.py"
        if EXPORT_PHY_EXTRACT_WAVEFORMS:
            try:
                subprocess.run(["phy", "extract-waveforms", str(params_py)], check=True)
                print("Phy: extracted waveforms for export.")
            except FileNotFoundError:
                print("WARNING: 'phy' command not found; skipping extract-waveforms.")
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: phy extract-waveforms failed: {exc}")
    if si_gui_folder:
        print(f"Run: python -m spikeinterface_gui \"{si_gui_folder}\"")

if __name__ == "__main__":
    main()
