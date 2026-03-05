"""
Minimal SpikeInterface pipeline to run SpykingCircus2 on tetrode recordings.
"""
# User guide: SpikeSorting/Pipelines/README_sc2_tet_pipeline.md

from __future__ import annotations

import argparse
import atexit
import builtins
import json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (Pipelines/..)
sys.path.insert(0, str(PROJECT_ROOT))

from Functions.analyzer_utils import (
    build_analyzer as build_analyzer_shared,
    maybe_remove_redundant_units as maybe_remove_redundant_units_shared,
)
from Functions.config_utils import (
    apply_env_overrides_from_env,
    print_pipeline_config_echo,
)
from Functions.phy_export import export_for_phy as export_for_phy_shared
from Functions.si_utils import (
    attach_oe_gain_to_uv_from_oebin,
    attach_oe_index_from_oebin,
    attach_geometry_if_needed as attach_geometry_if_needed_shared,
    ensure_probe_attached,
    maybe_scale_recording_to_uv as maybe_scale_recording_to_uv_shared,
    set_group_property,
)
from Functions.channel_utils import (
    build_oe_index_map,
    chunk_groups,
    detect_bad_channel_ids,
    load_bad_channels_from_path,
    load_channel_groups_from_path,
    resolve_bad_channel_ids,
    resolve_manual_groups,
    safe_channel_slice,
)
from Functions.fs_utils import safe_rmtree
from Functions.params_utils import merge_params, warn_unknown_sc2_overrides
from Functions.run_utils import (
    disable_run_logging,
    export_for_si_gui,
    initialize_run_io as initialize_run_io_shared,
    log_info,
    log_warn,
    reserve_run_folder,
)
from Functions.session_utils import (
    choose_recording_folder,
    first_seconds,
    pick_stream,
    resolve_config_sources as resolve_config_sources_shared,
)


def _pipeline_print(*args, **kwargs):
    """Route module print output through run-utils logging helpers."""
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", None)
    flush = kwargs.get("flush", False)

    # Preserve native print behavior for explicit formatting/flush semantics.
    if flush or sep != " " or end != "\n":
        builtins.print(*args, **kwargs)
        return

    # Preserve native behavior for non-standard output streams.
    if file not in (None, sys.stdout, sys.stderr):
        builtins.print(*args, **kwargs)
        return

    msg = sep.join(str(a) for a in args).rstrip("\n")
    msg_l = msg.lower()
    warning_like = (
        file is sys.stderr
        or msg_l.startswith("warning:")
        or msg_l.startswith("error:")
        or msg_l.startswith("traceback")
        or msg_l.startswith("userwarning")
        or ": userwarning:" in msg_l
        or "traceback" in msg_l
    )
    if warning_like:
        if msg.startswith("WARNING:"):
            msg = msg[len("WARNING:"):].lstrip()
        log_warn(msg)
    else:
        log_info(msg)

print = _pipeline_print  # type: ignore[assignment]

# ---------------------------------------------------------------------
# User Configuration
# ---------------------------------------------------------------------
# Flow: load -> group/bad-channels -> geometry -> preprocess -> sort -> analyze/QC -> export.

# Session and I/O
TEST_SECONDS = None  # None=full recording (use 300-600 for quick QC runs)
DEFAULT_ROOT_DIR = PROJECT_ROOT / "recordings"  # recordings root
SESSION_SUBPATH = None  # None=discover from root; set relative Open Ephys path for fixed runs
STREAM_NAME = None  # None=auto-pick first neural stream; else exact stream name
USE_CONFIG_JSONS = False  # prompt for config/*.json groups + bad-channels

DEFAULT_BASE_OUT = PROJECT_ROOT  # base output folder for sc2_outputs / si_gui_exports
SC2_OUT = None  # initialized in main()
SI_GUI_OUT = None  # initialized in main()

# Sorting mode and channel groups
SORT_BY_GROUP = False  # True=run SC2 separately per tetrode group; False=single run
STRICT_GROUPS = True  # True=error if no valid groups; False=chunk by order
CHANNELS_PER_TETRODE = 4  # fallback chunk size when CHANNEL_GROUPS is empty
# Policy: explicit groups are preferred over chunking to avoid shifts after bad-channel removal.
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
CHANNEL_GROUPS_PATH = None  # optional JSON path or env SPIKESORT_CHANNEL_GROUPS
# Group semantics are unchanged in both modes; only execution plan differs.

# Bad-channel policy
# Use OE labels (CH##) to avoid positional mismatch across sessions/slices.
BAD_CHANNELS = ["CH7", "CH2", "CH26", "CH4", "CH42", "CH50", "CH51", "CH52", "CH54", "CH56", "CH6", "CH8"]  # Example preset: Tatsu
#BAD_CHANNELS = ["CH58", "CH64", "CH62", "CH60", "CH63", "CH61", "CH59", "CH57", "CH47", "CH45", "CH43", "CH41"]  # Example preset: Fuyu
BAD_CHANNELS_PATH = None  # optional JSON path or env SPIKESORT_BAD_CHANNELS
AUTO_BAD_CHANNELS = False  # True=auto-detect bad channels (merged with manual list); False=skip
AUTO_BAD_CHANNELS_METHOD = "std"  # "std" or "mad" (passed to SI detect_bad_channels)
AUTO_BAD_CHANNELS_KWARGS = {}  # only used when AUTO_BAD_CHANNELS=True; extra args passed to detect_bad_channels ({}=defaults)

# Preprocessing toggles:
# - USE_SI_PREPROCESS controls SI preprocessing before sorter input.
# - USE_SC2_PREPROCESS controls SC2 internal preprocessing in sorter params.
# - RAW_MODE enforces strict raw input by force-disabling both preprocess toggles, notch, CAR, and SI whitening.
USE_SI_PREPROCESS = True  # True=apply SI preprocessing before SC2; False=skip SI preprocessing
USE_SC2_PREPROCESS = False  # True=let SC2 run its own preprocessing; False=disable SC2 preprocessing
RAW_MODE = False  # True=force strict raw policy

SI_BP_MIN_HZ = 300  # SI bandpass low cut (Hz)
SI_BP_MAX_HZ = 6000  # SI bandpass high cut (Hz)
SI_BP_FTYPE = "bessel"  # usually "bessel" or "butter"
SI_BP_ORDER = 2  # SI bandpass filter order
SI_BP_MARGIN_MS = 10  # SI bandpass margin (ms)
# Notch is applied after SI bandpass (SI path) or before sorter preprocessing (SC2 path).
APPLY_NOTCH = False  # True=enable notch filter; False=skip
NOTCH_FREQUENCIES = [50, 100, 150]  # notch frequencies (Hz)
NOTCH_Q = 30  # notch filter Q

# SI CAR policy: global is safer by default for tetrodes; per-tetrode can attenuate spikes.
SI_APPLY_CAR = False  # True=enable common average reference; False=skip
CAR_MODE = "global"  # "global" or "tetrode"
CAR_OPERATOR = "median"  # "median" (robust) or "average"

SI_APPLY_WHITEN = False  # True=apply SI whitening when USE_SI_PREPROCESS=True (double-whitening); False=skip
SI_WHITEN_MODE = "local"  # "global" or "local"
SI_WHITEN_RADIUS_UM = 100.0  # used when SI_WHITEN_MODE == "local"
MATERIALIZE_SI_PREPROCESS = False  # True=save preprocessed rec_sc2 to disk (faster reuse; uses extra space); False=in-memory

# Geometry and traceview layout
ATTACH_GEOMETRY = True  # Attach probe geometry for channel locations (plots, local whitening); does not change bandpass/CAR
# Internal SI/SC2 wrappers may still warn about missing probe on temporary snippets.
LINEARIZE_TRACEVIEW = True  # Traceview only (not Phy or preprocessing): True=flatten groups; False=keep group layout
TRACEVIEW_CONTACT_SPACING_UM = 20.0  # traceview-only spacing between contacts
TRACEVIEW_GROUP_SPACING_UM = 200.0  # traceview-only spacing between groups
TETRODE_PITCH_UM = 20.0  # synthetic within-tetrode spacing (2x2)
TETRODE_SPACING_DX_UM = 300.0  # synthetic spacing between tetrodes (x)
TETRODE_SPACING_DY_UM = 300.0  # synthetic spacing between tetrodes (y)

# Analyzer and QC
SAVE_ANALYZER = True  # True=persist analyzer to disk (binary_folder); False=skip
COMPUTE_QC_METRICS = True  # PC-based metrics add runtime; disable for quick smoke runs
QC_METRIC_NAMES = [  # no-PC metrics
    "firing_rate",
    "presence_ratio",
    "isi_violation",
    "snr",
    "amplitude_cutoff",
]
QC_PC_METRICS = {  # requires PCs
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    "nn_isolation",
    "nn_noise_overlap",
    "silhouette",
}
REMOVE_REDUNDANT_UNITS = False  # True=remove near-duplicate units post-sort; False=skip
REDUNDANT_THRESHOLD = 0.95  # higher = stricter duplicate detection
REDUNDANT_STRATEGY = "minimum_shift"  # SI remove_redundant_units remove_strategy

# Export targets and Phy export behavior
EXPORT_TO_PHY = True  # True=write Phy export after sorting; False=skip
EXPORT_TO_SI_GUI = False  # True=write SpikeInterface GUI export; False=skip
EXPORT_PHY_EXTRACT_WAVEFORMS = False  # runs `phy extract-waveforms` after export (requires Phy CLI)
SIMPLE_PHY_EXPORT = None  # None=auto(single group only) | True=force simple | False=force full mapping rewrite
EXPORT_PHY_CHANNEL_IDS_MODE = "oe_label"  # "oe_label"(recommended) | "oe_index" | "as_exported"
# Policy: keep channel_ids mapping explicit so Phy labels remain traceable to OE channels.
MATERIALIZE_EXPORT = False  # True=save rec_export to disk before analyzer/export (faster reuse; uses extra space); False=in-memory

# Phy export policy:
# - keep uV scaling for amplitude sanity in curation
# - Phy export uses analyzer.recording (no extra export-time bandpass branch)
# - hp_filtered metadata controls filtered/unfiltered interpretation in Phy
EXPORT_SCALE_TO_UV = True  # True=scale export recording to microvolts if gain info is present; False=leave native units
ANALYZER_FROM_SORTER = True  # True=use rec_sc2 for analyzer/QC + Phy export context; False=use rec_export

# SC2 parameter overrides (keys mirror sorter default params).
SC2_PARAM_OVERRIDES = {  # {}=use SC2 defaults
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

# SpikeAgent Override Bridge
# Used by wrapper launches that pass runtime overrides via env JSON.
PIPELINE_OVERRIDES_ENV = "SPIKESORT_PIPELINE_OVERRIDES"
# Allow-list of globals that can be overridden by SPIKESORT_PIPELINE_OVERRIDES.
ALLOWED_PIPELINE_OVERRIDES = {
    "TEST_SECONDS",
    "SESSION_SUBPATH",
    "STREAM_NAME",
    "USE_CONFIG_JSONS",
    "CHANNEL_GROUPS_PATH",
    "BAD_CHANNELS_PATH",
    "SORT_BY_GROUP",
    "AUTO_BAD_CHANNELS",
    "SI_APPLY_CAR",
    "CAR_MODE",
    "SI_APPLY_WHITEN",
    "USE_SI_PREPROCESS",
    "USE_SC2_PREPROCESS",
    "RAW_MODE",
    "EXPORT_TO_PHY",
    "EXPORT_TO_SI_GUI",
    "SAVE_ANALYZER",
    "EXPORT_PHY_CHANNEL_IDS_MODE",
}

# Debug toggles.
DEBUG_WARN_TRACE = False  # print stack traces for warnings (debug)
DEBUG_GEOMETRY_ATTACH = False  # log each geometry attach/reattach (debug)

EFFECTIVE_CONFIG_KEYS = (
    "TEST_SECONDS",
    "DEFAULT_ROOT_DIR",
    "SESSION_SUBPATH",
    "STREAM_NAME",
    "USE_CONFIG_JSONS",
    "DEFAULT_BASE_OUT",
    "SORT_BY_GROUP",
    "STRICT_GROUPS",
    "CHANNELS_PER_TETRODE",
    "CHANNEL_GROUPS",
    "CHANNEL_GROUPS_PATH",
    "BAD_CHANNELS",
    "BAD_CHANNELS_PATH",
    "AUTO_BAD_CHANNELS",
    "AUTO_BAD_CHANNELS_METHOD",
    "AUTO_BAD_CHANNELS_KWARGS",
    "USE_SI_PREPROCESS",
    "USE_SC2_PREPROCESS",
    "RAW_MODE",
    "SI_BP_MIN_HZ",
    "SI_BP_MAX_HZ",
    "SI_BP_FTYPE",
    "SI_BP_ORDER",
    "SI_BP_MARGIN_MS",
    "APPLY_NOTCH",
    "NOTCH_FREQUENCIES",
    "NOTCH_Q",
    "SI_APPLY_CAR",
    "CAR_MODE",
    "CAR_OPERATOR",
    "SI_APPLY_WHITEN",
    "SI_WHITEN_MODE",
    "SI_WHITEN_RADIUS_UM",
    "MATERIALIZE_SI_PREPROCESS",
    "ATTACH_GEOMETRY",
    "LINEARIZE_TRACEVIEW",
    "TRACEVIEW_CONTACT_SPACING_UM",
    "TRACEVIEW_GROUP_SPACING_UM",
    "TETRODE_PITCH_UM",
    "TETRODE_SPACING_DX_UM",
    "TETRODE_SPACING_DY_UM",
    "SAVE_ANALYZER",
    "COMPUTE_QC_METRICS",
    "QC_METRIC_NAMES",
    "QC_PC_METRICS",
    "REMOVE_REDUNDANT_UNITS",
    "REDUNDANT_THRESHOLD",
    "REDUNDANT_STRATEGY",
    "EXPORT_TO_PHY",
    "EXPORT_TO_SI_GUI",
    "EXPORT_PHY_EXTRACT_WAVEFORMS",
    "SIMPLE_PHY_EXPORT",
    "EXPORT_PHY_CHANNEL_IDS_MODE",
    "MATERIALIZE_EXPORT",
    "EXPORT_SCALE_TO_UV",
    "ANALYZER_FROM_SORTER",
    "SC2_PARAM_OVERRIDES",
    "DEBUG_WARN_TRACE",
    "DEBUG_GEOMETRY_ATTACH",
)


# ---------------------------------------------------------------------
# Pipeline Helpers (Local + Functions Utilities)
# ---------------------------------------------------------------------
# Per-function titles live in docstrings; headings below group related helpers.

# Environment Override Helper
def apply_env_pipeline_overrides():
    """Apply allow-listed env JSON overrides with strict type checks."""
    return apply_env_overrides_from_env(
        env_name=PIPELINE_OVERRIDES_ENV,
        allowed_keys=ALLOWED_PIPELINE_OVERRIDES,
        module_globals=globals(),
        bool_keys={
            "USE_CONFIG_JSONS",
            "SORT_BY_GROUP",
            "AUTO_BAD_CHANNELS",
            "SI_APPLY_CAR",
            "SI_APPLY_WHITEN",
            "USE_SI_PREPROCESS",
            "USE_SC2_PREPROCESS",
            "RAW_MODE",
            "EXPORT_TO_PHY",
            "EXPORT_TO_SI_GUI",
            "SAVE_ANALYZER",
        },
        str_or_none_keys={
            "SESSION_SUBPATH",
            "STREAM_NAME",
            "CHANNEL_GROUPS_PATH",
            "BAD_CHANNELS_PATH",
        },
        enum_keys={
            "CAR_MODE": {"global", "tetrode"},
            "EXPORT_PHY_CHANNEL_IDS_MODE": {"oe_label", "oe_index", "as_exported"},
        },
        positive_number_or_none_keys={"TEST_SECONDS"},
    )


def validate_preprocess_toggles() -> None:
    """Sanity-check preprocessing toggle combination and emit warnings."""
    if RAW_MODE:
        return
    if not USE_SI_PREPROCESS and SI_APPLY_CAR:
        log_warn(
            "USE_SI_PREPROCESS=False while SI_APPLY_CAR=True: SI CAR will still run "
            "in the no-SI path before sorting."
        )
    if not USE_SI_PREPROCESS and not USE_SC2_PREPROCESS:
        log_warn(
            "Both USE_SI_PREPROCESS and USE_SC2_PREPROCESS are False. "
            "Running without SI/SC2 preprocessing toggles "
            "(optional no-SI CAR/notch settings may still apply)."
        )
    if USE_SC2_PREPROCESS and not USE_SI_PREPROCESS and ANALYZER_FROM_SORTER:
        log_warn(
            "USE_SC2_PREPROCESS=True with USE_SI_PREPROCESS=False: SC2 may filter internally "
            "for sorting, but Phy hp_filtered tracks the exported analyzer recording "
            "(so hp_filtered remains False in this configuration)."
        )


def apply_raw_mode_policy() -> None:
    """Force a strict raw-input policy when RAW_MODE is enabled."""
    global USE_SI_PREPROCESS, USE_SC2_PREPROCESS, APPLY_NOTCH, SI_APPLY_CAR, SI_APPLY_WHITEN
    if not RAW_MODE:
        return

    forced = []

    def _force_false(name: str) -> None:
        module_globals = globals()
        old_val = bool(module_globals[name])
        if old_val:
            forced.append(name)
        module_globals[name] = False

    _force_false("USE_SI_PREPROCESS")
    _force_false("USE_SC2_PREPROCESS")
    _force_false("APPLY_NOTCH")
    _force_false("SI_APPLY_CAR")
    _force_false("SI_APPLY_WHITEN")

    if forced:
        log_info(
            "RAW_MODE enabled: forced False for "
            + ", ".join(sorted(forced))
        )
    else:
        log_info("RAW_MODE enabled: preprocess/CAR/notch toggles were already disabled.")


# Debug Helpers (Warning Trace, Geometry Logging)
def enable_warning_trace(limit: int = 12) -> None:
    """Install a warning hook that prints a short traceback (debug-only)."""
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        traceback.print_stack(limit=limit)
        print(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = _showwarning
    warnings.simplefilter("always")


def log_section(title: str) -> None:
    """Print a compact section header in run logs."""
    log_info("")
    log_info(f"=== {title} ===")


def apply_configured_si_car(recording, groups=None, *, context_label: str, print_disabled: bool = False):
    """Apply SI CAR according to CAR_MODE/CAR_OPERATOR config."""
    groups = groups or []
    if not SI_APPLY_CAR:
        if print_disabled:
            print(f"{context_label}: CAR disabled.")
        return recording
    try:
        mode = (CAR_MODE or "tetrode").lower()
        operator = (CAR_OPERATOR or "median").lower()
        if mode == "tetrode" and groups:
            # SpikeInterface uses reference="global" with groups to mean per-group CAR.
            print("WARNING: per-tetrode CAR can attenuate spikes; use only if verified on your data.")
            recording = spre.common_reference(
                recording,
                reference="global",
                operator=operator,
                groups=groups,
            )
            print(f"{context_label}: applied per-tetrode CAR ({operator}) on {len(groups)} tetrodes.")
        else:
            recording = spre.common_reference(recording, reference="global", operator=operator)
            print(f"{context_label}: applied global CAR ({operator}).")
    except Exception as exc:
        print(f"WARNING: CAR failed (continuing without CAR): {exc}")
    return recording

# Core Processing Helpers (Grouping, Geometry, Preprocessing, Analyzer, Export Metadata)
def validate_or_reattach_geometry(
    recording,
    groups,
    *,
    tetrodes_per_row: int,
    tetrode_offsets,
    label: str,
    require_probe: bool = False,
):
    """Validate geometry metadata and reattach only when missing/invalid."""
    if not ATTACH_GEOMETRY or not groups:
        return recording

    needs_reattach = False
    try:
        locs = np.asarray(recording.get_channel_locations(), dtype="float64")
        if (
            locs.ndim != 2
            or locs.shape[0] != recording.get_num_channels()
            or not np.all(np.isfinite(locs))
        ):
            needs_reattach = True
    except Exception:
        needs_reattach = True

    out = recording
    if needs_reattach:
        log_warn(f"{label}: channel locations missing/invalid; reattaching geometry.")
        out = attach_geometry_if_needed_shared(
            recording,
            groups,
            attach_geometry=ATTACH_GEOMETRY,
            tetrode_pitch_um=TETRODE_PITCH_UM,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
            scale_to_uv=False,
            debug=DEBUG_GEOMETRY_ATTACH,
            label=label,
        )

    if require_probe:
        try:
            out = ensure_probe_attached(out)
        except Exception as exc:
            raise RuntimeError(f"{label}: failed to attach probe ({exc})") from exc
    return out


def coerce_dummy_probe_from_locations_strict(rec, label: str):
    """Force dummy-probe construction from channel locations, failing loudly on errors."""
    try:
        locs = np.asarray(rec.get_channel_locations(), dtype="float64")
        if (
            locs.ndim != 2
            or locs.shape[0] != rec.get_num_channels()
            or not np.all(np.isfinite(locs))
        ):
            raise RuntimeError(
                f"invalid channel locations for dummy probe "
                f"(shape={getattr(locs, 'shape', None)})"
            )
        # SI API compatibility:
        # - some versions require explicit `locations`
        # - some accept keyword form
        # - some return the recording, others mutate in place and return None
        def _normalize_return(obj):
            return rec if obj is None else obj

        try:
            return _normalize_return(rec.set_dummy_probe_from_locations(locs))
        except TypeError:
            try:
                return _normalize_return(rec.set_dummy_probe_from_locations(locations=locs))
            except TypeError:
                return _normalize_return(rec.set_dummy_probe_from_locations())
    except Exception as exc:
        raise RuntimeError(
            f"{label}: failed to coerce dummy probe from channel locations ({exc})"
        ) from exc


def preprocess_for_sc2(recording, groups=None):
    """Apply SI-side preprocessing steps before sorter execution.

    Args:
        recording: SpikeInterface recording object.
        groups: Optional list of channel groups for CAR.

    Returns:
        Preprocessed recording object.

    Examples:
        >>> # With SI preprocessing
        >>> rec_prep = preprocess_for_sc2(recording, groups=[["CH1","CH2"], ["CH3","CH4"]])

        >>> # Without SI preprocessing (SC2 path controlled by USE_SC2_PREPROCESS)
        >>> rec_prep = preprocess_for_sc2(recording)
    """
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

        rec = apply_configured_si_car(
            rec,
            groups,
            context_label="Preprocessing (SI)",
            print_disabled=True,
        )

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

    # Non-SI path: optional notch only; SC2 preprocessing is controlled by USE_SC2_PREPROCESS.
    if APPLY_NOTCH and NOTCH_FREQUENCIES:
        rec = recording
        for freq in NOTCH_FREQUENCIES:
            rec = spre.notch_filter(rec, freq=freq, q=NOTCH_Q)
        print(f"Preprocessing: applied notch filters at {NOTCH_FREQUENCIES} Hz (Q={NOTCH_Q}).")
        return rec

    if USE_SC2_PREPROCESS:
        print("Passing recording to SC2: SC2 preprocessing is enabled.")
    else:
        print("Passing recording to SC2: SC2 preprocessing is disabled (raw/no-SI path).")
    return rec


# Analyzer Probe Helpers
def _attach_probe_from_group_layout(recording, group_prop, pitch=20.0, dx=150.0, dy=150.0):
    """Create a synthetic probe layout from group IDs when geometry is missing."""
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
    """Best-effort probe attachment for analyzer/QC (no failure)."""
    try:
        if recording.get_probe() is not None:
            return recording
    except Exception as exc:
        log_warn(f"probe check failed before analyzer; continuing with fallback attach path ({exc})")
    try:
        recording = ensure_probe_attached(recording)
        if recording.get_probe() is not None:
            return recording
    except Exception as exc:
        log_warn(f"ensure_probe_attached failed for analyzer; trying group-layout fallback ({exc})")
    try:
        group_prop = recording.get_property("group")
    except Exception as exc:
        log_warn(f"could not read 'group' property for analyzer probe fallback ({exc})")
        group_prop = None
    if group_prop is not None:
        try:
            return _attach_probe_from_group_layout(recording, group_prop)
        except Exception as exc:
            log_warn(f"group-layout probe fallback failed for analyzer ({exc})")
    return recording


# Shared Analyzer/Export Adapters
def build_analyzer(
    sorting,
    recording,
    base_folder: Path,
    label: str,
    wf_ms_before: float | None = None,
    wf_ms_after: float | None = None,
):
    """Create analyzer/QC via shared helper with tet pipeline settings."""
    recording = ensure_probe_for_analyzer(recording)
    return build_analyzer_shared(
        sorting=sorting,
        recording=recording,
        base_folder=base_folder,
        label=label,
        wf_ms_before=wf_ms_before,
        wf_ms_after=wf_ms_after,
        save_analyzer=SAVE_ANALYZER,
        compute_qc_metrics=COMPUTE_QC_METRICS,
        qc_metric_names=QC_METRIC_NAMES,
        qc_pc_metrics=QC_PC_METRICS,
        # Keep runtime conservative/stable for current workflow.
        n_jobs=1,
    )


def maybe_remove_redundant_units(analyzer, label: str):
    """Optional post-sort curation via shared helper."""
    return maybe_remove_redundant_units_shared(
        analyzer,
        label,
        remove_redundant=REMOVE_REDUNDANT_UNITS,
        duplicate_threshold=REDUNDANT_THRESHOLD,
        remove_strategy=REDUNDANT_STRATEGY,
    )


def export_for_phy(
    analyzer,
    base_folder: Path,
    label: str,
    groups,
    original_index_map: dict,
    oe_index_map: dict,
    group_ids=None,
):
    """Export Phy via shared helper with tet pipeline settings."""
    # Set hp_filtered metadata to match whether the exported data is already
    # high/band-pass filtered.
    if ANALYZER_FROM_SORTER:
        # Analyzer uses rec_sc2; hp_filtered tracks exported recording filtering from SI path.
        # This intentionally does not infer SC2 internal filtering when apply_preprocessing=True.
        phy_hp_filtered = bool(USE_SI_PREPROCESS)
    else:
        # Analyzer uses rec_export; no extra export-time bandpass is applied in this pipeline.
        phy_hp_filtered = False
    return export_for_phy_shared(
        analyzer=analyzer,
        base_folder=base_folder,
        label=label,
        groups=groups,
        original_index_map=original_index_map,
        oe_index_map=oe_index_map,
        group_ids=group_ids,
        simple_phy_export=SIMPLE_PHY_EXPORT,
        export_channel_ids_mode=EXPORT_PHY_CHANNEL_IDS_MODE,
        linearize_traceview=LINEARIZE_TRACEVIEW,
        traceview_contact_spacing_um=TRACEVIEW_CONTACT_SPACING_UM,
        traceview_group_spacing_um=TRACEVIEW_GROUP_SPACING_UM,
        scaled_to_uv=EXPORT_SCALE_TO_UV,
        phy_hp_filtered=phy_hp_filtered,
    )


def _to_jsonable(value):
    """Recursively convert values to JSON-serializable types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def collect_effective_settings() -> dict:
    """Capture resolved top-level settings after env overrides/policy enforcement."""
    module_globals = globals()
    return {key: _to_jsonable(module_globals.get(key)) for key in EFFECTIVE_CONFIG_KEYS}


def write_effective_config(
    *,
    sc2_run: Path,
    args,
    use_config_jsons: bool,
    root_dir: Path,
    selected_session_path: Path,
    selected_stream: str,
    applied_overrides: dict,
    ignored_overrides: list[str],
    groups,
    group_ids,
    grouping_details: dict,
    sc2_params: dict,
) -> None:
    """Persist resolved runtime config for reproducibility/debugging."""
    payload = {
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "script_path": str(Path(__file__).resolve()),
        "run_folder": str(sc2_run),
        "cli_args": {
            "root_dir": str(args.root_dir),
            "base_out": str(args.base_out),
            "channel_groups": str(args.channel_groups) if args.channel_groups else None,
            "bad_channels": str(args.bad_channels) if args.bad_channels else None,
            "no_config_json": bool(args.no_config_json),
            "dry_run_config": bool(args.dry_run_config),
        },
        "effective_settings": collect_effective_settings(),
        "selection": {
            "recording_root": str(root_dir),
            "session_path": str(selected_session_path),
            "stream_name": selected_stream,
            "use_config_jsons": bool(use_config_jsons),
        },
        "pipeline_overrides": {
            "applied": _to_jsonable(applied_overrides),
            "ignored": _to_jsonable(sorted(ignored_overrides)),
        },
        "grouping": {
            "group_ids": _to_jsonable(list(group_ids) if group_ids is not None else []),
            "groups": _to_jsonable(groups),
            "details": _to_jsonable(grouping_details),
        },
        "sc2_params": _to_jsonable(sc2_params),
    }

    out_path = sc2_run / "effective_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log_info(f"Effective config saved: {out_path}")


# Stage Orchestration Helpers (Called by Main in Execution Order)
def load_recording_with_indices(root_dir: Path):
    """Load session recording and build channel index maps."""
    log_section("Session")
    # Prompt-only mode: SESSION_SUBPATH takes precedence; otherwise prompt user.
    effective_selection = None if SESSION_SUBPATH else "prompt"
    data_path = choose_recording_folder(root_dir, SESSION_SUBPATH, effective_selection, 0)
    print(f"Recording Root: {root_dir}")
    print(f"SC2 Output: {SC2_OUT}")
    print(f"Using Open Ephys Folder: {data_path}")

    log_section("Recording")
    stream = pick_stream(data_path, STREAM_NAME)
    print(f"Using Stream: {stream}")

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

    # Attach OE stream-list indices from structure.oebin if extractor did not expose them.
    attach_oe_index_from_oebin(recording, data_path, stream)
    # Attach per-channel gain_to_uV from structure.oebin when bit_volts metadata is present.
    attach_oe_gain_to_uv_from_oebin(recording, data_path, stream)

    recording = first_seconds(recording, TEST_SECONDS)
    original_channel_order = list(recording.channel_ids)
    original_index_map = {}
    for idx, ch in enumerate(original_channel_order):
        original_index_map[ch] = idx
        original_index_map[str(ch)] = idx
    oe_index_map = build_oe_index_map(recording, original_index_map)
    channel_order = original_channel_order.copy()
    return recording, original_channel_order, original_index_map, oe_index_map, channel_order, data_path, stream


def resolve_groups_and_bad_channels(
    recording,
    channel_order,
    original_channel_order,
    original_index_map,
    args,
    channel_groups_path,
    bad_channels_path,
    env_groups_path,
    env_bad_path,
    sc2_run: Path,
):
    """Resolve groups/bad channels, filter recording, and return grouping state."""
    log_section("Channel Groups")
    manual_groups = CHANNEL_GROUPS
    groups_source = "inline CHANNEL_GROUPS"

    # Priority: CLI > env > config > inline.
    config_loaded = load_channel_groups_from_path(channel_groups_path) if channel_groups_path else None
    if config_loaded is not None:
        manual_groups = config_loaded
        groups_source = f"config CHANNEL_GROUPS_PATH {channel_groups_path}"

    env_loaded = load_channel_groups_from_path(env_groups_path) if env_groups_path else None
    if env_loaded is not None:
        manual_groups = env_loaded
        groups_source = f"env file {env_groups_path}"

    cli_loaded = load_channel_groups_from_path(args.channel_groups) if args.channel_groups else None
    if cli_loaded is not None:
        manual_groups = cli_loaded
        groups_source = f"CLI file {args.channel_groups}"

    if manual_groups:
        log_info(f"Channel Groups Source: {groups_source} | configured groups={len(manual_groups)}")
    else:
        log_info("Channel Groups Source: none configured (will use fallback chunking if needed).")

    base_groups = []
    if manual_groups:
        base_groups = resolve_manual_groups(recording, manual_groups)
        if not base_groups and groups_source != "inline CHANNEL_GROUPS":
            log_warn(f"no channel groups resolved from {groups_source}; falling back to inline CHANNEL_GROUPS.")
            manual_groups = CHANNEL_GROUPS
            groups_source = "inline CHANNEL_GROUPS (fallback)"
            base_groups = resolve_manual_groups(recording, manual_groups)
        if not base_groups:
            log_warn("no channel groups resolved; falling back to chunked groups.")
            manual_groups = []
    if not base_groups:
        if STRICT_GROUPS:
            raise RuntimeError(
                "No valid channel groups resolved. Provide CHANNEL_GROUPS or --channel-groups "
                "and ensure channel IDs match the recording."
            )
        base_groups = chunk_groups(original_channel_order, CHANNELS_PER_TETRODE)

    log_section("Bad Channels")
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
    log_info(f"Bad Channels Source: {bad_source}")

    manual_bad = resolve_bad_channel_ids(recording, bad_channels)
    auto_bad = detect_bad_channel_ids(
        recording,
        AUTO_BAD_CHANNELS,
        method=AUTO_BAD_CHANNELS_METHOD,
        **AUTO_BAD_CHANNELS_KWARGS,
    )
    auto_bad_sorted = np.sort(auto_bad)
    if AUTO_BAD_CHANNELS:
        log_info(f"Auto-detected bad channels ({auto_bad.size}): {auto_bad_sorted.tolist()}")
    else:
        if auto_bad.size:
            log_info(f"Note: AUTO_BAD_CHANNELS disabled, ignoring detected list: {auto_bad_sorted.tolist()}")

    # Normalize bad IDs to recording dtype before slicing.
    channel_dtype = np.asarray(channel_order).dtype if channel_order else object
    bad_arrays = [np.array(arr, dtype=channel_dtype) for arr in (manual_bad, auto_bad) if arr.size]
    removed_bad_count = 0
    bad_ids = np.array([], dtype=channel_dtype)
    bad_indices: list[int] = []
    if bad_arrays:
        bad_ids = np.unique(np.concatenate(bad_arrays))
        bad_indices = sorted([original_index_map[str(ch)] for ch in bad_ids if str(ch) in original_index_map])
        removed_bad_count = int(len(bad_ids))
        log_info(f"Removed {len(bad_ids)} bad channels: {bad_ids.tolist()}")
        log_info(f"Removed Indices In Original Order: {bad_indices}")
        keep_ids = [ch for ch in channel_order if ch not in bad_ids]
        recording = safe_channel_slice(recording, keep_ids)
        assert_channel_slice_order(recording, keep_ids, "post-bad-channel slice")
        channel_order = keep_ids
    else:
        channel_order = original_channel_order.copy()

    # Preserve original tetrode grid positions even if some tetrodes are fully removed.
    base_tetrode_count = len(base_groups)
    filtered_groups = []
    filtered_indices = []
    dropped_tiny_groups = []
    for idx, grp in enumerate(base_groups):
        subset = [ch for ch in grp if ch in channel_order]
        if not subset:
            continue
        # SC2 whitening paths require at least 2 channels per group.
        if len(subset) < 2:
            dropped_tiny_groups.append((idx, len(subset)))
            continue
        filtered_groups.append(subset)
        filtered_indices.append(idx)
    if dropped_tiny_groups:
        log_warn(
            "Dropping groups with <2 channels (SC2 whitening may fail on single-channel groups): "
            f"{dropped_tiny_groups}"
        )
    groups = filtered_groups
    if STRICT_GROUPS and not groups:
        raise RuntimeError(
            "No channel groups remain after bad-channel filtering. "
            "Check BAD_CHANNELS/CHANNEL_GROUPS and selected stream/session."
        )
    # Keep original tetrode IDs after filtering so labels stay stable across sessions.
    group_ids = filtered_indices.copy() if filtered_indices else list(range(len(groups)))
    grouped_ids = {ch for grp in groups for ch in grp}
    keep_ids = [ch for ch in channel_order if ch in grouped_ids]
    if len(keep_ids) != len(channel_order):
        missing = [ch for ch in channel_order if ch not in grouped_ids]
        if STRICT_GROUPS:
            raise RuntimeError(
                f"{len(missing)} channels are not covered by CHANNEL_GROUPS under STRICT_GROUPS=True "
                f"(first: {missing[:5]}). This often indicates a wrong stream/session selection "
                "or mismatched group config."
            )
        log_warn(
            f"{len(missing)} channels not present in groups; removing from recording "
            f"(first: {missing[:5]})"
        )
        recording = safe_channel_slice(recording, keep_ids)
        assert_channel_slice_order(recording, keep_ids, "post-group-coverage slice")
        channel_order = keep_ids
    small_groups = [gid for gid, grp in zip(group_ids, groups) if len(grp) < 3]
    if small_groups:
        log_warn(f"tetrodes with <3 channels: {small_groups}")
    if manual_groups:
        log_info(f"Tetrodes kept: {len(groups)} (source: {groups_source}); first group: {groups[0] if groups else 'n/a'}")
    else:
        log_info(f"Tetrodes kept: {len(groups)} (fallback chunking); first group: {groups[0] if groups else 'n/a'}")
    grouping_details = {
        "groups_source": groups_source,
        "bad_channels_source": bad_source,
        "manual_bad_channels_resolved": _to_jsonable(manual_bad),
        "auto_bad_channels_detected": _to_jsonable(auto_bad_sorted),
        "removed_bad_channels": _to_jsonable(bad_ids),
        "removed_bad_channel_indices_in_original_order": _to_jsonable(bad_indices),
        "base_tetrode_count": int(base_tetrode_count),
        "dropped_tiny_groups": _to_jsonable(dropped_tiny_groups),
        "kept_group_ids": _to_jsonable(group_ids),
    }

    if args.dry_run_config:
        log_section("Dry Run Plan")
        log_info(f"Session mode: {'explicit subpath' if SESSION_SUBPATH else 'prompt'}")
        log_info(f"Stream: {STREAM_NAME or 'auto'} | Test seconds: {TEST_SECONDS if TEST_SECONDS is not None else 'full'}")
        log_info(f"Group source: {groups_source} | Tetrodes kept: {len(groups)} / {base_tetrode_count}")
        log_info(f"Bad-channel source: {bad_source} | Removed channels: {removed_bad_count}")
        log_info(f"Sort mode: {'per-group' if SORT_BY_GROUP else 'bundle'}")
        log_info(
            f"Preprocess toggles: USE_SI_PREPROCESS={USE_SI_PREPROCESS}, "
            f"USE_SC2_PREPROCESS={USE_SC2_PREPROCESS}"
        )
        log_info(f"Analyzer source: {'rec_sc2' if ANALYZER_FROM_SORTER else 'rec_export'}")
        log_info(f"Phy export mode: {EXPORT_PHY_CHANNEL_IDS_MODE}")
        log_info(f"Planned SC2 run folder: {sc2_run}")
        log_info("Dry-run complete. Exiting before geometry/preprocessing/sorting/export.")
        return recording, groups, group_ids, channel_order, filtered_indices, base_tetrode_count, grouping_details, True

    return recording, groups, group_ids, channel_order, filtered_indices, base_tetrode_count, grouping_details, False


def compute_tetrode_layout(base_tetrode_count, filtered_indices, groups):
    """Compute tetrode grid parameters and offsets for geometry attachment."""
    tetrode_offsets = None
    tetrodes_per_row = max(1, int(np.ceil(np.sqrt(base_tetrode_count)))) if base_tetrode_count else 1
    if groups:
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
    return tetrodes_per_row, tetrode_offsets


def prepare_recordings_for_sorting_and_export(
    recording,
    groups,
    group_ids,
    tetrodes_per_row,
    tetrode_offsets,
    sc2_run: Path,
):
    """Attach geometry, build sorter/export/analyzer recordings, and return them."""
    log_section("Geometry")
    if ATTACH_GEOMETRY and groups:
        # Authoritative geometry attach on the post-sliced recording.
        recording = attach_geometry_if_needed_shared(
            recording,
            groups,
            attach_geometry=ATTACH_GEOMETRY,
            tetrode_pitch_um=TETRODE_PITCH_UM,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
            scale_to_uv=False,
            debug=DEBUG_GEOMETRY_ATTACH,
            label="recording",
        )
        recording = validate_or_reattach_geometry(
            recording,
            groups,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
            label="recording",
            require_probe=True,
        )
        print(f"Geometry checked on recording (tetrodes_per_row={tetrodes_per_row}).")

    # Optional SI CAR branch when SI preprocessing is disabled.
    if not USE_SI_PREPROCESS:
        recording = apply_configured_si_car(
            recording,
            groups,
            context_label="Preprocessing (no-SI path)",
            print_disabled=False,
        )

    log_section("Preprocessing")
    rec_sc2 = preprocess_for_sc2(recording, groups=groups)
    rec_sc2 = rec_sc2.astype("float32")

    rec_sc2 = validate_or_reattach_geometry(
        rec_sc2,
        groups,
        tetrodes_per_row=tetrodes_per_row,
        tetrode_offsets=tetrode_offsets,
        label="rec_sc2",
        require_probe=True,
    )
    if MATERIALIZE_SI_PREPROCESS:
        try:
            preproc_folder = sc2_run / "rec_sc2_preprocessed"
            rec_sc2 = rec_sc2.save(folder=preproc_folder, format="binary_folder", overwrite=True)
            print(f"Materialized rec_sc2 at {preproc_folder}")
            rec_sc2 = validate_or_reattach_geometry(
                rec_sc2,
                groups,
                tetrodes_per_row=tetrodes_per_row,
                tetrode_offsets=tetrode_offsets,
                label="rec_sc2 (materialized)",
                require_probe=True,
            )
        except Exception as exc:
            print(f"WARNING: failed to materialize rec_sc2: {exc}")

    log_section("Export/Analyzer Prep")
    # Export/analyzer recordings do not change sorter input.
    rec_export = recording
    if EXPORT_SCALE_TO_UV:
        rec_export = maybe_scale_recording_to_uv_shared(
            rec_export,
            label="Export Path",
            missing_message="Export Path: channel gains not found; export left in native units.",
            success_message="Export Path: scaled recording to microvolts for Phy/GUI export.",
        )
    rec_export = validate_or_reattach_geometry(
        rec_export,
        groups,
        tetrodes_per_row=tetrodes_per_row,
        tetrode_offsets=tetrode_offsets,
        label="rec_export",
        require_probe=True,
    )
    if MATERIALIZE_EXPORT:
        try:
            export_folder = sc2_run / "rec_export_prepared"
            rec_export = rec_export.save(folder=export_folder, format="binary_folder", overwrite=True)
            print(f"Materialized rec_export at {export_folder}")
            rec_export = validate_or_reattach_geometry(
                rec_export,
                groups,
                tetrodes_per_row=tetrodes_per_row,
                tetrode_offsets=tetrode_offsets,
                label="rec_export (materialized)",
                require_probe=True,
            )
        except Exception as exc:
            print(f"WARNING: failed to materialize rec_export: {exc}")
    set_group_property(rec_export, groups, group_ids)
    assert_group_property_mapping(rec_export, groups, group_ids, "rec_export")

    if ANALYZER_FROM_SORTER:
        rec_analyzer = rec_sc2
        # QC scaling branch only; sorter output is already fixed.
        if EXPORT_SCALE_TO_UV:
            rec_analyzer = maybe_scale_recording_to_uv_shared(
                rec_analyzer,
                label="Analyzer Path",
                missing_message="Analyzer Path: channel gains not found; analyzer left in native units.",
                success_message="Analyzer Path: scaled rec_sc2 to microvolts for QC.",
            )
        rec_analyzer = validate_or_reattach_geometry(
            rec_analyzer,
            groups,
            tetrodes_per_row=tetrodes_per_row,
            tetrode_offsets=tetrode_offsets,
            label="rec_analyzer",
            require_probe=True,
        )
        if MATERIALIZE_EXPORT:
            try:
                analyzer_folder = sc2_run / "rec_analyzer_prepared"
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

    # Re-assert group labels after transform wrappers in case properties were dropped.
    set_group_property(rec_analyzer, groups, group_ids)
    assert_group_property_mapping(rec_analyzer, groups, group_ids, "rec_analyzer")

    return rec_sc2, rec_export, rec_analyzer


# Small Utility Helpers (Stage Logic)
def assert_channel_slice_order(recording, expected_ids, label: str):
    """Fail fast when channel slicing reorders channels unexpectedly."""
    expected = list(expected_ids)
    actual = list(recording.channel_ids)
    if actual == expected:
        return
    if len(actual) != len(expected):
        raise RuntimeError(
            f"{label}: channel count mismatch after slice "
            f"(actual={len(actual)}, expected={len(expected)})."
        )
    mismatch = None
    for idx, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            mismatch = (idx, a, e)
            break
    if mismatch is None:
        mismatch = ("unknown", actual[:5], expected[:5])
    raise RuntimeError(
        f"{label}: channel order mismatch after slice at index {mismatch[0]} "
        f"(actual={mismatch[1]}, expected={mismatch[2]})."
    )


def assert_group_property_mapping(recording, groups, group_ids, label: str):
    """Fail fast when recording 'group' property does not match expected mapping."""
    if not groups:
        return
    if group_ids is None:
        gids = list(range(len(groups)))
    elif isinstance(group_ids, (int, np.integer, str)):
        gids = [group_ids]
    else:
        gids = list(group_ids)
    if len(gids) != len(groups):
        raise RuntimeError(
            f"{label}: group_ids/groups length mismatch "
            f"({len(gids)} vs {len(groups)})."
        )
    try:
        values = np.asarray(recording.get_property("group"))
    except Exception as exc:
        raise RuntimeError(f"{label}: missing 'group' property ({exc}).") from exc
    n_ch = recording.get_num_channels()
    if values.ndim != 1 or values.size != n_ch:
        raise RuntimeError(
            f"{label}: invalid 'group' property length "
            f"(shape={getattr(values, 'shape', None)}, channels={n_ch})."
        )
    index_map = {ch: i for i, ch in enumerate(recording.channel_ids)}
    mismatches = []
    for gid, grp in zip(gids, groups):
        expected_gid = int(gid)
        for ch in grp:
            idx = index_map.get(ch)
            if idx is None:
                raise RuntimeError(f"{label}: channel {ch!r} is not present in recording.channel_ids.")
            actual_gid = int(values[idx])
            if actual_gid != expected_gid:
                mismatches.append((ch, actual_gid, expected_gid))
                if len(mismatches) >= 5:
                    break
        if mismatches:
            break
    if mismatches:
        raise RuntimeError(
            f"{label}: 'group' property mismatch examples={mismatches}."
        )


def normalize_group_key(value):
    """Normalize group identifiers so dict lookups work for str/int keys."""
    try:
        return int(value)
    except Exception:
        return value


def get_group_recording(rec_dict, key, fallback):
    """Fetch a group recording by trying raw/int/str key variants."""
    if rec_dict is None:
        return fallback
    if key in rec_dict:
        return rec_dict[key]
    key_int = normalize_group_key(key)
    if key_int in rec_dict:
        return rec_dict[key_int]
    key_str = str(key)
    if key_str in rec_dict:
        return rec_dict[key_str]
    return fallback


def require_group_recording(rec_dict, key, dict_name: str):
    """Strict group recording fetch that raises on missing keys."""
    rec = get_group_recording(rec_dict, key, None)
    if rec is None:
        try:
            keys = list(rec_dict.keys()) if rec_dict is not None else []
        except Exception:
            keys = []
        raise RuntimeError(f"Group key {key!r} not found in {dict_name} keys={keys}")
    return rec


# Sorting/Postprocessing Helpers
def prepare_sc2_runtime_params(rec_sc2, groups, group_ids):
    """Prepare SC2 params and analyzer waveform window settings."""
    log_section("Sorting")
    # Ensure group property is present for downstream split/export behavior.
    set_group_property(rec_sc2, groups, group_ids)
    assert_group_property_mapping(rec_sc2, groups, group_ids, "rec_sc2")

    sc2_defaults = ss.Spykingcircus2Sorter.default_params()
    warn_unknown_sc2_overrides(sc2_defaults, SC2_PARAM_OVERRIDES)
    sc2_params = merge_params(sc2_defaults, SC2_PARAM_OVERRIDES)
    sc2_params["apply_preprocessing"] = bool(USE_SC2_PREPROCESS)
    if "preprocessing" in sc2_params:
        sc2_params["preprocessing"]["apply"] = bool(USE_SC2_PREPROCESS)
    # Keep SC2 execution conservative/stable unless explicitly overridden.
    sc2_params.setdefault("job_kwargs", {})
    sc2_params["job_kwargs"].setdefault("n_jobs", 1)
    if USE_SI_PREPROCESS and sc2_params.get("apply_preprocessing", True):
        print(
            "WARNING: both SI and SC2 preprocessing are enabled; "
            "this will apply filter/reference stages twice."
        )
    if SI_APPLY_CAR and sc2_params.get("apply_preprocessing", True):
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
    print("SC2 params -> job_kwargs:", sc2_params.get("job_kwargs"))
    if not sc2_params.get("apply_preprocessing", True) and isinstance(sc2_params.get("filtering"), dict):
        print(
            "SC2 params -> filtering overrides present but SC2 preprocessing is disabled "
            "(apply_preprocessing=False); SC2 bandpass/CMR step is skipped."
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
    return sc2_params, analyzer_ms_before, analyzer_ms_after


def run_sorting_stage(
    rec_sc2,
    rec_export,
    sc2_run: Path,
    sc2_params: dict,
    groups,
    group_ids,
    analyzer_ms_before,
    analyzer_ms_after,
    original_index_map: dict,
    oe_index_map: dict,
):
    """Run SC2 sorting and per-group analyzer/export flow when SORT_BY_GROUP=True."""
    sorting_sc2 = None
    phy_folders = []
    si_gui_folders = []
    sorter_folder = sc2_run / "sorter_output"
    # reserve_run_folder() only reserves a unique path; ensure parent exists
    # before passing a nested sorter folder to SI.
    sorter_folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        if SORT_BY_GROUP:
            print("Sorting per tetrode group (split_by='group').")
            try:
                group_prop = np.asarray(rec_sc2.get_property("group"))
                rec_group_values = sorted({normalize_group_key(v) for v in group_prop.tolist()})
                expected_group_values = sorted({normalize_group_key(v) for v in (group_ids or [])})
                print(
                    "Per-group key check -> rec_sc2 group values:",
                    rec_group_values,
                    "| expected group_ids:",
                    expected_group_values,
                )
                if set(rec_group_values) != set(expected_group_values):
                    log_warn(
                        "Per-group key mismatch between rec_sc2 'group' property and group_ids. "
                        "This can cause split-key lookup errors."
                    )
            except Exception as exc:
                log_warn(f"Could not validate per-group keys before split: {exc}")
            try:
                locs = np.asarray(rec_sc2.get_channel_locations(), dtype="float64")
                if locs.ndim != 2 or locs.shape[0] != rec_sc2.get_num_channels() or not np.all(np.isfinite(locs)):
                    raise RuntimeError(
                        "rec_sc2 channel locations are invalid "
                        f"(shape={getattr(locs, 'shape', None)})."
                    )
            except Exception as exc:
                raise RuntimeError(
                    "Per-group sorting requires valid channel locations for probe coercion. "
                    f"Attach geometry/probe before sorting ({exc})."
                ) from exc
            rec_sc2_dict = {}
            for gid, rec in rec_sc2.split_by("group").items():
                rec_sc2_dict[gid] = coerce_dummy_probe_from_locations_strict(
                    rec, f"rec_sc2[{gid}]"
                )
            rec_export_dict = rec_export.split_by("group")
            if group_ids is not None:
                group_map = {normalize_group_key(gid): grp for gid, grp in zip(group_ids, groups)}
            else:
                group_map = {normalize_group_key(idx): grp for idx, grp in enumerate(groups)}

            sortings_sc2 = ss.run_sorter(
                "spykingcircus2",
                rec_sc2_dict,
                folder=sorter_folder,
                remove_existing_folder=True,
                verbose=True,
                **sc2_params,
            )
            for gid, sorting_group in sortings_sc2.items():
                gid_key = normalize_group_key(gid)
                rec_exp = require_group_recording(rec_export_dict, gid_key, "rec_export_dict")
                rec_an = rec_exp if not ANALYZER_FROM_SORTER else require_group_recording(
                    rec_sc2_dict, gid_key, "rec_sc2_dict"
                )
                grp_channels = group_map.get(gid_key, list(rec_exp.channel_ids))

                # In per-group mode, keep analyzer units consistent with bundle mode:
                # when using rec_sc2 as analyzer source, apply optional uV scaling for QC.
                if ANALYZER_FROM_SORTER and EXPORT_SCALE_TO_UV:
                    rec_an = maybe_scale_recording_to_uv_shared(
                        rec_an,
                        label=f"Analyzer Path (group {gid})",
                        missing_message=(
                            f"Analyzer Path (group {gid}): channel gains not found; analyzer left in native units."
                        ),
                        success_message=f"Analyzer Path (group {gid}): scaled rec_sc2 to microvolts for QC.",
                    )

                # Re-assert group labels after transform wrappers.
                try:
                    gid_label = int(gid_key)
                except Exception:
                    gid_label = 0
                set_group_property(rec_an, [grp_channels], [gid_label])
                assert_group_property_mapping(rec_an, [grp_channels], [gid_label], f"rec_an(group {gid})")

                analyzer_group = build_analyzer(
                    sorting_group,
                    rec_an,
                    SC2_OUT,
                    f"sc2_g{gid}",
                    wf_ms_before=analyzer_ms_before,
                    wf_ms_after=analyzer_ms_after,
                )
                analyzer_group = maybe_remove_redundant_units(analyzer_group, f"sc2_g{gid}")
                if EXPORT_TO_PHY:
                    phy_folder, _ = export_for_phy(
                        analyzer_group,
                        SC2_OUT,
                        f"sc2_g{gid}",
                        [grp_channels],
                        original_index_map,
                        oe_index_map,
                        [gid],
                    )
                    phy_folders.append(phy_folder)
                if EXPORT_TO_SI_GUI:
                    si_gui_folders.append(export_for_si_gui(analyzer_group, SI_GUI_OUT, f"sc2_g{gid}"))
            print(f"SC2 per-group runs complete | output: {sc2_run}")
        else:
            sorting_sc2 = ss.run_sorter(
                "spykingcircus2",
                rec_sc2,
                folder=sorter_folder,
                remove_existing_folder=True,
                verbose=True,
                **sc2_params,
            )
            print(f"SC2 units: {sorting_sc2.get_num_units()} | output: {sorter_folder}")
    except Exception as exc:
        print(f"Error: SpykingCircus2 sorter failed: {exc}")
        print(f"Preserving run folder for debugging: {sc2_run}")
        print(f"Attempting to remove partial sorter output folder: {sorter_folder}")
        try:
            safe_rmtree(sorter_folder)
            print("Partial sorter output folder removed.")
        except Exception as exc2:
            print(f"WARNING: failed to remove partial sorter output folder: {exc2}")
        raise
    return sorting_sc2, phy_folders, si_gui_folders


def run_postprocessing_stage(
    sorting_sc2,
    rec_analyzer,
    groups,
    group_ids,
    analyzer_ms_before,
    analyzer_ms_after,
    original_index_map: dict,
    oe_index_map: dict,
    phy_folders,
    si_gui_folders,
):
    """Run analyzer/QC/export stage for single-run mode."""
    log_section("Postprocessing/Export")
    # Build analyzer, QC, and exports from sorter output.
    analyzer_sc2 = None
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
            phy_folder, _ = export_for_phy(
                analyzer_sc2,
                SC2_OUT,
                "sc2",
                groups,
                original_index_map,
                oe_index_map,
                group_ids,
            )
            phy_folders.append(phy_folder)
        if EXPORT_TO_SI_GUI:
            si_gui_folders.append(export_for_si_gui(analyzer_sc2, SI_GUI_OUT, "sc2"))
    return analyzer_sc2, phy_folders, si_gui_folders


def print_summary_stage(sorting_sc2, analyzer_sc2, phy_folders, si_gui_folders):
    """Print final run summary and viewer launch commands."""
    log_section("Summary")
    log_info("SC2 Pipeline Complete.")
    if phy_folders:
        if SORT_BY_GROUP:
            print("Summary: per-group exports written (see each group's QC/units within its folder).")
        for folder in phy_folders:
            print(f"Run: phy template-gui \"{folder / 'params.py'}\"")
            params_py = folder / "params.py"
            if EXPORT_PHY_EXTRACT_WAVEFORMS:
                try:
                    subprocess.run(["phy", "extract-waveforms", str(params_py)], check=True)
                    print(f"Phy: extracted waveforms for export ({folder.name}).")
                except FileNotFoundError:
                    print("WARNING: 'phy' command not found; skipping extract-waveforms.")
                except subprocess.CalledProcessError as exc:
                    print(f"WARNING: phy extract-waveforms failed: {exc}")
            if not SORT_BY_GROUP and sorting_sc2 is not None and analyzer_sc2 is not None:
                try:
                    n_units = len(sorting_sc2.get_unit_ids())
                except Exception:
                    n_units = "unknown"
                qc_metrics_path = getattr(analyzer_sc2, "_qc_metrics_path", None)
                if qc_metrics_path is None and getattr(analyzer_sc2, "folder", None):
                    qc_metrics_path = str(Path(analyzer_sc2.folder) / "qc_metrics.csv")
                print(
                    "Summary:",
                    f"units={n_units},",
                    f"qc_metrics={qc_metrics_path or 'n/a'}",
                )
            else:
                print("Summary: per-group Phy exports written.")
    if si_gui_folders:
        for folder in si_gui_folders:
            print(f"Run: python -m spikeinterface_gui \"{folder}\"")


# ---------------------------------------------------------------------
# Main Pipeline Flow
# ---------------------------------------------------------------------

def main():
    """Run the tetrode SC2 pipeline from CLI args and configured globals."""
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
    applied_overrides, ignored_overrides = apply_env_pipeline_overrides()
    # Apply strict RAW_MODE policy after env overrides so wrapper/env can enable it.
    apply_raw_mode_policy()
    validate_preprocess_toggles()
    if applied_overrides:
        log_info(f"Applied pipeline overrides from env: {sorted(applied_overrides.keys())}")
    if ignored_overrides:
        log_warn(f"Ignored unknown/blocked pipeline overrides from env: {sorted(ignored_overrides)}")

    if DEBUG_WARN_TRACE:
        enable_warning_trace()

    env_groups_path, env_bad_path, channel_groups_path, bad_channels_path, use_config_jsons = resolve_config_sources_shared(
        args,
        project_root=PROJECT_ROOT,
        use_config_jsons_default=USE_CONFIG_JSONS,
        channel_groups_path_default=CHANNEL_GROUPS_PATH,
        bad_channels_path_default=BAD_CHANNELS_PATH,
    )
    root_dir = args.root_dir
    base_out = args.base_out
    sc2_run = reserve_run_folder(base_out)
    global SC2_OUT, SI_GUI_OUT
    SC2_OUT, SI_GUI_OUT, run_log_handle, _ = initialize_run_io_shared(
        base_out,
        sc2_run,
        export_to_si_gui=EXPORT_TO_SI_GUI,
    )
    atexit.register(disable_run_logging, run_log_handle)
    print_pipeline_config_echo(
        raw_mode=RAW_MODE,
        use_si_preprocess=USE_SI_PREPROCESS,
        use_sc2_preprocess=USE_SC2_PREPROCESS,
        si_apply_whiten=SI_APPLY_WHITEN,
        si_apply_car=SI_APPLY_CAR,
        apply_notch=APPLY_NOTCH,
        car_mode=CAR_MODE,
        attach_geometry=ATTACH_GEOMETRY,
        export_scale_to_uv=EXPORT_SCALE_TO_UV,
        analyzer_from_sorter=ANALYZER_FROM_SORTER,
        auto_bad_channels=AUTO_BAD_CHANNELS,
        strict_groups=STRICT_GROUPS,
        sort_by_group=SORT_BY_GROUP,
        use_config_jsons=use_config_jsons,
        test_seconds=TEST_SECONDS,
        stream_name=STREAM_NAME,
    )

    (
        recording,
        original_channel_order,
        original_index_map,
        oe_index_map,
        channel_order,
        selected_session_path,
        selected_stream,
    ) = load_recording_with_indices(root_dir)

    (
        recording,
        groups,
        group_ids,
        channel_order,
        filtered_indices,
        base_tetrode_count,
        grouping_details,
        should_exit_early,
    ) = resolve_groups_and_bad_channels(
        recording,
        channel_order,
        original_channel_order,
        original_index_map,
        args,
        channel_groups_path,
        bad_channels_path,
        env_groups_path,
        env_bad_path,
        sc2_run,
    )
    if should_exit_early:
        return

    tetrodes_per_row, tetrode_offsets = compute_tetrode_layout(
        base_tetrode_count,
        filtered_indices,
        groups,
    )
    rec_sc2, rec_export, rec_analyzer = prepare_recordings_for_sorting_and_export(
        recording,
        groups,
        group_ids,
        tetrodes_per_row,
        tetrode_offsets,
        sc2_run,
    )

    sc2_params, analyzer_ms_before, analyzer_ms_after = prepare_sc2_runtime_params(rec_sc2, groups, group_ids)
    # Persist the fully resolved run state before sorter execution for reproducibility.
    write_effective_config(
        sc2_run=sc2_run,
        args=args,
        use_config_jsons=use_config_jsons,
        root_dir=root_dir,
        selected_session_path=selected_session_path,
        selected_stream=selected_stream,
        applied_overrides=applied_overrides,
        ignored_overrides=ignored_overrides,
        groups=groups,
        group_ids=group_ids,
        grouping_details=grouping_details,
        sc2_params=sc2_params,
    )
    sorting_sc2, phy_folders, si_gui_folders = run_sorting_stage(
        rec_sc2,
        rec_export,
        sc2_run,
        sc2_params,
        groups,
        group_ids,
        analyzer_ms_before,
        analyzer_ms_after,
        original_index_map,
        oe_index_map,
    )
    analyzer_sc2, phy_folders, si_gui_folders = run_postprocessing_stage(
        sorting_sc2,
        rec_analyzer,
        groups,
        group_ids,
        analyzer_ms_before,
        analyzer_ms_after,
        original_index_map,
        oe_index_map,
        phy_folders,
        si_gui_folders,
    )
    print_summary_stage(sorting_sc2, analyzer_sc2, phy_folders, si_gui_folders)

if __name__ == "__main__":
    main()
