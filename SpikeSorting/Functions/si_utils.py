"""SpikeInterface geometry/stream helpers.

Note: heavy dependencies are imported lazily to keep import-time light.
"""

import ast
import json
import re
from pathlib import Path

import numpy as np

from .run_utils import log_info, log_warn


def _load_spre():
    """Lazy-load spikeinterface.preprocessing to keep module import lightweight."""
    try:
        import spikeinterface.preprocessing as spre
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface is required for preprocessing helpers."
        ) from exc
    return spre


def _load_se():
    """Lazy-load spikeinterface.extractors to keep module import lightweight."""
    try:
        import spikeinterface.extractors as se
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface is required for Open Ephys stream discovery. "
            "Activate the spikeinterface environment before running the pipeline."
        ) from exc
    return se


def _load_probe():
    """Lazy-load Probe class to avoid heavy imports during module import."""
    try:
        from probeinterface import Probe
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "probeinterface is required for geometry/probe attachment."
        ) from exc
    return Probe


def discover_oe_stream_names(folder):
    """Return stream names declared in an Open Ephys folder."""
    se = _load_se()
    try:
        rec = se.read_openephys(folder)
        getter = getattr(rec, "get_annotation", None)
        if callable(getter):
            names = getter("streams_names") or getter("stream_names") or []
        else:
            annotations = getattr(rec, "_annotations", {}) or {}
            names = annotations.get("streams_names") or annotations.get("stream_names") or []
        if not names:
            names = [getattr(rec, "stream_name", None)] if hasattr(rec, "stream_name") else []
            names = [name for name in names if name]
        return names
    except ValueError as exc:
        message = str(exc)
        match = re.search(r"`stream_names`:\s*(\[[^\]]+\])", message)
        if match:
            try:
                return list(ast.literal_eval(match.group(1)))
            except Exception as parse_exc:
                log_warn(
                    "failed to parse stream names from extractor ValueError; "
                    f"re-raising original error ({parse_exc})."
                )
        raise


def normalize_stream_name(name: str | None) -> str:
    """Normalize stream-name variants across Open Ephys/SI representations."""
    if name is None:
        return ""
    text = str(name).strip()
    if "#" in text:
        text = text.split("#", 1)[-1].strip()
    return text.casefold()


def _select_oebin_stream_entry(continuous: list, stream_name: str):
    """Best-effort stream entry resolution for structure.oebin.

    Matching priority:
    1) exact string match
    2) normalized-name match (handles node/prefix wrappers)
    3) suffix containment

    Safety policy:
    - If multiple entries tie for best score, return None (ambiguous) instead
      of picking an arbitrary first match.
    """
    target = str(stream_name)
    target_norm = normalize_stream_name(stream_name)
    best_entries = []
    best_score = -1
    for entry in continuous:
        candidate = str(entry.get("stream_name"))
        if candidate == target:
            score = 3
        elif target_norm and normalize_stream_name(candidate) == target_norm:
            score = 2
        elif target.endswith(candidate) or candidate.endswith(target):
            score = 1
        else:
            score = 0
        if score > best_score:
            best_entries = [entry]
            best_score = score
        elif score == best_score and score > 0:
            best_entries.append(entry)
    if best_score <= 0:
        return None
    if len(best_entries) > 1:
        candidates = [str(e.get("stream_name")) for e in best_entries]
        log_warn(
            "multiple structure.oebin streams matched requested stream name; "
            "skipping metadata attach to avoid ambiguity. "
            f"requested={stream_name!r}, matched={candidates}"
        )
        return None
    return best_entries[0]


def attach_geom(recording, groups, tetrodes_per_row=None, pitch=20.0, dx=150.0, dy=150.0, tetrode_offsets=None):
    """Attach 2x2-style tetrode geometry and return a new recording.

    Gotcha: channels missing from `groups` keep default coordinates and trigger warnings.
    """
    groups = list(groups)
    if tetrode_offsets is not None and len(tetrode_offsets) != len(groups):
        raise ValueError("tetrode_offsets must match number of tetrodes")

    assigned = {ch for grp in groups for ch in grp}
    missing = [ch for ch in recording.channel_ids if ch not in assigned]
    if missing:
        log_warn(f"{len(missing)} channels not present in groups (first: {missing[:5]})")

    index_map = {ch: i for i, ch in enumerate(recording.channel_ids)}
    positions = np.zeros((len(index_map), 2), dtype=float)

    if tetrodes_per_row is None:
        tetrodes_per_row = max(1, int(np.ceil(np.sqrt(len(groups)))))

    for tetrode, group in enumerate(groups):
        base = np.array([[0.0, 0.0], [pitch, 0.0], [0.0, pitch], [pitch, pitch]], dtype=float)[: len(group)]
        if tetrode_offsets is not None:
            offset = np.array(tetrode_offsets[tetrode], dtype=float)
        else:
            row, col = divmod(tetrode, tetrodes_per_row)
            offset = np.array([col * dx, row * dy], dtype=float)
        for j, ch in enumerate(group):
            if ch not in index_map:
                log_warn(f"channel {ch} missing from recording; skipping geometry assignment.")
                continue
            positions[index_map[ch]] = base[j] + offset

    Probe = _load_probe()
    probe = Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7})
    probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
    return recording.set_probe(probe, in_place=False)


# Optional fallback: single-grid geometry.
# - Places channels on one flat grid by channel order (ncols, dx_um, dy_um).
# - Returns one all-channel group (group_ids=[0]), unlike tetrode grouping.
# - Not used by default in current pipelines; call explicitly only for A/B tests.
def attach_single_grid_geom(recording, channel_order=None, ncols=4, dx_um=10.0, dy_um=200.0):
    """Attach flat grid geometry and return `(new_recording, groups, group_ids)`.

    Gotcha: this creates one all-channel group and is not tetrode-specific.
    """
    if ncols <= 0:
        raise ValueError("ncols must be >= 1")

    channel_order = list(channel_order) if channel_order is not None else list(recording.channel_ids)
    index_map = {ch: i for i, ch in enumerate(recording.channel_ids)}
    positions = np.zeros((recording.get_num_channels(), 2), dtype=float)

    for idx, ch in enumerate(channel_order):
        if ch not in index_map:
            log_warn(f"channel {ch} missing from recording; skipping grid assignment.")
            continue
        row, col = divmod(idx, ncols)
        positions[index_map[ch]] = (col * float(dx_um), row * float(dy_um))

    Probe = _load_probe()
    probe = Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7})
    probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
    rec = recording.set_probe(probe, in_place=False)
    groups = [channel_order.copy()]
    group_ids = [0]
    return rec, groups, group_ids


def set_group_property(recording, groups, group_ids=None):
    """Set per-channel `group` property in place.

    Gotcha: channels not assigned to any group are set to `-1` and reported.

    Example:
        set_group_property(rec, [["CH40", "CH38", "CH36", "CH34"]], [0])
    """
    if group_ids is None:
        group_ids = list(range(len(groups)))
    elif len(group_ids) != len(groups):
        raise ValueError("group_ids must match number of groups")
    index_map = {ch: i for i, ch in enumerate(recording.channel_ids)}
    values = np.full(len(recording.channel_ids), fill_value=-1, dtype=int)
    for group_label, group in zip(group_ids, groups):
        for ch in group:
            if ch in index_map:
                values[index_map[ch]] = int(group_label)
    recording.set_property("group", values)
    if np.any(values == -1):
        missing_idx = np.where(values == -1)[0]
        missing_ch = [recording.channel_ids[i] for i in missing_idx[:10]]
        log_warn(f"unassigned channels in group property (examples: {missing_ch})")


def ensure_probe_attached(recording, radius=5):
    """Ensure recording has a probe; attach from channel locations if missing.

    Gotcha: returns same object when probe exists, otherwise returns a new object.
    """
    try:
        if recording.get_probe() is not None:
            return recording
    except Exception as exc:
        log_warn(
            "could not query probe on recording; falling back to channel-location "
            f"probe attach ({exc})."
        )
    try:
        locs = recording.get_channel_locations()
    except Exception as exc:
        raise RuntimeError(f"Cannot attach probe: channel locations unavailable ({exc})")
    Probe = _load_probe()
    probe = Probe(ndim=locs.shape[1])
    probe.set_contacts(positions=locs, shapes="circle", shape_params={"radius": radius})
    probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
    return recording.set_probe(probe, in_place=False)


def ensure_geom_and_units(recording, groups, tetrodes_per_row=None, scale_to_uv=True, tetrode_offsets=None, pitch=20.0):
    """Attach geometry and check gain metadata without rescaling traces.

    Gotcha: this validates metadata only; it does not convert data to microvolts.
    """
    rec = attach_geom(
        recording,
        groups,
        tetrodes_per_row,
        pitch=pitch,
        tetrode_offsets=tetrode_offsets,
    )
    if scale_to_uv:
        gain = None
        try:
            gain = rec.get_property("gain_to_uV")
        except Exception:
            gain = None
        if gain is None:
            # Fall back to channel gains if available (metadata only; no rescaling here).
            try:
                gain = rec.get_channel_gains()
            except Exception:
                gain = None
        if gain is None:
            log_warn("gain_to_uV/gains missing; leaving recording in native units.")
    return rec


def attach_geometry_if_needed(
    recording,
    groups,
    *,
    attach_geometry: bool,
    tetrode_pitch_um: float,
    tetrodes_per_row: int,
    tetrode_offsets,
    scale_to_uv: bool = False,
    debug: bool = False,
    label: str | None = None,
):
    """Attach probe geometry when enabled; otherwise return recording unchanged."""
    if not attach_geometry or not groups:
        return recording
    out = ensure_geom_and_units(
        recording,
        groups,
        pitch=tetrode_pitch_um,
        tetrodes_per_row=tetrodes_per_row,
        tetrode_offsets=tetrode_offsets,
        scale_to_uv=scale_to_uv,
    )
    if debug:
        label_text = f" ({label})" if label else ""
        log_info(f"Geometry attach{label_text}: groups={len(groups)}.")
    return out


def prepare_uv_scaling_params(recording, gains, offsets, *, label: str):
    """Validate gain/offset arrays and convert likely V/bit gains to uV/bit."""
    n_ch = recording.get_num_channels()

    use_oebin_gain = False
    oebin_gains = None
    try:
        oebin_gains = recording.get_property("oe_gain_to_uV")
    except Exception:
        oebin_gains = None
    if oebin_gains is not None:
        try:
            oebin_arr = np.asarray(oebin_gains, dtype="float64")
            if oebin_arr.ndim == 0:
                oebin_arr = np.full(n_ch, float(oebin_arr), dtype="float64")
            elif oebin_arr.size == n_ch:
                oebin_arr = oebin_arr.reshape(n_ch).astype("float64", copy=False)
            else:
                oebin_arr = None
            if oebin_arr is not None and np.all(np.isfinite(oebin_arr)) and float(np.max(np.abs(oebin_arr))) > 0.0:
                gains_arr = oebin_arr
                use_oebin_gain = True
                log_info(
                    f"{label}: using oe_gain_to_uV from structure.oebin "
                    f"(median={float(np.median(np.abs(gains_arr))):.3e} uV/bit)."
                )
            else:
                log_info(f"{label}: oe_gain_to_uV present but invalid; falling back to channel gains metadata.")
        except Exception:
            log_info(f"{label}: failed to parse oe_gain_to_uV; falling back to channel gains metadata.")

    converted_from_volts = False
    if not use_oebin_gain:
        gains_arr = np.asarray(gains, dtype="float64")
        if gains_arr.ndim == 0:
            gains_arr = np.full(n_ch, float(gains_arr), dtype="float64")
        elif gains_arr.size == n_ch:
            gains_arr = gains_arr.reshape(n_ch).astype("float64", copy=False)
        else:
            log_info(f"{label}: channel gains length {gains_arr.size} != n_channels {n_ch}; leaving native units.")
            return None, None

        if not np.all(np.isfinite(gains_arr)):
            log_info(f"{label}: channel gains contain NaN/Inf; leaving native units.")
            return None, None

        abs_gains = np.abs(gains_arr)
        gmin = float(np.min(abs_gains))
        gmed = float(np.median(abs_gains))
        gmax = float(np.max(abs_gains))
        log_info(f"{label}: channel gain stats (abs): min={gmin:.3e}, median={gmed:.3e}, max={gmax:.3e}")

        if gmax == 0.0:
            log_info(f"{label}: all channel gains are zero; leaving native units.")
            return None, None

        # Open Ephys/Intan often provide V/bit (~1e-7 to 1e-5). Convert to uV/bit.
        if 1e-9 <= gmed <= 1e-4:
            gains_arr = gains_arr * 1e6
            converted_from_volts = True
            converted_median = float(np.median(np.abs(gains_arr)))
            log_info(
                f"{label}: gains look like V/bit; multiplied by 1e6 to convert to uV/bit "
                f"(new median={converted_median:.3e})."
            )
        else:
            log_info(f"{label}: gains assumed to already be in uV/bit.")

    if offsets is None:
        offsets_arr = np.zeros(n_ch, dtype="float64")
        log_info(f"{label}: channel offsets missing; using zeros.")
    else:
        offsets_arr = np.asarray(offsets, dtype="float64")
        if offsets_arr.ndim == 0:
            offsets_arr = np.full(n_ch, float(offsets_arr), dtype="float64")
        elif offsets_arr.size == n_ch:
            offsets_arr = offsets_arr.reshape(n_ch).astype("float64", copy=False)
        else:
            offsets_arr = np.zeros(n_ch, dtype="float64")
            log_info(f"{label}: channel offsets length mismatch; using zeros.")
        if not np.all(np.isfinite(offsets_arr)):
            offsets_arr = np.zeros(n_ch, dtype="float64")
            log_info(f"{label}: channel offsets contain NaN/Inf; using zeros.")

    if use_oebin_gain and np.any(offsets_arr != 0):
        log_info(f"{label}: non-zero channel offsets ignored when using oe_gain_to_uV.")
        offsets_arr = np.zeros(n_ch, dtype="float64")

    if converted_from_volts and np.any(offsets_arr != 0):
        offsets_arr = offsets_arr * 1e6
        log_info(f"{label}: offsets multiplied by 1e6 to match converted gain units.")

    return gains_arr.astype("float32"), offsets_arr.astype("float32")


def maybe_scale_recording_to_uv(
    recording,
    *,
    label: str,
    missing_message: str,
    success_message: str,
):
    """Best-effort scale recording to uV using channel gain/offset metadata."""
    try:
        gains = recording.get_channel_gains()
    except Exception:
        gains = None
    try:
        offsets = recording.get_channel_offsets()
    except Exception:
        offsets = None
    if gains is None:
        log_info(missing_message)
        return recording

    gains_arr, offsets_arr = prepare_uv_scaling_params(
        recording,
        gains,
        offsets,
        label=label,
    )
    if gains_arr is not None:
        spre = _load_spre()
        recording = spre.scale(recording, gain=gains_arr, offset=offsets_arr, dtype="float32")
        log_info(success_message)
    return recording


def attach_oe_index_from_oebin(recording, data_path: Path, stream_name: str) -> None:
    """Attach `oe_channel_index` from `structure.oebin` in place when available.

    Gotcha: stored index is stream-order position, not guaranteed hardware channel number.

    Example:
        attach_oe_index_from_oebin(rec, data_path, "Record Node 125#Acquisition_Board-100.Rhythm Data")
    """
    try:
        prop_keys = recording.get_property_keys()
    except Exception as exc:
        log_warn(f"could not inspect recording property keys; skipping OE index attach ({exc}).")
        prop_keys = []
    if "oe_channel_index" in prop_keys:
        return

    oebin_path = Path(data_path) / "structure.oebin"
    if not oebin_path.exists():
        return

    try:
        meta = json.loads(oebin_path.read_text())
        continuous = meta.get("continuous", [])
        stream_entry = _select_oebin_stream_entry(continuous, stream_name)
        if stream_entry is None:
            return

        channels_meta = stream_entry.get("channels", [])
        if not isinstance(channels_meta, list) or not channels_meta:
            return

        name_to_idx = {}
        for idx, ch_meta in enumerate(channels_meta):
            name = ch_meta.get("channel_name")
            if isinstance(name, str) and name:
                name_to_idx[name] = idx
        if not name_to_idx:
            return

        ch_ids = list(recording.channel_ids)
        try:
            labels = list(recording.get_property("channel_name")) if "channel_name" in prop_keys else None
        except Exception as exc:
            log_warn(f"failed to read channel_name property; using channel_ids as labels ({exc}).")
            labels = None
        if labels is None or len(labels) != len(ch_ids):
            labels = [str(ch) for ch in ch_ids]

        oe_idx = []
        for ch, label in zip(ch_ids, labels):
            idx = name_to_idx.get(str(label))
            if idx is None:
                idx = name_to_idx.get(str(ch))
            if idx is None:
                return
            oe_idx.append(int(idx))

        recording.set_property("oe_channel_index", np.asarray(oe_idx, dtype=np.int32))
        log_info("OE index map: attached oe_channel_index from structure.oebin stream order.")
    except Exception as exc:
        log_warn(f"failed to attach oe_channel_index from structure.oebin: {exc}")


def attach_oe_gain_to_uv_from_oebin(recording, data_path: Path, stream_name: str) -> None:
    """Attach per-channel `oe_gain_to_uV` from `structure.oebin` in place when available.

    Gotcha: this is intended for explicit uV scaling and does not alter trace data.

    Example:
        attach_oe_gain_to_uv_from_oebin(rec, data_path, "Record Node 125#Acquisition_Board-100.Rhythm Data")
    """
    try:
        prop_keys = recording.get_property_keys()
    except Exception as exc:
        log_warn(f"could not inspect recording property keys; skipping OE gain attach ({exc}).")
        prop_keys = []
    if "oe_gain_to_uV" in prop_keys:
        return

    oebin_path = Path(data_path) / "structure.oebin"
    if not oebin_path.exists():
        return

    def _bit_volts_to_gain_uv(value):
        """Convert `bit_volts` to uV/bit with heuristic unit handling.

        Open Ephys metadata can appear in either:
        - uV/bit (commonly around 0.195), or
        - V/bit (commonly around 1.95e-7)
        """
        try:
            raw = float(value)
        except Exception:
            return None, None
        if not np.isfinite(raw) or raw == 0.0:
            return None, None
        abs_raw = abs(raw)
        # Likely V/bit -> convert to uV/bit.
        if 1e-9 <= abs_raw <= 1e-4:
            return raw * 1e6, "v_per_bit"
        # Likely already uV/bit.
        return raw, "uv_per_bit"

    try:
        meta = json.loads(oebin_path.read_text())
        continuous = meta.get("continuous", [])
        stream_entry = _select_oebin_stream_entry(continuous, stream_name)
        if stream_entry is None:
            return

        channels_meta = stream_entry.get("channels", [])
        if not isinstance(channels_meta, list) or not channels_meta:
            return

        name_to_gain_uv = {}
        metadata_gain_count = 0
        converted_count = 0
        direct_count = 0
        for ch_meta in channels_meta:
            name = ch_meta.get("channel_name")
            bit_volts = ch_meta.get("bit_volts")
            if not isinstance(name, str) or not name:
                continue
            if bit_volts is None:
                continue
            gain_uv, mode = _bit_volts_to_gain_uv(bit_volts)
            if gain_uv is None:
                continue
            if np.isfinite(gain_uv) and gain_uv != 0.0:
                name_to_gain_uv[name] = gain_uv
                metadata_gain_count += 1
                if mode == "v_per_bit":
                    converted_count += 1
                elif mode == "uv_per_bit":
                    direct_count += 1
        if not name_to_gain_uv:
            return

        ch_ids = list(recording.channel_ids)
        try:
            labels = list(recording.get_property("channel_name")) if "channel_name" in prop_keys else None
        except Exception as exc:
            log_warn(f"failed to read channel_name property; using channel_ids as labels ({exc}).")
            labels = None
        if labels is None or len(labels) != len(ch_ids):
            labels = [str(ch) for ch in ch_ids]

        gain_values = []
        for ch, label in zip(ch_ids, labels):
            gain_uv = name_to_gain_uv.get(str(label))
            if gain_uv is None:
                gain_uv = name_to_gain_uv.get(str(ch))
            if gain_uv is None:
                return
            gain_values.append(float(gain_uv))

        gains_arr = np.asarray(gain_values, dtype=np.float32)
        recording.set_property("oe_gain_to_uV", gains_arr)
        if converted_count and direct_count:
            log_warn(
                "OE gain map: mixed bit_volts units detected in structure.oebin "
                f"(converted={converted_count}, direct={direct_count})."
            )
        elif converted_count:
            log_info(
                "OE gain map: converted bit_volts from V/bit to uV/bit "
                f"(metadata channels={metadata_gain_count}, recording channels={len(gains_arr)})."
            )
        elif direct_count:
            log_info(
                "OE gain map: interpreted bit_volts as uV/bit "
                f"(metadata channels={metadata_gain_count}, recording channels={len(gains_arr)})."
            )
        log_info(
            "OE gain map: attached oe_gain_to_uV from structure.oebin "
            f"(median={float(np.median(np.abs(gains_arr))):.3e} uV/bit)."
        )
    except Exception as exc:
        log_warn(f"failed to attach oe_gain_to_uV from structure.oebin: {exc}")


__all__ = [
    "discover_oe_stream_names",
    "normalize_stream_name",
    "attach_geom",
    "attach_geometry_if_needed",
    "attach_single_grid_geom",
    "set_group_property",
    "ensure_probe_attached",
    "ensure_geom_and_units",
    "prepare_uv_scaling_params",
    "maybe_scale_recording_to_uv",
    "attach_oe_index_from_oebin",
    "attach_oe_gain_to_uv_from_oebin",
]
