import ast
import re
from pathlib import Path

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from probeinterface import Probe


def discover_oe_stream_names(folder):
    """Return the list of stream names declared in an Open Ephys folder."""
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
            except Exception:
                pass
        raise


def attach_geom(recording, groups, tetrodes_per_row=None, pitch=20.0, dx=150.0, dy=150.0, tetrode_offsets=None):
    """Attach a Probe with simple 2x2 tetrode layout."""
    groups = list(groups)
    if tetrode_offsets is not None and len(tetrode_offsets) != len(groups):
        raise ValueError("tetrode_offsets must match number of tetrodes")

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
            positions[index_map[ch]] = base[j] + offset

    probe = Probe(ndim=2)
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7})
    device_idx = np.array([index_map[ch] for ch in recording.channel_ids], dtype=int)
    probe.set_device_channel_indices(device_idx)
    return recording.set_probe(probe)


def set_group_property(recording, groups, group_ids=None):
    """Store a group label as a channel property named 'group' (labels may be non-contiguous)."""
    if group_ids is None:
        group_ids = list(range(len(groups)))
    elif len(group_ids) != len(groups):
        raise ValueError("group_ids must match number of groups")
    index_map = {ch: i for i, ch in enumerate(recording.channel_ids)}
    values = np.zeros(len(recording.channel_ids), dtype=int)
    for group_label, group in zip(group_ids, groups):
        for ch in group:
            if ch in index_map:
                values[index_map[ch]] = int(group_label)
    recording.set_property("group", values)


def ensure_probe_attached(recording, radius=5):
    """Attach a Probe object if missing (uses channel locations)."""
    try:
        if recording.get_probe() is not None:
            return recording
    except Exception:
        pass
    try:
        locs = recording.get_channel_locations()
    except Exception as exc:
        raise RuntimeError(f"Cannot attach probe: channel locations unavailable ({exc})")
    probe = Probe(ndim=locs.shape[1])
    probe.set_contacts(positions=locs, shapes="circle", shape_params={"radius": radius})
    probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
    return recording.set_probe(probe, in_place=False)


def ensure_geom_and_units(recording, groups, tetrodes_per_row=None, scale_to_uv=True, tetrode_offsets=None):
    """Attach geometry and check for gain_to_uV metadata; does not rescale traces."""
    rec = attach_geom(recording, groups, tetrodes_per_row, tetrode_offsets=tetrode_offsets)
    if scale_to_uv:
        try:
            gain = rec.get_property("gain_to_uV")
        except Exception:
            gain = None
        if gain is None:
            print("Warning: gain_to_uV missing; leaving recording in native units.")
    return rec


__all__ = [
    "discover_oe_stream_names",
    "attach_geom",
    "set_group_property",
    "ensure_probe_attached",
    "ensure_geom_and_units",
]
