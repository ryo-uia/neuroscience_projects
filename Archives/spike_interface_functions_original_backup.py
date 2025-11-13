import re
import ast
from pathlib import Path

import numpy as np
import spikeinterface.extractors as se
from probeinterface import Probe


def discover_oe_stream_names(folder: Path) -> list[str]:
    """
    Return list of OE stream names for this folder.
    Works even when SI raises on multi-stream by parsing the error message.
    """
    try:
        rec = se.read_openephys(folder)  # may raise if multiple streams
        # If we got here, there was only one stream or SI allowed opening without specifying.
        # Try public accessor; fall back to _annotations.
        get_ann = getattr(rec, "get_annotation", None)
        if callable(get_ann):
            names = get_ann("streams_names") or get_ann("stream_names") or []
        else:
            ann = getattr(rec, "_annotations", {}) or {}
            names = ann.get("streams_names") or ann.get("stream_names") or []
        # Some SI versions return empty here for single-stream; then get from rec.
        if not names:
            # As a last resort, print repr and try to infer; otherwise assume single.
            names = [getattr(rec, "stream_name", None)] if hasattr(rec, "stream_name") else []
            names = [n for n in names if n]
        return names
    except ValueError as e:
        msg = str(e)
        # Typical message contains a Python list right after `stream_names`:
        # `stream_names`: ['Record Node ... Rhythm Data', 'Record Node ... Rhythm Data_ADC']
        m = re.search(r"`stream_names`:\s*(\[[^\]]+\])", msg)
        if m:
            try:
                names = ast.literal_eval(m.group(1))
                return names
            except Exception:
                pass
        # If we can't parse, re-raise with context.
        raise


def attach_geom(recording, groups, tetrodes_per_row, pitch=20.0, dx=150.0, dy=150.0):
    # 4-ch layout inside a tetrode
    idx_map = {ch: i for i, ch in enumerate(recording.channel_ids)}
    pos = np.zeros((len(idx_map), 2), dtype=float)
    for t, g in enumerate(groups):
        base_xy = np.array([[0, 0], [pitch, 0], [0, pitch], [pitch, pitch]], dtype=float)[: len(g)]
        row, col = divmod(t, tetrodes_per_row)
        offset = np.array([col * dx, row * dy], dtype=float)
        for j, ch in enumerate(g):
            pos[idx_map[ch]] = base_xy[j] + offset
    pr = Probe(ndim=2)
    pr.set_contacts(
        positions=pos,
        shapes="circle",
        shape_params={"radius": 7},
    )
    # VERY IMPORTANT for SI: map probe contacts to device channel indices
    device_inds = np.array([idx_map[ch] for ch in recording.channel_ids], dtype=int)
    pr.set_device_channel_indices(device_inds)
    recording = recording.set_probe(pr)
    return recording

