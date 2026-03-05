"""Phy export utilities for spike sorting pipelines.

This module provides functions for exporting spike sorting results to Phy format
with proper channel ID mapping and traceview geometry.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .fs_utils import safe_rmtree
from .run_utils import log_info, log_warn


def _load_export_to_phy():
    """Lazy-load export_to_phy so module import stays lightweight."""
    try:
        from spikeinterface.exporters import export_to_phy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface is required for Phy export. "
            "Activate the spikeinterface environment before exporting."
        ) from exc
    return export_to_phy


def _read_is_filtered_annotation(recording):
    """Best-effort read of recording annotation `is_filtered`."""
    try:
        return recording.get_annotation("is_filtered")
    except Exception:
        pass
    try:
        annotations = getattr(recording, "_annotations", None)
        if isinstance(annotations, dict) and "is_filtered" in annotations:
            return annotations.get("is_filtered")
    except Exception:
        pass
    return None


def _set_params_bool(params_path: Path, key: str, value: bool):
    """Set or append a boolean key in Phy params.py."""
    try:
        text = params_path.read_text(encoding="utf-8")
    except Exception as exc:
        log_warn(f"could not read {params_path.name} to set {key}: {exc}")
        return
    replacement = f"{key} = {str(bool(value))}"
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        if text and not text.endswith("\n"):
            text += "\n"
        text += replacement + "\n"
    try:
        params_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        log_warn(f"could not write {params_path.name} to set {key}: {exc}")


def _write_channel_id_map(
    folder: Path,
    channel_ids_rec,
    lookup_index,
) -> None:
    """Write phy_idx -> OE mapping file for traceability."""
    try:
        mapping_path = folder / "channel_id_map.tsv"
        with mapping_path.open("w", encoding="utf-8") as f:
            f.write("phy_idx\toe_index\toe_label\n")
            for phy_idx, ch in enumerate(channel_ids_rec):
                oe_index = lookup_index(ch, phy_idx)
                ch_label = str(ch)
                f.write(f"{phy_idx}\t{oe_index}\t{ch_label}\n")
        log_info(f"Phy mapping: {mapping_path}")
    except Exception as exc:
        log_warn(f"failed to write channel_id_map.tsv: {exc}")

def _apply_single_template_phy_workaround(folder: Path) -> None:
    """Work around phylib AssertionError for single-template exports.

    Some phylib builds squeeze singleton dimensions when reading arrays.
    With one template, template_ind.npy can become shape-mismatched and trip:
    assert cols.shape == (n_templates, n_channels_loc)

    For n_templates == 1, remove template_ind.npy so Phy falls back to dense
    template loading.
    """
    template_ind_path = folder / "template_ind.npy"
    templates_path = folder / "templates.npy"
    if not template_ind_path.exists() or not templates_path.exists():
        return
    try:
        templates = np.load(templates_path, mmap_mode="r")
    except Exception as exc:
        log_warn(f"could not inspect templates.npy for single-template workaround: {exc}")
        return

    if templates.ndim < 1 or int(templates.shape[0]) != 1:
        return

    backup_path = folder / "template_ind.single_template_backup.npy"
    try:
        if backup_path.exists():
            backup_path.unlink()
        template_ind_path.replace(backup_path)
        log_warn(
            "Phy single-template workaround applied: moved template_ind.npy to "
            "template_ind.single_template_backup.npy."
        )
    except Exception as exc:
        log_warn(f"failed to apply single-template workaround: {exc}")

def build_phy_export_sparsity(analyzer, export_sparse_channels: int | None):
    """Build optional Phy export sparsity from config (best channels per unit).

    Args:
        analyzer: SortingAnalyzer object.
        export_sparse_channels: Number of channels per unit for sparse export.
            None or <=0 for dense export.

    Returns:
        ChannelSparsity object or None for dense export.
    """
    if export_sparse_channels is None:
        return None

    try:
        n_sparse = int(export_sparse_channels)
    except Exception:
        log_warn(
            f"export_sparse_channels={export_sparse_channels!r} is invalid; "
            "using dense Phy export."
        )
        return None

    if n_sparse <= 0:
        log_info("Phy export: dense mode (export_sparse_channels<=0).")
        return None

    try:
        n_channels = int(analyzer.recording.get_num_channels())
    except Exception:
        n_channels = n_sparse
    n_use = max(1, min(n_sparse, n_channels))
    if n_use < n_sparse:
        log_warn(
            f"Phy export: clamped export_sparse_channels from {n_sparse} to {n_use} "
            f"(recording has {n_channels} channels)."
        )

    try:
        from spikeinterface.core import ChannelSparsity
        sparsity = ChannelSparsity.from_best_channels(analyzer, num_channels=n_use, peak_sign="neg")
        log_info(f"Phy export sparsity: top {n_use} channel(s) per unit.")
        return sparsity
    except Exception as exc:
        log_warn(f"failed to build Phy export sparsity ({exc}); using dense export.")
        return None


def export_for_phy(
    analyzer,
    base_folder: Path,
    label: str,
    groups,
    original_index_map: dict,
    oe_index_map: dict,
    *,
    group_ids=None,
    group_sizes_by_id=None,
    simple_phy_export: bool | None = None,
    export_channel_ids_mode: str = "oe_label",
    linearize_traceview: bool = True,
    traceview_contact_spacing_um: float = 20.0,
    traceview_group_spacing_um: float = 200.0,
    export_sparse_channels: int | None = None,
    label_stereotrodes_as_mua: bool = False,
    stereotrode_mua_label: str = "mua",
    default_cluster_label: str = "unsorted",
    scaled_to_uv: bool = True,
    scale_to_uv: bool | None = None,
    phy_hp_filtered: bool | None = None,
):
    """Export analyzer to Phy and rewrite channel_ids mapping when configured.

    Args:
        analyzer: SortingAnalyzer object.
        base_folder: Output base folder.
        label: Label for the export folder.
        groups: List of channel groups.
        original_index_map: Mapping of channel IDs to original indices.
        oe_index_map: Mapping of channel IDs to OE indices.
        group_ids: Optional list of group IDs.
        group_sizes_by_id: Optional mapping of group IDs to sizes.
        simple_phy_export: Force simple export mode (None=auto).
        export_channel_ids_mode: Channel ID mode ("oe_label", "oe_index", "as_exported").
        linearize_traceview: Whether to linearize traceview.
        traceview_contact_spacing_um: Contact spacing for traceview.
        traceview_group_spacing_um: Group spacing for traceview.
        export_sparse_channels: Number of sparse channels per unit.
        label_stereotrodes_as_mua: Whether to label stereotrode clusters as MUA.
        stereotrode_mua_label: Label for stereotrode MUA clusters.
        default_cluster_label: Default cluster label.
        scaled_to_uv: Whether the upstream export/analyzer recording has already
            been scaled to microvolts.
        scale_to_uv: Deprecated alias for `scaled_to_uv` (backward compatibility).
        phy_hp_filtered: Optional explicit value to write as `hp_filtered` in
            exported `params.py`. None infers from recording annotation.

    Returns:
        Tuple of (export_folder, group_unique).

    Example:
        folder, _ = export_for_phy(
            analyzer,
            base_folder=Path("sc2_outputs"),
            label="sc2",
            groups=groups,
            original_index_map=original_index_map,
            oe_index_map=oe_index_map,
            export_channel_ids_mode="oe_label",
        )
    """
    export_to_phy = _load_export_to_phy()
    folder = base_folder / f"phy_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_sparsity = build_phy_export_sparsity(analyzer, export_sparse_channels)
    # copy_binary=True so Phy extracts snippets from the exported binary, not the raw recording.
    export_kwargs = dict(
        output_folder=folder,
        remove_if_exists=True,
        copy_binary=True,
        sparsity=export_sparsity,
    )
    export_recording = analyzer.recording
    export_source = "analyzer.recording"
    if scale_to_uv is not None:
        scaled_to_uv = bool(scale_to_uv)
    inferred_hp_filtered = _read_is_filtered_annotation(export_recording)
    resolved_hp_filtered = (
        bool(phy_hp_filtered)
        if phy_hp_filtered is not None
        else (bool(inferred_hp_filtered) if inferred_hp_filtered is not None else None)
    )
    export_to_phy(analyzer, **export_kwargs)
    log_info(
        f"Phy export source: {export_source} | scaled_to_uV={scaled_to_uv}"
    )
    if resolved_hp_filtered is not None:
        log_info(f"Phy export flag: hp_filtered={resolved_hp_filtered}")
    else:
        log_warn("Phy export flag: hp_filtered unresolved (keeping exporter default).")
    if simple_phy_export is None:
        simple_flag = groups is not None and len(groups) <= 1
    else:
        simple_flag = simple_phy_export
    # Simple Phy export = keep Phy default contiguous channel_ids for a single-group run (no mapping rewrite).
    simple_export = bool(simple_flag) and groups is not None and len(groups) <= 1
    params_path = folder / "params.py"

    cache_dir = folder / ".phy"
    if cache_dir.exists():
        safe_rmtree(cache_dir)

    channel_ids_rec = list(export_recording.channel_ids)
    group_unique = None
    try:
        group_prop_check = export_recording.get_property("group")
        if group_prop_check is None:
            log_warn("export recording has no 'group' property.")
        else:
            group_unique = np.unique(group_prop_check)
    except Exception as exc:
        log_warn(f"could not read 'group' property: {exc}")

    def lookup_index(ch, fallback):
        if ch in oe_index_map:
            return int(oe_index_map[ch])
        ch_str = str(ch)
        if ch_str in oe_index_map:
            return int(oe_index_map[ch_str])
        if ch in original_index_map:
            return int(original_index_map[ch])
        if ch_str in original_index_map:
            return int(original_index_map[ch_str])
        return int(fallback)

    # Always emit mapping file before any early-return branch.
    _write_channel_id_map(folder, channel_ids_rec, lookup_index)

    if simple_export and export_channel_ids_mode == "as_exported":
        if params_path.exists() and resolved_hp_filtered is not None:
            _set_params_bool(params_path, "hp_filtered", resolved_hp_filtered)
        _apply_single_template_phy_workaround(folder)
        return folder, None

    if export_channel_ids_mode not in ("oe_index", "oe_label", "as_exported"):
        log_warn(
            f"export_channel_ids_mode={export_channel_ids_mode!r} is invalid; "
            "falling back to 'oe_index'."
        )
        export_ids_mode = "oe_index"
    else:
        export_ids_mode = export_channel_ids_mode

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
        # and channel_ids for labels; here "oe_index" prefers the numeric OE index when available.
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
                log_warn(f"could not overwrite channel_ids.npy: {exc}")
        if simple_export:
            params_path.write_text(text)
            if resolved_hp_filtered is not None:
                _set_params_bool(params_path, "hp_filtered", resolved_hp_filtered)
            _apply_single_template_phy_workaround(folder)
            return folder, group_unique
        pattern_map = re.compile(r"channel_map\s*=.*")
        replacement_map = f"channel_map = np.array({channel_map.tolist()}, dtype=np.int32)"
        text, count_map = pattern_map.subn(replacement_map, text, count=1)
        if not count_map:
            text += f"\n{replacement_map}\n"
        try:
            np.save(folder / "channel_map.npy", channel_map.astype(np.int32))
        except Exception as exc:
            log_warn(f"could not overwrite channel_map.npy: {exc}")

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

    if linearize_traceview:
        contact_spacing = float(traceview_contact_spacing_um)
        group_spacing = float(traceview_group_spacing_um)
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
                log_warn(f"could not overwrite {name}: {exc}")

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
            log_warn(f"could not overwrite channel_groups.npy: {exc}")
        try:
            np.save(folder / "channel_shanks.npy", channel_shanks_out.astype(np.int32))
        except Exception as exc:
            log_warn(f"could not overwrite channel_shanks.npy: {exc}")
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
                log_warn("template_ind shape mismatch; using peak_local indices.")
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
        log_warn(f"could not recompute cluster channel groups: {exc}")

    # Optionally pre-label stereotrode clusters as MUA for Phy curation.
    try:
        if label_stereotrodes_as_mua and cluster_channel_groups is not None:
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
                    label_out = stereotrode_mua_label if group_val in stereotrode_groups else default_cluster_label
                    f.write(f"{cid}\t{label_out}\n")
            log_info(
                f"Labeled stereotrode clusters as '{stereotrode_mua_label}' in cluster_group.tsv "
                f"({len(stereotrode_groups)} stereotrode groups)."
            )
    except Exception as exc:
        log_warn(f"could not label stereotrode clusters as MUA: {exc}")

    if params_path.exists() and resolved_hp_filtered is not None:
        _set_params_bool(params_path, "hp_filtered", resolved_hp_filtered)

    _apply_single_template_phy_workaround(folder)
    return folder, group_unique


__all__ = [
    "build_phy_export_sparsity",
    "export_for_phy",
]

