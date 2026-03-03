"""Compatibility layer re-exporting split pipeline helper modules.

This module preserves the historical import path:
    from Functions.pipeline_utils import ...

New code should prefer importing from:
    - Functions.session_utils
    - Functions.channel_utils
    - Functions.params_utils
    - Functions.fs_utils
"""

from __future__ import annotations

from .channel_utils import (
    build_oe_index_map,
    chunk_groups,
    detect_bad_channel_ids,
    filter_groups_with_valid_ids,
    filter_groups_with_indices,
    load_bad_channels_from_path,
    load_channel_groups_from_path,
    resolve_bad_channel_ids,
    resolve_manual_groups,
    safe_channel_slice,
)
from .fs_utils import safe_rmtree
from .params_utils import merge_params, warn_unknown_sc2_overrides
from .session_utils import (
    choose_config_json,
    choose_recording_folder,
    discover_recording_folders,
    first_seconds,
    pick_stream,
)

__all__ = [
    "pick_stream",
    "choose_config_json",
    "first_seconds",
    "safe_channel_slice",
    "chunk_groups",
    "resolve_manual_groups",
    "load_channel_groups_from_path",
    "load_bad_channels_from_path",
    "resolve_bad_channel_ids",
    "detect_bad_channel_ids",
    "merge_params",
    "discover_recording_folders",
    "choose_recording_folder",
    "safe_rmtree",
    "build_oe_index_map",
    "warn_unknown_sc2_overrides",
    "filter_groups_with_valid_ids",
    "filter_groups_with_indices",
]
