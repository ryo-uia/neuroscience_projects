"""Configuration helpers for pipeline env overrides and config echo output."""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np

from .run_utils import log_info, log_warn


def apply_env_overrides_from_env(
    *,
    env_name: str,
    allowed_keys: set[str],
    module_globals: dict,
    bool_keys: Iterable[str] = (),
    str_or_none_keys: Iterable[str] = (),
    enum_keys: dict[str, set[str]] | None = None,
    positive_number_or_none_keys: Iterable[str] = (),
):
    """Apply allow-listed JSON overrides from environment to module globals."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return {}, []
    try:
        data = json.loads(raw)
    except Exception as exc:
        log_warn(f"invalid {env_name} JSON; ignoring overrides ({exc}).")
        return {}, []
    if not isinstance(data, dict):
        log_warn(f"{env_name} must be a JSON object; ignoring overrides.")
        return {}, []

    bool_key_set = set(bool_keys)
    str_or_none_key_set = set(str_or_none_keys)
    pos_or_none_key_set = set(positive_number_or_none_keys)
    enum_key_map = enum_keys or {}

    def _validate_override_value(key: str, value):
        if key in bool_key_set:
            if isinstance(value, bool):
                return True, value, ""
            return False, None, "must be a JSON boolean (true/false)"

        if key in str_or_none_key_set:
            if value is None or isinstance(value, str):
                return True, value, ""
            return False, None, "must be a string or null"

        if key in pos_or_none_key_set:
            if value is None:
                return True, None, ""
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    as_float = float(value)
                except Exception:
                    return False, None, "must be null or a positive number"
                if np.isfinite(as_float) and as_float > 0:
                    return True, value, ""
            return False, None, "must be null or a positive number"

        if key in enum_key_map:
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in enum_key_map[key]:
                    return True, normalized, ""
            allowed = sorted(enum_key_map[key])
            return False, None, f"must be one of {allowed}"

        return True, value, ""

    applied = {}
    ignored = []
    for key, value in data.items():
        if key in allowed_keys and key in module_globals:
            valid, normalized_value, reason = _validate_override_value(key, value)
            if not valid:
                log_warn(f"Ignoring override '{key}': {reason} (got {value!r})")
                ignored.append(key)
                continue
            module_globals[key] = normalized_value
            applied[key] = normalized_value
        else:
            ignored.append(key)
    return applied, ignored


def print_pipeline_config_echo(
    *,
    raw_mode,
    use_si_preprocess,
    use_sc2_preprocess,
    si_apply_whiten,
    si_apply_car,
    apply_notch,
    car_mode,
    attach_geometry,
    export_scale_to_uv,
    analyzer_from_sorter,
    auto_bad_channels,
    strict_groups,
    sort_by_group,
    use_config_jsons,
    test_seconds,
    stream_name,
):
    """Emit standardized config echo lines."""
    log_info(
        "Config (preprocess): RAW_MODE= "
        f"{raw_mode} USE_SI_PREPROCESS= {use_si_preprocess} USE_SC2_PREPROCESS= {use_sc2_preprocess} "
        f"SI_APPLY_WHITEN= {si_apply_whiten} SI_APPLY_CAR= {si_apply_car} APPLY_NOTCH= {apply_notch} "
        f"CAR_MODE= {car_mode}"
    )
    log_info(
        "Config (geometry): ATTACH_GEOMETRY= "
        f"{attach_geometry}"
    )
    log_info(
        "Config (export): EXPORT_SCALE_TO_UV= "
        f"{export_scale_to_uv} ANALYZER_FROM_SORTER= {analyzer_from_sorter}"
    )
    log_info(
        "Config (session/groups): AUTO_BAD_CHANNELS= "
        f"{auto_bad_channels} STRICT_GROUPS= {strict_groups} SORT_BY_GROUP= {sort_by_group} "
        f"USE_CONFIG_JSONS= {use_config_jsons} TEST_SECONDS= {test_seconds} STREAM_NAME= {stream_name or 'auto'}"
    )


__all__ = [
    "apply_env_overrides_from_env",
    "print_pipeline_config_echo",
]

