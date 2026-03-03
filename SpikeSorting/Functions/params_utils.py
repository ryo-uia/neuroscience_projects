"""Parameter dictionary utilities for sorter configuration."""

from __future__ import annotations

from copy import deepcopy

from .run_utils import log_warn


def merge_params(defaults: dict, overrides: dict) -> dict:
    """Deep-merge overrides into defaults without mutating the original dict."""
    if not overrides:
        return deepcopy(defaults)
    merged = deepcopy(defaults)

    def _apply(target: dict, source: dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _apply(target[key], value)
            else:
                target[key] = value

    _apply(merged, overrides)
    return merged


def warn_unknown_sc2_overrides(defaults: dict, overrides: dict, prefix: str = "") -> None:
    """Warn when SC2 overrides include keys not present in default params."""
    if not isinstance(overrides, dict):
        return
    for key, value in overrides.items():
        path = f"{prefix}{key}"
        if key not in defaults:
            log_warn(f"SC2 override '{path}' not in default params; it may be ignored.")
            continue
        default_val = defaults.get(key)
        if isinstance(value, dict):
            if isinstance(default_val, dict):
                warn_unknown_sc2_overrides(default_val, value, prefix=f"{path}.")
            else:
                log_warn(f"SC2 override '{path}' is a dict but default is not; it may be ignored.")


__all__ = [
    "merge_params",
    "warn_unknown_sc2_overrides",
]
