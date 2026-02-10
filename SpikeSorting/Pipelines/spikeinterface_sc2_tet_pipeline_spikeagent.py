"""
SpikeAgent-style entrypoint for the tetrode SC2 pipeline.

This is an example wrapper that keeps the original pipeline logic untouched and
applies a machine-readable job config before calling the existing main().
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
BASE_PIPELINE_PATH = THIS_FILE.parent / "spikeinterface_sc2_tet_pipeline.py"

# Only allow explicit, known global overrides from job config.
ALLOWED_OVERRIDES = {
    "TEST_SECONDS",
    "SESSION_SUBPATH",
    "SESSION_SELECTION",
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
    "EXPORT_TO_PHY",
    "EXPORT_TO_SI_GUI",
    "SAVE_ANALYZER",
    "EXPORT_PHY_CHANNEL_IDS_MODE",
}


def load_base_pipeline():
    spec = importlib.util.spec_from_file_location("tet_base_pipeline", BASE_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base pipeline: {BASE_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_job_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Job config must be a JSON object.")
    return data


def apply_pipeline_overrides(base, cfg: dict) -> list[str]:
    ignored = []
    overrides = cfg.get("pipeline_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("'pipeline_overrides' must be an object when provided.")

    for key, value in overrides.items():
        if key in ALLOWED_OVERRIDES and hasattr(base, key):
            setattr(base, key, value)
        else:
            ignored.append(key)
    return ignored


def build_base_argv(cfg: dict) -> list[str]:
    # Map machine config -> original pipeline CLI flags.
    cli = cfg.get("cli_args", {})
    if cli is None:
        cli = {}
    if not isinstance(cli, dict):
        raise ValueError("'cli_args' must be an object when provided.")

    argv = [str(BASE_PIPELINE_PATH)]
    if cli.get("root_dir"):
        argv += ["--root-dir", str(cli["root_dir"])]
    if cli.get("base_out"):
        argv += ["--base-out", str(cli["base_out"])]
    if cli.get("channel_groups"):
        argv += ["--channel-groups", str(cli["channel_groups"])]
    if cli.get("bad_channels"):
        argv += ["--bad-channels", str(cli["bad_channels"])]
    if cli.get("no_config_json", False):
        argv += ["--no-config-json"]
    return argv


def print_effective_summary(base, cfg: dict, ignored_overrides: list[str], base_argv: list[str]) -> None:
    print("SpikeAgent example wrapper (tetrode)")
    print(f"Base pipeline: {BASE_PIPELINE_PATH}")
    print(f"Job name: {cfg.get('job_name', 'unnamed')}")
    print("Resolved base argv:", " ".join(base_argv))
    print(
        "Effective pipeline settings:",
        {
            "TEST_SECONDS": getattr(base, "TEST_SECONDS", None),
            "SESSION_SUBPATH": getattr(base, "SESSION_SUBPATH", None),
            "SESSION_SELECTION": getattr(base, "SESSION_SELECTION", None),
            "STREAM_NAME": getattr(base, "STREAM_NAME", None),
            "USE_CONFIG_JSONS": getattr(base, "USE_CONFIG_JSONS", None),
            "CHANNEL_GROUPS_PATH": getattr(base, "CHANNEL_GROUPS_PATH", None),
            "BAD_CHANNELS_PATH": getattr(base, "BAD_CHANNELS_PATH", None),
            "SORT_BY_GROUP": getattr(base, "SORT_BY_GROUP", None),
            "AUTO_BAD_CHANNELS": getattr(base, "AUTO_BAD_CHANNELS", None),
            "USE_SI_PREPROCESS": getattr(base, "USE_SI_PREPROCESS", None),
            "SI_APPLY_CAR": getattr(base, "SI_APPLY_CAR", None),
            "SI_APPLY_WHITEN": getattr(base, "SI_APPLY_WHITEN", None),
            "EXPORT_TO_PHY": getattr(base, "EXPORT_TO_PHY", None),
            "EXPORT_TO_SI_GUI": getattr(base, "EXPORT_TO_SI_GUI", None),
            "EXPORT_PHY_CHANNEL_IDS_MODE": getattr(base, "EXPORT_PHY_CHANNEL_IDS_MODE", None),
        },
    )
    if ignored_overrides:
        print("Ignored unknown/blocked pipeline_overrides:", sorted(ignored_overrides))


def main() -> None:
    parser = argparse.ArgumentParser(description="SpikeAgent example wrapper for tetrode SC2 pipeline")
    parser.add_argument(
        "--job-config",
        type=Path,
        required=True,
        help="JSON job file with 'pipeline_overrides' and optional 'cli_args'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved config and exit without sorting.",
    )
    args = parser.parse_args()

    cfg = load_job_config(args.job_config)
    base = load_base_pipeline()
    ignored = apply_pipeline_overrides(base, cfg)
    base_argv = build_base_argv(cfg)

    print_effective_summary(base, cfg, ignored, base_argv)
    if args.dry_run:
        print("Dry-run complete (no sorting started).")
        return

    # Call original pipeline entrypoint with synthetic argv.
    orig_argv = sys.argv
    try:
        sys.argv = base_argv
        base.main()
    finally:
        sys.argv = orig_argv


if __name__ == "__main__":
    main()
