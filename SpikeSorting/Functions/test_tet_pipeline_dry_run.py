"""Pipeline-level dry-run regression test for tet pipeline orchestration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipelines import spikeinterface_sc2_tet_pipeline as tet


class _FakeRecording:
    """Minimal recording stub for dry-run group/bad-channel resolution."""

    def __init__(self, channel_ids):
        self.channel_ids = list(channel_ids)

    def get_property_keys(self):
        return ["channel_name"]

    def get_property(self, key):
        if key == "channel_name":
            return list(self.channel_ids)
        raise KeyError(key)

    def get_num_channels(self):
        return len(self.channel_ids)

    def channel_slice(self, keep_channel_ids=None, channel_ids=None):
        keep = keep_channel_ids if keep_channel_ids is not None else channel_ids
        return _FakeRecording(list(keep))


class TetPipelineDryRunTests(unittest.TestCase):
    def test_dry_run_respects_cli_group_precedence_and_skips_sorting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            groups_cfg = tmp_path / "groups_config.json"
            groups_env = tmp_path / "groups_env.json"
            groups_cli = tmp_path / "groups_cli.json"

            groups_cfg.write_text(json.dumps([["CH1", "CH2"]]), encoding="utf-8")
            groups_env.write_text(json.dumps([["CH1", "CH2", "CH3"]]), encoding="utf-8")
            groups_cli.write_text(json.dumps([["CH1", "CH2", "CH3", "CH4"]]), encoding="utf-8")

            channels = ["CH1", "CH2", "CH3", "CH4"]
            rec = _FakeRecording(channels)
            original_index_map = {}
            for i, ch in enumerate(channels):
                original_index_map[ch] = i
                original_index_map[str(ch)] = i
            oe_index_map = dict(original_index_map)

            captured = {}
            original_resolve = tet.resolve_groups_and_bad_channels

            def _wrapped_resolve(*args, **kwargs):
                out = original_resolve(*args, **kwargs)
                captured["grouping_details"] = out[6]
                captured["should_exit_early"] = out[7]
                return out

            with patch.object(
                tet,
                "load_recording_with_indices",
                return_value=(
                    rec,
                    list(channels),
                    original_index_map,
                    oe_index_map,
                    list(channels),
                    tmp_path / "session",
                    "Record Node 125#Acquisition_Board-100.Rhythm Data",
                ),
            ), patch.object(
                tet,
                "reserve_run_folder",
                return_value=tmp_path / "sc2_run_test",
            ), patch.object(
                tet,
                "initialize_run_io_shared",
                return_value=(
                    tmp_path / "sc2_outputs",
                    tmp_path / "si_gui_exports",
                    None,
                    tmp_path / "run.log",
                ),
            ), patch.object(
                tet,
                "resolve_groups_and_bad_channels",
                side_effect=_wrapped_resolve,
            ), patch.object(
                tet,
                "run_sorting_stage",
                side_effect=AssertionError("run_sorting_stage must not be called in --dry-run-config"),
            ) as run_sorter_mock, patch.object(
                tet,
                "BAD_CHANNELS",
                [],
            ), patch.object(
                tet,
                "CHANNEL_GROUPS_PATH",
                groups_cfg,
            ), patch.object(
                tet.atexit,
                "register",
                lambda *a, **k: None,
            ), patch.dict(
                os.environ,
                {
                    "SPIKESORT_CHANNEL_GROUPS": str(groups_env),
                    "SPIKESORT_PIPELINE_OVERRIDES": "",
                },
                clear=False,
            ), patch.object(
                sys,
                "argv",
                [
                    "spikeinterface_sc2_tet_pipeline.py",
                    "--dry-run-config",
                    "--root-dir",
                    str(tmp_path),
                    "--base-out",
                    str(tmp_path),
                    "--channel-groups",
                    str(groups_cli),
                ],
            ):
                tet.main()

            self.assertTrue(captured.get("should_exit_early"))
            groups_source = captured["grouping_details"]["groups_source"]
            self.assertIn("CLI file", groups_source)
            self.assertIn("groups_cli.json", groups_source)
            run_sorter_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
