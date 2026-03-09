"""Regression tests for channel/session helper edge-case behavior."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Functions.channel_utils import (
    load_bad_channels_from_path,
    load_channel_groups_from_path,
)
from Functions.session_utils import pick_stream
from Functions.testing_utils import workspace_tempdir


class TestChannelUtilsJsonSemantics(unittest.TestCase):
    def test_empty_channel_groups_json_is_explicit_empty(self) -> None:
        with workspace_tempdir("channel_groups_json") as tmp:
            p = tmp / "groups.json"
            p.write_text(json.dumps([]), encoding="utf-8")
            groups = load_channel_groups_from_path(p)
            self.assertEqual(groups, [])

    def test_empty_bad_channels_json_is_explicit_empty(self) -> None:
        with workspace_tempdir("bad_channels_json") as tmp:
            p = tmp / "bad.json"
            p.write_text(json.dumps([]), encoding="utf-8")
            bad = load_bad_channels_from_path(p)
            self.assertEqual(bad, [])


class TestSessionUtilsPreferredStreamValidation(unittest.TestCase):
    def test_pick_stream_rejects_unknown_preferred(self) -> None:
        streams = [
            "Record Node 125#Acquisition_Board-100.Rhythm Data",
            "Record Node 125#Acquisition_Board-100.ADC Data",
        ]
        with patch("Functions.session_utils.discover_oe_stream_names", return_value=streams):
            with self.assertRaises(RuntimeError):
                pick_stream(Path("."), "NotARealStream")

    def test_pick_stream_accepts_normalized_preferred(self) -> None:
        streams = [
            "Record Node 125#Acquisition_Board-100.Rhythm Data",
            "Record Node 125#Acquisition_Board-100.ADC Data",
        ]
        with patch("Functions.session_utils.discover_oe_stream_names", return_value=streams):
            stream = pick_stream(Path("."), "Acquisition_Board-100.Rhythm Data")
            self.assertEqual(stream, "Record Node 125#Acquisition_Board-100.Rhythm Data")


if __name__ == "__main__":
    unittest.main()
