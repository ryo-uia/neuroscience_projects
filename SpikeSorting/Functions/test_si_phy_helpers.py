"""Regression tests for SI stream matching and Phy mapping helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Functions.phy_export import _write_channel_id_map
from Functions.si_utils import _select_oebin_stream_entry


class SiUtilsStreamMatchingTests(unittest.TestCase):
    def test_exact_stream_name_match(self) -> None:
        continuous = [
            {"stream_name": "Record Node 101#Acquisition_Board-100.Rhythm Data"},
            {"stream_name": "Record Node 101#Acquisition_Board-100.ADC Data"},
        ]
        out = _select_oebin_stream_entry(
            continuous, "Record Node 101#Acquisition_Board-100.Rhythm Data"
        )
        self.assertIsNotNone(out)
        self.assertEqual(out.get("stream_name"), continuous[0]["stream_name"])

    def test_unique_normalized_match(self) -> None:
        continuous = [
            {"stream_name": "Record Node 101#Acquisition_Board-100.Rhythm Data"},
            {"stream_name": "Record Node 101#Acquisition_Board-100.ADC Data"},
        ]
        out = _select_oebin_stream_entry(continuous, "Acquisition_Board-100.Rhythm Data")
        self.assertIsNotNone(out)
        self.assertEqual(out.get("stream_name"), continuous[0]["stream_name"])

    def test_ambiguous_normalized_match_returns_none(self) -> None:
        continuous = [
            {"stream_name": "Record Node 101#Acquisition_Board-100.Rhythm Data"},
            {"stream_name": "Record Node 125#Acquisition_Board-100.Rhythm Data"},
        ]
        out = _select_oebin_stream_entry(continuous, "Acquisition_Board-100.Rhythm Data")
        self.assertIsNone(out)

    def test_no_match_returns_none(self) -> None:
        continuous = [
            {"stream_name": "Record Node 101#Acquisition_Board-100.Rhythm Data"},
            {"stream_name": "Record Node 101#Acquisition_Board-100.ADC Data"},
        ]
        out = _select_oebin_stream_entry(continuous, "Acquisition_Board-200.Rhythm Data")
        self.assertIsNone(out)


class PhyExportMappingFileTests(unittest.TestCase):
    def test_write_channel_id_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            channel_ids = ["CH40", "CH38", "CH36"]
            index_map = {"CH40": 0, "CH38": 1, "CH36": 2}

            _write_channel_id_map(
                folder,
                channel_ids,
                lambda ch, fallback: index_map.get(str(ch), fallback),
            )

            out = folder / "channel_id_map.tsv"
            self.assertTrue(out.exists())
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "phy_idx\toe_index\toe_label")
            self.assertEqual(lines[1], "0\t0\tCH40")
            self.assertEqual(lines[2], "1\t1\tCH38")
            self.assertEqual(lines[3], "2\t2\tCH36")


if __name__ == "__main__":
    unittest.main()
