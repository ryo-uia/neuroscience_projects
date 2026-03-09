"""Regression tests for SI stream matching and Phy mapping helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Functions.phy_export import _write_channel_id_map
from Functions.si_utils import _select_oebin_stream_entry, attach_oe_gain_to_uv_from_oebin
from Functions.testing_utils import workspace_tempdir


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
        with workspace_tempdir("phy_channel_id_map") as folder:
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


class _FakeRecordingForOEBin:
    def __init__(self, channel_ids):
        self.channel_ids = list(channel_ids)
        self._properties = {"channel_name": list(channel_ids)}

    def get_property_keys(self):
        return list(self._properties.keys())

    def get_property(self, key):
        return self._properties[key]

    def set_property(self, key, value):
        self._properties[key] = value


class OEBinGainAttachTests(unittest.TestCase):
    def _write_oebin(self, root: Path, bit_volts_values):
        channels = []
        for i, bv in enumerate(bit_volts_values, start=1):
            channels.append({"channel_name": f"CH{i}", "bit_volts": bv})
        payload = {
            "continuous": [
                {
                    "stream_name": "Record Node 125#Acquisition_Board-100.Rhythm Data",
                    "channels": channels,
                }
            ]
        }
        (root / "structure.oebin").write_text(json.dumps(payload), encoding="utf-8")

    def test_bit_volts_already_uv_per_bit(self):
        with workspace_tempdir("oebin_gain_uv") as root:
            self._write_oebin(root, [0.195, 0.195, 0.195])
            rec = _FakeRecordingForOEBin(["CH1", "CH2", "CH3"])

            attach_oe_gain_to_uv_from_oebin(
                rec,
                root,
                "Record Node 125#Acquisition_Board-100.Rhythm Data",
            )

            gains = rec.get_property("oe_gain_to_uV")
            self.assertAlmostEqual(float(gains[0]), 0.195, places=6)
            self.assertAlmostEqual(float(gains[1]), 0.195, places=6)
            self.assertAlmostEqual(float(gains[2]), 0.195, places=6)

    def test_bit_volts_in_v_per_bit_is_converted(self):
        with workspace_tempdir("oebin_gain_v") as root:
            self._write_oebin(root, [1.95e-7, 1.95e-7, 1.95e-7])
            rec = _FakeRecordingForOEBin(["CH1", "CH2", "CH3"])

            attach_oe_gain_to_uv_from_oebin(
                rec,
                root,
                "Record Node 125#Acquisition_Board-100.Rhythm Data",
            )

            gains = rec.get_property("oe_gain_to_uV")
            self.assertAlmostEqual(float(gains[0]), 0.195, places=6)
            self.assertAlmostEqual(float(gains[1]), 0.195, places=6)
            self.assertAlmostEqual(float(gains[2]), 0.195, places=6)


if __name__ == "__main__":
    unittest.main()
