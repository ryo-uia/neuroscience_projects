"""Regression tests for shared run/log utility behavior."""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Functions.run_utils import reserve_run_folder
from Functions.testing_utils import workspace_tempdir


class _FixedDatetime:
    @classmethod
    def now(cls):
        return datetime(2026, 3, 8, 14, 42, 7)


class RunUtilsTests(unittest.TestCase):
    def test_reserve_run_folder_creates_directory_and_suffixes_collisions(self):
        with workspace_tempdir("run_utils") as tmp_path:
            with patch("Functions.run_utils.datetime", _FixedDatetime):
                run_a = reserve_run_folder(tmp_path)
                run_b = reserve_run_folder(tmp_path)

            self.assertTrue(run_a.is_dir())
            self.assertTrue(run_b.is_dir())
            self.assertEqual(run_a.name, "sc2_run_20260308_144207")
            self.assertEqual(run_b.name, "sc2_run_20260308_144207_01")


if __name__ == "__main__":
    unittest.main()
