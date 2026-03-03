"""Regression tests for pipeline env-override parsing behavior."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Functions.config_utils import apply_env_overrides_from_env


ENV_NAME = "TEST_PIPELINE_OVERRIDES"


class ConfigOverrideParsingTests(unittest.TestCase):
    """Validate strict typing + enum behavior for env JSON overrides."""

    def _run_override(self, payload: str):
        module_globals = {
            "USE_SC2_PREPROCESS": True,
            "CAR_MODE": "global",
            "TEST_SECONDS": None,
        }
        with patch.dict(os.environ, {ENV_NAME: payload}, clear=False):
            applied, ignored = apply_env_overrides_from_env(
                env_name=ENV_NAME,
                allowed_keys={"USE_SC2_PREPROCESS", "CAR_MODE", "TEST_SECONDS"},
                module_globals=module_globals,
                bool_keys={"USE_SC2_PREPROCESS"},
                enum_keys={"CAR_MODE": {"global", "tetrode"}},
                positive_number_or_none_keys={"TEST_SECONDS"},
            )
        return module_globals, applied, ignored

    def test_boolean_string_is_rejected(self):
        module_globals, applied, ignored = self._run_override(
            '{"USE_SC2_PREPROCESS":"False"}'
        )
        self.assertEqual(applied, {})
        self.assertIn("USE_SC2_PREPROCESS", ignored)
        self.assertTrue(module_globals["USE_SC2_PREPROCESS"])

    def test_boolean_json_value_is_accepted(self):
        module_globals, applied, ignored = self._run_override(
            '{"USE_SC2_PREPROCESS":false}'
        )
        self.assertEqual(ignored, [])
        self.assertEqual(applied.get("USE_SC2_PREPROCESS"), False)
        self.assertFalse(module_globals["USE_SC2_PREPROCESS"])

    def test_enum_validation_and_normalization(self):
        module_globals, applied, ignored = self._run_override(
            '{"CAR_MODE":"TETRODE"}'
        )
        self.assertEqual(ignored, [])
        self.assertEqual(applied.get("CAR_MODE"), "tetrode")
        self.assertEqual(module_globals["CAR_MODE"], "tetrode")

        module_globals, applied, ignored = self._run_override(
            '{"CAR_MODE":"invalid_mode"}'
        )
        self.assertEqual(applied, {})
        self.assertIn("CAR_MODE", ignored)
        self.assertEqual(module_globals["CAR_MODE"], "global")

    def test_test_seconds_validation(self):
        module_globals, applied, ignored = self._run_override(
            '{"TEST_SECONDS":300}'
        )
        self.assertEqual(ignored, [])
        self.assertEqual(applied.get("TEST_SECONDS"), 300)
        self.assertEqual(module_globals["TEST_SECONDS"], 300)

        module_globals, applied, ignored = self._run_override(
            '{"TEST_SECONDS":null}'
        )
        self.assertEqual(ignored, [])
        self.assertIsNone(applied.get("TEST_SECONDS"))
        self.assertIsNone(module_globals["TEST_SECONDS"])

        module_globals, applied, ignored = self._run_override(
            '{"TEST_SECONDS":"300"}'
        )
        self.assertEqual(applied, {})
        self.assertIn("TEST_SECONDS", ignored)
        self.assertIsNone(module_globals["TEST_SECONDS"])

    def test_unknown_key_is_ignored(self):
        module_globals, applied, ignored = self._run_override(
            '{"NOT_ALLOWED_KEY":123}'
        )
        self.assertEqual(applied, {})
        self.assertIn("NOT_ALLOWED_KEY", ignored)
        self.assertTrue(module_globals["USE_SC2_PREPROCESS"])
        self.assertEqual(module_globals["CAR_MODE"], "global")


if __name__ == "__main__":
    unittest.main()

