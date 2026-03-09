"""Regression tests for pipeline env-override parsing behavior."""

from __future__ import annotations

import importlib
import json
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
            "SC2_PARAM_OVERRIDES": {
                "apply_motion_correction": False,
                "detection": {
                    "method": "matched_filtering",
                    "method_kwargs": {"peak_sign": "neg", "detect_threshold": 6},
                },
            },
        }
        with patch.dict(os.environ, {ENV_NAME: payload}, clear=False):
            applied, ignored = apply_env_overrides_from_env(
                env_name=ENV_NAME,
                allowed_keys={"USE_SC2_PREPROCESS", "CAR_MODE", "TEST_SECONDS", "SC2_PARAM_OVERRIDES"},
                module_globals=module_globals,
                bool_keys={"USE_SC2_PREPROCESS"},
                enum_keys={"CAR_MODE": {"global", "tetrode"}},
                positive_number_or_none_keys={"TEST_SECONDS"},
                merge_object_keys={"SC2_PARAM_OVERRIDES"},
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

    def test_json_object_merge_override_preserves_existing_sc2_params(self):
        module_globals, applied, ignored = self._run_override(
            '{"SC2_PARAM_OVERRIDES":{"deterministic_peaks_detection":true,"detection":{"method_kwargs":{"detect_threshold":7}}}}'
        )
        self.assertEqual(ignored, [])
        merged = module_globals["SC2_PARAM_OVERRIDES"]
        self.assertFalse(merged["apply_motion_correction"])
        self.assertEqual(merged["detection"]["method"], "matched_filtering")
        self.assertEqual(merged["detection"]["method_kwargs"]["peak_sign"], "neg")
        self.assertEqual(merged["detection"]["method_kwargs"]["detect_threshold"], 7)
        self.assertTrue(merged["deterministic_peaks_detection"])
        self.assertEqual(applied.get("SC2_PARAM_OVERRIDES"), merged)

    def test_tet_pipeline_analyzer_waveform_overrides_accept_numbers_and_none(self):
        module_name = "Pipelines.spikeinterface_sc2_tet_pipeline"
        mod = importlib.import_module(module_name)
        payload = json.dumps(
            {
                "ANALYZER_WF_MS_BEFORE": 1.25,
                "ANALYZER_WF_MS_AFTER": None,
            }
        )
        try:
            with patch.dict(os.environ, {mod.PIPELINE_OVERRIDES_ENV: payload}, clear=False):
                mod = importlib.reload(mod)
                applied, ignored = mod.apply_env_pipeline_overrides()
                self.assertEqual(ignored, [])
                self.assertEqual(applied.get("ANALYZER_WF_MS_BEFORE"), 1.25)
                self.assertIsNone(applied.get("ANALYZER_WF_MS_AFTER"))
                self.assertEqual(mod.ANALYZER_WF_MS_BEFORE, 1.25)
                self.assertIsNone(mod.ANALYZER_WF_MS_AFTER)
        finally:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(mod.PIPELINE_OVERRIDES_ENV, None)
            importlib.reload(mod)

    def test_tet_pipeline_sc2_param_overrides_accept_nested_json_object(self):
        module_name = "Pipelines.spikeinterface_sc2_tet_pipeline"
        mod = importlib.import_module(module_name)
        payload = json.dumps(
            {
                "SC2_PARAM_OVERRIDES": {
                    "deterministic_peaks_detection": True,
                    "detection": {"method_kwargs": {"detect_threshold": 7}},
                }
            }
        )
        try:
            with patch.dict(os.environ, {mod.PIPELINE_OVERRIDES_ENV: payload}, clear=False):
                mod = importlib.reload(mod)
                applied, ignored = mod.apply_env_pipeline_overrides()
                self.assertEqual(ignored, [])
                self.assertEqual(applied["SC2_PARAM_OVERRIDES"]["detection"]["method_kwargs"]["detect_threshold"], 7)
                self.assertTrue(applied["SC2_PARAM_OVERRIDES"]["deterministic_peaks_detection"])
                self.assertEqual(mod.SC2_PARAM_OVERRIDES["detection"]["method"], "matched_filtering")
                self.assertEqual(mod.SC2_PARAM_OVERRIDES["detection"]["method_kwargs"]["peak_sign"], "neg")
                self.assertEqual(mod.SC2_PARAM_OVERRIDES["detection"]["method_kwargs"]["detect_threshold"], 7)
                self.assertTrue(mod.SC2_PARAM_OVERRIDES["deterministic_peaks_detection"])
        finally:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(mod.PIPELINE_OVERRIDES_ENV, None)
            importlib.reload(mod)

    def test_mixed_pipeline_analyzer_waveform_overrides_accept_numbers_and_none(self):
        module_name = "Pipelines.spikeinterface_sc2_mixed_pipeline"
        mod = importlib.import_module(module_name)
        payload = json.dumps(
            {
                "ANALYZER_WF_MS_BEFORE": 1.25,
                "ANALYZER_WF_MS_AFTER": None,
            }
        )
        try:
            with patch.dict(os.environ, {mod.PIPELINE_OVERRIDES_ENV: payload}, clear=False):
                mod = importlib.reload(mod)
                applied, ignored = mod.apply_env_pipeline_overrides()
                self.assertEqual(ignored, [])
                self.assertEqual(applied.get("ANALYZER_WF_MS_BEFORE"), 1.25)
                self.assertIsNone(applied.get("ANALYZER_WF_MS_AFTER"))
                self.assertEqual(mod.ANALYZER_WF_MS_BEFORE, 1.25)
                self.assertIsNone(mod.ANALYZER_WF_MS_AFTER)
        finally:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(mod.PIPELINE_OVERRIDES_ENV, None)
            importlib.reload(mod)

    def test_mixed_pipeline_sc2_param_overrides_accept_nested_json_object(self):
        module_name = "Pipelines.spikeinterface_sc2_mixed_pipeline"
        mod = importlib.import_module(module_name)
        payload = json.dumps(
            {
                "SC2_PARAM_OVERRIDES": {
                    "deterministic_peaks_detection": True,
                    "detection": {"method_kwargs": {"detect_threshold": 7}},
                }
            }
        )
        try:
            with patch.dict(os.environ, {mod.PIPELINE_OVERRIDES_ENV: payload}, clear=False):
                mod = importlib.reload(mod)
                applied, ignored = mod.apply_env_pipeline_overrides()
                self.assertEqual(ignored, [])
                self.assertEqual(applied["SC2_PARAM_OVERRIDES"]["detection"]["method_kwargs"]["detect_threshold"], 7)
                self.assertTrue(applied["SC2_PARAM_OVERRIDES"]["deterministic_peaks_detection"])
                self.assertEqual(mod.SC2_PARAM_OVERRIDES["detection"]["method"], "matched_filtering")
                self.assertEqual(mod.SC2_PARAM_OVERRIDES["detection"]["method_kwargs"]["peak_sign"], "neg")
                self.assertEqual(mod.SC2_PARAM_OVERRIDES["detection"]["method_kwargs"]["detect_threshold"], 7)
                self.assertTrue(mod.SC2_PARAM_OVERRIDES["deterministic_peaks_detection"])
        finally:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(mod.PIPELINE_OVERRIDES_ENV, None)
            importlib.reload(mod)

    def test_mixed_pipeline_preprocess_toggle_overrides_accept_booleans(self):
        module_name = "Pipelines.spikeinterface_sc2_mixed_pipeline"
        mod = importlib.import_module(module_name)
        payload = json.dumps(
            {
                "USE_SC2_PREPROCESS": False,
                "RAW_MODE": True,
                "SORT_BY_GROUP": True,
                "FAIL_ON_PER_GROUP_ERROR": True,
            }
        )
        try:
            with patch.dict(os.environ, {mod.PIPELINE_OVERRIDES_ENV: payload}, clear=False):
                mod = importlib.reload(mod)
                applied, ignored = mod.apply_env_pipeline_overrides()
                self.assertEqual(ignored, [])
                self.assertFalse(applied["USE_SC2_PREPROCESS"])
                self.assertTrue(applied["RAW_MODE"])
                self.assertTrue(applied["SORT_BY_GROUP"])
                self.assertTrue(applied["FAIL_ON_PER_GROUP_ERROR"])
                self.assertFalse(mod.USE_SC2_PREPROCESS)
                self.assertTrue(mod.RAW_MODE)
                self.assertTrue(mod.SORT_BY_GROUP)
                self.assertTrue(mod.FAIL_ON_PER_GROUP_ERROR)
        finally:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(mod.PIPELINE_OVERRIDES_ENV, None)
            importlib.reload(mod)

    def test_mixed_pipeline_raw_mode_policy_forces_preprocess_toggles_false(self):
        module_name = "Pipelines.spikeinterface_sc2_mixed_pipeline"
        mod = importlib.import_module(module_name)
        try:
            mod.USE_SI_PREPROCESS = True
            mod.USE_SC2_PREPROCESS = True
            mod.APPLY_NOTCH = True
            mod.SI_APPLY_CAR = True
            mod.SI_APPLY_WHITEN = True
            mod.RAW_MODE = True
            mod.apply_raw_mode_policy()
            self.assertFalse(mod.USE_SI_PREPROCESS)
            self.assertFalse(mod.USE_SC2_PREPROCESS)
            self.assertFalse(mod.APPLY_NOTCH)
            self.assertFalse(mod.SI_APPLY_CAR)
            self.assertFalse(mod.SI_APPLY_WHITEN)
        finally:
            importlib.reload(mod)

    def test_mixed_pipeline_sc2_runtime_params_respect_use_sc2_preprocess(self):
        module_name = "Pipelines.spikeinterface_sc2_mixed_pipeline"
        mod = importlib.import_module(module_name)
        try:
            mod.USE_SI_PREPROCESS = False
            mod.USE_SC2_PREPROCESS = False
            sc2_params, _, _, _, _ = mod.resolve_sc2_runtime_params()
            self.assertFalse(sc2_params["apply_preprocessing"])
            if "preprocessing" in sc2_params:
                self.assertFalse(sc2_params["preprocessing"]["apply"])

            mod.USE_SC2_PREPROCESS = True
            sc2_params, _, _, _, _ = mod.resolve_sc2_runtime_params()
            self.assertTrue(sc2_params["apply_preprocessing"])
            if "preprocessing" in sc2_params:
                self.assertTrue(sc2_params["preprocessing"]["apply"])
        finally:
            importlib.reload(mod)


if __name__ == "__main__":
    unittest.main()
