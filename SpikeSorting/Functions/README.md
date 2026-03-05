# SpikeSorting Function Helpers

This folder is the shared toolbox behind the SC2 tetrode/mixed pipelines.
If a run behaves oddly, this is usually where the root cause lives.

## Use This Import Style

- Prefer direct imports from split modules (`session_utils`, `channel_utils`, etc.).
- `pipeline_utils.py` still exists as a compatibility shim for older scripts.
- Existing imports like `from Functions.pipeline_utils import ...` are still valid, but new code should avoid adding more shim dependencies.

## Module Map (Practical)

- `session_utils.py`
  - Session discovery, prompt selection, stream selection, test-duration slicing.
- `channel_utils.py`
  - Channel slicing compatibility, group resolution, bad-channel parsing/detection, OE index fallback mapping.
- `si_utils.py`
  - Geometry/probe attachment, in-place group property assignment, OE metadata attach.
- `analyzer_utils.py`
  - SortingAnalyzer build, waveform/template/PC/QC computation, optional redundant-unit cleanup.
- `phy_export.py`
  - Phy export, channel mapping rewrite, `hp_filtered` handling, optional sparse export.
- `run_utils.py`
  - Run folder reservation, tee logging, standardized helper logging (`log_info`, `log_warn`), SI GUI export.
- `config_utils.py`
  - Env override application/validation and standardized config-echo logging.
- `params_utils.py`
  - Deep-merge param dictionaries and SC2 override key checks.
- `fs_utils.py`
  - Retry-safe folder deletion.
- `pipeline_utils.py`
  - Compatibility re-export layer only.

## Conventions You Should Assume

- Naming:
  - Use `*_valid_ids` for channel-ID filtering.
  - `filter_groups_with_indices(...)` is kept as backward-compatible alias only.
- Mutability:
  - In-place functions: `set_group_property()`, `attach_oe_index_from_oebin()`.
  - Return-new-recording functions: `attach_geom()`, `attach_single_grid_geom()`, `ensure_geom_and_units()`.
  - `ensure_probe_attached()` may return the same object or a new one depending on probe presence.
- Logging:
  - Helpers should use `log_info()` / `log_warn()` from `run_utils` for consistent run output.
- Imports:
  - Heavy dependencies are lazy-loaded where possible to keep module import cheap.

## Known Gotchas

- OE index meaning:
  - `attach_oe_index_from_oebin()` stores stream-list position from `structure.oebin`.
  - Do not treat that as guaranteed hardware channel number unless your metadata schema explicitly does.
  - If `oe_channel_index` is not attached, fallback channel-order mapping is expected and non-fatal for tet runs.
  - If multiple streams tie during `structure.oebin` matching, helpers now skip metadata attach (fail-safe) instead of picking an arbitrary first match.
- Geometry defaults:
  - Channels not covered by groups can end up at `(0, 0)`; strict-group workflows should prevent this.
- Probe attachment:
  - `ensure_probe_attached()` needs channel locations. If missing, analyzer/export stages can fail.
- Phy filter flag:
  - `phy_export.py` writes `hp_filtered` even in simple-export return paths.
  - This avoids filtered/unfiltered interpretation drift in Phy.
- One-path mode:
  - In SC2 tet pipeline, `ANALYZER_FROM_SORTER=True` means analyzer and Phy export both use sorter-input context (`rec_sc2`).
  - `rec_export` may still be prepared, but it is not the Phy/analyzer source in that mode.

## Common Failure Modes (Fast Triage)

- `No neural streams found`
  - Check stream names in `structure.oebin` and/or set `STREAM_NAME` explicitly.
- `No valid channel groups resolved`
  - Check `CHANNEL_GROUPS`/JSON labels against the selected session + stream channel IDs.
- `No channel groups remain after bad-channel filtering`
  - Bad-channel list is too aggressive for that session, or group map is stale.
- `gain_to_uV/gains missing`
  - Export stays in native units; do not assume microvolts.
- Weird Phy filtering behavior
  - In tet pipeline, verify `ANALYZER_FROM_SORTER`, `USE_SI_PREPROCESS`, and `USE_SC2_PREPROCESS` match the intended preprocessing/export path.
  - Keep `ANALYZER_FROM_SORTER=True` for single-path behavior (analyzer + Phy share `rec_sc2` context).
  - `scaled_to_uv` controls export scaling intent and `hp_filtered` is written from resolved filter state.
- Empty JSON config semantics
  - `load_channel_groups_from_path()` returns explicit empty list for `[]` (no groups), not fallback.
  - `load_bad_channels_from_path()` returns explicit empty list for `[]` (no bad channels), not fallback.

## Quick API Reference

- `session_utils.py`
  - `pick_stream(data_path, preferred)`
  - `choose_config_json(label, candidates, default_path)`
  - `first_seconds(recording, seconds)`
  - `discover_recording_folders(root)`
  - `choose_recording_folder(root, subpath, selection, index)`
- `channel_utils.py`
  - `safe_channel_slice(recording, keep_ids)`
  - `chunk_groups(channel_ids, size)`
  - `resolve_manual_groups(recording, manual_groups)`
  - `load_channel_groups_from_path(path)`
  - `load_bad_channels_from_path(path)`
  - `resolve_bad_channel_ids(recording, manual_list)`
  - `detect_bad_channel_ids(recording, enabled, method=None, **kwargs)`
  - `build_oe_index_map(recording, fallback_map)`
  - `filter_groups_with_valid_ids(groups, valid_ids)`
- `si_utils.py`
  - `discover_oe_stream_names(folder)`
  - `attach_geom(recording, groups, ...)`
  - `attach_single_grid_geom(recording, channel_order, ...)`
  - `set_group_property(recording, groups, group_ids)`
  - `ensure_probe_attached(recording, radius=5)`
  - `ensure_geom_and_units(recording, groups, ...)`
  - `attach_oe_index_from_oebin(recording, data_path, stream_name)`
- `run_utils.py`
  - `TeeStream`
  - `reserve_run_folder(base_out)`
  - `enable_run_logging(log_path)`
  - `initialize_run_io(base_out, sc2_run, export_to_si_gui)`
  - `disable_run_logging(handle)`
  - `log_info(message)`
  - `log_warn(message)`
  - `export_for_si_gui(analyzer, base_folder, label)`
- `config_utils.py`
  - `apply_env_overrides_from_env(...)`
  - `print_pipeline_config_echo(...)`

## Regression Test

- Collaborator env baseline (repo root):
  - `conda env create -f spikeinterface_env.yml` (or `conda env update -n spikeinterface -f spikeinterface_env.yml --prune`)
- Override parsing regression checks (bool/enum/number/unknown key):
  - `python -m unittest discover -s SpikeSorting/Functions -p "test_*.py"`
- `analyzer_utils.py`
  - `build_analyzer(...)`
  - `maybe_remove_redundant_units(...)`
- `phy_export.py`
  - `build_phy_export_sparsity(...)`
  - `export_for_phy(...)`
