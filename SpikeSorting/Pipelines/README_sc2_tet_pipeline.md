# SC2 Tetrode Pipeline

This document covers `SpikeSorting/Pipelines/spikeinterface_sc2_tet_pipeline.py`.

## Required Inputs

- Open Ephys recordings under a root folder, with at least one `structure.oebin` file.
- A valid stream in each session (auto-detected by default, or set `STREAM_NAME`).
- Channel grouping and bad-channel definitions:
  - inline defaults in the script, or
  - JSON files via `--channel-groups` / `--bad-channels`, or
  - env vars (`SPIKESORT_CHANNEL_GROUPS`, `SPIKESORT_BAD_CHANNELS`).
- Python environment with `spikeinterface` + `spykingcircus2` available.
- Optional: Phy CLI (`phy`) if using `EXPORT_PHY_EXTRACT_WAVEFORMS=True`.
  - Reproducible curation env spec: `newphy2_env.yml` (separate from sorting env).

JSON formats:

- Channel groups: list of lists, e.g. `[["CH40","CH38","CH36","CH34"], ["CH48","CH46","CH44","CH42"]]`
- Bad channels: list, e.g. `["CH41","CH43"]`

## Collaborator Setup (Reproducible)

From repo root (`C:\Users\ryoi\Documents\Code` on this machine):

```powershell
# Create/update the shared SpikeInterface env
conda env create -f spikeinterface_env.yml
# or, if env already exists:
conda env update -n spikeinterface -f spikeinterface_env.yml --prune

conda activate spikeinterface
```

Quick validation on collaborator PCs:

```powershell
# Helper-module import smoke
python SpikeSorting/Functions/smoke_test_imports.py

# Env override parsing regression checks
python -m unittest discover -s SpikeSorting/Functions -p "test_*.py"

# Verify SC2 sorter is available in this env
python -c "import spikeinterface.sorters as ss; print('spykingcircus2' in ss.available_sorters(), 'spykingcircus2' in ss.installed_sorters())"
```

Notes:
- `spikeinterface_env.yml` is now a portable collaborator spec (no machine-specific `prefix`).
- Phy GUI is typically kept in a separate env (for example `newphy2`).
  - Create/update with: `conda env create -f newphy2_env.yml` (or `conda env update -n newphy2 -f newphy2_env.yml --prune`)
- Shared env quick-reference: `README_envs.md` (repo root).

## Channel Map Policy (Hirose 64ch)

For Open Ephys FlexDrive/Hirose recordings, treat `CHANNEL_GROUPS` as the tetrode-truth map in `CH##` labels
(OE GUI channel numbering, starts at 1).

Use `structure.oebin` as session-specific metadata/validation:
- stream channel order validation (`CH##`, plus any non-neural channels like `ADC*`)
- `oe_channel_index` attachment for exports
- `bit_volts -> oe_gain_to_uV` attachment for scaling

Do not rely on `structure.oebin` alone to infer tetrode membership in all setups.
For per-tetrode sorting, keep explicit `CHANNEL_GROUPS` and validate against `structure.oebin`.

Pre-sort verification is now in:
- `SpikeSorting/Analysis/sc2_presort_checks.ipynb`
- Section: `CHANNEL_GROUPS vs structure.oebin Verification`
- Toggle: `VERIFY_CHANNEL_GROUPS_AGAINST_OEBIN = True`

## Helper Modules

The pipeline uses helper modules in `SpikeSorting/Functions/`:

| Module | Purpose |
|--------|---------|
| `session_utils.py` | Session discovery/selection and stream selection |
| `channel_utils.py` | Channel groups, bad-channel handling, channel slicing/mapping |
| `si_utils.py` | Probe geometry, Open Ephys metadata attachment (`oe_channel_index`, `oe_gain_to_uV`) |
| `config_utils.py` | Env override parsing/validation and standardized config echo logging |
| `params_utils.py` | Nested parameter merge + unknown override warnings |
| `run_utils.py` | Run logging, output folder management, SI GUI export |
| `analyzer_utils.py` | SortingAnalyzer build + QC metrics |
| `phy_export.py` | Phy export and channel mapping metadata |
| `pipeline_utils.py` | Compatibility shim re-exporting the split helpers |

## Minimal Run Command

```powershell
# If not already in the SpikeInterface environment:
# conda activate spikeinterface
python SpikeSorting/Pipelines/spikeinterface_sc2_tet_pipeline.py
```

Fallback for non-activated shells:

```powershell
.\run_spikeinterface.ps1 SpikeSorting/Pipelines/spikeinterface_sc2_tet_pipeline.py
```

Useful checks:

```powershell
# Resolve session/groups/bad channels only, then exit
python SpikeSorting/Pipelines/spikeinterface_sc2_tet_pipeline.py --dry-run-config

# Explicit roots
python SpikeSorting/Pipelines/spikeinterface_sc2_tet_pipeline.py --root-dir SpikeSorting/recordings --base-out SpikeSorting
```

Notes:

- By default, the script prompts for session selection when `SESSION_SUBPATH` is not set.
- For quick tests, set `TEST_SECONDS=300` in the script; set back to `None` for full runs.
- In this workspace, VS Code is pinned to the `spikeinterface` interpreter, so plain `python ...` is the default path.
- Advanced orchestration (SpikeAgent and job-wrapper) is documented in `SpikeSorting/Pipelines/README_sc2_tet_advanced.md`.

## Config Reference (Common Options)

| Setting | Options | Recommended Default | Notes |
|---|---|---|---|
| `SESSION_SUBPATH` | `None` or relative OE session path | Set for non-interactive runs | When set, session prompt is skipped and this path is used directly. |
| `STREAM_NAME` | `None` or exact stream name | Exact stream for automation | `None` auto-picks first neural stream. |
| `TEST_SECONDS` | `None` or positive number of seconds | `300` for smoke tests, `None` final | Must be `> 0` when set. |
| `USE_CONFIG_JSONS` | `True` / `False` | `False` | `True` can prompt for JSON files. |
| `SORT_BY_GROUP` | `True` / `False` | `False` | `True` runs per-tetrode sorting. |
| `USE_SI_PREPROCESS` | `True` / `False` | `True` | Enables SI preprocessing path before sorter input. |
| `USE_SC2_PREPROCESS` | `True` / `False` | `False` | Enables SC2 internal preprocessing (`apply_preprocessing`). Both toggles can be `True` for intentional double preprocessing tests, and both can be `False` for raw/no-internal mode. |
| `RAW_MODE` | `True` / `False` | `False` | Strict raw mode. When `True`, the pipeline force-disables SI/SC2 preprocessing, notch, CAR, and SI whitening. |
| `SI_APPLY_CAR` | `True` / `False` | `False` | Enable only after validation. |
| `CAR_MODE` | `"global"`, `"tetrode"` | `"global"` | Used when CAR is enabled. |
| `SI_APPLY_WHITEN` | `True` / `False` | `False` | Avoid double-whitening in most cases. |
| `EXPORT_PHY_CHANNEL_IDS_MODE` | `"oe_label"`, `"oe_index"`, `"as_exported"` | `"oe_label"` | Affects `channel_ids` metadata for downstream analysis. |
| `SAVE_ANALYZER` | `True` / `False` | `True` | Saves analyzer folder for QC reuse. |

Additional runtime safeguards:
- Groups with fewer than 2 channels are dropped automatically before sorting.
- Groups with fewer than 3 channels are kept but logged as a warning.
- In per-group mode, each split recording is coerced to a single dummy probe from channel locations (strict; failure stops the run).
- Per-group mode requires valid channel locations (normally keep `ATTACH_GEOMETRY=True`).
- Group lookups are strict in per-group mode (missing group key raises immediately; no fallback to full recording).
- SC2 `job_kwargs.n_jobs` defaults to `1` for stable multiprocessing behavior.
- If both `USE_SI_PREPROCESS=True` and `USE_SC2_PREPROCESS=True`, filter/reference stages are applied twice (intentional test mode; logged with warnings).
- If both `USE_SI_PREPROCESS=False` and `USE_SC2_PREPROCESS=False`, the run proceeds in raw/no-internal mode (optional no-SI CAR/notch settings may still apply).
- If `USE_SI_PREPROCESS=False` and `SI_APPLY_CAR=True`, SI CAR still runs in the no-SI path (intentional CAR-only behavior; logged with warning).
- Phy `hp_filtered` describes the exported analyzer recording path, not SC2 internal preprocessing used during sorting.
- Empty JSON semantics are explicit: `[]` in `--channel-groups` means no groups (not fallback), and `[]` in `--bad-channels` means no bad channels (not fallback).
- If `structure.oebin` stream matching is ambiguous, OE metadata attach (`oe_channel_index` / `oe_gain_to_uV`) is skipped to avoid incorrect mapping.

## Recommended Baseline (Tetrode)

Key recommendations:
- `SORT_BY_GROUP=False` (bundle run first), then cross-check with `SORT_BY_GROUP=True`
- `USE_SI_PREPROCESS=True`, `USE_SC2_PREPROCESS=False`, `SI_APPLY_CAR=False`, `SI_APPLY_WHITEN=False`
- `EXPORT_PHY_CHANNEL_IDS_MODE="oe_label"` for clearest channel traceability

## Key Outputs

Under `base_out` (default `SpikeSorting/`):

- `sc2_outputs/run_logs/sc2_run_*.log`: run log.
- `sc2_outputs/sc2_run_*/sorter_output`: sorter output folder.
- `sc2_outputs/sc2_run_*/effective_config.json`: resolved run settings snapshot (post env overrides and `RAW_MODE` policy), selected session/stream, grouping/bad-channel details, and final `sc2_params`.
- `sc2_outputs/sc2_run_*/rec_*` (only when `MATERIALIZE_*` flags are enabled): run-scoped materialized recordings.
- `sc2_outputs/phy_sc2_*/` (or `phy_sc2_g*/` when `SORT_BY_GROUP=True`): Phy exports.
  - Includes `params.py`, `recording.dat`, `channel_id_map.tsv`.
- `sc2_outputs/analyzer_sc2_*/` (only when `SAVE_ANALYZER=True`).
- `si_gui_exports/*` when `EXPORT_TO_SI_GUI=True`.

Scaling note:
- Export/analyzer scaling to microvolts first prefers Open Ephys `structure.oebin` (`bit_volts -> oe_gain_to_uV`), then falls back to gain heuristics when metadata is missing.

The script prints launch commands at the end:

- `phy template-gui "<.../params.py>"`
- `python -m spikeinterface_gui "<...>"`

## Common Failure Modes

- `ModuleNotFoundError: spikeinterface`
  - Cause: wrong Python environment (imports happen at startup).
  - Fix: activate/run with your spikeinterface env Python.
- `No Open Ephys sessions found under ...`
  - Cause: wrong `--root-dir` / missing `structure.oebin`.
  - Fix: point to the recordings root.
- Session prompt blocks automation
  - Cause: session selection is prompt-driven when `SESSION_SUBPATH` is unset.
  - Fix: set `SESSION_SUBPATH` to a specific session path for non-interactive runs.
- Grouping errors with `STRICT_GROUPS=True`
  - Cause: invalid/empty groups after bad-channel filtering, or channels not covered by groups (often wrong stream/session).
  - Fix: update `CHANNEL_GROUPS`/bad channels JSON.
- Single-channel groups dropped
  - Cause: after bad-channel removal, some groups can shrink to one channel.
  - Behavior: pipeline auto-drops groups with `<2` channels (SC2 whitening safety).
  - Fix: adjust bad-channel list or grouping config if too many groups are dropped.
- Geometry/probe attach failure (strict path)
  - Cause: missing or invalid channel locations after slicing/preprocess/materialization.
  - Behavior: pipeline raises an error before sorting/analyzer export.
  - Fix: verify geometry attachment inputs and channel-group consistency.
- Per-group key mismatch
  - Cause: `group_ids` does not match recording `group` property values after transforms.
  - Behavior: per-group mode raises instead of silently using full recording.
  - Fix: check group assignment and bad-channel filtering outputs.
- `WARNING: 'phy' command not found; skipping extract-waveforms.`
  - Non-fatal unless you require pre-extracted waveforms.
- `WARNING: invalid SPIKESORT_PIPELINE_OVERRIDES JSON; ignoring overrides.`
  - Cause: malformed env JSON override payload.
  - Fix: provide a valid JSON object or unset the env var.
- `Ignored unknown/blocked pipeline overrides` or `Ignoring override 'KEY': ...`
  - Cause: override key not allow-listed, or value type invalid (for example `"False"` string instead of JSON `false` boolean).
  - Fix: use allow-listed keys and correct JSON types in override payloads.
- Unexpected preprocessing behavior in strict raw tests
  - Cause: expected strict raw mode but `RAW_MODE=False`.
  - Fix: set `RAW_MODE=True` (or verify `effective_config.json` for final resolved toggles).
- Unexpected CAR in a no-SI-preprocess run
  - Cause: `USE_SI_PREPROCESS=False` with `SI_APPLY_CAR=True` still applies SI CAR in the no-SI path.
  - Fix: set `SI_APPLY_CAR=False` (or enable `RAW_MODE=True` for strict raw behavior).
- Ambiguous Open Ephys stream during metadata attach
  - Cause: multiple `structure.oebin` streams tie for requested stream name after normalization.
  - Behavior: pipeline skips attaching OE metadata properties rather than guessing.
  - Fix: set an exact `STREAM_NAME` matching the stream shown in extractor output.
- Sorter failure cleanup behavior
  - Behavior: pipeline preserves the run folder for debugging and removes only partial `sorter_output`.
  - Rationale: keeps `effective_config.json` and run diagnostics available after failures.
