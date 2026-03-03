# Environment Setup (Sorting + Curation)

This repository uses two separate conda environments:

- `spikeinterface` for sorting and pipeline execution.
- `newphy2` for Phy GUI curation.

## 1) Sorting Environment (`spikeinterface`)

From repo root:

```powershell
conda env create -f spikeinterface_env.yml
# or update existing
conda env update -n spikeinterface -f spikeinterface_env.yml --prune
conda activate spikeinterface
```

Validation:

```powershell
python SpikeSorting/Functions/smoke_test_imports.py
python -m unittest discover -s SpikeSorting/Functions -p "test_*.py"
python -c "import spikeinterface.sorters as ss; print('spykingcircus2' in ss.available_sorters(), 'spykingcircus2' in ss.installed_sorters())"
```

## 2) Curation Environment (`newphy2`)

From repo root:

```powershell
conda env create -f newphy2_env.yml
# or update existing
conda env update -n newphy2 -f newphy2_env.yml --prune
conda activate newphy2
```

Validation:

```powershell
phy --help
```

## Workflow

1. Run sorting pipelines in `spikeinterface`.
2. Open Phy exports in `newphy2`:

```powershell
phy template-gui <path-to-params.py>
```

Notes:

- Keep these envs separate to avoid package conflicts.
- `spikeinterface_env.yml` and `newphy2_env.yml` are portable specs (no machine-specific `prefix`).
