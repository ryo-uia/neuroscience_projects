"""Analyzer utilities for spike sorting pipelines.

This module provides functions for creating and managing SortingAnalyzer objects
with QC metrics computation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .run_utils import log_info, log_warn
from .si_utils import ensure_probe_attached


def _load_create_sorting_analyzer():
    """Lazy-load create_sorting_analyzer to avoid heavy import at module import time."""
    try:
        from spikeinterface.core import create_sorting_analyzer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface is required to build analyzers. "
            "Activate the spikeinterface environment before running this pipeline."
        ) from exc
    return create_sorting_analyzer


def _maybe_load_compute_quality_metrics():
    """Best-effort lazy import for optional quality metrics fallback path."""
    try:
        from spikeinterface.qualitymetrics import compute_quality_metrics
    except Exception:
        return None
    return compute_quality_metrics


def _maybe_load_curation():
    """Best-effort lazy import for optional redundant-unit curation path."""
    try:
        import spikeinterface.curation as sc
    except Exception:
        return None
    return sc


def build_analyzer(
    sorting: "BaseSorting",
    recording: "BaseRecording",
    base_folder: Path,
    label: str,
    *,
    wf_ms_before: float | None = None,
    wf_ms_after: float | None = None,
    save_analyzer: bool = False,
    compute_qc_metrics: bool = True,
    qc_metric_names: list[str] | None = None,
    qc_pc_metrics: set[str] | None = None,
    max_spikes_per_unit: int = 500,
    random_seed: int = 42,
    n_jobs: int = 4,
    chunk_duration: str = "2s",
):
    """Create a SortingAnalyzer, compute waveforms/PCs/QC, and optionally save outputs.

    Args:
        sorting: Sorting object.
        recording: Recording object.
        base_folder: Output base folder.
        label: Label for the analyzer folder.
        wf_ms_before: Waveform window before peak (ms).
        wf_ms_after: Waveform window after peak (ms).
        save_analyzer: Whether to persist analyzer to disk.
        compute_qc_metrics: Whether to compute QC metrics.
        qc_metric_names: List of QC metric names to compute.
        qc_pc_metrics: Set of QC metrics that require PCs.
        max_spikes_per_unit: Maximum spikes per unit for random sampling.
        random_seed: Random seed for reproducibility.
        n_jobs: Number of parallel jobs.
        chunk_duration: Chunk duration for parallel processing.

    Returns:
        SortingAnalyzer object.
    """
    create_sorting_analyzer = _load_create_sorting_analyzer()

    # Default QC metrics
    if qc_metric_names is None:
        qc_metric_names = [
            "firing_rate",
            "presence_ratio",
            "isi_violation",
            "snr",
            "amplitude_cutoff",
        ]
    if qc_pc_metrics is None:
        qc_pc_metrics = {
            "isolation_distance",
            "l_ratio",
            "d_prime",
            "nearest_neighbor",
            "nn_isolation",
            "nn_noise_overlap",
            "silhouette",
        }

    folder = base_folder / f"analyzer_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    qc_metrics_path = None

    # Attach probe for analyzer
    try:
        recording = ensure_probe_attached(recording)
        log_info("Analyzer recording probe attached.")
    except Exception as exc:
        log_warn(f"failed to attach probe for analyzer: {exc}")

    analyzer_format = "binary_folder" if save_analyzer else "memory"
    if save_analyzer:
        folder.mkdir(parents=True, exist_ok=True)

    analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format=analyzer_format,
        folder=folder,
        overwrite=True,
    )

    # Compute random spikes
    analyzer.compute(
        {"random_spikes": {"max_spikes_per_unit": max_spikes_per_unit, "seed": random_seed}},
        verbose=True,
    )

    # Compute waveforms
    wf_kwargs = {}
    if wf_ms_before is not None:
        wf_kwargs["ms_before"] = wf_ms_before
    if wf_ms_after is not None:
        wf_kwargs["ms_after"] = wf_ms_after
    analyzer.compute(
        "waveforms",
        **wf_kwargs,
        dtype="float32",
        verbose=True,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
    )

    # Compute templates and PCs
    analyzer.compute("templates")
    analyzer.compute("principal_components")

    # Compute QC metrics
    if compute_qc_metrics:
        qc_done = False
        compute_quality_metrics = _maybe_load_compute_quality_metrics()
        try:
            try:
                analyzer.compute("noise_levels")
            except Exception as exc:
                log_warn(f"noise_levels failed: {exc}")

            skip_pc = not any(name in qc_pc_metrics for name in qc_metric_names)
            try:
                analyzer.compute(
                    "quality_metrics",
                    metric_names=qc_metric_names,
                    skip_pc_metrics=skip_pc,
                )
            except TypeError:
                analyzer.compute("quality_metrics", metric_names=qc_metric_names)

            try:
                qm = analyzer.get_extension("quality_metrics").get_data()
            except Exception:
                qm = None

            if qm is not None:
                log_info("QC metrics computed via analyzer extension.")
                out_path = (
                    Path(analyzer.folder) / "qc_metrics.csv"
                    if analyzer.folder
                    else (folder / "qc_metrics.csv")
                )
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    qm.to_csv(out_path)
                    log_info(f"QC metrics saved: {out_path}")
                    qc_metrics_path = out_path
                except Exception as exc:
                    log_warn(f"QC metrics save failed: {exc}")
                qc_done = True
            else:
                log_warn("QC metrics extension computed but returned no data.")
        except Exception as exc:
            log_warn(f"QC metrics extension path failed; trying fallback: {exc}")

        # Fallback to compute_quality_metrics if extension failed
        if not qc_done:
            if compute_quality_metrics is None:
                log_warn("QC metrics unavailable; skipping quality metrics.")
            else:
                try:
                    try:
                        analyzer.compute("noise_levels")
                    except Exception as exc:
                        log_warn(f"noise_levels failed in fallback path: {exc}")
                    qm = compute_quality_metrics(analyzer, metric_names=qc_metric_names)
                    log_info("QC metrics computed via compute_quality_metrics fallback.")
                    out_path = (
                        Path(analyzer.folder) / "qc_metrics.csv"
                        if analyzer.folder
                        else (folder / "qc_metrics.csv")
                    )
                    try:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        qm.to_csv(out_path)
                        log_info(f"QC metrics saved: {out_path}")
                        qc_metrics_path = out_path
                    except Exception as exc:
                        log_warn(f"QC metrics save failed: {exc}")
                except Exception as exc:
                    log_warn(f"QC metrics failed: {exc}")

    if qc_metrics_path is not None:
        setattr(analyzer, "_qc_metrics_path", str(qc_metrics_path))

    log_info(f"Analyzer computed for {label} at {folder}")
    return analyzer


def maybe_remove_redundant_units(
    analyzer,
    label: str,
    *,
    remove_redundant: bool = False,
    duplicate_threshold: float = 0.95,
    remove_strategy: str = "minimum_shift",
):
    """Optionally remove redundant units via spikeinterface.curation.

    Args:
        analyzer: SortingAnalyzer object.
        label: Label for logging.
        remove_redundant: Whether to remove redundant units.
        duplicate_threshold: Threshold for duplicate detection.
        remove_strategy: Strategy for removing duplicates.

    Returns:
        SortingAnalyzer object with redundant units removed.
    """
    sc = _maybe_load_curation()
    if not remove_redundant:
        return analyzer
    if sc is None:
        log_warn("spikeinterface.curation unavailable; skipping redundant-unit removal.")
        return analyzer
    try:
        clean_sorting = sc.remove_redundant_units(
            analyzer,
            duplicate_threshold=duplicate_threshold,
            remove_strategy=remove_strategy,
        )
        analyzer = analyzer.select_units(clean_sorting.unit_ids)
        log_info(f"Redundant units removed for {label} (threshold={duplicate_threshold}).")
    except Exception as exc:
        log_warn(f"redundant-unit removal failed for {label}: {exc}")
    return analyzer


__all__ = [
    "build_analyzer",
    "maybe_remove_redundant_units",
]
