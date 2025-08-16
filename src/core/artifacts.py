"""
Artifact helpers: manifests, 'latest' pointers, and standard paths.

This module centralizes small JSON artifacts that connect pipeline stages
(fetch → split → resize → validate → train → evaluate).

Why centralize?
---------------
- Consistent filenames and schemas across stages.
- One logging style and error handling.
- Easier testing and CI.

Contents
--------
Manifests:
- write_training_summary(...)
- write_evaluation_summary(...)

Pointers (standardized):
- write_fetch_pointer(...), read_fetch_pointer(...)
- write_mapping_pointer(...), read_mapping_pointer(...)

Low level:
- write_latest_pointer(payload, dst_dir, stem)
- safe_json_dump(obj, path)
- safe_json_load(path)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.utils.logging_utils import get_logger
from src.utils.paths import OUTPUTS_DIR

log = get_logger(__name__)


# ---------------------------- time helpers -----------------------------------
def _utc_now() -> str:
    """Return current UTC time in ISO-8601."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact() -> str:
    """Return UTC timestamp as YYYYMMDD_HHMMSSZ."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


# ------------------------- JSON (safe) helpers -------------------------------
def safe_json_dump(obj: Any, path: Path) -> None:
    """
    Write JSON to disk with UTF-8 encoding and parent dir creation.

    Parameters
    ----------
    obj : Any
        JSON-serializable Python object (dataclasses auto-converted).
    path : Path
        Destination path.

    Logging
    -------
    - 'artifact.write' on success
    - 'artifact.write_failed' on error
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if is_dataclass(obj):
            obj = asdict(obj)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        log.info("artifact.write", extra={"path": str(path)})
    except Exception as e:
        log.error("artifact.write_failed", extra={"path": str(path), "error": str(e)})
        raise


def safe_json_load(path: Path) -> Dict[str, Any]:
    """
    Read JSON from disk with UTF-8 and basic error logging.

    Returns
    -------
    dict
        Parsed JSON payload.
    """
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        log.debug("artifact.read", extra={"path": str(p)})
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON must be an object/dict.")
        return data
    except Exception as e:
        log.error("artifact.read_failed", extra={"path": str(p), "error": str(e)})
        raise


def write_latest_pointer(payload: Dict[str, Any], dst_dir: Path, stem: str) -> Path:
    """
    Write `<stem>_latest.json` next to versioned files to act as a cursor.

    Parameters
    ----------
    payload : dict
        JSON-serializable dict to write.
    dst_dir : Path
        Directory where the pointer should live.
    stem : str
        Prefix for the file (e.g., 'training_summary' or 'evaluation_summary').

    Returns
    -------
    Path
        Path to the written latest file.
    """
    latest_path = Path(dst_dir) / f"{stem}_latest.json"
    safe_json_dump(payload, latest_path)
    log.info("artifact.latest_updated", extra={"path": str(latest_path), "stem": stem})
    return latest_path


# ---------------------------- training summary -------------------------------
def write_training_summary(
    *,
    out_dir: Path,
    run_id: Optional[str],
    args_dict: Dict[str, Any],
    class_names: list[str],
    class_to_idx: dict[str, int],
    per_class_counts: dict[str, int],
    best_f1: float,
    best_epoch: int,
    checkpoint_path: Path,
    env_info: Dict[str, Any],
) -> Path:
    """
    Write a standardized training summary manifest and update a 'latest' pointer.

    Parameters
    ----------
    out_dir : Path
        Directory where summaries are stored (e.g., outputs/training/).
    run_id : str | None
        Optional run identifier to embed in the filename & payload.
    args_dict : dict
        CLI/config arguments for traceability (stringified where needed).
    class_names : list[str]
        Ordered class names used by the model.
    class_to_idx : dict[str, int]
        Mapping from class name to integer index.
    per_class_counts : dict[str, int]
        Counts per class in the dataset (pre split).
    best_f1 : float
        Best validation macro F1 achieved.
    best_epoch : int
        Epoch index at which best F1 occurred.
    checkpoint_path : Path
        Path to the saved best model weights (.pth).
    env_info : dict
        Environment snapshot (e.g., from core.env.get_env_info().to_dict()).

    Returns
    -------
    Path
        Path to the versioned training summary JSON.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_now()
    fname = f"training_summary_{(run_id or 'no-runid')}_{_utc_compact()}.json"
    summary_path = out_dir / fname

    payload = {
        "timestamp_utc": ts,
        "run_id": run_id,
        "args": args_dict,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "class_distribution": per_class_counts,
        "results": {"best_f1": round(float(best_f1), 6), "best_epoch": int(best_epoch)},
        "artifacts": {"checkpoint": str(Path(checkpoint_path).resolve())},
        "env": env_info,
    }

    safe_json_dump(payload, summary_path)
    write_latest_pointer(payload, out_dir, stem="training_summary")
    return summary_path


# --------------------------- evaluation summary ------------------------------
def write_evaluation_summary(
    *,
    out_dir: Path,
    run_id: Optional[str],
    args_dict: Dict[str, Any],
    class_names: list[str],
    metrics: Dict[str, float],
) -> Path:
    """
    Write a standardized evaluation summary manifest and update a 'latest' pointer.

    Parameters
    ----------
    out_dir : Path
        Directory where evaluation outputs are stored (e.g., outputs/evaluation/).
    run_id : str | None
        Optional run identifier.
    args_dict : dict
        CLI/config arguments used for evaluation.
    class_names : list[str]
        Ordered class names used to compute metrics/reporting.
    metrics : dict
        Scalar metrics (e.g., acc, precision_macro, recall_macro, f1_macro).

    Returns
    -------
    Path
        Path to the versioned evaluation summary JSON.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_now()
    fname = f"evaluation_summary_{(run_id or 'no-runid')}_{_utc_compact()}.json"
    summary_path = out_dir / fname

    payload = {
        "timestamp_utc": ts,
        "run_id": run_id,
        "args": args_dict,
        "class_names": class_names,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
    }

    safe_json_dump(payload, summary_path)
    write_latest_pointer(payload, out_dir, stem="evaluation_summary")
    return summary_path


# ============================= POINTERS ======================================
# Standard locations:
#   outputs/pointers/fetch/<owner>/<slug>/
#       latest.json
#       history/fetch_YYYYMMDD_HHMMSSZ.json
#   outputs/pointers/mapping/<owner>/<slug>/
#       latest.json
#       history/mapping_YYYYMMDD_HHMMSSZ.json

# ---------------------------- common utilities -------------------------------
_DATASET_RE = re.compile(r"^(?P<owner>[^/]+)/(?P<slug>[^/]+)$")


def _split_dataset_slug(dataset: Optional[str]) -> Tuple[str, str]:
    """
    Split 'owner/slug' into ('owner', 'slug'). If not parseable, return ('_unknown_', '_unknown_').
    """
    if not dataset:
        return "_unknown_", "_unknown_"
    m = _DATASET_RE.match(dataset.strip())
    if not m:
        log.warning("artifacts.dataset_slug_invalid", extra={"dataset": dataset})
        return "_unknown_", "_unknown_"
    return m.group("owner"), m.group("slug")


def _pointer_root(kind: str, dataset: Optional[str]) -> Path:
    """
    Compute the root directory for pointer files.

    Parameters
    ----------
    kind : {"fetch","mapping"}
        Pointer family.
    dataset : str | None
        'owner/slug'. If None/invalid, falls back to '_unknown_'.

    Returns
    -------
    Path
        outputs/pointers/<kind>/<owner>/<slug>/
    """
    owner, slug = _split_dataset_slug(dataset)
    return OUTPUTS_DIR / "pointers" / kind / owner / slug


def _write_latest_and_history(kind: str, payload: Dict[str, Any], base_dir: Path) -> Dict[str, Path]:
    """
    Write latest.json and a timestamped copy under history/.

    Returns
    -------
    dict with keys: latest_path, history_path
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    latest = base / "latest.json"
    safe_json_dump(payload, latest)

    hist_dir = base / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist = hist_dir / f"{kind}_{_utc_compact()}.json"
    safe_json_dump(payload, hist)

    log.info("artifacts.latest_updated", extra={"kind": kind, "path": str(latest)})
    log.info("artifacts.history_written", extra={"kind": kind, "path": str(hist)})
    return {"latest_path": latest, "history_path": hist}


def _resolve_pointer_path(path_or_dir: Path) -> Path:
    """
    If a directory is provided, read its 'latest.json'. If a file is provided, use it directly.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "latest.json"
    return p


# ------------------------------- validators ----------------------------------
def _validate_fetch_pointer(ptr: Dict[str, Any]) -> None:
    """
    Validate minimal fetch pointer schema.

    Required keys:
      - dataset (owner/slug)
      - dataset_root
      - fetched_at_utc
    Optional keys:
      - training_dir, testing_dir, version, run_id
    """
    missing = [k for k in ("dataset", "dataset_root", "fetched_at_utc") if k not in ptr]
    if missing:
        log.error("artifacts.pointer_invalid", extra={"kind": "fetch", "missing": missing})
        raise ValueError(f"Invalid fetch pointer: missing keys {missing}")


def _validate_mapping_pointer(ptr: Dict[str, Any]) -> None:
    """
    Validate minimal mapping pointer schema.

    Required keys:
      - classes (list[str])
      - path (path to index_remap.json)
      - written_at_utc
    Optional keys:
      - index_remap (dict[int,str]), num_classes, dataset, run_id
    """
    missing = [k for k in ("classes", "path", "written_at_utc") if k not in ptr]
    if missing:
        log.error("artifacts.pointer_invalid", extra={"kind": "mapping", "missing": missing})
        raise ValueError(f"Invalid mapping pointer: missing keys {missing}")

    if "num_classes" in ptr and isinstance(ptr["num_classes"], int):
        if ptr.get("classes") and len(ptr["classes"]) != ptr["num_classes"]:
            raise ValueError(
                f"Invalid mapping pointer: num_classes={ptr['num_classes']} "
                f"!= len(classes)={len(ptr['classes'])}"
            )


# ----------------------------- fetch pointer ---------------------------------
def write_fetch_pointer(
    *,
    dataset: str,
    dataset_root: Path,
    training_dir: Optional[Path] = None,
    testing_dir: Optional[Path] = None,
    version: Optional[str] = None,
    run_id: Optional[str] = None,
    dst_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Write standardized fetch pointer (latest + history).

    Parameters
    ----------
    dataset : str
        Kaggle slug 'owner/slug'.
    dataset_root : Path
        Root directory where the dataset was fetched/cached.
    training_dir : Path | None
        Optional resolved path to the training split folder.
    testing_dir : Path | None
        Optional resolved path to the testing split folder.
    version : str | None
        Optional dataset version/commit/hash.
    run_id : str | None
        Orchestrated run id for provenance.
    dst_dir : Path | None
        If provided, pointer is written there. Otherwise derived from dataset:
        outputs/pointers/fetch/<owner>/<slug>/

    Returns
    -------
    dict
        {'latest_path': Path, 'history_path': Path}
    """
    payload = {
        "dataset": dataset,
        "dataset_root": str(Path(dataset_root).resolve()),
        "training_dir": str(Path(training_dir).resolve()) if training_dir else None,
        "testing_dir": str(Path(testing_dir).resolve()) if testing_dir else None,
        "version": version,
        "run_id": run_id,
        "fetched_at_utc": _utc_now(),
    }
    base_dir = Path(dst_dir) if dst_dir else _pointer_root("fetch", dataset)
    out = _write_latest_and_history("fetch", payload, base_dir)
    return out


def read_fetch_pointer(path_or_dir: Path) -> Dict[str, Any]:
    """
    Read and validate a fetch pointer from a directory (latest.json) or a file path.

    Returns
    -------
    dict
        Validated pointer payload.
    """
    p = _resolve_pointer_path(path_or_dir)
    payload = safe_json_load(p)
    _validate_fetch_pointer(payload)
    log.info("artifacts.pointer_read", extra={"kind": "fetch", "path": str(p)})
    return payload


# ---------------------------- mapping pointer --------------------------------
def write_mapping_pointer(
    *,
    classes: list[str],
    index_remap_path: Path,
    dataset: Optional[str] = None,
    index_remap: Optional[Dict[int, str]] = None,
    run_id: Optional[str] = None,
    dst_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Write standardized mapping pointer (latest + history).

    Parameters
    ----------
    classes : list[str]
        Ordered class names corresponding to model outputs.
    index_remap_path : Path
        Absolute/relative path to the index_remap.json (authoritative map).
    dataset : str | None
        Optional 'owner/slug' to place pointer under a dataset-specific directory.
    index_remap : dict[int,str] | None
        Optional inline copy of the index remap for convenience.
    run_id : str | None
        Orchestrated run id for provenance.
    dst_dir : Path | None
        If provided, pointer is written there. Otherwise derived from dataset:
        outputs/pointers/mapping/<owner>/<slug>/ (or _unknown_/_unknown_ if None).

    Returns
    -------
    dict
        {'latest_path': Path, 'history_path': Path}
    """
    payload = {
        "classes": list(classes),
        "num_classes": len(classes),
        "index_remap": index_remap,  # optional
        "path": str(Path(index_remap_path).resolve()),
        "dataset": dataset,
        "run_id": run_id,
        "written_at_utc": _utc_now(),
    }
    base_dir = Path(dst_dir) if dst_dir else _pointer_root("mapping", dataset)
    out = _write_latest_and_history("mapping", payload, base_dir)
    return out


def read_mapping_pointer(path_or_dir: Path) -> Dict[str, Any]:
    """
    Read and validate a mapping pointer from a directory (latest.json) or a file path.

    Returns
    -------
    dict
        Validated pointer payload.
    """
    p = _resolve_pointer_path(path_or_dir)
    payload = safe_json_load(p)
    _validate_mapping_pointer(payload)
    log.info("artifacts.pointer_read", extra={"kind": "mapping", "path": str(p)})
    return payload
