"""
Artifact helpers: manifests, 'latest' pointers, and standard paths.

This module provides a single place to write/read small JSON artifacts that
connect pipeline stages (fetch → split → resize → validate → train → evaluate).

Why centralize?
---------------
- Consistent filenames and schemas across stages.
- One logging style and error handling.
- Easier testing and CI.

Contents
--------
- write_training_summary(...)
- write_evaluation_summary(...)
- write_latest_pointer(payload, dst_dir, stem)
- safe_json_dump(obj, path)
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging_utils import get_logger
from src.utils.paths import OUTPUTS_DIR

log = get_logger(__name__)


def _utc_now() -> str:
    """Return current UTC time in ISO-8601."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_json_dump(obj: Any, path: Path) -> None:
    """
    Write JSON to disk with UTF-8 encoding and parent dir creation.

    Parameters
    ----------
    obj : Any
        JSON-serializable Python object.
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
        # dataclass convenience
        if is_dataclass(obj):
            obj = asdict(obj)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        log.info("artifact.write", extra={"path": str(path)})
    except Exception as e:
        log.error("artifact.write_failed", extra={"path": str(path), "error": str(e)})
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
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_now()
    fname = f"training_summary_{(run_id or 'no-runid')}_{ts}.json"
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
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_now()
    fname = f"evaluation_summary_{(run_id or 'no-runid')}_{ts}.json"
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
