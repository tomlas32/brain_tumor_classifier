from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from src.utils.logging_utils import get_logger
from src.utils.paths import OUTPUTS_DIR

log = get_logger(__name__)


def default_index_remap_path() -> Path:
    """
    Conventional location for the active mapping produced by split:
        outputs/mappings/latest.json
    """
    p = OUTPUTS_DIR / "mappings" / "latest.json"
    log.debug("mapping.default_path", extra={"path": str(p)})
    return p

def read_index_remap(path: Path | str) -> Dict[str, str]:
    """
    Load an index_remap.json mapping index (str) -> class_name (str).
    Validates schema & ordering keys.
    """
    p = Path(path)
    log.debug("mapping.read.start", extra={"path": str(p)})

    if not p.exists():
        log.error("mapping.read.missing", extra={"path": str(p)})
        raise FileNotFoundError(f"index_remap not found: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("mapping.read.failed", extra={"path": str(p), "error": str(e)})
        raise RuntimeError(f"Failed to read index_remap at {p}: {e}") from e

    if not isinstance(data, dict) or not data:
        log.error("mapping.read.bad_schema", extra={"path": str(p), "type": type(data).__name__})
        raise ValueError(f"index_remap at {p} must be a non-empty dict")

    bad_keys = [k for k in data.keys() if not (isinstance(k, str) and k.isdigit())]
    bad_vals = [k for k, v in data.items() if not (isinstance(v, str) and v)]
    if bad_keys or bad_vals:
        log.error("mapping.read.invalid_entries", extra={"bad_keys": bad_keys, "bad_vals": bad_vals})
        raise ValueError("index_remap contains invalid keys or values")

    log.debug("mapping.read.ok", extra={"num_classes": len(data)})
    return data


def expected_classes_from_remap(idx_to_class: Dict[str, str]) -> List[str]:
    """
    Return classes ordered by integer index (0..N-1).
    """
    try:
        ordered = [cls for _, cls in sorted(idx_to_class.items(), key=lambda kv: int(kv[0]))]
        log.debug("mapping.expected_classes", extra={"num_classes": len(ordered), "classes": ordered})
        return ordered
    except Exception as e:
        log.error("mapping.expected_classes.error", extra={"error": str(e)})
        raise ValueError(f"index_remap has non-contiguous or invalid indices: {e}") from e
    
def verify_dataset_classes(
    ds_classes: List[str],
    expected_classes: List[str],
    *,
    strict: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Compare dataset classes (e.g., ImageFolder.classes) against expected.
    - strict=True: raise on mismatch
    - strict=False: return (False, message) on mismatch; (True, None) if ok
    """
    if ds_classes == expected_classes:
        log.debug("mapping.verify.ok", extra={"classes": ds_classes})
        return True, None

    msg = (
        "Class mapping mismatch.\n"
        f"Expected: {expected_classes}\n"
        f"Found:    {ds_classes}"
    )
    if strict:
        log.error("mapping.verify.mismatch_strict", extra={"expected": expected_classes, "found": ds_classes})
        raise RuntimeError(msg)
    else:
        log.warning("mapping.verify.mismatch_non_strict", extra={"expected": expected_classes, "found": ds_classes})
        return False, msg


def write_index_remap(
    classes: List[str],
    *,
    dataset: Optional[str] = None,
    use_dataset_subdir: bool = False,
) -> Path:
    """
    Create and save a deterministic index_remap.json from an ordered class list.

    Primary location (recommended):
      - outputs/mappings[/<owner>/<slug>]/latest.json
        and a timestamped history file: YYYYMMDD_HHMMSSZ__<N>cls.json

    Returns:
      Path to 'latest.json'
    """
    if not classes:
        log.error("mapping.write.empty_classes")
        raise ValueError("write_index_remap: classes list cannot be empty")

    index_remap = {str(i): cls for i, cls in enumerate(classes)}
    base = _mapping_base_dir(dataset, use_dataset_subdir)

    latest_path = base / "latest.json"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    history_path = base / f"{ts}__{len(classes)}cls.json"

    payload = json.dumps(index_remap, indent=2)

    try:
        latest_path.write_text(payload, encoding="utf-8")
        history_path.write_text(payload, encoding="utf-8")
    except Exception as e:
        log.error("mapping.write.failed", extra={"base": str(base), "error": str(e)})
        raise

    log.info("mapping.write.ok", extra={
        "latest_path": str(latest_path),
        "history_path": str(history_path),
        "num_classes": len(classes),
        "classes": classes,
        "dataset": dataset if use_dataset_subdir else None,
    })
    return latest_path


def copy_index_remap(src: Path | str, dst_dir: Path | str) -> Path:
    """
    Copy an existing mapping into a destination directory (e.g., models/<run_id>/).
    """
    src = Path(src)
    dst_dir = Path(dst_dir)
    log.debug("mapping.copy.start", extra={"src": str(src), "dst_dir": str(dst_dir)})

    if not src.exists():
        log.error("mapping.copy.missing_source", extra={"src": str(src)})
        raise FileNotFoundError(f"copy_index_remap: source not found: {src}")

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "index_remap.json"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as e:
        log.error("mapping.copy.failed", extra={"src": str(src), "dst_dir": str(dst_dir), "error": str(e)})
        raise

    log.info("mapping.copy.ok", extra={"src": str(src), "dst": str(dst)})
    return dst


def _mapping_base_dir(dataset: Optional[str], use_dataset_subdir: bool) -> Path:
    """
    Decide where to store mapping artefacts under outputs/mappings.
    - If use_dataset_subdir=True and dataset is 'owner/slug', nest under that.
    """
    base = OUTPUTS_DIR / "mappings"
    if use_dataset_subdir and dataset:
        owner, slug = (dataset.split("/", 1) if "/" in dataset else ("unknown", dataset))
        base = base / owner / slug
    base.mkdir(parents=True, exist_ok=True)
    log.debug("mapping.base_dir", extra={"base": str(base)})
    return base