"""
Validate a *resized* dataset directory tree after the `resize` step.

What this script checks
-----------------------
1) **Class labels**: each image’s parent folder name must be in `index_remap.json`.
2) **Image readability**: file opens with PIL; truncated / unreadable files are errors.
3) **Mode**: images should be RGB (post-resize standardization).
4) **Dimensions**: every image must be exactly (size x size).
5) **All-black / all-white**: hard errors if all pixels are 0 or 255.
6) **Low variance**: warn if per-image std < threshold (very low contrast).
7) **Duplicates (optional)**: SHA-1 based detection of identical content.
8) **Extensions**: default image extensions (or `+ext` additions); `'all'` disables filtering.

Logging & exit codes
--------------------
- Run-aware, stage-aware logs: every log line includes [validate|RUN_ID].
- `--fail-on error` (default): exit 1 if any errors are found.
- `--fail-on warning`: exit 1 if any errors OR warnings are found.
- Returns 2 when inputs are missing (guardrails).

Typical pipeline order
----------------------
fetch → split → resize → **validate** → train → evaluate

Examples
--------
# Validate resized training set with defaults (224px, default extensions)
python -m src.pipeline.validate --in-dir data/training_resized

# Accept any file extension and enable duplicate detection
python -m src.pipeline.validate --exts all --dup-check

# Stricter mode: fail on warnings as well
python -m src.pipeline.validate --fail-on warning
"""

from __future__ import annotations

import json
import hashlib
import time, argparse, os
from pathlib import Path
from typing import Dict, Tuple, Set

import numpy as np
from PIL import Image, UnidentifiedImageError

from datetime import datetime, timezone

from src.utils.logging_utils import get_logger, configure_logging
from src.utils.parser_utils import parse_exts, add_common_logging_args, DEFAULT_EXTS
from src.utils.paths import DATA_DIR, OUTPUTS_DIR
from src.core.mapping import read_index_remap, expected_classes_from_remap

VALIDATION_REPORTS_DIR = OUTPUTS_DIR / "validation_reports"

log = get_logger(__name__)

def _load_valid_classes(index_remap_path: Path) -> Set[str]:
    """
    Resolve the **allowed class names** from an index remap file.

    Accepts the canonical format produced by `split.py`:
        {"0":"glioma","1":"meningioma","2":"notumor","3":"pituitary"}

    Returns
    -------
    set[str]
        The set of allowed class names.
    """
    idx_to_class = read_index_remap(index_remap_path)
    ordered = expected_classes_from_remap(idx_to_class)
    return set(ordered)


def _is_all_black_or_white(img_arr: np.ndarray) -> str | None:
    """
    Returns "BLACK" if all pixels are 0, "WHITE" if all 255, else None.
    Works for RGB arrays (H, W, 3) produced by PIL -> np.asarray.
    """
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
        if np.max(img_arr) == 0:
            return "BLACK"
        if np.min(img_arr) == 255:
            return "WHITE"
    return None


def _file_sha1(p: Path) -> str:
    """
    Compute SHA-1 hash for duplicate detection (I/O efficient chunked read).
    """
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(
    in_dir: str | Path = DATA_DIR / "training_resized",
    index_remap_path: str | Path = OUTPUTS_DIR / "mappings" / "latest.json",
    size: int = 224,
    exts: str = DEFAULT_EXTS,  # parsed by parse_exts()
    dup_check: bool = False,
    warn_low_std: float = 3.0,      # warn if per-image std < this (very low contrast)
    min_file_bytes: int = 1024,     # warn if file under 1 KB (likely broken)
) -> dict:
    """
    Validate a *resized* dataset directory tree after your resize step.

    Parameters
    ----------
    in_dir : Path
        Root directory to validate (typically data/training_resized or testing_resized).
    index_remap_path : Path
        Path to index_remap.json that defines the allowed class names.
    size : int
        Expected square dimension (e.g., 224).
    exts : str
        Extension argument interpreted by `parse_exts()`:
          * default list (e.g., .jpg,.jpeg,.png)
          * '+webp,+gif' to add formats
          * 'all' to accept any extension (empty-set sentinel)
    dup_check : bool
        Enable SHA-1 duplicate detection (warning-level findings).
    warn_low_std : float
        Warn if per-image pixel std is below this threshold.
    min_file_bytes : int
        Warn if file size is below this many bytes.

    Returns
    -------
    dict
        Summary with counts, per-class tallies, and elapsed seconds.

    Notes
    -----
    - Logs ERROR for fatal issues; WARNING for suspicious-but-usable items.
    - Keeps scanning after errors to surface as many issues as possible.
    """
    t0 = time.time()

    in_dir = Path(in_dir)
    index_remap_path = Path(index_remap_path)
    allowed_labels = _load_valid_classes(index_remap_path)
    size_expected: Tuple[int, int] = (size, size)

    # Normalize extensions once (empty set => accept all)
    exts_set = parse_exts(exts)  # returns set[str] or empty set for "all"

    n_files_seen = 0
    n_images_ok = 0
    n_errors = 0
    n_warnings = 0
    per_class_counts: Dict[str, int] = {}
    seen_hashes: Dict[str, Path] = {}

    log.info("validate.start", extra={
        "in_dir": str(in_dir),
        "size_expected": size_expected,
        "exts_arg": exts,
        "exts_effective": sorted(exts_set) if exts_set else "ALL",
        "dup_check": dup_check,
        "warn_low_std": warn_low_std,
        "min_file_bytes": min_file_bytes
    })

    for p in in_dir.rglob("*"):
        if not p.is_file():
            continue
        n_files_seen += 1

        # Extension check (skip if 'all' -> exts_set == empty)
        if exts_set and p.suffix.lower() not in exts_set:
            log.error(f"[BAD_EXT] {p} - Extension {p.suffix.lower()} not in {sorted(exts_set)}")
            n_errors += 1
            continue

        # Label from parent folder
        label = p.parent.name
        if label not in allowed_labels:
            log.error(f"[BAD_LABEL] {p} - Folder '{label}' not in allowed classes {sorted(allowed_labels)}")
            n_errors += 1  # continue checks to surface more issues

        # File size sanity
        try:
            nbytes = p.stat().st_size
            if nbytes < min_file_bytes:
                log.warning(f"[TINY_FILE] {p} - {nbytes} bytes")
                n_warnings += 1
        except Exception as e:
            log.warning(f"[STAT_FAIL] {p} - Could not stat file: {e}")
            n_warnings += 1

        # Read & verify image
        try:
            with Image.open(p) as im:
                im.verify()  # detect truncation
            with Image.open(p) as im:
                im = im.convert("RGB")  # normalize for analysis/logging
                actual_mode = im.mode
                actual_size = im.size
                arr = np.asarray(im)
        except UnidentifiedImageError:
            log.error(f"[UNREADABLE] {p} - PIL cannot identify image")
            n_errors += 1
            continue
        except Exception as e:
            log.error(f"[READ_FAIL] {p} - Failed to read: {e}")
            n_errors += 1
            continue

        # Mode check (should be RGB post-resize)
        if actual_mode != "RGB":
            log.error(f"[NOT_RGB] {p} - Mode={actual_mode}")
            n_errors += 1

        # Size check
        if actual_size != size_expected:
            log.error(f"[BAD_SIZE] {p} - Got {actual_size}, expected {size_expected}")
            n_errors += 1

        # All black/white detection
        flat = _is_all_black_or_white(arr)
        if flat == "BLACK":
            log.error(f"[ALL_BLACK] {p} - All pixels are zero")
            n_errors += 1
        elif flat == "WHITE":
            log.error(f"[ALL_WHITE] {p} - All pixels are 255")
            n_errors += 1

        # Low-variance (very low contrast) warning
        std_val = float(arr.std())
        if std_val < warn_low_std:
            log.warning(f"[LOW_STD] {p} - std={std_val:.3f} < {warn_low_std}")
            n_warnings += 1

        # Duplicate detection (optional)
        if dup_check:
            try:
                sig = _file_sha1(p)
                if sig in seen_hashes:
                    log.warning(f"[DUPLICATE] {p} - Duplicate of {seen_hashes[sig]}")
                    n_warnings += 1
                else:
                    seen_hashes[sig] = p
            except Exception as e:
                log.warning(f"[HASH_FAIL] {p} - SHA1 failed: {e}")
                n_warnings += 1

        per_class_counts[label] = per_class_counts.get(label, 0) + 1

        # OK if readable, RGB, expected size, and not all-black/white
        is_ok = (actual_mode == "RGB" and actual_size == size_expected and flat is None)
        if is_ok:
            n_images_ok += 1

    elapsed = time.time() - t0

    # Summary
    log.info("validate.summary", extra={
        "in_dir": str(in_dir),
        "elapsed_sec": round(elapsed, 2),
        "scanned": n_files_seen,
        "ok": n_images_ok,
        "errors": n_errors,
        "warnings": n_warnings
    })
    if per_class_counts:
        for k in sorted(per_class_counts):
            log.info(f"[CLASS_COUNT] {k}: {per_class_counts[k]}")

    return {
        "in_dir": str(in_dir),
        "size_expected": size_expected,
        "exts_effective": sorted(exts_set) if exts_set else "ALL",
        "scanned": n_files_seen,
        "ok": n_images_ok,
        "errors": n_errors,
        "warnings": n_warnings,
        "per_class_counts": per_class_counts,
        "elapsed_sec": elapsed,
    }


def main(argv=None) -> int:
    """
    Entry point:
    1) Parse CLI args and configure logging with [validate|RUN_ID].
    2) Guardrails: ensure input dir and mapping exist.
    3) Run validation and apply `--fail-on` policy for exit code.
    """
    parser = argparse.ArgumentParser(description="Validate a resized dataset before training")
    parser.add_argument("--in-dir", type=Path, default=DATA_DIR / "training_resized",
                        help="Path to resized dataset directory (e.g., data/training_resized)")
    parser.add_argument("--index-remap", type=Path, default=OUTPUTS_DIR / "mappings" / "latest.json",
                        help="Path to index_remap.json that defines allowed classes")
    parser.add_argument("--size", type=int, default=224, help="Expected image size (square)")
    parser.add_argument("--exts", type=str, default=DEFAULT_EXTS,
                        help="Comma-separated extensions. Use +ext to add; 'all' to accept any.")
    parser.add_argument("--dup-check", action="store_true", help="Enable duplicate detection (SHA-1)")
    parser.add_argument("--warn-low-std", type=float, default=3.0,
                        help="Warn if per-image std is below this threshold")
    parser.add_argument("--min-file-bytes", type=int, default=1024,
                        help="Warn if file size is below this many bytes")
    parser.add_argument("--fail-on", choices=["error", "warning", "never"], default="error",
                        help="Exit with nonzero code if these severities occur")
    parser.add_argument(
    "--no-write-report",
    dest="write_report",
    action="store_false",
    help="Disable writing a JSON validation report to outputs/validation_reports/ (enabled by default).",
)
    # default ON
    parser.set_defaults(write_report=True)

    # shared logging flags: --log-level, --log-file
    add_common_logging_args(parser)

    args = parser.parse_args(argv)

    # Run-aware logging (ties logs across stages)
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="validate")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="validate")

    # Guardrails
    if not args.in_dir.exists():
        log.error("validate.in_dir_missing", extra={"in_dir": str(args.in_dir)})
        print(f"--in-dir not found: {args.in_dir}")
        return 2
    if not args.index_remap.exists():
        log.error("validate.index_remap_missing", extra={"index_remap": str(args.index_remap)})
        print(f"--index-remap not found: {args.index_remap}")
        return 2

    summary = validate_dataset(
        in_dir=args.in_dir,
        index_remap_path=args.index_remap,
        size=args.size,
        exts=args.exts,
        dup_check=args.dup_check,
        warn_low_std=args.warn_low_std,
        min_file_bytes=args.min_file_bytes,
    )

    if args.write_report:
        VALIDATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        report_path = VALIDATION_REPORTS_DIR / f"validation_{run_id}_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info("validation_report_written", extra={"path": str(report_path)})

    # Compact human summary to stdout
    print(
        f"Validated {summary['scanned']} files | "
        f"OK: {summary['ok']} | "
        f"Errors: {summary['errors']} | "
        f"Warnings: {summary['warnings']}"
    )

    if args.fail_on == "error" and summary["errors"] > 0:
        return 1
    if args.fail_on == "warning" and (summary["errors"] > 0 or summary["warnings"] > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())