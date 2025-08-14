from __future__ import annotations

import json
import hashlib
import time, argparse
from pathlib import Path
from typing import Dict, Tuple, Set

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils.logging_utils import get_logger, configure_logging
from src.utils.parser_utils import parse_exts, add_common_logging_args, DEFAULT_EXTS
from src.utils.paths import DATA_DIR, OUTPUTS_DIR

log = get_logger(__name__)

def _load_valid_classes(index_remap_path: Path) -> Set[str]:
    """
    Expects index_remap.json to map label indices -> class names, e.g.
    {"0":"glioma","1":"meningioma","2":"notumor","3":"pituitary"}.
    Returns the set of allowed class names.
    """
    with open(index_remap_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {str(v) for v in mapping.values()}


def _is_all_black_or_white(img_arr: np.ndarray) -> str | None:
    """
    Returns "BLACK" if all pixels are 0, "WHITE" if all 255, else None.
    """
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
        if np.max(img_arr) == 0:
            return "BLACK"
        if np.min(img_arr) == 255:
            return "WHITE"
    return None


def _file_sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(
    in_dir: str | Path = DATA_DIR / "training",
    index_remap_path: str | Path = OUTPUTS_DIR / "mappings" / "latest.json",
    size: int = 224,
    exts: str = DEFAULT_EXTS,  # parsed by parse_exts()
    dup_check: bool = False,
    warn_low_std: float = 3.0,      # warn if per-image std < this (very low contrast)
    min_file_bytes: int = 1024,     # warn if file under 1 KB (likely broken)
) -> dict:
    """
    Validate a *resized* dataset directory tree after your resize step.

    - Uses src.utils.parser_utils.parse_exts to normalize/interpret --exts.
      * '+webp,+gif' adds to defaults
      * 'all' means accept any extension (no filtering)  ← empty set sentinel
    - Logs ERROR for fatal issues; WARNING for suspicious-but-usable items.
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

    parser = argparse.ArgumentParser(description="Validate a resized dataset before training")
    parser.add_argument("--in-dir", type=Path, default=DATA_DIR / "training",
                        help="Path to resized dataset directory")
    parser.add_argument("--index-remap", type=Path, default=OUTPUTS_DIR / "mappings" / "latest.json",
                        help="Path to index_remap.json")
    parser.add_argument("--size", type=int, default=224, help="Expected image size")
    parser.add_argument("--exts", type=str, default=DEFAULT_EXTS,
                        help="Comma-separated extensions. Use +ext to add; 'all' to accept any.")
    parser.add_argument("--dup-check", action="store_true", help="Enable duplicate detection")
    parser.add_argument("--warn-low-std", type=float, default=3.0,
                        help="Warn if per-image std is below this threshold")
    parser.add_argument("--min-file-bytes", type=int, default=1024,
                        help="Warn if file size is below this many bytes")
    parser.add_argument("--fail-on", choices=["error", "warning", "never"], default="error",
                        help="Exit with nonzero code if these severities occur")

    # shared logging flags: --log-level, --log-file
    add_common_logging_args(parser)

    args = parser.parse_args(argv)

    # Configure rotating/fixed logging as in your other scripts
    configure_logging(log_level=args.log_level, log_file=args.log_file)

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