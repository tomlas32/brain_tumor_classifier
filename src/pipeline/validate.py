"""
Validate a dataset pre and post processing.

What this script checks
-----------------------
1) **Class labels**: each image's parent folder name must be in `index_remap.json`.
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
fetch → merge → validate (pre) → cleanup → resize → validate (post) → split → train → evaluate.

Examples
--------
# Config-first
python -m src.pipeline.validate --config configs/validate.yaml

# Config + overrides
python -m src.pipeline.validate --config configs/validate.yaml \
  --override fail_on=error --override exts=.jpg,.png

# Legacy (no config)
python -m src.pipeline.validate --in-dir data/testing_resized --exts all --dup-check
"""

from __future__ import annotations

import json
import hashlib
import time, argparse, os, cv2
from pathlib import Path
from typing import Dict, Tuple, Set

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity as ssim

from datetime import datetime, timezone

from src.utils.logging_utils import get_logger, configure_logging
from src.utils.parser_utils import (
    parse_exts, 
    add_common_logging_args, 
    add_common_config_args,
    add_dataset_args, 
    add_mapping_args,
    DEFAULT_EXTS
    )
from src.utils.paths import DATA_DIR, OUTPUTS_DIR, MERGED_DIR, PROCESSED_DIR

from src.core.mapping import read_index_remap, expected_classes_from_remap
from src.core.config import build_validate_config, to_dict
from src.core.artifacts import read_mapping_pointer
from src.pipeline.resize import resize_and_pad

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

def _encoded_output_sha1(p: Path, size: int = 224) -> str:
    """
    Hash the BYTES that resize.py would actually write:
      1) read with cv2
      2) resize_and_pad(...)  -> identical normalization
      3) encode using the same format implied by the source suffix (jpg/png)
      4) SHA-1 over the encoded byte buffer
    """
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread failed: {p}")

    sq = resize_and_pad(img, size=size)  # same as resize stage
    ext = p.suffix.lower()

    # choose encoder by source suffix (resize preserves the original extension)
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", sq, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # OpenCV default
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", sq, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])  # OpenCV default
    else:
        # fallback: treat as JPEG to approximate cv2.imwrite behavior for other 3‑channel formats
        ok, buf = cv2.imencode(".jpg", sq, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if not ok:
        raise ValueError(f"cv2.imencode failed for {p} ({ext})")

    h = hashlib.sha1()
    h.update(buf.tobytes())
    return h.hexdigest()

def _ssim_similarity(p1: Path, p2: Path, size: int = 224) -> float:
    """Compute SSIM similarity between two images resized & padded to same size."""
    import cv2
    img1 = cv2.imread(str(p1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(p2), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return 0.0
    sq1 = resize_and_pad(img1, size=size)
    sq2 = resize_and_pad(img2, size=size)
    gray1 = cv2.cvtColor(sq1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(sq2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)


def _phash_of_path(p: Path, size: int = 224, hash_size: int = 8, highfreq: int = 4) -> int:
    """
    Returns a 64-bit perceptual hash as an int.
    - highfreq * hash_size => DCT region size (e.g., 32x32 when 4*8).
    """
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread failed: {p}")

    sq = resize_and_pad(img, size=size)
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    f = hash_size * highfreq
    gray = cv2.resize(gray, (f, f), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(gray.astype("float32"))
    dct_low = dct[:hash_size, :hash_size]
    median = float(np.median(dct_low))
    bits = (dct_low > median).astype(np.uint8).flatten()
    # pack 64 bits into int
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)


def _phash_of_path_flip(p: Path, size: int = 224, **kw) -> int:
    """Same as _phash_of_path but on a horizontally flipped normalized square."""
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread failed: {p}")
    sq = resize_and_pad(img, size=size)
    sq = cv2.flip(sq, 1)
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    hash_size = kw.get("hash_size", 8); highfreq = kw.get("highfreq", 4)
    f = hash_size * highfreq
    gray = cv2.resize(gray, (f, f), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(gray.astype("float32"))
    dct_low = dct[:hash_size, :hash_size]
    median = float(np.median(dct_low))
    bits = (dct_low > median).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)


def _encoded_output_sha1_flip(p: Path, size: int = 224) -> str:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread failed: {p}")

    sq = resize_and_pad(img, size=size)
    sq = cv2.flip(sq, 1)  # horizontal flip

    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", sq, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", sq, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok, buf = cv2.imencode(".jpg", sq, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"cv2.imencode failed for flipped {p} ({ext})")

    h = hashlib.sha1()
    h.update(buf.tobytes())
    return h.hexdigest()


def _hamming_int(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def validate_dataset(
    in_dir: str | Path = MERGED_DIR,
    index_remap_path: str | Path | None = OUTPUTS_DIR / "mappings" / "latest.json",
    size: int = 224,
    exts: str = DEFAULT_EXTS,  # parsed by parse_exts()
    dup_check: bool = False,
    warn_low_std: float = 3.0,
    min_file_bytes: int = 1024,
    *,
    enforce_size: bool = True,    # pre: False, post: True
    require_rgb: bool = True,     # pre: False, post: True
    phash: bool = False,  
    phash_thresh: int = 8,
    ssim_thresh: float = 0.90,
) -> dict:
    """
    Validate a dataset directory tree.

    - If index_remap_path is None, BAD_LABEL checks are skipped (pre-validate).
    - If enforce_size/require_rgb are False, those checks are skipped (pre-validate).
    """
    from collections import defaultdict
    t0 = time.time()

    in_dir = Path(in_dir)

    allowed_labels: Set[str] | None = None
    if index_remap_path is not None:
        try:
            allowed_labels = _load_valid_classes(Path(index_remap_path))
        except Exception as e:
            # Let guardrails handle if caller insists on a path that doesn't parse
            raise

    size_expected: Tuple[int, int] = (size, size)
    exts_set = parse_exts(exts)  # empty set => accept any

    n_files_seen = 0
    n_images_ok = 0
    n_errors = 0
    n_warnings = 0
    per_class_counts: Dict[str, int] = {}

    seen_file_hashes: Dict[str, Path] = {}
    seen_content_hashes: Dict[str, Path] = {}
    seen_phashes: list[tuple[int, Path]] = []

    errors_by_type = defaultdict(int)
    warnings_by_type = defaultdict(int)
    findings = []
    dup_groups: Dict[str, list[str]] = {}
    subset_name = Path(in_dir).name

    def _record(kind: str, code: str, path: Path, *, label: str | None = None,
                details: dict | None = None, sha1: str | None = None,
                duplicate_of: str | None = None):
        rec = {
            "path": str(path),
            "label": label,
            "subset": subset_name,
            "kind": kind,
            "code": code,
        }
        if details: rec["details"] = details
        if sha1: rec["sha1"] = sha1
        if duplicate_of: rec["duplicate_of"] = str(duplicate_of)
        findings.append(rec)
        (errors_by_type if kind == "error" else warnings_by_type)[code] += 1

    log.info("validate.start", extra={
        "in_dir": str(in_dir),
        "size_expected": size_expected,
        "exts_arg": exts,
        "exts_effective": sorted(exts_set) if exts_set else "ALL",
        "dup_check": dup_check,
        "warn_low_std": warn_low_std,
        "min_file_bytes": min_file_bytes,
        "enforce_size": enforce_size,
        "require_rgb": require_rgb,
        "has_allowed_labels": allowed_labels is not None,
    })

    for p in sorted(in_dir.rglob("*"), key=lambda x: str(x)):
        if not p.is_file():
            continue
        n_files_seen += 1

        # Extension filter
        if exts_set and p.suffix.lower() not in exts_set:
            log.error(f"[BAD_EXT] {p} - {p.suffix.lower()} not in {sorted(exts_set)}")
            n_errors += 1
            _record("error", "BAD_EXT", p, label=p.parent.name,
                    details={"ext": p.suffix.lower(), "allowed": sorted(exts_set)})
            continue

        label = p.parent.name

        # BAD_LABEL only if we have an allowed set
        if allowed_labels is not None and label not in allowed_labels:
            log.error(f"[BAD_LABEL] {p} - '{label}' not in {sorted(allowed_labels)}")
            n_errors += 1
            _record("error", "BAD_LABEL", p, label=label,
                    details={"allowed": sorted(allowed_labels)})

        # Tiny file warning (pre I/O)
        try:
            nbytes = p.stat().st_size
            if nbytes < min_file_bytes:
                log.warning(f"[TINY_FILE] {p} - {nbytes} bytes")
                n_warnings += 1
                _record("warning", "TINY_FILE", p, label=label,
                        details={"bytes": int(nbytes), "min_bytes": int(min_file_bytes)})
        except Exception as e:
            log.warning(f"[STAT_FAIL] {p} - {e}")
            n_warnings += 1
            _record("warning", "STAT_FAIL", p, label=label, details={"error": str(e)})

        # Read image
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                actual_mode = im.mode
                actual_size = im.size
                # For analysis (all-black/white + std), use RGB array
                arr = np.asarray(im.convert("RGB"))
        except UnidentifiedImageError:
            log.error(f"[UNREADABLE] {p}")
            n_errors += 1
            _record("error", "UNREADABLE", p, label=label)
            continue
        except Exception as e:
            log.error(f"[READ_FAIL] {p} - {e}")
            n_errors += 1
            _record("error", "READ_FAIL", p, label=label, details={"error": str(e)})
            continue

        # Mode check (optional in pre)
        if require_rgb and actual_mode != "RGB":
            log.error(f"[NOT_RGB] {p} - mode={actual_mode}")
            n_errors += 1
            _record("error", "NOT_RGB", p, label=label, details={"mode": actual_mode})

        # Size check (optional in pre)
        if enforce_size and actual_size != size_expected:
            log.error(f"[BAD_SIZE] {p} - {actual_size} != {size_expected}")
            n_errors += 1
            _record("error", "BAD_SIZE", p, label=label,
                    details={"got": list(actual_size), "expected": list(size_expected)})

        # All black/white
        flat = _is_all_black_or_white(arr)
        if flat == "BLACK":
            log.error(f"[ALL_BLACK] {p}")
            n_errors += 1
            _record("error", "ALL_BLACK", p, label=label)
        elif flat == "WHITE":
            log.error(f"[ALL_WHITE] {p}")
            n_errors += 1
            _record("error", "ALL_WHITE", p, label=label)

        # Low variance
        std_val = float(arr.std())
        if std_val < warn_low_std:
            log.warning(f"[LOW_STD] {p} - std={std_val:.3f} < {warn_low_std}")
            n_warnings += 1
            _record("warning", "LOW_STD", p, label=label,
                    details={"std": round(std_val, 6), "threshold": float(warn_low_std)})

        # Duplicate detection (optional)
        if dup_check:
            try:
                file_sig = _file_sha1(p)                       # exact file bytes
                content_sig = _encoded_output_sha1(p, size=size)  # normalized pixels
                content_sig_flip = _encoded_output_sha1_flip(p, size=size)

                is_file_dup = file_sig in seen_file_hashes
                is_content_dup = (content_sig in seen_content_hashes) or (content_sig_flip in seen_content_hashes)

                dup_hit = False
                first_path: Path | None = None
                used_sig = None

                if is_file_dup:
                    dup_hit = True
                    first_path = seen_file_hashes[file_sig]
                    used_sig = file_sig
                elif is_content_dup:
                    dup_hit = True
                    first_path = seen_content_hashes.get(content_sig) or seen_content_hashes.get(content_sig_flip)
                    used_sig = content_sig if content_sig in seen_content_hashes else content_sig_flip

                if dup_hit and first_path is not None:
                    log.error(f"[DUPLICATE] {p} dup of {first_path}")
                    n_errors += 1
                    _record(
                        "error", "DUPLICATE", p, label=label,
                        sha1=used_sig,                     # key used by cleanup
                        duplicate_of=str(first_path),
                        details={"file_sha1": file_sig, "content_sha1": content_sig, "mode": "both"},
                    )
                    dup_groups.setdefault(used_sig, [str(first_path)])
                    if str(p) not in dup_groups[used_sig]:
                        dup_groups[used_sig].append(str(p))
                else:
                    # mark first occurrences
                    seen_file_hashes.setdefault(file_sig, p)
                    seen_content_hashes.setdefault(content_sig, p)
                    seen_content_hashes.setdefault(content_sig_flip, p)

                    if phash:
                        try:
                            ph = _phash_of_path(p, size=size)
                            ph_flip = _phash_of_path_flip(p, size=size)

                            best_path = None
                            best_dist = None
                            best_ssim = None

                            for prev_ph, prev_path in seen_phashes:
                                d1 = _hamming_int(ph, prev_ph)
                                d2 = _hamming_int(ph_flip, prev_ph)
                                d = min(d1, d2)
                                if d > phash_thresh:
                                    continue

                                # Prefer same-class + verify similarity
                                score = _ssim_similarity(p, prev_path, size=size)
                                same_class = (p.parent.name == prev_path.parent.name)

                                if not same_class or score < ssim_thresh:
                                    # Keep the visibility log, but do not record a warning
                                    log.debug(f"[PHASH_REJECTED] {p} vs {prev_path} (d={d}, ssim={score:.3f})")
                                    continue

                                # Track the best acceptable candidate (lowest Hamming, then highest SSIM)
                                if (best_path is None) or (d < best_dist) or (d == best_dist and score > best_ssim):
                                    best_path, best_dist, best_ssim = prev_path, d, score

                            if best_path is not None:
                                log.warning(f"[NEAR_DUP_PHASH] {p} ~ {best_path} (d={best_dist}, ssim={best_ssim:.3f})")
                                n_warnings += 1
                                _record(
                                    "warning", "NEAR_DUP_PHASH", p, label=label,
                                    details={
                                        "hamming": int(best_dist),
                                        "threshold": int(phash_thresh),
                                        "ssim": round(best_ssim, 3)
                                    },
                                    duplicate_of=str(best_path),
                                )

                            # Always index current image’s pHashes after evaluating
                            seen_phashes.append((ph, p))
                            seen_phashes.append((ph_flip, p))

                        except Exception as e:
                            log.warning(f"[PHASH_FAIL] {p} - {e}")
                            n_warnings += 1
                            _record("warning", "PHASH_FAIL", p, label=label, details={"error": str(e)})

            except Exception as e:
                log.warning(f"[HASH_FAIL] {p} - {e}")
                n_warnings += 1
                _record("warning", "HASH_FAIL", p, label=label, details={"error": str(e)})

        per_class_counts[label] = per_class_counts.get(label, 0) + 1

        # OK criteria depend on toggles
        ok = (
            (not require_rgb or actual_mode == "RGB") and
            (not enforce_size or actual_size == size_expected) and
            flat is None
        )
        if ok:
            n_images_ok += 1

    elapsed = time.time() - t0

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
        "errors_by_type": dict(errors_by_type),
        "warnings_by_type": dict(warnings_by_type),
        "per_class_counts": per_class_counts,
        "elapsed_sec": elapsed,
        "findings": findings,
        "duplicate_groups": (
            [{"sha1": k, "paths": v} for k, v in dup_groups.items()] if dup_groups else []
        ),
    }



def main(argv=None) -> int:
    """
    Entry point:
    1) Parse CLI args and configure logging with [validate|RUN_ID].
    2) Guardrails: ensure input dir and mapping exist.
    3) Run validation and apply `--fail-on` policy for exit code.
    """
    parser = argparse.ArgumentParser(description="Validate a dataset (pre or post processing)")
    parser.add_argument("--dup-check", dest="dup_check", action="store_true", help="Enable duplicate detection")
    parser.add_argument("--no-dup-check", dest="dup_check", action="store_false", help="Disable duplicate detection")
    parser.add_argument("--warn-low-std", type=float, default=None,
                        help="Warn if per-image std is below this threshold")
    parser.add_argument("--min-file-bytes", type=int, default=None,
                        help="Warn if file size is below this many bytes")
    parser.add_argument("--fail-on", choices=["error", "warning", "never"], default=None,
                        help="Exit with nonzero code if these severities occur")
    parser.add_argument(
    "--no-write-report",
    dest="write_report",
    action="store_false",
    help="Disable writing a JSON validation report to outputs/validation_reports/ (enabled by default).",
)
    # default ON
    parser.set_defaults(
        write_report=None,        
        dup_check=None,          
        phash=None,                
        phash_thresh=None,         
        ssim_thresh=None,          
    )
    parser.add_argument("--report-tag", type=str, default=None,
                    help="Optional tag to append to report filename, e.g. 'pre' or 'post'.")
    parser.add_argument("--phash", dest="phash", action="store_true",
                    help="Enable perceptual hashing (near-duplicate detection)")
    parser.add_argument("--no-phash", dest="phash", action="store_false",
                    help="Disable perceptual hashing (near-duplicate detection)")
    parser.add_argument("--phash-thresh", type=int, default=None,
                        help="Max Hamming distance for pHash near-duplicates (default: 8)")
    parser.add_argument("--ssim-thresh", type=float, default=None,
                    help="SSIM confirmation threshold for pHash near-duplicates (default: 0.90)")

    # shared mapping flags: --index-remap, --mapping-pointer
    add_mapping_args(parser)
    # shared dataset flags: --in-dir, --size, --exts
    add_dataset_args(parser, with_size=True, with_exts=True)
    # shared config flags: --config, --override, `--dry-run`
    add_common_config_args(parser, include_dry_run=True)
    # shared logging flags: --log-level, --log-file
    add_common_logging_args(parser)

    args = parser.parse_args(argv)

    # Config-first
    cfg = build_validate_config(args.config, overrides=args.override)
    log.info("config.resolved", extra={"config": to_dict(cfg)})

    # Run-aware logging (ties logs across stages)
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    
    log_level = cfg.log_level or args.log_level or "INFO"
    log_file  = cfg.log_file or args.log_file or None

    configure_logging(
        log_level=log_level,
        file_mode="fixed" if log_file else "auto",
        log_file=log_file,
        run_id=run_id,
        stage="merge",
    )

    # choose in_dir: explicit > arg > mode-based default
    if cfg.in_dir is not None:
        in_dir = cfg.in_dir
    elif args.in_dir is not None:
        in_dir = args.in_dir
    else:
        # If strict checks are enabled, assume post-validate → processed; else pre → merged
        strict = bool(getattr(cfg, "enforce_size", True) and getattr(cfg, "require_rgb", True))
        in_dir = PROCESSED_DIR if strict else MERGED_DIR
    
    
    mapping_pointer = cfg.mapping_pointer or args.mapping_pointer
    index_remap = cfg.index_remap or args.index_remap
    size = args.size if args.size is not None else (cfg.size or 224)
    dup_check = args.dup_check if args.dup_check is not None else bool(cfg.dup_check)
    warn_low_std   = args.warn_low_std   if args.warn_low_std   is not None else cfg.warn_low_std
    min_file_bytes = args.min_file_bytes if args.min_file_bytes is not None else cfg.min_file_bytes
    fail_on = (args.fail_on or cfg.fail_on or "error").lower()
    write_report = args.write_report if args.write_report is not None else cfg.write_report
    dry = bool(getattr(args, "dry_run", False) or getattr(cfg, "dry_run", False))
    phash        = args.phash        if args.phash        is not None else bool(cfg.phash)
    phash_thresh = args.phash_thresh if args.phash_thresh is not None else cfg.phash_thresh  # (default 8 in cfg)
    ssim_thresh  = args.ssim_thresh  if args.ssim_thresh  is not None else cfg.ssim_thresh   # (default 0.90 in cfg)
    
    # exts: allow list or 'all' from config; otherwise CLI string
    exts_source = args.exts if args.exts is not None else (cfg.exts or DEFAULT_EXTS)
    exts_set = parse_exts(exts_source)

    
    # ---DRY RUN---
    if dry:
        in_dir_p = Path(in_dir) if in_dir else None
        in_exists = in_dir_p.exists() if in_dir_p else False

        # Pure, tolerant counts (no image I/O). Empty exts_set => accept any.
        def _count_by_class(root: Path, exts: set[str]) -> dict[str, int]:
            if not root or not root.exists():
                return {}
            tallies: dict[str, int] = {}
            for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                cnt = 0
                for q in class_dir.rglob("*"):
                    if q.is_file() and (not exts or q.suffix.lower() in exts):
                        cnt += 1
                if cnt:
                    tallies[class_dir.name] = cnt
            return tallies

        per_class = _count_by_class(in_dir_p, exts_set) if in_exists else {}
        total = sum(per_class.values())

        allowed = None
        if index_remap and Path(index_remap).exists():
            try:
                allowed = sorted(_load_valid_classes(Path(index_remap)))
            except Exception as e:
                log.warning("validate.dry.index_remap_read_failed",
                            extra={"index_remap": str(index_remap), "error": str(e)})

        log.info("dry_run.validate.plan", extra={
            "in_dir": str(in_dir_p) if in_dir_p else None,
            "exists": {"in_dir": in_exists},
            "index_remap": str(index_remap) if index_remap else None,
            "index_remap_exists": bool(index_remap and Path(index_remap).exists()),
            "size_expected": (size, size),
            "exts_effective": sorted(exts_set) if exts_set else ["<any>"],
            "total_files": total,
            "per_class": per_class,
            "allowed_labels": allowed,
        })

        print("\n[DRY-RUN] Validate plan")
        print(f"  in_dir:        {in_dir_p if in_dir_p else '<none>'}  ({'exists' if in_exists else 'MISSING'})")
        print(f"  index_remap:   {index_remap if index_remap else '<none>'}  "
            f"({ 'exists' if (index_remap and Path(index_remap).exists()) else 'MISSING'})")
        print(f"  size:          {size} x {size}")
        print(f"  exts:          {', '.join(sorted(exts_set)) if exts_set else '<any>'}")
        if per_class:
            print("  Per-class counts:")
            for k in sorted(per_class):
                print(f"    {k:15s} -> {per_class[k]:5d}")
        else:
            print("  Per-class counts: <none>")
        if allowed is not None:
            print("  Allowed labels (from index_remap): " + ", ".join(allowed))
        print("\n[DRY-RUN] No images will be opened; no report will be written.")
        return 0
    # ---END DRY RUN---

    if mapping_pointer:
        try:
            mp = read_mapping_pointer(mapping_pointer)
            index_remap = Path(mp["path"])
            log.info("validate.mapping_pointer_used", extra={
                "pointer": str(mapping_pointer),
                "index_remap": str(index_remap),
                "num_classes": mp.get("num_classes"),
            })
        except Exception as e:
            log.warning("validate.mapping_pointer_error_nonfatal",
                        extra={"pointer": str(mapping_pointer), "error": str(e)})

    # Guardrails
    if not Path(in_dir).exists():
        log.error("validate.in_dir_missing", extra={"in_dir": str(in_dir)})
        print(f"--in-dir not found: {in_dir}")
        return 2
    if index_remap is not None and not Path(index_remap).exists():
        log.error("validate.index_remap_missing", extra={"index_remap": str(index_remap)})
        print(f"--index-remap not found: {index_remap}")
        return 2

    summary = validate_dataset(
        in_dir=in_dir,
        index_remap_path=index_remap if index_remap else None,
        size=size,
        exts=exts_source,
        dup_check=dup_check,
        warn_low_std=warn_low_std,
        min_file_bytes=min_file_bytes,
        enforce_size=bool(getattr(cfg, "enforce_size", True)),
        require_rgb=bool(getattr(cfg, "require_rgb", True)),
        phash=phash,                
        phash_thresh=phash_thresh,
        ssim_thresh=float(ssim_thresh),  
    )

    if write_report:
        VALIDATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_tag = getattr(cfg, "report_tag", None) or getattr(args, "report_tag", None)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        suffix = f"_{report_tag}" if report_tag else ""
        report_path = VALIDATION_REPORTS_DIR / f"validation_{run_id}_{ts}{suffix}.json"
        
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

    if summary.get("errors_by_type"):
        print("  Errors by type:", {k: summary["errors_by_type"][k] for k in sorted(summary["errors_by_type"])})
    if summary.get("warnings_by_type"):
        print("  Warnings by type:", {k: summary["warnings_by_type"][k] for k in sorted(summary["warnings_by_type"])})


    if fail_on == "error" and summary["errors"] > 0:
        return 1
    if fail_on == "warning" and (summary["errors"] > 0 or summary["warnings"] > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())