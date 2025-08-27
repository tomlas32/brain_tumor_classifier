"""
Merge Kaggle 'Training' and 'Testing' into a single canonical dataset:

    data/merged/<class>/*

- Reads standardized fetch pointer (outputs/pointers/fetch/<owner>/<slug>/latest.json).
- Scans both roots, filters by --exts (supports 'all' and '+webp' semantics via parser_utils).
- Copies files into MERGED_DIR with collision-safe naming.
- Writes a JSON manifest under outputs/merge/ with per-class counts and totals.
- Supports --dry-run to print a plan without writing.

Author: Tomasz Lasota
Date: 2025-08-19
Version: 1.0
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse, shutil, time, os, json
from datetime import datetime, timezone
from typing import Dict, List

from src.utils.paths import DATA_DIR, OUTPUTS_DIR, MERGED_DIR
from src.core.config import build_merge_config, to_dict
from src.utils.paths import DEFAULT_DATASET
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import (
    add_common_logging_args, 
    add_exts_arg, 
    parse_exts,
    add_common_config_args,
    )
from src.core.artifacts import read_fetch_pointer

log = get_logger(__name__)

def _pointer_path_for(slug: str) -> Path:
    owner, name = (slug.split("/", 1) if "/" in slug else ("_unknown_", slug))
    return OUTPUTS_DIR / "pointers" / "fetch" / owner / name / "latest.json"

def _empty_dir(d: Path) -> None:
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

def gather_by_class(root: Path, exts: set[str]) -> Dict[str, List[Path]]:
    """
    Return dict: class_name -> list[Path] for images under root/<class>/*
    Accepts any extension if exts is empty set (parse_exts('all')).
    """
    mapping: Dict[str, List[Path]] = defaultdict(list)
    if not root.exists():
        log.warning("merge.input_root_missing", extra={"root": str(root)})
        return mapping
    accept_any = len(exts) == 0
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and (accept_any or p.suffix.lower() in exts):
                mapping[cls].append(p)
    return mapping

def safe_copy(src: Path, dst: Path) -> Path:
    """Copy src -> dst, avoiding filename collisions with __{i} suffix. Returns final dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        i = 1
        while True:
            cand = dst.with_name(f"{dst.stem}__{i}{dst.suffix}")
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)
    return dst

def _render_plan(combined: Dict[str, List[Path]]) -> str:
    total = sum(len(v) for v in combined.values())
    lines = []
    lines.append("\n[DRY-RUN] Merge plan")
    lines.append("  Destination:  data/merged")
    lines.append("  Per-class counts:")
    for cls in sorted(combined.keys()):
        lines.append(f"    {cls:15s} -> {len(combined[cls]):5d}")
    lines.append(f"\n  Total files to copy: {total}")
    lines.append("\n[DRY-RUN] No files will be created, moved, or modified.")
    return "\n".join(lines)

def _write_manifest(manifest: dict) -> Path:
    out_dir = OUTPUTS_DIR / "merge"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    latest = out_dir / "latest.json"
    stamped = out_dir / f"manifest_{ts}.json"
    with stamped.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    # Update latest pointer
    with latest.open("w", encoding="utf-8") as f:
        json.dump({"latest": str(stamped.resolve()), "manifest": manifest}, f, ensure_ascii=False, indent=2)
    return stamped

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge Kaggle Training/Testing into data/merged.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Kaggle slug (owner/dataset) used to auto-locate the fetch pointer.")
    parser.add_argument("--pointer", type=Path, default=None,
                        help="Optional explicit path to the fetch pointer JSON (overrides --dataset).")
    parser.add_argument("--clear-dest", action="store_true",
                        help="Delete all existing files/dirs in DATA_DIR/merged before writing.")
    
    add_common_config_args(parser)         # --config, --override, --dry-run
    add_common_logging_args(parser)        # --log-level, --log-file
    add_exts_arg(parser)                   # --exts with your '+webp' semantics
    args = parser.parse_args(argv)

    # Structured logging
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="merge")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="merge")
    t0 = time.time()

    # Resolve config (config-first with CLI fallback)
    cfg = build_merge_config(args.config, overrides=args.override)
    log.info("config.resolved", extra={"config": to_dict(cfg)})

    dataset_slug = os.getenv("DATASET_SLUG", cfg.dataset or args.dataset)
    pointer = cfg.pointer or args.pointer or _pointer_path_for(dataset_slug)
    exts_source = cfg.exts if cfg.exts is not None else args.exts
    exts = parse_exts(exts_source)  # empty set => accept any
    clear_dest = bool(cfg.clear_dest or args.clear_dest)
    do_dry = bool(getattr(cfg, "dry_run", False) or getattr(args, "dry_run", False))

    # Load fetch pointer (unless doing an empty dry-run)
    if do_dry and not Path(pointer).exists():
        # Show an empty plan with informative placeholders
        log.info("dry_run.merge.no_pointer", extra={"pointer": str(pointer), "dataset": dataset_slug})
        print("\n[DRY-RUN] Merge plan\n  (No fetch pointer found; nothing to do)")
        return 0

    try:
        fetch_ptr = read_fetch_pointer(pointer)
    except Exception as e:
        log.error("merge.pointer_read_error", extra={"pointer": str(pointer), "error": str(e)})
        return 2

    dataset_root = Path(fetch_ptr["dataset_root"])
    src_training = Path(fetch_ptr.get("training_dir") or (dataset_root / "Training"))
    src_testing  = Path(fetch_ptr.get("testing_dir")  or (dataset_root / "Testing"))

    if not src_training.exists() or not src_testing.exists():
        log.error("merge.source_dirs_missing", extra={
            "src_training": str(src_training),
            "src_testing": str(src_testing),
            "dataset_root": str(dataset_root),
        })
        return 2

    log.info("merge.fetch_pointer_loaded", extra={
        "pointer": str(pointer),
        "dataset": fetch_ptr.get("dataset"),
        "dataset_root": str(dataset_root),
        "src_training": str(src_training),
        "src_testing": str(src_testing),
        "exts": sorted(exts) if exts else ["<any>"],
    })

    # Pool both roots
    combined = defaultdict(list)
    for src_root in (src_training, src_testing):
        for cls, paths in gather_by_class(src_root, exts).items():
            combined[cls].extend(paths)

    # DRY-RUN plan
    if do_dry:
        log.info("dry_run.merge.plan", extra={
            "dest": str(MERGED_DIR),
            "totals": {cls: len(paths) for cls, paths in combined.items()},
            "total": sum(len(v) for v in combined.values()),
        })
        print(_render_plan(combined))
        return 0

    # Prepare destination
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    if clear_dest:
        _empty_dir(MERGED_DIR)
        log.info("merge.cleared_dest", extra={"dest": str(MERGED_DIR)})

    # Copy
    per_class_counts = {}
    for cls in sorted(combined.keys()):
        count = 0
        for src in sorted(combined[cls]):
            dst = MERGED_DIR / cls / src.name
            safe_copy(src, dst)
            count += 1
        per_class_counts[cls] = count
        log.debug("merge.class_done", extra={"class": cls, "copied": count})

    elapsed = time.time() - t0
    total = sum(per_class_counts.values())
    manifest = {
        "run_id": run_id,
        "pointer": str(pointer),
        "dataset": fetch_ptr.get("dataset"),
        "dataset_root": str(dataset_root),
        "source_training": str(src_training),
        "source_testing": str(src_testing),
        "dest_merged": str(MERGED_DIR),
        "exts": sorted(list(exts)) if exts else ["<any>"],
        "per_class": per_class_counts,
        "total_copied": total,
        "elapsed_s": round(elapsed, 2),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = _write_manifest(manifest)

    log.info("merge.done", extra={
        "dest": str(MERGED_DIR),
        "total": total,
        "elapsed_s": round(elapsed, 2),
        "manifest": str(manifest_path),
    })

    print("Merge complete →", str(MERGED_DIR))
    for cls in sorted(per_class_counts.keys()):
        print(f"{cls:15s} -> {per_class_counts[cls]:5d}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
