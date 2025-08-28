"""
Split the canonical processed store (data/processed/<class>/*) into train/val/test.

Writes:
    <output_root>/training/<class>/*
    <output_root>/validation/<class>/*
    <output_root>/testing/<class>/*

- De-duplicates file paths, shuffles with --seed.
- Ensures at least one test image for any non-empty class.
- Avoids filename collisions by appending a numeric suffix.

Example:
# Config-first
python -m src.pipeline.split --config configs/split.yaml

# Config + overrides
python -m src.pipeline.split --config configs/split.yaml \
  --override test_frac=0.2 --override clear_dest=false

# Legacy (no config)
python -m src.pipeline.split --dataset owner/name --test-frac 0.2

Author: Tomasz Lasota
Date: 2025-08-16
Version: 1.3  
"""

from pathlib import Path
import argparse, random, shutil, os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from src.utils.paths import DATA_DIR, DEFAULT_DATASET, PROCESSED_DIR
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import (
    add_common_logging_args, 
    add_exts_arg, 
    parse_exts, 
    add_common_config_args
    )

from src.core.mapping import write_index_remap as mapping_write_index_remap, copy_index_remap
from src.core.config import build_split_config
from src.core.artifacts import write_mapping_pointer

from src.pipeline.split_planner import plan_split, make_log_extra, render_human

log = get_logger(__name__)


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
    - If `exts` is empty, accept any extension.
    """
    mapping: Dict[str, List[Path]] = defaultdict(list)
    if not root.exists():
        log.warning("Input root does not exist: %s", root)
        return mapping
    accept_any = len(exts) == 0
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and (accept_any or p.suffix.lower() in exts):
                mapping[cls].append(p)
    return mapping


def safe_copy(src: Path, dst: Path):
    """Copy src -> dst, avoiding filename collisions with __{i} suffix."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = dst
    if out.exists():
        i = 1
        while True:
            cand = out.with_name(f"{out.stem}__{i}{out.suffix}")
            if not cand.exists():
                out = cand
                break
            i += 1
    shutil.copy2(src, out)


def _class_names_from_dir(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _write_index_and_pointer(
    classes: List[str],
    run_id: str,
    dataset_slug: str,
    save_remap_to_project_root: bool,
    mapping_use_dataset_subdir: bool,
    mapping_write_split_copy: bool,
) -> Path:
    """
    Write index_remap.json + pointer.
    """
    split_root = DATA_DIR
    latest_path = mapping_write_index_remap(
        classes,
        dataset=dataset_slug if mapping_use_dataset_subdir else None,
        use_dataset_subdir=bool(mapping_use_dataset_subdir),
    )
    if mapping_write_split_copy:
        copy_index_remap(latest_path, split_root)
    if save_remap_to_project_root:
        copy_index_remap(latest_path, Path("index_remap.json").parent)

    # Standardized mapping pointer (no fetch_ptr in processed mode)
    ordered_classes = _class_names_from_dir(DATA_DIR / "training")
    write_mapping_pointer(
        classes=ordered_classes,
        index_remap_path=latest_path,
        dataset=dataset_slug,
        index_remap=None,
        run_id=run_id,
        dst_dir=None,
    )
    return latest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Split data/processed into train/val/test.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Used only to label mapping pointers (owner/slug).")
    parser.add_argument("--test-frac", type=float, default=None,
                        help="Fraction per class for final test set (0-1).")
    parser.add_argument("--val-frac", type=float, default=None,
                        help="Fraction per class for validation set (0-1).")
    parser.add_argument("--balance", choices=["none", "equalize"], default=None,
                    help="none: keep original class sizes; equalize: cap each class to the smallest class before splitting.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--clear-dest", action="store_true",
                        help="Delete existing data/{training,validation,testing} before writing.")
    add_exts_arg(parser)
    add_common_config_args(parser, include_dry_run=True)
    add_common_logging_args(parser)

    args = parser.parse_args(argv or [])
    # Resolve config
    cfg = build_split_config(args.config, args.override or [])
    # Logging
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    log_level = cfg.log_level or args.log_level or "INFO"
    log_file  = cfg.log_file or args.log_file or None

    configure_logging(
        log_level=log_level,
        file_mode="fixed" if log_file else "auto",
        log_file=log_file,
        run_id=run_id,
        stage="split",
    )

    log.info("split.dispatch", extra={"argv": argv})

    dataset_slug = cfg.dataset or args.dataset
    test_frac = cfg.test_frac if getattr(cfg, "test_frac", None) is not None else args.test_frac
    val_frac = cfg.val_frac if getattr(cfg, "val_frac", None) is not None else args.val_frac
    seed = cfg.seed if getattr(cfg, "seed", None) is not None else args.seed
    clear_dest = bool(getattr(cfg, "clear_dest", False) or getattr(args, "clear_dest", False))
    exts = parse_exts(args.exts)  # set[str] or empty set for "all"

    # Guard: fractions must be provided by YAML or CLI
    if test_frac is None or val_frac is None:
        log.error("split.fracs_missing", extra={"test_frac": test_frac, "val_frac": val_frac})
        print("❌ Missing fractions: please set data.test_frac and data.val_frac in YAML or via CLI.")
        return 2

    if not (0.0 <= test_frac < 1.0 and 0.0 <= val_frac < 1.0 and test_frac + val_frac < 1.0):
        log.error("split.invalid_fracs", extra={"test_frac": test_frac, "val_frac": val_frac})
        print(f"❌ Invalid fractions: test={test_frac}, val={val_frac}. Require 0≤each<1 and test+val<1.")
        return 2

    # Source = canonical processed store
    src_root = PROCESSED_DIR
    if not src_root.exists():
        log.error("split.processed_missing", extra={"src_root": str(src_root)})
        print("❌ data/processed not found. Run resize first.")
        return 2

    # Pool by class
    combined = defaultdict(list)
    src_map = gather_by_class(src_root, exts)
    for cls, paths in src_map.items():
        combined[cls].extend(paths)
    
    balance = args.balance  # config-first not needed; single flag is fine here
    cap_per_class = None
    if balance == "equalize":
        # compute the minimum class size across all classes (unique file paths per class)
        cap_per_class = min(len({str(p) for p in paths}) for paths in combined.values() if paths)
        log.info("split.equalize_cap", extra={"cap_per_class": cap_per_class})

    # DRY-RUN (data exists)
    if getattr(args, "dry_run", False) or getattr(cfg, "dry_run", False):
        plan = plan_split(combined, test_frac=test_frac, val_frac=val_frac, seed=seed, balance=balance)
        extra = make_log_extra(
            plan,
            dataset_slug=dataset_slug,
            pointer=None,
            src_training=src_root,
            src_testing=Path("-"),
            exts=args.exts,
            test_frac=test_frac,
            val_frac=val_frac,
            balance=balance,
            seed=seed,
            clear_dest=clear_dest,
            out_training=DATA_DIR / "training",
            out_validation=DATA_DIR / "validation",
            out_testing=DATA_DIR / "testing",
            mapping_use_dataset_subdir=getattr(cfg, "mapping_use_dataset_subdir", False),
            mapping_write_split_copy=getattr(cfg, "mapping_write_split_copy", False),
            save_remap_to_project_root=getattr(cfg, "save_remap_to_project_root", False),
        )
        log.info("dry_run.split.plan", extra=extra)
        print(render_human(plan, extra))
        return 0
    # --- end DRY RUN ---

    random.seed(seed)

    train_out = DATA_DIR / "training"
    val_out = DATA_DIR / "validation"
    test_out = DATA_DIR / "testing"

    if clear_dest:
        for d in (train_out, val_out, test_out):
            _empty_dir(d)
        log.info("split.cleared_dest", extra={
            "train_out": str(train_out), "val_out": str(val_out), "test_out": str(test_out)
        })

    # Actual per-class split and copy
    summary = []
    for cls, paths in sorted(combined.items()):
        uniq = sorted({str(p) for p in paths})
        random.shuffle(uniq)

        # Optional class balancing: cap to the smallest class size
        if cap_per_class is not None and len(uniq) > cap_per_class:
            # random.sample returns a new list; keep size == cap_per_class
            uniq = random.sample(uniq, cap_per_class)
        n = len(uniq)

        if n == 0:
            continue

        n_test = max(1, int(n * test_frac))
        n_val = int(n * val_frac)
        if n - (n_test + n_val) < 1 and n > 1:
            if n_val > 0:
                n_val = max(0, n - n_test - 1)
            if n - (n_test + n_val) < 1 and n_test > 1:
                n_test = 1

        test_paths = [Path(p) for p in uniq[:n_test]]
        val_paths = [Path(p) for p in uniq[n_test:n_test + n_val]] if n_val > 0 else []
        train_paths = [Path(p) for p in uniq[n_test + n_val:]]

        for p in test_paths:
            safe_copy(p, test_out / cls / p.name)
        for p in val_paths:
            safe_copy(p, val_out / cls / p.name)
        for p in train_paths:
            safe_copy(p, train_out / cls / p.name)

        summary.append((cls, len(train_paths), len(val_paths), len(test_paths)))
        log.debug("split.class_summary", extra={
            "class": cls, "train": len(train_paths), "val": len(val_paths), "test": len(test_paths)
        })

    # Mapping (index_remap + pointer)
    
    latest_map_path = _write_index_and_pointer(
        classes=_class_names_from_dir(train_out),
        run_id=run_id,
        dataset_slug=dataset_slug,
        save_remap_to_project_root=getattr(cfg, "save_remap_to_project_root", False),
        mapping_use_dataset_subdir=getattr(cfg, "mapping_use_dataset_subdir", False),
        mapping_write_split_copy=getattr(cfg, "mapping_write_split_copy", False),
    )

    # Console summary
    print("\nPer-class counts:")
    for cls, ntr, nva, nte in summary:
        print(f"{cls:15s} -> train: {ntr:5d} | val: {nva:5d} | test: {nte:5d}")

    log.info("split.done", extra={
        "dataset": dataset_slug,
        "mapping_latest": str(latest_map_path),
        "outputs": {"training": str(train_out), "validation": str(val_out), "testing": str(test_out)},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())