"""
Combine two class-structured image roots into a single per-class split.

Pools images from --training-dir and --testing-dir, creates a random split with
--test-frac per class, and writes to:

    <output_root>/training/<class>/*
    <output_root>/testing/<class>/*

- De-duplicates file paths, shuffles with --seed.
- Ensures at least one test image for any non-empty class.
- Avoids filename collisions by appending a numeric suffix.

Notes:
-----
- Reads standardized *fetch pointer* from outputs/pointers/fetch/<owner>/<slug>/latest.json
- Writes standardized *mapping pointer* to outputs/pointers/mapping/<owner>/<slug>/{latest.json,history/...}


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
import argparse, random, shutil, time, os
from collections import defaultdict
from datetime import datetime, timezone

from src.utils.paths import DATA_DIR, OUTPUTS_DIR
from src.utils.configs import DEFAULT_DATASET
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args, add_exts_arg, parse_exts

from src.core.mapping import write_index_remap as mapping_write_index_remap, copy_index_remap
from src.core.config import build_split_config, to_dict
from src.core.artifacts import read_fetch_pointer, write_mapping_pointer

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

def gather_by_class(root: Path, exts: set[str]):
    """Return dict: class_name -> list[Path] for images under root/<class>/*

    Notes
    -----
    - If `exts` is empty, accept any file extension (matches parser_utils.parse_exts('all')).
    """
    mapping = defaultdict(list)
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
    if dst.exists():
        i = 1
        while True:
            cand = dst.with_name(f"{dst.stem}__{i}{dst.suffix}")
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)


def _class_names_from_dir(dir_path: Path) -> list[str]:
    """Return sorted subfolder names (classes) under dir_path."""
    if not dir_path.exists():
        return []
    return sorted([d.name for d in dir_path.iterdir() if d.is_dir()])

def _check_subset_classes(root: Path, expected: set[str], subset: str) -> tuple[set, set]:
    """
    Compare classes in root/subset vs expected set.
    Returns (missing, extra).
    """
    p = root / subset
    found = set(_class_names_from_dir(p))
    missing = expected - found
    extra = found - expected
    return missing, extra


def _build_and_write_mapping(
    split_root: Path,
    *,
    prefer_subset: str = "training",
    dataset: str | None = None,
    use_dataset_subdir: bool = False,
    write_split_copy: bool = False,
    also_save_project_root: bool = False,
    project_root_path: Path = Path("index_remap.json"),
) -> Path:
    # 1) get ordered classes from prefer_subset
    classes = _class_names_from_dir(split_root / prefer_subset)
    if not classes:
        msg = f"No class folders found under {split_root / prefer_subset}"
        log.error("split.index_remap.no_classes", extra={"subset": prefer_subset, "path": str(split_root)})
        raise RuntimeError(msg)

    # 2) verify other subsets share the same set
    for subset in ("testing",):
        missing, extra = _check_subset_classes(split_root, set(classes), subset)
        if missing or extra:
            msg = f"Class mismatch in '{subset}' vs '{prefer_subset}': missing={sorted(missing)} extra={sorted(extra)}"
            log.error("split.class_mismatch", extra={"subset": subset, "message": msg})
            raise RuntimeError(msg)

    # 3) write via shared core mapping
    latest_path = mapping_write_index_remap(
        classes,
        dataset=dataset,
        use_dataset_subdir=use_dataset_subdir,
    )

    # 4) optional extra copies
    if write_split_copy:
        copy_index_remap(latest_path, split_root)
    if also_save_project_root:
        copy_index_remap(latest_path, project_root_path.parent)

    return latest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Re-split class-structured image roots.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                    help="Kaggle slug (owner/dataset) used to auto-locate the pointer.")
    parser.add_argument("--pointer", type=Path, default=None,
                        help="Optional explicit path to the pointer JSON (overrides --dataset).")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction per class for final test set (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--clear-dest", action="store_true",
                    help="Delete all existing files/dirs in DATA_DIR/training and DATA_DIR/testing before writing.")
    parser.add_argument("--save-remap-to-project-root", action="store_true",
                    help="Also save index_remap.json at project root (./index_remap.json).")
    parser.add_argument("--config", type=Path, default=None,
                    help="Optional YAML config file for split (config-first).")
    parser.add_argument("--override", action="append", default=[],
                    help="Override config values as key=val (e.g., test_frac=0.25 clear_dest=true). "
                         "Repeat for multiple overrides.")
    
    add_common_logging_args(parser)  # --log-level, --log-file
    add_exts_arg(parser)
    args = parser.parse_args(argv)

     # --- run-aware logging ---
    # Run-aware logging (ties logs across stages)
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="split")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="split")
    t0 = time.time()

    # Resolve config (config-first with CLI fallback)
    cfg = build_split_config(args.config, overrides=args.override)
    log.info("config.resolved", extra={"config": to_dict(cfg)})

    # Effective values (config wins; fall back to CLI/env defaults)
    dataset_slug = os.getenv("DATASET_SLUG", cfg.dataset or args.dataset)
    pointer = cfg.pointer or args.pointer or _pointer_path_for(dataset_slug)
    test_frac = cfg.test_frac if cfg.test_frac is not None else args.test_frac
    seed = cfg.seed if cfg.seed is not None else args.seed
    clear_dest = bool(cfg.clear_dest or args.clear_dest)
    save_remap_to_project_root = bool(cfg.save_remap_to_project_root or args.save_remap_to_project_root)

    # exts from config if provided; otherwise from CLI; both go through parse_exts()
    exts_source = cfg.exts if cfg.exts is not None else args.exts
    exts = parse_exts(exts_source)  # empty set means “accept any”

    # Seed *after* resolving config
    random.seed(seed)

    log.debug("split.args_effective", extra={
        "dataset": dataset_slug,
        "pointer": str(pointer),
        "test_frac": test_frac,
        "seed": seed,
        "exts": sorted(exts) if exts else ["<any>"],
        "clear_dest": clear_dest,
        "save_remap_to_project_root": save_remap_to_project_root,
        "mapping_use_dataset_subdir": bool(cfg.mapping_use_dataset_subdir),
        "mapping_write_split_copy": bool(cfg.mapping_write_split_copy),
    })

    try:
        fetch_ptr = read_fetch_pointer(pointer)
    except Exception as e:
        log.error("split.pointer_read_error", extra={"pointer": str(pointer), "error": str(e)})
        return 2

    dataset_root = Path(fetch_ptr["dataset_root"])
    # Prefer explicit dirs if present; otherwise try conventional subfolders
    src_training = Path(fetch_ptr.get("training_dir") or (dataset_root / "Training"))
    src_testing  = Path(fetch_ptr.get("testing_dir")  or (dataset_root / "Testing"))

    if not src_training.exists() or not src_testing.exists():
        log.error("split.source_dirs_missing", extra={
            "src_training": str(src_training),
            "src_testing": str(src_testing),
            "dataset_root": str(dataset_root),
        })
        return 2

    log.info("split.fetch_pointer_loaded", extra={
        "pointer": str(pointer),
        "dataset": fetch_ptr.get("dataset"),
        "dataset_root": str(dataset_root),
        "src_training": str(src_training),
        "src_testing": str(src_testing),
    })

    log.info("split.started", extra={"dataset": dataset_slug, "pointer": str(pointer), 
                                     "test_frac": test_frac, "seed": seed, "exts": sorted(exts) if exts else ["<any>"]})

    train_out = DATA_DIR / "training"
    test_out  = DATA_DIR / "testing"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    if clear_dest:
        _empty_dir(train_out)
        _empty_dir(test_out)
        log.info("split.cleared_dest", extra={"train_out": str(train_out), "test_out": str(test_out)})

    # 1) Pool images from both roots
    combined = defaultdict(list)
    for src_root in (src_training, src_testing):
        src_map = gather_by_class(src_root, exts)
        for cls, paths in src_map.items():
            combined[cls].extend(paths)
    log.debug("split.combined_counts", extra={k: len(v) for k, v in combined.items()})

     # 2) Per-class split (test then train)
    summary = []
    for cls, paths in sorted(combined.items()):
        uniq = sorted({str(p) for p in paths})
        random.shuffle(uniq)

        n = len(uniq)
        if n == 0:
            continue
        n_test = max(1, int(n * test_frac))

        test_paths = [Path(p) for p in uniq[:n_test]]
        train_paths = [Path(p) for p in uniq[n_test:]]

        for p in test_paths:
            safe_copy(p, test_out / cls / p.name)
        for p in train_paths:
            safe_copy(p, train_out / cls / p.name)

        summary.append((cls, len(train_paths), len(test_paths)))
        log.debug("split.class_summary", extra={"class": cls, "train": len(train_paths), "test": len(test_paths)})

    # 3) Build & write mapping (shared core), optionally copy elsewhere
    try:
        latest_map_path = _build_and_write_mapping(
            split_root=DATA_DIR,
            prefer_subset="training",
            dataset=dataset_slug if cfg.mapping_use_dataset_subdir else None,
            use_dataset_subdir=bool(cfg.mapping_use_dataset_subdir),
            write_split_copy=bool(cfg.mapping_write_split_copy),
            also_save_project_root=save_remap_to_project_root,
            project_root_path=Path("index_remap.json"),
        )

        # After index_remap.json is finalized, write standardized mapping pointer.
        ordered_classes = _class_names_from_dir(DATA_DIR / "training")
        try:
            ptr_out = write_mapping_pointer(
                classes=ordered_classes,
                index_remap_path=latest_map_path,
                dataset=fetch_ptr.get("dataset"),
                index_remap=None,  # or embed {idx: cls} if you want
                run_id=run_id,
                dst_dir=None,      # → outputs/pointers/mapping/<owner>/<slug>/
            )
            log.info("split.mapping_pointer_written", extra={
                "latest": str(ptr_out.get("latest_path")),
                "history": str(ptr_out.get("history_path")),
                "run_id": run_id,
            })
        except Exception as e:
            log.warning("split.mapping_pointer_write_failed", extra={"error": str(e), "run_id": run_id})

    except Exception as e:
        log.error("split.index_remap.failed", extra={"error": str(e)})
        return 2

    # Final summary
    elapsed = time.time() - t0
    log.info("split.done", extra={
        "train_out": str(train_out),
        "test_out": str(test_out),
        "elapsed_s": round(elapsed, 2),
    })
    print("Split complete. Output at:", str(DATA_DIR))

    for cls, ntr, nte in summary:
        print(f"{cls:15s} -> train: {ntr:5d} | test: {nte:5d}")
        log.debug("split.summary", extra={"class": cls, "train": ntr, "test": nte})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())