"""
Combine two class-structured image roots into a single per-class split.

Pools images from --training-dir and --testing-dir, creates a random split with
--test-frac per class, and writes to:

    <output_root>/training/<class>/*
    <output_root>/testing/<class>/*

- De-duplicates file paths, shuffles with --seed.
- Ensures at least one test image for any non-empty class.
- Avoids filename collisions by appending a numeric suffix.

Author: Tomasz Lasota
Date: 2025-08-14
Version: 1.1  
"""

from pathlib import Path
import argparse, random, shutil, time, json, os, random
from collections import defaultdict
from datetime import datetime, timezone

from src.utils.paths import DATA_DIR, OUTPUTS_DIR
from src.utils.configs import DEFAULT_DATASET
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args, add_exts_arg, parse_exts

log = get_logger(__name__)

def _pointer_path_for(slug: str) -> Path:
    owner, name = (slug.split("/", 1) if "/" in slug else ("unknown", slug))
    return OUTPUTS_DIR / "downloads_pointer" / owner / name / "latest.json"

def _empty_dir(d: Path) -> None:
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

def gather_by_class(root: Path, exts: set[str]):
    """Return dict: class_name -> list[Path] for images under root/<class>/*"""
    mapping = defaultdict(list)
    if not root.exists():
        log.warning("Input root does not exist: %s", root)
        return mapping
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
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

def write_index_remap(
    split_root: Path,
    *,
    prefer_subset: str = "training",
    use_outputs_dir: bool = True,
    use_dataset_subdir: bool = False,   # set True to mirror fetch.py owner/slug layout
    dataset: str | None = None,         # required if use_dataset_subdir=True
    # ---- optional extra copies (off by default) ----
    write_split_copy: bool = False,     # write split_root / index_remap.json
    also_save_project_root: bool = False,
    project_root_path: Path = Path("index_remap.json"),
) -> Path:
    """
    Create and save a deterministic index_remap.json from folders in split_root/{prefer_subset}.

    Primary location (recommended):
      - outputs/mappings[/<owner>/<slug>]/latest.json  (and a timestamped history file)

    Optional extra copies:
      - split_root/index_remap.json (if write_split_copy=True)
      - project_root_path (if also_save_project_root=True)

    Validates that other subsets (e.g., 'testing') have the same class set.
    Returns the path to the 'latest.json' written under outputs/mappings (or the split copy if
    use_outputs_dir=False).
    """
    # 1) Collect and validate classes
    classes = _class_names_from_dir(split_root / prefer_subset)
    if not classes:
        msg = f"No class folders found under {split_root / prefer_subset}"
        log.error("split.index_remap.no_classes", extra={"subset": prefer_subset, "path": str(split_root)})
        raise RuntimeError(msg)

    expected = set(classes)
    for subset in ("testing",):
        missing, extra = _check_subset_classes(split_root, expected, subset)
        if missing or extra:
            msg = f"Class mismatch in '{subset}' vs '{prefer_subset}': missing={sorted(missing)} extra={sorted(extra)}"
            log.error("split.class_mismatch", extra={"subset": subset, "message": msg})
            raise RuntimeError(msg)

    # 2) Build mapping
    index_remap = {str(i): cls for i, cls in enumerate(classes)}

    # 3) Decide primary destination
    if use_outputs_dir:
        # Base: outputs/mappings[/<owner>/<slug>]
        base = OUTPUTS_DIR / "mappings"
        if use_dataset_subdir:
            if not dataset:
                raise ValueError("dataset must be provided when use_dataset_subdir=True")
            owner, slug = (dataset.split("/", 1) if "/" in dataset else ("unknown", dataset))
            base = base / owner / slug
        base.mkdir(parents=True, exist_ok=True)

        latest_path = base / "latest.json"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        history_path = base / f"{ts}__{len(classes)}cls.json"

        # Write both latest & history
        latest_path.write_text(json.dumps(index_remap, indent=2), encoding="utf-8")
        history_path.write_text(json.dumps(index_remap, indent=2), encoding="utf-8")

        log.info("split.index_remap.write", extra={
            "latest_path": str(latest_path),
            "history_path": str(history_path),
            "classes": classes,
            "num_classes": len(classes),
            "dataset": dataset if use_dataset_subdir else None
        })

        primary_path = latest_path
    else:
        # Legacy behavior: write under the split root
        dst_in_split = split_root / "index_remap.json"
        dst_in_split.write_text(json.dumps(index_remap, indent=2), encoding="utf-8")
        log.info("split.index_remap.write_split", extra={
            "path": str(dst_in_split),
            "classes": classes,
            "num_classes": len(classes)
        })
        primary_path = dst_in_split

    # 4) Optional extra copies
    if write_split_copy:
        dst_in_split = split_root / "index_remap.json"
        dst_in_split.write_text(json.dumps(index_remap, indent=2), encoding="utf-8")
        log.info("split.index_remap.copy_split_root", extra={"path": str(dst_in_split)})

    if also_save_project_root:
        project_root_path.write_text(json.dumps(index_remap, indent=2), encoding="utf-8")
        log.info("split.index_remap.copy_project_root", extra={"path": str(project_root_path)})

    return primary_path


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
    
    add_common_logging_args(parser)  # --log-level, --log-file
    add_exts_arg(parser)
    args = parser.parse_args(argv)
    
    t0 = time.time()
    random.seed(args.seed)

    configure_logging(log_level=args.log_level, log_file=args.log_file)

    exts = parse_exts(args.exts)

    dataset_slug = os.getenv("DATASET_SLUG", args.dataset)
    pointer = args.pointer or _pointer_path_for(dataset_slug)

    if not pointer.exists():
        log.error("split.pointer_missing", extra={"pointer": str(pointer)})
        return 2
    
    with pointer.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    
    try:
        src_training = Path(meta["training_dir"])
        src_testing  = Path(meta["testing_dir"])
    except KeyError as e:
        log.error("split.pointer_key_error", extra={"missing": str(e), "pointer": str(pointer)})
        return 2

    train_out = DATA_DIR / "training"
    test_out  = DATA_DIR / "testing"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    if args.clear_dest:
        _empty_dir(train_out)
        _empty_dir(test_out)
        log.info("split.cleared_dest", extra={"train_out": str(train_out), "test_out": str(test_out)})

    # 1) Pool images from both roots
    combined = defaultdict(list)
    for src_root in (src_training, src_testing):
        src_map = gather_by_class(src_root, exts)
        for cls, paths in src_map.items():
            combined[cls].extend(paths)

    # 2) Per-class split (test then train)
    summary = []
    for cls, paths in sorted(combined.items()):
        uniq = sorted({str(p) for p in paths})
        random.shuffle(uniq)

        n = len(uniq)
        if n == 0:
            continue
        n_test = max(1, int(n * args.test_frac))

        test_paths = [Path(p) for p in uniq[:n_test]]
        train_paths = [Path(p) for p in uniq[n_test:]]

        for p in test_paths:
            safe_copy(p, test_out / cls / p.name)
        for p in train_paths:
            safe_copy(p, train_out / cls / p.name)

        summary.append((cls, len(train_paths), len(test_paths)))
    
    try:
        write_index_remap(
            split_root=DATA_DIR,
            prefer_subset="training",
            use_outputs_dir=True,        # write to outputs/mappings/
            use_dataset_subdir=False,    # no owner/slug nesting
            dataset=None,
            write_split_copy=False,      # optional
            also_save_project_root=False # optional
        )
    except Exception as e:
        log.error("split.index_remap.failed", extra={"error": str(e)})
        return 2

    # 3) Summary
    elapsed = time.time() - t0
    log.info(
    "split:done",
    extra={
        "train_out": str(train_out),
        "test_out": str(test_out),
        "elapsed_s": round(elapsed, 2),
    },
)
    print("Split complete. Output at:", str(DATA_DIR))

    for cls, ntr, nte in summary:
        print(f"{cls:15s} -> train: {ntr:5d} | test: {nte:5d}")
        log.debug("split:summary", extra={"class": cls, "train": ntr, "test": nte})
    
    return 0 # --exit-zero success


if __name__ == "__main__":
    raise SystemExit(main())