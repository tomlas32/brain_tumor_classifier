"""
Combine two class-structured image roots into a single per-class split.

Pools images from --training-dir and --testing-dir, creates a random split with
--test-frac per class, and writes to:

    <output_root>/training/<class>/*
    <output_root>/testing/<class>/*

- De-duplicates file paths, shuffles with --seed.
- Ensures at least one test image for any non-empty class.
- Avoids filename collisions by appending a numeric suffix.
"""

from pathlib import Path
import argparse, random, shutil, time, json, os
from collections import defaultdict

from src.utils.paths import DATA_DIR, OUTPUTS_DIR
from src.utils.configs import DEFAULT_DATASET
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Re-split class-structured image roots.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                    help="Kaggle slug (owner/dataset) used to auto-locate the pointer.")
    parser.add_argument("--pointer", type=Path, default=None,
                        help="Optional explicit path to the pointer JSON (overrides --dataset).")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction per class for final test set (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--exts", type=str,
                        default=".png,.jpg,.jpeg,.bmp,.tif,.tiff",
                        help="Comma-separated extensions (lowercased).")
    parser.add_argument("--clear-dest", action="store_true",
                    help="Delete all existing files/dirs in DATA_DIR/training and DATA_DIR/testing before writing.")
    
    t0 = time.time()
    random.seed(args.seed)

    add_common_logging_args(parser)  # --log-level, --log-file
    args = parser.parse_args(argv)

    configure_logging(log_level=args.log_level, log_file=args.log_file)

    exts = {
        (e if e.startswith(".") else f".{e}")
        for e in (x.strip().lower() for x in args.exts.split(","))
        if e
    }

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

    # 3) Summary
    elapsed = time.time() - t0
    log.info("split:done", extra={"output_root": str(args.output_root), "elapsed_s": round(elapsed, 2)})
    print("Split complete. Output at:", args.output_root)
    for cls, ntr, nte in summary:
        print(f"{cls:15s} -> train: {ntr:5d} | test: {nte:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())