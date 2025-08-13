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
import argparse, random, shutil, time
from collections import defaultdict

from src.utils.paths import DATA_DIR
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args

log = get_logger(__name__)


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
    parser.add_argument("--training-dir", type=Path, default=DATA_DIR / "training",
                        help="Root with class subfolders for training images.")
    parser.add_argument("--testing-dir", type=Path, default=DATA_DIR / "testing",
                        help="Root with class subfolders for testing images.")
    parser.add_argument("--output-root", type=Path, default=DATA_DIR / "combined_split_simple",
                        help="Where training/ and testing/ will be created.")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction per class for final test set (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--exts", type=str,
                        default=".png,.jpg,.jpeg,.bmp,.tif,.tiff",
                        help="Comma-separated extensions (lowercased).")
    add_common_logging_args(parser)  # --log-level, --log-file

    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, log_file=args.log_file)

    exts = {
        (e if e.startswith(".") else f".{e}")
        for e in (x.strip().lower() for x in args.exts.split(","))
        if e
    }

    t0 = time.time()
    random.seed(args.seed)

    train_out = args.output_root / "training"
    test_out = args.output_root / "testing"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    # 1) Pool images from both roots
    combined = defaultdict(list)
    for src_root in [args.training_dir, args.testing_dir]:
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