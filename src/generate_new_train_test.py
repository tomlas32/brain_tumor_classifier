"""
Combine two class-structured image roots into a single per-class split.

This script pools images from `training_dir` and `testing_dir` (each containing
exactly one subfolder per class), then creates a new random split with
20% of images per class assigned to a final test set and the remaining 80% to
a training set. Files are copied to:

    <output_root>/training/<class>/*
    <output_root>/testing/<class>/*

Key behavior:
- De-duplicates file paths and shuffles with a fixed SEED for reproducibility.
- Ensures at least one test image for any non-empty class.
- Avoids filename collisions by appending a numeric suffix.
- Supports common image extensions defined in EXTS.

Configuration:
- Edit `training_dir`, `testing_dir`, `output_root`.
- Optionally adjust `TEST_FRAC`, `SEED`, and `EXTS`.

Assumptions:
- Class names (subfolder names) are consistent across both roots.
- Images are readable by PIL and use extensions listed in EXTS.

Usage:
- Save and run the script (e.g., `python split_pool.py`), then point
  `torchvision.datasets.ImageFolder` at the new `training/` and `testing/`.
"""



from pathlib import Path
import shutil, random
from collections import defaultdict

# ---- INPUTS ----
training_dir = Path(r"C:/Users/tomla/Documents/Projects/brain_tumor_classifier/data/training/")
testing_dir  = Path(r"C:/Users/tomla/Documents/Projects/brain_tumor_classifier/data/testing/")
output_root  = Path(r"C:/Users/tomla/Documents/Projects/brain_tumor_classifier/data/combined_split_simple")

TEST_FRAC = 0.20
SEED = 42
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # adjust if needed

# ---- OUTPUT DIRS ----
train_out = output_root / "training"
test_out  = output_root / "testing"
train_out.mkdir(parents=True, exist_ok=True)
test_out.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

def gather_by_class(root: Path):
    """Return dict: class_name -> list[Path] for images under root/<class>/*"""
    mapping = defaultdict(list)
    if not root.exists():
        return mapping
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                mapping[cls].append(p)
    return mapping

# 1) Pool images from both roots
combined = defaultdict(list)
for src_root in [training_dir, testing_dir]:
    src_map = gather_by_class(src_root)
    for cls, paths in src_map.items():
        combined[cls].extend(paths)

# 2) Per-class split (20% test, 80% train)
def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # avoid filename collisions
    if dst.exists():
        i = 1
        while True:
            cand = dst.with_name(f"{dst.stem}__{i}{dst.suffix}")
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)

summary = []
for cls, paths in sorted(combined.items()):
    # de-duplicate and shuffle
    uniq = sorted({str(p) for p in paths})
    random.shuffle(uniq)

    n = len(uniq)
    if n == 0:
        continue
    n_test = max(1, int(n * TEST_FRAC))  # at least 1 if class has images

    test_paths  = [Path(p) for p in uniq[:n_test]]
    train_paths = [Path(p) for p in uniq[n_test:]]

    # copy to output structure
    for p in test_paths:
        safe_copy(p, test_out / cls / p.name)
    for p in train_paths:
        safe_copy(p, train_out / cls / p.name)

    summary.append((cls, len(train_paths), len(test_paths)))

# 3) Print summary
print("Split complete. Output at:", output_root)
for cls, ntr, nte in summary:
    print(f"{cls:15s} -> train: {ntr:5d} | test: {nte:5d}")
