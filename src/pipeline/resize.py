"""
Resize (and pad) class-structured images produced by the split step.

This script scans per-class folders under training/testing inputs, resizes each
image to a square of the requested size while preserving aspect ratio, pads with
black to fill the square, and writes the results into mirrored output folders.

Typical pipeline order:
    1) fetch.py   → download dataset
    2) split.py   → create DATA_DIR/training and DATA_DIR/testing
    3) resize.py  → create DATA_DIR/training_resized and DATA_DIR/testing_resized

Features
--------
- Guards to ensure inputs exist and contain images (prevents running before split).
- Per-class directory traversal with collision-safe writing.
- Extension handling consistent with `parser_utils` (supports "+ext" and "all").
- Run-aware, stage-aware logging; DEBUG lines add per-class/process detail.
- CI-friendly non-zero exit codes on error.

Command-line arguments
----------------------
--train-in PATH       Input root for training images (default: DATA_DIR/training).
--train-out PATH      Output root for resized training images (default: DATA_DIR/training_resized).
--test-in PATH        Input root for testing images (default: DATA_DIR/testing).
--test-out PATH       Output root for resized testing images (default: DATA_DIR/testing_resized).
--size INT            Output square size in pixels (default: 224).
--exts STR            Comma-separated extensions; '+ext' adds to defaults; 'all' disables filtering.
--log-level LEVEL     Logging verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO).
--log-file PATH       Optional log file path; if omitted, stdout + auto-named rotating file.

Exit codes
----------
0  Success.
2  Neither training nor testing input directories exist (split step likely not run).
3  Inputs exist but no matching images were found (check --exts or run split).

Examples
--------
# Use defaults (224 px, default extensions)
python -m src.pipeline.resize

# Accept any file type under the inputs
python -m src.pipeline.resize --exts all

# Add WEBP and GIF on top of defaults
python -m src.pipeline.resize --exts +webp,+gif

# Custom size and explicit I/O roots
python -m src.pipeline.resize --size 256 --train-in data/training --train-out data/training_256

Notes
-----
- Aspect ratio is preserved; padding is added to reach an exact square.
- Interpolation switches automatically: INTER_AREA for downscaling, INTER_LINEAR for upscaling.
- Output directories are created only after input validation passes.
"""



import argparse
from pathlib import Path
from datetime import datetime, timezone
import os

import cv2
import numpy as np

from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args, add_exts_arg, parse_exts
from src.utils.paths import DATA_DIR

log = get_logger(__name__)

# Default I/O roots
TRAIN_INPUT = DATA_DIR / "training"
TEST_INPUT  = DATA_DIR / "testing"
TRAIN_OUT   = DATA_DIR / "training_resized"
TEST_OUT    = DATA_DIR / "testing_resized"

def resize_and_pad(img: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio, then pad to a square (size × size).

    Parameters
    ----------
    img : np.ndarray
        Input image (H×W×C) as loaded by OpenCV (BGR).
    size : int, optional
        Target square edge length in pixels (default 224).

    Returns
    -------
    np.ndarray
        A size×size×3 BGR image where the original content is preserved with
        aspect ratio and black padding fills the remaining area.

    Details
    -------
    - The resizing scale is computed as `size / max(H, W)` to avoid stretching.
    - Interpolation policy:
        * cv2.INTER_AREA  when scale < 1 (downscaling) — better for decimation.
        * cv2.INTER_LINEAR when scale >= 1 (upscaling) — smoother enlargement.
    - Padding is symmetric (top/bottom, left/right) with black (0,0,0) pixels.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    # DEBUG hint (only emitted if --log-level DEBUG)
    log.debug("resize.interp_choice", extra={"orig": (h, w), "target": size, "scale": round(scale, 4),
                                             "interp": "INTER_AREA" if scale < 1 else "INTER_LINEAR"})

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

def _count_images(root: Path, exts: set[str]) -> int:
    """
    Count images under a root matching the provided extensions.

    Parameters
    ----------
    root : Path
        Directory to scan recursively.
    exts : set[str]
        Lowercased extensions including the dot (e.g., {'.jpg', '.png'}). If the set
        is empty, any file suffix is accepted (this is how 'all' is represented).

    Returns
    -------
    int
        Number of files considered images for processing.
    """
    if not root.exists():
        return 0
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and (not exts or p.suffix.lower() in exts)
    )

def process_images(input_dir: Path, output_dir: Path, exts: set[str], target_size: int = 224) -> tuple[int, int]:
    """
    Resize and pad all matching images under `input_dir`, mirroring the class
    subfolder structure in `output_dir`.

    Parameters
    ----------
    input_dir : Path
        Root containing per-class subfolders of images to process.
    output_dir : Path
        Destination root; class subfolders will be created as needed.
    exts : set[str]
        Allowed extensions (empty set = accept any).
    target_size : int
        Edge length of the output square images.

    Returns
    -------
    (found, resized) : tuple[int, int]
        Total files found (matching `exts`) and total successfully written.

    Logging
    -------
    - WARNING: unreadable images (skipped).
    - ERROR:   failed writes.
    - DEBUG:   per-class tallies.
    """
    total_found = total_resized = 0
    if not input_dir.exists():
        log.warning("resize.input_missing", extra={"input_dir": str(input_dir)})
        return 0, 0

    for subfolder in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        cls_found = cls_resized = 0

        for img_path in subfolder.rglob("*"):
            if img_path.is_file() and (not exts or img_path.suffix.lower() in exts):
                total_found += 1
                cls_found += 1

                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    log.warning("resize.unreadable_image", extra={"path": str(img_path)})
                    continue

                resized = resize_and_pad(img, size=target_size)

                save_path = output_dir / subfolder.name / img_path.name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if cv2.imwrite(str(save_path), resized):
                    total_resized += 1
                    cls_resized += 1
                else:
                    log.error("resize.save_failed", extra={"path": str(save_path)})

        if cls_found:
            log.debug("resize.class_summary", extra={
                "class": subfolder.name, "found": cls_found, "resized": cls_resized, "out_dir": str(output_dir / subfolder.name)
            })

    return total_found, total_resized


def main(argv=None) -> int:
    """
    Entry point: validates inputs, configures logging, and processes training/testing trees.

    Behavior
    --------
    1) Parses args (input roots, output roots, size, extensions, logging).
    2) Configures logging with stage='resize' and a run ID (from RUN_ID env or UTC timestamp).
    3) Guards:
        - If neither input root exists → exit 2.
        - If both exist but no matching images → exit 3.
    4) Processes images and prints a concise human summary to stdout.
    """
    parser = argparse.ArgumentParser(description="Resize class-structured images (after split).")
    parser.add_argument("--train-in",  type=Path, default=TRAIN_INPUT, help="Input training root (from split).")
    parser.add_argument("--train-out", type=Path, default=TRAIN_OUT,   help="Output root for resized training images.")
    parser.add_argument("--test-in",   type=Path, default=TEST_INPUT,  help="Input testing root (from split).")
    parser.add_argument("--test-out",  type=Path, default=TEST_OUT,    help="Output root for resized testing images.")
    parser.add_argument("--size", type=int, default=224, help="Output square size in pixels (e.g., 224).")
    add_common_logging_args(parser)  # --log-level, --log-file
    add_exts_arg(parser)             # --exts semantics (supports '+ext' and 'all')
    args = parser.parse_args(argv)

    # Run-aware logging (ties logs across stages in Docker/CI)
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    configure_logging(log_level=args.log_level, log_file=args.log_file, run_id=run_id, stage="resize")

    exts = parse_exts(args.exts)  # empty set means “accept any” (matches 'all')

    # --- Guards: ensure split has produced inputs with images ---
    train_ct = _count_images(args.train_in, exts)
    test_ct  = _count_images(args.test_in,  exts)

    if not args.train_in.exists() and not args.test_in.exists():
        log.error("resize.inputs_missing", extra={"train_in": str(args.train_in), "test_in": str(args.test_in)})
        print("❌ Neither training nor testing input directories exist. Run split first.")
        return 2

    if train_ct == 0 and test_ct == 0:
        log.error("resize.no_images_found", extra={
            "train_in": str(args.train_in), "test_in": str(args.test_in),
            "exts": sorted(exts) or ["<any>"],
        })
        print("❌ No images found to resize (did you run split? Check extensions with --exts).")
        return 3

    # Create outputs only when we’re sure we have something to do
    args.train_out.mkdir(parents=True, exist_ok=True)
    args.test_out.mkdir(parents=True, exist_ok=True)

    train_found, train_resized = process_images(args.train_in, args.train_out, exts, args.size)
    test_found,  test_resized  = process_images(args.test_in,  args.test_out,  exts, args.size)

    total_found   = train_found + test_found
    total_resized = train_resized + test_resized

    log.info("resize.done", extra={
        "size": args.size,
        "train_in": str(args.train_in), "train_out": str(args.train_out),
        "test_in": str(args.test_in),   "test_out": str(args.test_out),
        "found": total_found, "resized": total_resized
    })

    print(f"\n✅ Total files found: {total_found}")
    print(f"✅ Total files resized and saved: {total_resized}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())