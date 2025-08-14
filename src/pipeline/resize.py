import cv2, argparse
from pathlib import Path
import numpy as np
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args, add_exts_arg, parse_exts
from src.utils.paths import DATA_DIR

log = get_logger(__name__)

TRAIN_INPUT = DATA_DIR / "training"
TEST_INPUT  = DATA_DIR / "testing"
TRAIN_OUT   = DATA_DIR / "training_resized"
TEST_OUT    = DATA_DIR / "testing_resized"

def resize_and_pad(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize image while maintaining aspect ratio, then pad to a square (size x size)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

def _count_images(root: Path, exts: set[str]) -> int:
    """Count images under root matching exts. If exts is empty, accept any suffix."""
    if not root.exists():
        return 0
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and (not exts or p.suffix.lower() in exts)
    )

def process_images(input_dir: Path, output_dir: Path, exts: set[str], target_size: int = 224):
    """Resize and pad images in all subfolders of input_dir, saving to output_dir."""
    total_found = total_resized = 0
    if not input_dir.exists():
        log.warning("resize.input_missing", extra={"input_dir": str(input_dir)})
        return 0, 0

    for subfolder in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        for img_path in subfolder.rglob("*"):
            if img_path.is_file() and (not exts or img_path.suffix.lower() in exts):
                total_found += 1
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    log.warning("resize.unreadable_image", extra={"path": str(img_path)})
                    continue

                resized = resize_and_pad(img, size=target_size)
                save_path = output_dir / subfolder.name / img_path.name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if cv2.imwrite(str(save_path), resized):
                    total_resized += 1
                else:
                    log.error("resize.save_failed", extra={"path": str(save_path)})

    return total_found, total_resized


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resize class-structured images (after split).")
    parser.add_argument("--train-in",  type=Path, default=TRAIN_INPUT, help="Input training root (from split).")
    parser.add_argument("--train-out", type=Path, default=TRAIN_OUT,   help="Output root for resized training images.")
    parser.add_argument("--test-in",   type=Path, default=TEST_INPUT,  help="Input testing root (from split).")
    parser.add_argument("--test-out",  type=Path, default=TEST_OUT,    help="Output root for resized testing images.")
    parser.add_argument("--size", type=int, default=224, help="Output square size in pixels (e.g., 224).")
    add_common_logging_args(parser)
    add_exts_arg(parser)
    args = parser.parse_args(argv)

    configure_logging(log_level=args.log_level, log_file=args.log_file)
    exts = parse_exts(args.exts)  # set(); empty set means “accept any”

    # --- Guards: ensure split has produced inputs with images ---
    train_ct = _count_images(args.train_in, exts)
    test_ct  = _count_images(args.test_in,  exts)

    if not args.train_in.exists() and not args.test_in.exists():
        log.error("resize.inputs_missing", extra={"train_in": str(args.train_in), "test_in": str(args.test_in)})
        print("❌ Neither training nor testing input directories exist. Run split first.")
        return 2

    if train_ct == 0 and test_ct == 0:
        log.error("resize.no_images_found", extra={
            "train_in": str(args.train_in), "test_in": str(args.test_in), "exts": sorted(exts) or ["<any>"]
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