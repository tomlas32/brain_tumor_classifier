"""
Evaluate a trained classifier on a class-structured test set.

What this script does
---------------------
1) Loads the resized test dataset (from `resize.py`) and image transforms.
2) **Aligns class encoding** using `index_remap.json`:
   - If mapping is missing/unreadable → WARN, proceed with dataset order.
   - If same set but different order → re-map dataset to expected order (WARN).
   - If sets differ → WARN, proceed (metrics may be misleading).
3) Loads the trained model weights and runs evaluation.
4) Saves:
   - Confusion matrices (counts + row-normalized),
   - A `.txt` classification report,
   - Image galleries (top mistakes & top correct per true class),
   - Grad-CAM overlays for both groups,
   - `evaluation_summary.json` with headline metrics.

Typical pipeline order
----------------------
fetch → split → resize → validate → train → **evaluate**

Examples
--------
python -m src.pipeline.evaluate \
  --eval-in data/testing_resized \
  --trained-model models/best_valF1_0.9123_epoch14.pth \
  --model resnet18 --image-size 224 --batch-size 64
"""



from __future__ import annotations

from src.utils.logging_utils import get_logger, configure_logging
from src.utils.parser_utils import add_common_logging_args, add_common_eval_args
from torch.utils.data import Subset
import torch
from torchvision.datasets import ImageFolder

import argparse, numpy as np, json, time
from pathlib import Path
from src.utils.paths import OUTPUTS_DIR

from src.core.artifacts import write_evaluation_summary
from src.core.env import bootstrap_env, log_env_once
from src.core.mapping import read_index_remap, expected_classes_from_remap
from src.core.transforms import build_transforms
from src.core.model import build_model, load_weights, get_device 
from src.core.data import make_eval_loader
from src.core.metrics import evaluate, save_classification_report, save_confusions
from src.core.viz import show_calls_gallery, show_gradcam_gallery
from src.evaluate.runner import EvalRunnerInputs, run as run_evaluation

from datetime import datetime, timezone
import os as _os

log = get_logger(__name__)


def make_parser_evaluate() -> argparse.ArgumentParser:
    """
    Evaluate a trained classifier on a class-structured test folder.

    Examples
    --------
    python -m src.pipeline.evaluate \
      --eval-in data/testing_resized \
      --trained-model models/best_valF1_0.9123_epoch14.pth \
      --model resnet18 --image-size 224 --batch-size 64
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on a test set.")
    parser.add_argument("--image-size", type=int, 
                        default=224, help="Square size used in preprocessing")
    parser.add_argument("--model", 
                        choices=["resnet18","resnet34","resnet50"], 
                        default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=OUTPUTS_DIR / "mappings" / "latest.json",
        help="Path to index remap mapping JSON (default: OUTPUTS_DIR/mappings/latest.json)",
    )
    parser.add_argument(
    "--top-per-class",
    type=int,
    default=6,
    help="Number of items per true class to include in galleries/Grad-CAM (default: 6)",
    )
    parser.add_argument(
        "--no-galleries",
        action="store_true",
        help="Disable plain image galleries of top correct/mistaken predictions",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Disable Grad-CAM visualizations for top correct/mistaken predictions",
    )
    add_common_eval_args(parser)     # --eval-in, --eval-out, --trained-model
    add_common_logging_args(parser)  # --log-level, --log-file
    return parser


def main(argv=None):
    """
    CLI entry:
    1) Parse args and configure run/stage-aware logging.
    2) Build test dataset and align class encoding using the mapping.
    3) Evaluate and write artifacts (confusion matrices, reports, galleries, Grad-CAM, summary).
    """
    # 1) Parse CLI
    parser = make_parser_evaluate()
    args = parser.parse_args(argv)

    # 2) Configure logging with run_id + stage='evaluate'
    run_id = _os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="evaluate")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="evaluate")
    log.info("evaluate.start", extra={"args": {k: (str(v) if isinstance(v, Path) else v) for k,v in vars(args).items()}})

    bootstrap_env(seed=args.seed)
    log_env_once()

    inputs = EvalRunnerInputs(
        image_size=args.image_size,
        eval_in=Path(args.eval_in),
        mapping_path=Path(args.mapping_path),
        model_name=args.model,
        weights_path=Path(args.trained_model),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        eval_out=Path(args.eval_out),
        run_id=run_id,
        args_dict={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        make_galleries=not args.no_galleries,
        make_gradcam=not args.no_gradcam,
        top_per_class=args.top_per_class,
    )

    acc, prec, rec, f1 = run_evaluation(inputs)
    log.info("evaluation_complete", extra={
        "acc": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())