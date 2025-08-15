"""
Train a CNN on resized MRI images (brain tumor classifier).

This script:
1) Loads a resized, class-structured training dataset (from `resize.py`).
2) Verifies class mapping against `index_remap.json` written by `split.py`.
3) Creates a stratified train/val split with deterministic seeding.
4) Trains a ResNet (18/34/50) with AMP, StepLR, and AdamW.
5) Saves only the single best checkpoint (by validation F1) and a summary JSON.

Key design choices
------------------
- Strict mapping verification: training *must* match `index_remap.json`.
- Run-aware logging: every log line includes [stage|run_id] for Docker/CI stitching.
- Single source of truth: mapping read/verify/copy comes from `src.core.mapping`.

Typical pipeline order
----------------------
fetch → split → resize → validate → **train** → evaluate

Examples
--------
# minimal (defaults)
python -m src.pipeline.train

# custom epochs / lr and output dirs
python -m src.pipeline.train --epochs 20 --lr 3e-4 --out-models models/brain_tumor

# explicit resized roots
python -m src.pipeline.train --train-in data/training_resized

# explicit mapping file (otherwise uses outputs/mappings/latest.json)
python -m src.pipeline.train --index-remap outputs/mappings/latest.json
"""



from __future__ import annotations

from pathlib import Path
import argparse, time, copy, math, os
from datetime import datetime, timezone
import numpy as np

import torch

from src.utils.logging_utils import configure_logging, get_logger
from src.core.artifacts import write_training_summary
from src.core.model import build_model, get_device
from src.core.transforms import build_transforms
from src.core.data import make_loader
from src.core.env import (
    bootstrap_env, 
    log_env_once, 
    get_env_info
)
from src.core.mapping import (
    default_index_remap_path,
)
from src.utils.parser_utils import (
    add_common_logging_args,
    add_common_dataset_args,
    add_common_train_args,
    add_model_args,
)
from src.train.runner import TrainRunnerInputs, run as run_training

log = get_logger(__name__)


def make_parser_train() -> argparse.ArgumentParser:
    """
    Create CLI parser for the training step.

    Examples
    --------
    # minimal (uses defaults)
    python -m src.pipeline.train

    # custom epochs / lr and save models to a custom dir
    python -m src.pipeline.train --epochs 20 --lr 3e-4 --out-models models/brain_tumor

    # point to specific resized folders
    python -m src.pipeline.train --train-in data/training_resized --test-in data/testing_resized
    """
    parser = argparse.ArgumentParser(description="Train a CNN on resized MRI images.")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square size after resize/pad (must match your preprocessing)")
    add_common_dataset_args(parser)   # roots, batch size, workers, val_frac, seed, out dirs, etc.
    add_common_train_args(parser)     # epochs, lr, weight_decay, scheduler, amp, etc.
    add_model_args(parser)            # model name + pretrained flag
    add_common_logging_args(parser)   # log level/file
    parser.add_argument("--index-remap", type=Path, default=None,
                        help="Path to index_remap.json (defaults to outputs/mappings/latest.json)")
    return parser


def main(argv=None) -> int:
    """
    CLI entry point:
    - Parses args and configures logging (with RUN_ID + stage='train').
    - Verifies mapping, builds loaders, model, and trains.
    - Writes a summary and copies mapping next to the checkpoint.
    """
    parser = make_parser_train()
    args = parser.parse_args(argv)

    # Run-aware logging (ties logs across stages)
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="train")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="train")

    bootstrap_env(seed=args.seed)
    log_env_once()

    # ---- Build runner inputs and execute training ----
    inputs = TrainRunnerInputs(
        image_size=args.image_size,
        train_in=args.train_in,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
        model_name=args.model,
        pretrained=bool(args.pretrained),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        amp=bool(args.amp),
        out_models=args.out_models,
        out_summary=args.out_summary,
        index_remap=args.index_remap,
        run_id=run_id,
        args_dict={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )


    best_f1, best_epoch, ckpt_path = run_training(inputs)
    log.info("training_complete", extra={
        "best_f1": round(best_f1, 4),
        "best_epoch": int(best_epoch),
        "checkpoint": str(ckpt_path),
    })
    print(f"✅ Training complete. Best F1={best_f1:.4f} at epoch {best_epoch}")
    print(f"📦 Best checkpoint: {ckpt_path}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())