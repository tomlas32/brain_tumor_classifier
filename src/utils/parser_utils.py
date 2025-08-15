"""
Reusable argparse helpers for the project.

This module provides small utilities to extend existing
`argparse.ArgumentParser` instances with common options and to parse
shared argument types (e.g., image file extensions).

Exports
-------
- DEFAULT_EXTS : str
    Default comma-separated extensions used by image-processing steps.
- add_common_logging_args(parser)
    Add standard logging options (--log-level, --log-file) to `parser`.
- add_exts_arg(parser)
    Add the --exts option to `parser` (supports '+ext' to add to defaults).
- parse_exts(exts_str) -> set[str]
    Parse/normalize a comma-separated extension string. Supports:
      * "+ext1,+ext2" to add to DEFAULT_EXTS
      * "all" to disable filtering (caller interprets empty set as "accept all")
- normalize_ext(e) -> str
    Normalize a single extension to lowercase and ensure it starts with '.'.

Usage
-----
    from argparse import ArgumentParser
    from src.utils.parser_utils import (
        add_common_logging_args, add_exts_arg, parse_exts
    )

    parser = ArgumentParser(description="My tool")
    add_common_logging_args(parser)
    add_exts_arg(parser)
    args = parser.parse_args()

    exts = parse_exts(args.exts)  # → {".png", ".jpg", ...} or set() if 'all'

Author: Tomasz Lasota
Date: 2025-08-14
Version: 1.1
"""


import argparse
from src.utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from pathlib import Path

DEFAULT_EXTS = ".png,.jpg,.jpeg,.bmp,.tif,.tiff"

def add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    """Attach standard logging args to any parser."""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to rotating log file. Leave empty for automatic per-script naming.",
    )

def add_exts_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add a --exts argument for specifying allowed file extensions.

    Default: ".png,.jpg,.jpeg,.bmp,.tif,.tiff"
    Use '+ext' to add to defaults, e.g. '+webp,+gif'
    """
    parser.add_argument(
        "--exts",
        type=str,
        default=DEFAULT_EXTS,  # ← use the constant
        help=(
            "Comma-separated extensions (lowercased). "
            "Use +ext to add to defaults, e.g. '+webp,+gif'. "
            "Use 'all' to disable filtering."
        ),
    )


def parse_exts(exts_str: str) -> set[str]:
    """
    Convert a comma-separated extension string into a normalized set.
    - If it starts with '+', extensions are ADDED to DEFAULT_EXTS.
    - If it equals 'all' (case-insensitive), no filtering is applied (returns empty set to mean 'accept all').
    """
    s = exts_str.strip().lower()
    if s == "all":
        return set()  # interpret as 'accept any extension'

    if s.startswith("+"):
        base = {normalize_ext(e) for e in DEFAULT_EXTS.split(",") if e}
        extras = {
            normalize_ext(e.lstrip("+"))  # ← strip '+' per item
            for e in s.split(",")
            if e
        }
        return base | extras

    return {normalize_ext(e) for e in s.split(",") if e}

def normalize_ext(e: str) -> str:
    e = e.strip().lower().lstrip("+")  # defensive: strip '+' if present
    return e if e.startswith(".") else f".{e}"


def add_common_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=5, help="StepLR: step_size (epochs)")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR: gamma")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True,
                        help="Disable automatic mixed precision (AMP)")
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet weights (default: True)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Disable pretrained weights")
    parser.add_argument("--out-models", type=Path, default=MODELS_DIR,
                        help="Directory to save best weights (default: models/)")
    parser.add_argument("--out-summary", type=Path, default=OUTPUTS_DIR / "training",
                        help="Directory to save training_summary.json (default: outputs/training/)")
    return parser


def add_common_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train-in", type=Path, default=DATA_DIR / "training_resized",
                        help="Input root with class subfolders for TRAIN (default: data/training_resized)")
    parser.add_argument("--test-in", type=Path, default=DATA_DIR / "testing_resized",
                        help="Optional input root with class subfolders for TEST (default: data/testing_resized)")
    parser.add_argument("--original-test-in", type=Path, default=None,
                        help="Optional 'original' external test set root (class subfolders)")
    parser.add_argument("--val-frac", type=float, default=0.20,
                        help="Validation fraction taken from --train-in (stratified)")
    return parser