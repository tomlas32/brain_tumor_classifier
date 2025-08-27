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
from src.core.cleanup_policy import DEFAULT_POLICY, DEFAULT_ACT_ON
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


def parse_exts(exts) -> set[str]:
    """
    Normalize extensions to a set of lowercase, dot-prefixed suffixes.

    Accepts:
      - 'all' (str, case-insensitive)  → empty set (caller interprets as 'accept any')
      - '+webp,+gif' (str)             → DEFAULT_EXTS ∪ {'.webp','.gif'}
      - '.jpg,.png' (str)              → {'.jpg','.png'}
      - ['.jpg','.png'] (list/tuple/set) → {'.jpg','.png'}
      - None                           → defaults to parsing DEFAULT_EXTS

    Notes:
      - '+' semantics apply only for string inputs (to extend DEFAULT_EXTS).
      - For iterables, items are normalized but not merged with DEFAULT_EXTS.
    """
    def normalize_ext(e: str) -> str:
        e = e.strip().lower()
        return e if e.startswith(".") else f".{e}"

    if exts is None:
        # use project default list from DEFAULT_EXTS (comma string)
        return {normalize_ext(e) for e in DEFAULT_EXTS.split(",") if e}

    if isinstance(exts, str):
        s = exts.strip().lower()
        if s == "all":
            return set()  # accept any
        if s.startswith("+"):
            base = {normalize_ext(e) for e in DEFAULT_EXTS.split(",") if e}
            extras = {normalize_ext(e.lstrip("+")) for e in s.split(",") if e}
            return base | extras
        # plain comma list
        return {normalize_ext(e) for e in s.split(",") if e}

    # list / tuple / set
    return {normalize_ext(str(e)) for e in exts if str(e).strip()}


def add_common_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None,
                        help="(Deprecated) Prefer config.loop.epochs. If provided, overrides config.")
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
                        help="Use ImageNet pretrained weights (opt-in)")
    parser.add_argument("--out-models", type=Path, default=MODELS_DIR,
                        help="Directory to save best weights (default: models/)")
    parser.add_argument("--out-summary", type=Path, default=OUTPUTS_DIR / "training",
                        help="Directory to save training_summary.json (default: outputs/training/)")
    return parser


def add_common_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train-in", type=Path, default=DATA_DIR / "training",
                    help="Input root with class subfolders for TRAIN (default: data/training)")
    parser.add_argument("--test-in", type=Path, default=DATA_DIR / "testing",
                    help="Optional input root with class subfolders for TEST (default: data/testing)")
    parser.add_argument("--original-test-in", type=Path, default=None,
                        help="Optional 'original' external test set root (class subfolders)")
    parser.add_argument("--val-frac", type=float, default=0.20,
                        help="Validation fraction taken from --train-in (stratified)")
    return parser


def add_common_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--eval-in", type=Path, default=DATA_DIR / "testing",
                    help="Input root with class subfolders for evaluation (default: data/testing)")
    parser.add_argument("--eval-out", type=Path, default=OUTPUTS_DIR / "evaluation",
                        help="Output directory for evaluation (default: outputs/evaluation)")
    parser.add_argument("--trained-model", type=Path)
    return parser


def add_common_cleanup_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Extend an existing parser with cleanup-stage arguments.
    Uses defaults from src.core.cleanup_policy.
    """
    parser.add_argument(
        "--report",
        type=str,
        default="latest",
        help="Path to a validation report JSON, or 'latest' to use most recent under outputs/validation_reports.",
    )
    parser.add_argument(
        "--policy",
        choices=["strict", "within_class", "report_only"],
        default=DEFAULT_POLICY,
        help="Quarantine policy.",
    )
    parser.add_argument(
        "--why",
        choices=["errors", "warnings", "both"],
        default=DEFAULT_ACT_ON,
        help="Which severities to act on.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not move files.",
    )
    parser.add_argument(
        "--report-tag",
        choices=["pre", "post"],
        default=None,
        help="If --report=latest, prefer a validation report tagged with this suffix (e.g., *_pre.json).",
    )
    return parser


def add_common_config_args(
    parser: argparse.ArgumentParser,
    *,
    include_dry_run: bool = False,
    dry_help: str = "Plan only; do not execute or write artifacts."
) -> None:
    """Attach config/override (and optional --dry-run) flags to any parser."""
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML config file (config-first)."
    )
    parser.add_argument(
        "--override", "-o", action="append", default=[],
        help="Override config values as key=val. Repeatable."
    )
    if include_dry_run:
        parser.add_argument(
            "--dry-run", action="store_true", help=dry_help
        )
