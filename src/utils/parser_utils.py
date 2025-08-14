"""
Reusable argument parser utilities for the project.

Provides helper functions to build and extend argparse.ArgumentParser
instances for various pipeline steps and CLI tools.

Functions:
    add_common_logging_args(parser):
        - Adds standard logging options (--log-level, --log-file) to an existing parser.

Usage:
    from src.utils.parser_utils import  add_common_logging_args

    parser = add_common_logging_args()
    args = parser.parse_args()


Author: Tomasz Lasota
Date: 2025-0-13
Version: 1.0    
"""


import argparse


def add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    """Attach standard logging args to any parser."""
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                        help="Logging verbosity")
    
    parser.add_argument("--log-file", default=None,
                        help="Path to rotating log file. Leave empty for automatic per-script naming.")


def add_exts_arg(parser):
    """
    Add a --exts argument for specifying allowed file extensions.

    Default: ".png,.jpg,.jpeg,.bmp,.tif,.tiff"
    """
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg,.bmp,.tif,.tiff",
        help="Comma-separated extensions (lowercased)."
    )


def parse_exts(exts_str: str) -> set[str]:
    """
    Convert a comma-separated extension string into a normalized set.

    Ensures each extension starts with '.', is lowercase, and trims whitespace.
    """
    return {
        (e if e.startswith(".") else f".{e}")
        for e in (x.strip().lower() for x in exts_str.split(","))
        if e
    }