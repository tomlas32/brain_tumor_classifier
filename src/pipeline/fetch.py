# src/pipeline/fetch.py
"""
Fetch a Kaggle dataset via KaggleHub with mixed logging (stdout + rotating file).

What this script does
---------------------
1) Configures logging so every line carries [fetch|RUN_ID] for easy stitching in Docker/CI.
2) Downloads a Kaggle dataset into a cache directory (defaults to DATA_DIR).
3) Writes a standardized *fetch pointer* under outputs/pointers/fetch/<owner>/<slug>/:
  - latest.json
  - history/fetch_YYYYMMDD_HHMMSSZ.json

Requirements
------------
- pip install kagglehub
- Kaggle API credentials configured (e.g., ~/.kaggle/kaggle.json)

Typical pipeline order
----------------------
**fetch** → split → resize → validate → train → evaluate

CLI arguments
-------------
--dataset STR       Kaggle slug, e.g. 'owner/dataset' (default from configs).
--cache-dir PATH    Where KaggleHub will cache the download (defaults to DATA_DIR).
--no-pointer        Skip writing the pointer JSONs (latest + history).
--pointer-dir PATH  Override destination dir for pointer JSONs.
--log-level LEVEL   DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO).
--log-file PATH     If set, use a fixed log file; otherwise auto per-script under outputs/logs/.

Exit codes
----------
0  Success
1  Failure (exception during download or pointer write)

Examples
--------
# Basic usage with defaults
python -m src.pipeline.fetch

# Custom dataset slug and cache dir
python -m src.pipeline.fetch --dataset owner/name --cache-dir data

# Write handoff pointer into a custom folder
python -m src.pipeline.fetch --pointer-dir outputs/pointers/my_exp

# Config-first
python -m src.pipeline.fetch --config configs/fetch.yaml

# Config + overrides (flip pointer off)
python -m src.pipeline.fetch --config configs/fetch.yaml \
  --override write_pointer=false

# Legacy still works (no config)
python -m src.pipeline.fetch --dataset owner/name --cache-dir data
"""

from __future__ import annotations

import argparse, os, time
from datetime import datetime, timezone
from pathlib import Path

import kagglehub

from src.utils.configs import DEFAULT_DATASET
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args
from src.utils.paths import DATA_DIR

from src.core.config import build_fetch_config, to_dict
from src.core.artifacts import write_fetch_pointer

log = get_logger(__name__)


def make_parser_fetch_kaggle() -> argparse.ArgumentParser:
    """
    CLI for KaggleHub fetch.

    Examples
    --------
    python -m src.pipeline.fetch --dataset owner/name
    python -m src.pipeline.fetch --cache-dir data --no-pointer
    """
    parser = argparse.ArgumentParser(description="Fetch Kaggle dataset with KaggleHub")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Kaggle slug, e.g. 'owner/dataset' (default: {DEFAULT_DATASET})")
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR,
                        help="Directory to store the downloaded dataset (default: DATA_DIR)")
    parser.add_argument("--no-pointer", action="store_true",
                        help="Skip writing outputs/downloads_pointer/.../latest.json")
    parser.add_argument("--pointer-dir", type=Path, default=None,
                        help="Override destination directory for pointer JSONs")
    parser.add_argument("--config", type=Path, default=None,
                    help="Optional YAML config file for fetch (config-first).")
    parser.add_argument("--override", action="append", default=[],
                        help="Override config values as key=val (e.g., dataset=owner/slug write_pointer=false). "
                            "Repeat for multiple overrides.")
    add_common_logging_args(parser)  # --log-level, --log-file
    return parser


# ------------------------------ core logic ------------------------------

def fetch_kaggle(
    dataset: str = DEFAULT_DATASET,
    cache_dir: Path | None = None,
) -> Path:
    """
    Download a Kaggle dataset via KaggleHub into `cache_dir` and return the local path.

    Notes
    -----
    - Sets the env var KAGGLEHUB_CACHE to the chosen cache_dir so KaggleHub uses it.
    - Emits a short stdout message (for simple pipelines) and detailed logs.
    """
    t0 = time.time()
    cache_dir = DATA_DIR if cache_dir is None else Path(cache_dir)

    try:
        log.info("fetch_kaggle.start", extra={"dataset": dataset, "cache_dir": str(cache_dir)})
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Point KaggleHub to our cache dir
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir.resolve())
        print(f"Downloading dataset '{dataset}' to: {cache_dir.resolve()}")
        log.debug("env_set", extra={"KAGGLEHUB_CACHE": os.environ["KAGGLEHUB_CACHE"]})

        log.info("download.begin", extra={"dataset": dataset})
        path = kagglehub.dataset_download(dataset)
        local_path = Path(path).resolve()

        elapsed = round(time.time() - t0, 2)
        log.info("download.success", extra={"local_path": str(local_path), "elapsed_s": elapsed})
        return local_path

    except Exception:
        elapsed = round(time.time() - t0, 2)
        log.error("download.failed", extra={"elapsed_s": elapsed}, exc_info=True)
        raise


# ------------------------------ entry point ------------------------------

def main(argv=None) -> int:
    """
    Entry:
    - Parse CLI and configure logging with [fetch|RUN_ID].
    - Download the dataset and (optionally) write the handoff pointer JSONs.
    """
    parser = make_parser_fetch_kaggle()
    args = parser.parse_args(argv)

    # Run-aware logging (ties logs across stages)
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="fetch")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="fetch")

    # Resolve config (config-first with CLI fallback)
    cfg = build_fetch_config(args.config, overrides=args.override)
    log.info("config.resolved", extra={"config": to_dict(cfg)})

    dataset = cfg.dataset or args.dataset
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else Path(args.cache_dir)
    write_pointer = bool(cfg.write_pointer)
    pointer_dir = cfg.pointer_dir or args.pointer_dir
    
    try:
        target = fetch_kaggle(dataset=dataset, cache_dir=cache_dir)
        if write_pointer and not args.no_pointer:
            try:
                out = write_fetch_pointer(
                    dataset=dataset,
                    dataset_root=target,
                    # If you want to auto-detect these at fetch time, see note below.
                    training_dir=None,
                    testing_dir=None,
                    version=None,
                    run_id=run_id,
                    dst_dir=pointer_dir,  # if None → outputs/pointers/fetch/<owner>/<slug>/
                )
                log.info("fetch.pointer_written", extra={
                    "dataset": dataset,
                    "latest": str(out.get("latest_path")),
                    "history": str(out.get("history_path")),
                })
            except Exception as e:
                log.warning("fetch.pointer_write_failed", extra={"error": str(e)})

        print(target)
        return 0

    except Exception as e:
        log.error("fetch.failed", extra={"error": str(e)}, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
