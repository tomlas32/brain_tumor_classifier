# src/pipeline/fetch.py
"""
Fetch a Kaggle dataset via KaggleHub with mixed logging (stdout + rotating file).

What this script does
---------------------
1) Configures logging so every line carries [fetch|RUN_ID] for easy stitching in Docker/CI.
2) Downloads a Kaggle dataset into a cache directory (defaults to DATA_DIR).
3) Writes a handoff JSON pointer under outputs/downloads_pointer/<owner>/<slug>/:
   - latest.json
   - timestamped history file

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

import argparse, json, os, time
from datetime import datetime, timezone
from pathlib import Path

import kagglehub

from src.utils.configs import DEFAULT_DATASET
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args
from src.utils.paths import DATA_DIR, OUTPUTS_DIR

from src.core.config import build_fetch_config, to_dict

log = get_logger(__name__)
POINTER_DIR = OUTPUTS_DIR / "downloads_pointer"


# ------------------------------ helpers ------------------------------

def _find_subdir(root: Path, name: str) -> Path:
    """
    Return root/name; if absent, try a case-insensitive match among subdirs.

    Raises
    ------
    FileNotFoundError, NotADirectoryError
    """
    if not root.exists():
        log.error("find_subdir.missing_root", extra={"root": str(root), "name": name})
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        log.error("find_subdir.not_a_dir", extra={"root": str(root), "name": name})
        raise NotADirectoryError(str(root))

    target_path = root / name
    log.debug("find_subdir.start", extra={"root": str(root), "name": name, "target_path": str(target_path)})

    # Exact match
    if target_path.is_dir():
        log.info("find_subdir.found", extra={"target_path": str(target_path)})
        return target_path

    # Case-insensitive match
    try:
        subdirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
    except OSError as e:
        log.error("find_subdir.iter_error", extra={"root": str(root), "error": str(e)})
        raise

    name_cf = name.casefold()
    match = next((p for p in subdirs if p.name.casefold() == name_cf), None)
    if match is not None:
        log.info("find_subdir.ci_found", extra={"path": str(match)})
        return match

    available = [p.name for p in subdirs]
    log.error("find_subdir.not_found", extra={"root": str(root), "name": name, "available": available})
    raise FileNotFoundError(f"Expected subdir '{name}' under {root}; available: {available}")


def write_latest_fetch_json(
    dataset: str,
    dataset_root: Path,
    cache_dir: Path,
    dst_dir: Path | None = None,
) -> Path:
    """
    Write a handoff JSON describing the fetched dataset (latest + history).

    Fields
    ------
    dataset        Kaggle slug (owner/slug) used for download
    version        Integer if root directory is a pure 'vNN' string, otherwise folder name
    dataset_root   Absolute path to the local dataset root returned by KaggleHub
    training_dir   Path to the 'Training' subfolder (case-insensitive search)
    testing_dir    Path to the 'Testing' subfolder (case-insensitive search)
    cache_dir      Absolute path to the chosen cache directory
    fetched_at     ISO-8601 UTC timestamp of when the pointer was written

    Returns
    -------
    Path to latest.json
    """
    dataset_root = Path(dataset_root).resolve()
    training_dir = _find_subdir(dataset_root, "Training")
    testing_dir  = _find_subdir(dataset_root, "Testing")

    # Parse dataset name and a simple version from folder name (e.g., 'v2' → 2)
    root_name = dataset_root.name
    ver_str = root_name.lstrip("vV")
    version = int(ver_str) if ver_str.isdigit() else root_name

    # Organize pointers by Kaggle slug
    try:
        owner, slug = dataset.split("/", 1)
    except ValueError:
        owner, slug = "unknown", dataset

    payload = {
        "dataset": dataset,
        "version": version,
        "dataset_root": str(dataset_root),
        "training_dir": str(training_dir),
        "testing_dir":  str(testing_dir),
        "cache_dir":    str(Path(cache_dir).resolve()),
        "fetched_at":   datetime.now(timezone.utc).isoformat(),
    }

    pointer_base = (dst_dir or (POINTER_DIR / owner / slug))
    pointer_base.mkdir(parents=True, exist_ok=True)

    latest_path  = pointer_base / "latest.json"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    history_path = pointer_base / f"{ts}__v{version}.json"

    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("handoff_written.latest", extra={"path": str(latest_path)})

    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("handoff_written.history", extra={"path": str(history_path)})

    return latest_path


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
        if write_pointer and not args.no_pointer:  # keep legacy flag respected too
            write_latest_fetch_json(
                dataset=dataset,
                dataset_root=target,
                cache_dir=cache_dir,
                dst_dir=pointer_dir,
            )
        print(target)
        return 0

    except Exception as e:
        log.error("fetch.failed", extra={"error": str(e)}, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
