"""
Fetch a Kaggle dataset via KaggleHub with mixed logging (stdout + rotating file).

- Logs key steps and timings to both console and a rotating log file.
- Keeps pipeline compatibility (stdout) while preserving local logs.
- Returns the resolved local dataset path on success; raises on failure.

Requirements:
    pip install kagglehub
    (and Kaggle API credentials configured)

Author: Tomasz Lasota
Date: 2025-0-13
Version: 1.2
"""


import kagglehub, os, time, argparse, json
from pathlib import Path
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args
from datetime import datetime, timezone
from src.utils.paths import DATA_DIR, OUTPUTS_DIR
from src.utils.configs import DEFAULT_DATASET

log = get_logger(__name__)
POINTER_DIR = OUTPUTS_DIR / "downloads_pointer"

def _find_subdir(root: Path, name: str) -> Path:
    """Return root/name; if not present, try case-insensitive match among subdirs."""

    if not root.exists():
        log.error("find_subdir.missing_root", extra={"root": str(root), "name": name})
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        log.error("find_subdir.not_a_dir", extra={"root": str(root), "name": name})
        raise NotADirectoryError(str(root))
    
    target_path = root / name
    log.debug("find_subdir.start", extra={"root": str(root), "name": name, "target_path": str(target_path)})

    # 1) Exact match
    if target_path.is_dir():
        log.info("find_subdir: found", extra={"target_path": str(target_path)})
        return target_path
    
    # 2) Case-insensitive match
    try:
        subdirs = sorted((p for p in root.iterdir() if p.is_dir()),
                         key=lambda p: p.name.casefold())
    except OSError as e:
        log.error("find_subdir.iter_error", extra={"root": str(root), "error": str(e)})
        raise

    name_cf = name.casefold()
    match = next((p for p in subdirs if p.name.casefold() == name_cf), None)
    if match is not None:
        log.info("find_subdir.ci_found", extra={"path": str(match)})
        return match

    available = [p.name for p in subdirs]
    log.error("find_subdir.not_found",
              extra={"root": str(root), "name": name, "available": available})
    raise FileNotFoundError(f"Expected subdir '{name}' under {root}; available: {available}")

def write_latest_fetch_json(dataset: str, dataset_root: Path, cache_dir: Path,
                            dst_dir: Path | None = None) -> Path:
    
    dataset_root = Path(dataset_root).resolve()
    training_dir = _find_subdir(dataset_root, "Training")
    testing_dir  = _find_subdir(dataset_root, "Testing")

    # parse dataset name and version
    root_name = dataset_root.name
    ver_str = root_name.lstrip("vV") # in case it starts with 'v' or 'V'
    version = int(ver_str) if ver_str.isdigit() else root_name

    # organise pointers by Kaggle slug
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
        "fetched_at":   datetime.now(timezone.utc).isoformat()
    }

    pointer_base = (dst_dir or (POINTER_DIR / owner / slug))
    pointer_base.mkdir(parents=True, exist_ok=True)

    # paths
    latest_path   = pointer_base / "latest.json"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    history_path  = pointer_base / f"{ts}__v{version}.json"

    # write both
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("handoff_written.latest", extra={"path": str(latest_path)})

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("handoff_written.history", extra={"path": str(history_path)})

    
    return latest_path

# Parser with additional args for kagglehub
def make_parser_fetch_kaggle() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Kaggle dataset with KaggleHub")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Default: {DEFAULT_DATASET}")
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR,
                        help="Directory to store the downloaded dataset")
    parser.add_argument("--no-pointer", action="store_true",
                        help="Skip writing outputs/downloads_pointer/.../latest.json")
    parser.add_argument("--pointer-dir", type=Path, default=None,
                        help="Override destination directory for pointer JSONs")
    add_common_logging_args(parser)
    return parser

# Fetch a Kaggle dataset
def fetch_kaggle(dataset: str = DEFAULT_DATASET,
                 cache_dir: Path | None = None) -> Path:
    
    """
     Downloads a dataset via KaggleHub into `cache_dir` and returns the local path.
    """

    t0 = time.time()
    cache_dir = DATA_DIR if cache_dir is None else Path(cache_dir)

    try:
        # Set custom kaggle data download destination directory
        log.info("fetch_kaggle:start", extra={"dataset": dataset, "cache_dir": str(cache_dir)})
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Point KaggleHub cache to data directory
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir.resolve())
        print(f"Downloading dataset '{dataset}' to: {cache_dir.resolve()}")
        log.debug("env_set", extra={"KAGGLEHUB_CACHE": os.environ["KAGGLEHUB_CACHE"]})

        log.info("download_begin", extra={"dataset": dataset})
        path = kagglehub.dataset_download(dataset)
        local_path = Path(path).resolve()

        elapsed = round(time.time() - t0, 2)
        log.info("download_success", extra={"local_path": str(local_path), "elapsed_s": elapsed})
        return local_path

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        log.error("download_failed", extra={"elapsed_s": elapsed}, exc_info=True)
        raise

def main(argv=None) -> int:

    parser = make_parser_fetch_kaggle()
    args = parser.parse_args()

    # Logging: auto file per script by default; fixed if user passes --log-file; stdout always on
    if args.log_file:  # user provided a path → fixed mode
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file)
    else:              # default → auto per-script name in outputs/logs/
        configure_logging(log_level=args.log_level, file_mode="auto")
    
    try:
        # Attempt the download
        target = fetch_kaggle(dataset=args.dataset, cache_dir=Path(args.cache_dir))

        # Write pointer JSONs
        write_latest_fetch_json(
            dataset=args.dataset,
            dataset_root=target,
            cache_dir=Path(args.cache_dir)
        )

        print(target)  # for shell/pipeline consumption
        return 0  # success

    except Exception as e:
        log.error(f"Fetch failed: {e}", exc_info=True)
        return 1  # failure

if __name__ == "__main__":

    raise SystemExit(main())
    

    
    
    

