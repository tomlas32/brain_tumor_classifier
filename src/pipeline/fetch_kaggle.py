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
Version: 1.1
"""


import kagglehub, os, sys, argparse, time, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

log = logging.getLogger(__name__)

def configure_logging(log_level: str = "INFO",
                      log_file: Path = Path("logs/fetch_kaggle.log")) -> None:
    """
    Configure logging to stdout AND to a rotating file.
    """
    # Ensure log directory exists
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Common formatter
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    # Console handler (stdout)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    # Rotating file handler (5 MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    # Root logger setup
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers if configure_logging is called twice
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

def fetch_kaggle(dataset: str = "sartajbhuvaji/brain-tumor-classification-mri",
                 cache_dir: Path = Path("data")) -> Path:
    
    """
     Downloads a dataset via KaggleHub into `cache_dir` and returns the local path.
    """

    t0 = time.time()
    cache_dir = Path(cache_dir)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Kaggle dataset with KaggleHub")
    parser.add_argument(
        "--dataset",
        default="sartajbhuvaji/brain-tumor-classification-mri",
        help="Kaggle dataset slug, e.g. 'owner/dataset'",
    )
    parser.add_argument(
        "--cache-dir",
        default="data",
        help="Directory to store the downloaded dataset",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--log-file",
        default="logs/fetch_kaggle.log",
        help="Path to the rotating log file (will be created if it doesn't exist)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    configure_logging(log_level=args.log_level, log_file=Path(args.log_file))

     # Run
    target = fetch_kaggle(dataset=args.dataset, cache_dir=Path(args.cache_dir))
    # Print the final path for easy shell consumption
    print(target)

