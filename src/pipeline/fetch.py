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


import kagglehub, os, time, argparse
from pathlib import Path
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.parser_utils import add_common_logging_args
from src.utils.paths import DATA_DIR
from src.utils.configs import DEFAULT_DATASET

log = get_logger(__name__)

def make_parser_fetch_kaggle() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Kaggle dataset with KaggleHub")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Default: {DEFAULT_DATASET}")
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR,
                        help="Directory to store the downloaded dataset")
    add_common_logging_args(parser)
    return parser

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


if __name__ == "__main__":

    parser = make_parser_fetch_kaggle()
    args = parser.parse_args()
    

    # Logging: auto file per script by default; fixed if user passes --log-file; stdout always on
    if args.log_file:  # user provided a path → fixed mode
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file)
    else:              # default → auto per-script name in outputs/logs/
        configure_logging(log_level=args.log_level, file_mode="auto")
    
    # Run
    target = fetch_kaggle(dataset=args.dataset, cache_dir=args.cache_dir)
    print(target)  # keep for shell/pipeline consumption

