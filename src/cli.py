import typer
from typing import Optional
from pathlib import Path
from src.pipeline import fetch as fetch_mod
from src.pipeline import split as split_mod
# from src.pipeline.resize import resize_and_pad_batch
# from src.pipeline.train import train_model
# from src.pipeline.evaluate import evaluate_model
# from src.pipeline.export import export_artifacts
from src.utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CONFIGS_DIR
from src.utils.configs import DEFAULT_DATASET

app = typer.Typer()

@app.command()
def fetch(
    dataset: str = DEFAULT_DATASET,
    cache_dir: Path = DATA_DIR,
    write_pointer: bool = True,
    pointer_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    
    """
    Download a Kaggle dataset into the local data directory.

    This command wraps the `fetch.py` pipeline step, downloading the specified
    Kaggle dataset via KaggleHub into `DATA_DIR` (or a custom location) and
    optionally writing a pointer JSON for downstream steps.

    Parameters
    ----------
    dataset : str, optional
        Kaggle dataset slug in the form 'owner/dataset'.
        Defaults to the project's DEFAULT_DATASET.
    cache_dir : Path, optional
        Directory to store the downloaded dataset. Defaults to DATA_DIR.
    write_pointer : bool, optional
        If True (default), write `latest.json` and a timestamped history file
        into the pointer directory.
    pointer_dir : Path, optional
        Custom directory for pointer files. Overrides the default
        `OUTPUTS_DIR/downloads_pointer/<owner>/<slug>/`.
    log_level : str, optional
        Logging verbosity. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str, optional
        Path to a log file. If omitted, logs go to stdout and an auto-named file.

    Examples
    --------
    # Download the default dataset into DATA_DIR and write pointer
    python -m src.cli fetch

    # Download into a custom directory without writing a pointer
    python -m src.cli fetch --cache-dir /tmp/mydata --write-pointer False

    # Download and write pointer into a custom directory
    python -m src.cli fetch --pointer-dir /custom/dir

    """
    argv = [
        "--dataset", dataset,
        "--cache-dir", str(cache_dir),
        "--log-level", log_level,
    ]
    if not write_pointer:
        argv += ["--no-pointer"]
    if pointer_dir is not None:
        argv += ["--pointer-dir", str(pointer_dir)]
    if log_file:
        argv += ["--log-file", str(log_file)]

    code = fetch_mod.main(argv)
    raise typer.Exit(code)

@app.command()
def split(
    dataset: str = DEFAULT_DATASET,
    pointer: Optional[Path] = None,
    test_frac: float = 0.20,
    seed: int = 42,
    exts: str = ".png,.jpg,.jpeg,.bmp,.tif,.tiff",
    clear_dest: bool = False,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Re-split pooled images into DATA_DIR/training and DATA_DIR/testing.

    Examples:
    python -m src.cli split                  # use default extensions
    python -m src.cli split --exts webp      # replace defaults with webp
    python -m src.cli split --exts +webp,+gif  # add to defaults
    """
    argv = [
        "--dataset", dataset,
        "--test-frac", str(test_frac),
        "--seed", str(seed),
        "--exts", exts,
        "--log-level", log_level,
    ]
    if pointer:
        argv += ["--pointer", str(pointer)]
    if clear_dest:
        argv += ["--clear-dest"]
    if log_file:
        argv += ["--log-file", str(log_file)]

    code = split_mod.main(argv)   # calls your split.py main(argv)
    raise typer.Exit(code)        # propagate exit status to shell/CI
    

# @app.command()
# def resize(in_dir: str = str(DATA_DIR / "combined_split_simple"),
#            out_dir: str = str(DATA_DIR / "training"),
#            size: int = 224):
#     resize_and_pad_batch(in_dir, out_dir, size)

# @app.command()
# def train(cfg: str = str(CONFIGS_DIR / "train.yaml")):
#     train_model(cfg, models_dir=MODELS_DIR, outputs_dir=OUTPUTS_DIR)

# @app.command()
# def evaluate(cfg: str = str(CONFIGS_DIR / "eval.yaml")):
#     evaluate_model(cfg, models_dir=MODELS_DIR, outputs_dir=OUTPUTS_DIR)

# @app.command()
# def export(src_weights: str, dst_dir: str = str(MODELS_DIR)):
#     export_artifacts(src_weights, dst_dir)

if __name__ == "__main__":
    app()