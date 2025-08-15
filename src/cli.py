import typer
from typing import Optional
from pathlib import Path
from src.pipeline import fetch as fetch_mod
from src.pipeline import split as split_mod
from src.pipeline import resize as resize_mod
from src.pipeline import validate as validate_mod
from src.pipeline import train as train_mod
# from src.pipeline.evaluate import evaluate_model
# from src.pipeline.export import export_artifacts
from src.utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CONFIGS_DIR
from src.utils.configs import DEFAULT_DATASET
from src.utils.parser_utils import DEFAULT_EXTS


DEFAULT_INDEX_REMAP = OUTPUTS_DIR / "mappings" / "latest.json"

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

    code = fetch_mod.main(argv)    # calls fetch.py main(argv)
    raise typer.Exit(code)         # # propagate exit status to shell/CI

@app.command()
def split(
    dataset: str = DEFAULT_DATASET,
    pointer: Optional[Path] = None,
    test_frac: float = 0.20,
    seed: int = 42,
    exts: str = DEFAULT_EXTS,
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

    code = split_mod.main(argv)   # calls split.py main(argv)
    raise typer.Exit(code)        # propagate exit status to shell/CI
    

@app.command()
def resize(
    size: int = 224,
    train_in_dir: Path = DATA_DIR / "training",
    train_out_dir: Path = DATA_DIR / "training_resized",
    test_in_dir: Path = DATA_DIR / "testing",
    test_out_dir: Path = DATA_DIR / "testing_resized",
    exts: str = DEFAULT_EXTS,
    log_level: str = "INFO",
    log_file: Optional[str] = None

):
    """
    Resize and pad images in training and testing directories to a fixed square size.

    This wraps the `resize.py` pipeline step and preserves aspect ratio
    with black padding. Defaults assume `split.py` has been run first.

    Examples:
    python -m src.cli resize                       # default size 224, default exts
    python -m src.cli resize --size 256            # change output size
    python -m src.cli resize --exts all            # accept all file extensions
    python -m src.cli resize --exts +webp,+gif     # add extra extensions to defaults
    """
    argv = [
        "--size", str(size),
        "--train-in", str(train_in_dir),
        "--train-out", str(train_out_dir),
        "--test-in", str(test_in_dir),
        "--test-out", str(test_out_dir),
        "--exts", exts,
        "--log-level", log_level,
    ]

    if log_file:
        argv += ["--log-file", str(log_file)]

    code = resize_mod.main(argv)   # calls resize.py main(argv)
    raise typer.Exit(code)        # propagate exit status to shell/CI


@app.command()
def validate(
    in_dir: Path = DATA_DIR / "training",
    index_remap: Path = DEFAULT_INDEX_REMAP,  # outputs/mappings/latest.json
    size: int = 224,
    exts: str = typer.Option(
        DEFAULT_EXTS,
        help="Comma-separated extensions. Use +ext to add (e.g. '+webp'); use 'all' to accept any."
    ),
    dup_check: bool = False,  # default off (hashing can be slow); enable when needed
    fail_on: str = typer.Option("error", help="Fail on: 'error' | 'warning' | 'never'"),
    warn_low_std: float = 3.0,
    min_file_bytes: int = 1024,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Validate the integrity and consistency of the resized dataset before training.

    This command wraps `src.pipeline.validate.main(argv)` to match the structure
    and behavior of other pipeline commands, ensuring consistent logging,
    argument parsing, and exit code handling across both the CLI and standalone usage.

    Examples
    --------
    # Validate using defaults (DATA_DIR/training and latest mapping in outputs/mappings/)
    python -m src.cli validate

    # Validate with duplicate detection enabled
    python -m src.cli validate --dup-check

    # Fail if any warnings or errors are found
    python -m src.cli validate --fail-on warning

    # Validate a custom directory and mapping file
    python -m src.cli validate --in-dir data/training_resized --index-remap outputs/mappings/my_map.json
    """
    argv = [
        "--in-dir", str(in_dir),
        "--index-remap", str(index_remap),
        "--size", str(size),
        "--exts", exts,
        "--warn-low-std", str(warn_low_std),
        "--min-file-bytes", str(min_file_bytes),
        "--fail-on", fail_on,
        "--log-level", log_level,
    ]
    if dup_check:
        argv += ["--dup-check"]
    if log_file:
        argv += ["--log-file", str(log_file)]

    code = validate_mod.main(argv)   # calls validate.py main(argv)
    raise typer.Exit(code)           # propagate exit status to shell/CI

@app.command()
def train(
    # I/O
    train_in: Path = DATA_DIR / "training_resized",
    out_models: Path = MODELS_DIR,
    out_summary: Path = OUTPUTS_DIR / "training",

    # data/split
    val_frac: float = 0.20,
    image_size: int = 224,

    # training
    batch_size: int = 32,
    num_workers: int = 4,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    step_size: int = 5,
    gamma: float = 0.5,
    seed: int = 42,
    amp: bool = True,

    # model
    model: str = "resnet18",           # choices: resnet18 | resnet34 | resnet50
    pretrained: bool = True,

    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Train a CNN on the resized training set.

    Wraps `src.training.train.main(argv)` to keep CLI consistent with other steps.
    Uses the class mapping from `outputs/mappings/latest.json` (written by `split`).

    Examples
    --------
    # Default training
    python -m src.cli train

    # Heavier model, more epochs
    python -m src.cli train --model resnet50 --epochs 30

    # Change LR/scheduler and disable AMP
    python -m src.cli train --lr 3e-4 --step-size 10 --gamma 0.3 --amp False

    # Custom paths
    python -m src.cli train --train-in data/training_resized --out-models models/brain_tumor
    """
    argv = [
        "--train-in", str(train_in),
        "--val-frac", str(val_frac),
        "--image-size", str(image_size),

        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--step-size", str(step_size),
        "--gamma", str(gamma),
        "--seed", str(seed),

        "--model", model,
        "--out-models", str(out_models),
        "--out-summary", str(out_summary),

        "--log-level", log_level,
    ]

    # booleans as flags (match train.py parser)
    if not amp:
        argv += ["--no-amp"]
    if pretrained:
        argv += ["--pretrained"]
    else:
        argv += ["--no-pretrained"]
    if log_file:
        argv += ["--log-file", str(log_file)]

    code = train_mod.main(argv)   # calls src/training/train.py:main(argv)
    raise typer.Exit(code)

# @app.command()
# def evaluate(cfg: str = str(CONFIGS_DIR / "eval.yaml")):
#     evaluate_model(cfg, models_dir=MODELS_DIR, outputs_dir=OUTPUTS_DIR)


if __name__ == "__main__":
    app()