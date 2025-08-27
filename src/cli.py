"""
Typer-based CLI façade for the pipeline.

Each command:
- Collects arguments (CLI flags, optional --config YAML, optional --override key=val).
- Dispatches into the corresponding stage module via its main(argv).
- Ensures structured logging with a dispatch entry (stage + argv).

IMPORTANT:
This CLI is *config-first*. Any option that also exists in a stage YAML defaults
to None and is only forwarded if explicitly provided — so the CLI will NOT
silently override your YAML values.
"""

from typing import Optional, List
from pathlib import Path
import typer

from src.utils.paths import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    MERGED_DIR,
    PROCESSED_DIR,
    DEFAULT_DATASET,
    DEFAULT_INDEX_REMAP,
)
from src.utils.parser_utils import DEFAULT_EXTS
from src.utils.logging_utils import get_logger

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


# ---------------------------
# MERGE
# ---------------------------
@app.command()
def merge(
    dataset: Optional[str] = None,
    pointer: Optional[Path] = None,
    exts: Optional[str] = None,
    clear_dest: Optional[bool] = None,
    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values as key=val (e.g., clear_dest=true exts=all). Repeatable."),
    dry_run: bool = False,
):
    """
    Merge Kaggle's original Training/Testing into a single canonical set:

        data/merged/<class>/*

    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import merge as merge_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # Only forward overrides when explicitly provided
    if dataset is not None:
        argv += ["--dataset", dataset]
    if exts is not None:
        argv += ["--exts", exts]
    if pointer is not None:
        argv += ["--pointer", str(pointer)]
    if clear_dest is True:
        argv += ["--clear-dest"]
    elif clear_dest is False:
        argv += ["--no-clear-dest"]

    log.info("cli.dispatch", extra={"stage": "merge", "argv": argv})
    code = int(merge_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# FETCH
# ---------------------------
@app.command()
def fetch(
    dataset: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    write_pointer: Optional[bool] = None,
    pointer_dir: Optional[Path] = None,
    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values as key=val (e.g., dataset=owner/slug write_pointer=false). Repeatable."),
    dry_run: bool = False,
):
    """
    Download a Kaggle dataset (via KaggleHub) into DATA_DIR (or custom cache_dir).
    Optionally write a pointer for downstream stages.

    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import fetch as fetch_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # overrides only when provided
    if dataset is not None:
        argv += ["--dataset", dataset]
    if cache_dir is not None:
        argv += ["--cache-dir", str(cache_dir)]
    if write_pointer is True:
        argv += ["--write-pointer"]
    elif write_pointer is False:
        argv += ["--no-write-pointer"]
    if pointer_dir is not None:
        argv += ["--pointer-dir", str(pointer_dir)]

    log.info("cli.dispatch", extra={"stage": "fetch", "argv": argv})
    code = int(fetch_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# SPLIT
# ---------------------------
@app.command()
def split(
    in_dir: Optional[Path] = None,
    out_train: Optional[Path] = None,
    out_test: Optional[Path] = None,
    val_frac: Optional[float] = None,
    test_frac: Optional[float] = None,
    seed: Optional[int] = None,
    stratify: Optional[bool] = None,
    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values as key=val (e.g., val_frac=0.2 test_frac=0.1). Repeatable."),
    dry_run: bool = False,
):
    """
    Split a merged dataset into train/val/test pointers.

    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import split as split_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # overrides only when provided
    if in_dir is not None:
        argv += ["--in-dir", str(in_dir)]
    if out_train is not None:
        argv += ["--out-train", str(out_train)]
    if out_test is not None:
        argv += ["--out-test", str(out_test)]
    if val_frac is not None:
        argv += ["--val-frac", str(val_frac)]
    if test_frac is not None:
        argv += ["--test-frac", str(test_frac)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if stratify is True:
        argv += ["--stratify"]
    elif stratify is False:
        argv += ["--no-stratify"]

    log.info("cli.dispatch", extra={"stage": "split", "argv": argv})
    code = int(split_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# RESIZE
# ---------------------------
@app.command()
def resize(
    size: Optional[int] = None,
    train_in_dir: Optional[Path] = None,      # e.g., MERGED_DIR
    train_out_dir: Optional[Path] = None,     # e.g., PROCESSED_DIR
    test_in_dir: Optional[Path] = None,
    test_out_dir: Optional[Path] = None,
    exts: Optional[str] = None,
    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values as key=val (e.g., size=256 exts=all). Repeatable."),
    dry_run: bool = False,
):
    """
    Resize/pad images to a fixed square size (aspect preserved via padding).

    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import resize as resize_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # overrides only when provided
    if size is not None:
        argv += ["--size", str(size)]
    if train_in_dir is not None:
        argv += ["--train-in", str(train_in_dir)]
    if train_out_dir is not None:
        argv += ["--train-out", str(train_out_dir)]
    if test_in_dir is not None:
        argv += ["--test-in", str(test_in_dir)]
    if test_out_dir is not None:
        argv += ["--test-out", str(test_out_dir)]
    if exts is not None:
        argv += ["--exts", exts]

    log.info("cli.dispatch", extra={"stage": "resize", "argv": argv})
    code = int(resize_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# VALIDATE
# ---------------------------
@app.command()
def validate(
    in_dir: Optional[Path] = None,
    index_remap: Optional[Path] = None,
    size: Optional[int] = None,
    exts: Optional[str] = None,
    dup_check: Optional[bool] = None,
    phash: Optional[bool] = None,
    phash_thresh: Optional[int] = None,
    ssim_thresh: Optional[float] = None,
    fail_on: Optional[str] = None,
    warn_low_std: Optional[float] = None,
    min_file_bytes: Optional[int] = None,
    enforce_size: Optional[bool] = None,
    require_rgb: Optional[bool] = None,
    write_report: Optional[bool] = None,
    report_tag: Optional[str] = None,
    dup_mode: Optional[str] = None,
    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values as key=val (e.g., size=256 exts=all fail_on=warning). Repeatable."),
    dry_run: bool = False,
):
    """
    Validate dataset structure/quality (pre / post). Includes optional duplicate detection.

    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import validate as validate_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # overrides only when provided
    if in_dir is not None:
        argv += ["--in-dir", str(in_dir)]
    if index_remap is not None:
        argv += ["--index-remap", str(index_remap)]
    if size is not None:
        argv += ["--size", str(size)]
    if exts is not None:
        argv += ["--exts", exts]

    if dup_check is True:
        argv += ["--dup-check"]
    elif dup_check is False:
        argv += ["--no-dup-check"]

    if phash is True:
        argv += ["--phash"]
    elif phash is False:
        argv += ["--no-phash"]

    if phash_thresh is not None:
        argv += ["--phash-thresh", str(phash_thresh)]
    if ssim_thresh is not None:
        argv += ["--ssim-thresh", str(ssim_thresh)]

    if fail_on is not None:
        argv += ["--fail-on", fail_on]
    if warn_low_std is not None:
        argv += ["--warn-low-std", str(warn_low_std)]
    if min_file_bytes is not None:
        argv += ["--min-file-bytes", str(min_file_bytes)]

    if enforce_size is True:
        argv += ["--enforce-size"]
    elif enforce_size is False:
        argv += ["--no-enforce-size"]

    if require_rgb is True:
        argv += ["--require-rgb"]
    elif require_rgb is False:
        argv += ["--no-require-rgb"]

    if write_report is True:
        argv += ["--write-report"]
    elif write_report is False:
        argv += ["--no-write-report"]

    if report_tag is not None:
        argv += ["--report-tag", report_tag]
    if dup_mode is not None:
        argv += ["--dup-mode", dup_mode]

    log.info("cli.dispatch", extra={"stage": "validate", "argv": argv})
    code = int(validate_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# TRAIN
# ---------------------------
@app.command()
def train(
    # I/O (YAML-first)
    train_in: Optional[Path] = None,
    val_in: Optional[Path] = None,
    out_models: Optional[Path] = None,
    out_summary: Optional[Path] = None,

    # data/split (YAML-first)
    val_frac: Optional[float] = None,
    image_size: Optional[int] = None,

    # training (YAML-first)
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    step_size: Optional[int] = None,
    gamma: Optional[float] = None,
    seed: Optional[int] = None,
    amp: Optional[bool] = None,

    # model (YAML-first)
    model: Optional[str] = None,
    pretrained: Optional[bool] = None,

    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,

    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values: key=val (e.g., model.name=resnet50). Repeatable."),
    dry_run: bool = False,
):
    """
    Train a CNN on the prepared dataset (config-first).
    Saves checkpoints and a training summary.

    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import train as train_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # overrides only when provided
    if train_in is not None:
        argv += ["--train-in", str(train_in)]
    if val_in is not None:
        argv += ["--val-in", str(val_in)]
    if out_models is not None:
        argv += ["--out-models", str(out_models)]
    if out_summary is not None:
        argv += ["--out-summary", str(out_summary)]

    if val_frac is not None:
        argv += ["--val-frac", str(val_frac)]
    if image_size is not None:
        argv += ["--image-size", str(image_size)]

    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    if num_workers is not None:
        argv += ["--num-workers", str(num_workers)]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    if lr is not None:
        argv += ["--lr", str(lr)]
    if weight_decay is not None:
        argv += ["--weight-decay", str(weight_decay)]
    if step_size is not None:
        argv += ["--step-size", str(step_size)]
    if gamma is not None:
        argv += ["--gamma", str(gamma)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if amp is True:
        argv += ["--amp"]
    elif amp is False:
        argv += ["--no-amp"]

    if model is not None:
        argv += ["--model", model]
    if pretrained is True:
        argv += ["--pretrained"]
    elif pretrained is False:
        argv += ["--no-pretrained"]

    log.info("cli.dispatch", extra={"stage": "train", "argv": argv})
    code = int(train_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# EVALUATE
# ---------------------------
@app.command()
def evaluate(
    # data
    eval_in: Optional[Path] = None,
    image_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,

    # model
    name: Optional[str] = None,
    weights_path: Optional[Path] = None,

    # io & viz
    eval_out: Optional[Path] = None,
    top_per_class: Optional[int] = None,
    no_galleries: Optional[bool] = None,
    no_gradcam: Optional[bool] = None,

    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,

    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values: key=val (e.g., io.eval_out=outputs/evaluation). Repeatable."),
    dry_run: bool = False,
):
    """
    Evaluate a trained model on a test set. Can generate galleries and Grad-CAMs.

    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import evaluate as evaluate_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # overrides only when provided
    if eval_in is not None:
        argv += ["--eval-in", str(eval_in)]
    if image_size is not None:
        argv += ["--image-size", str(image_size)]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    if num_workers is not None:
        argv += ["--num-workers", str(num_workers)]
    if seed is not None:
        argv += ["--seed", str(seed)]

    if name is not None:
        argv += ["--name", name]
    if weights_path is not None:
        argv += ["--weights-path", str(weights_path)]

    if eval_out is not None:
        argv += ["--eval-out", str(eval_out)]
    if top_per_class is not None:
        argv += ["--top-per-class", str(top_per_class)]
    if no_galleries is True:
        argv += ["--no-galleries"]
    elif no_galleries is False:
        argv += ["--galleries"]
    if no_gradcam is True:
        argv += ["--no-gradcam"]
    elif no_gradcam is False:
        argv += ["--gradcam"]

    log.info("cli.dispatch", extra={"stage": "evaluate", "argv": argv})
    code = int(evaluate_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# CLEANUP (quarantine based on validate report)
# ---------------------------
@app.command()
def cleanup(
    report: Optional[str] = None,             # e.g., "latest" or a specific path
    policy: Optional[str] = None,             # "strict" | "within_class" | "report_only"
    why: Optional[str] = None,                # "errors" | "warnings" | "both"
    report_tag: Optional[str] = None,
    # logging
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    # config-first
    config: Optional[Path] = typer.Option(None, help="Optional YAML config file (config-first)."),
    override: List[str] = typer.Option([], "--override", "-o",
        help="Override config values as key=val (e.g., policy=within_class why=both). Repeatable."),
    dry_run: bool = False,
):
    """
    Quarantine bad files based on a validate.py report (usually the latest pre/post).
    YAML-first. Only explicitly-set flags override YAML.
    """
    from src.pipeline import cleanup as cleanup_mod

    argv: List[str] = ["--log-level", log_level]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if config is not None:
        argv += ["--config", str(config)]
    for ov in override or []:
        argv += ["--override", ov]
    if dry_run:
        argv += ["--dry-run"]

    # overrides only when provided
    if report is not None:
        argv += ["--report", report]
    if policy is not None:
        argv += ["--policy", policy]
    if why is not None:
        argv += ["--why", why]
    if report_tag is not None:
        argv += ["--report-tag", report_tag]

    log.info("cli.dispatch", extra={"stage": "cleanup", "argv": argv})
    code = int(cleanup_mod.main(argv))
    raise typer.Exit(code)


# ---------------------------
# PIPELINE (master orchestrator)
# ---------------------------
@app.command()
def pipeline(
    # Master config + overrides
    config: Optional[Path] = typer.Option(
        None,
        help="Master YAML for the full pipeline (fetch → merge → validate (pre) → cleanup "
             "→ resize → validate (post) → split → train → evaluate).",
    ),
    override: List[str] = typer.Option(
        [], "--override", "-o",
        help="Override master config: dotted keys like train.data.image_size=256 (repeatable).",
    ),
    # Execution controls
    dry_run: bool = typer.Option(False, help="Plan only; do not run any stage."),
    skip: List[str] = typer.Option(
        [],
        help="Stages to skip entirely (ignored if --resume-from is set). "
             "Choices: fetch, merge, validate_pre, cleanup, resize, validate_post, split, train, evaluate",
    ),
    resume_from: Optional[str] = typer.Option(None, help="Start from this stage; earlier stages are skipped."),
):
    """
    Run the full pipeline via a single master YAML. Config-first.

    Examples
    --------
    - Dry run (plan only):
      python -m src.cli pipeline --config configs/pipeline.yaml --dry-run

    - Full run:
      python -m src.cli pipeline --config configs/pipeline.yaml

    - Resume from resize:
      python -m src.cli pipeline --config configs/pipeline.yaml --resume-from resize

    - Skip fetch/split:
      python -m src.cli pipeline --config configs/pipeline.yaml --skip fetch --skip split

    - Tweak specific keys:
      python -m src.cli pipeline --config configs/pipeline.yaml \
        -o train.loop.epochs=20 -o evaluate.io.top_per_class=8
    """
    from src.pipeline.orchestrator import run_pipeline

    argv_preview = {
        "config": str(config) if config else None,
        "override": override,
        "dry_run": bool(dry_run),
        "skip": skip,
        "resume_from": resume_from,
    }
    log.info("cli.dispatch", extra={"stage": "pipeline", "argv": argv_preview})

    code = run_pipeline(
        master_yaml=config,
        overrides=override,
        dry_run=dry_run,
        skip=skip,
        resume_from=resume_from,
    )
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
