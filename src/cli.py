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
    Re-split pooled images into DATA_DIR/training and DATA_DIR/testing
    using the same flags as split.py.
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