"""
Train a CNN on resized MRI images (brain tumor classifier).

This script:
1) Loads a resized, class-structured training dataset (from `resize.py`).
2) Verifies class mapping against `index_remap.json` written by `split.py`.
3) Creates a stratified train/val split with deterministic seeding.
4) Trains a ResNet (18/34/50) with AMP, StepLR, and AdamW.
5) Saves only the single best checkpoint (by validation F1) and a summary JSON.

Key design choices
------------------
- Strict mapping verification: training *must* match `index_remap.json`.
- Run-aware logging: every log line includes [stage|run_id] for Docker/CI stitching.
- Single source of truth: mapping read/verify/copy comes from `src.core.mapping`.

Typical pipeline order
----------------------
fetch → split → resize → validate → **train** → evaluate

Examples
--------
# minimal (defaults)
python -m src.pipeline.train

# custom epochs / lr and output dirs
python -m src.pipeline.train --epochs 20 --lr 3e-4 --out-models models/brain_tumor

# explicit resized roots
python -m src.pipeline.train --train-in data/training_resized

# explicit mapping file (otherwise uses outputs/mappings/latest.json)
python -m src.pipeline.train --index-remap outputs/mappings/latest.json
"""



from __future__ import annotations

from pathlib import Path
import argparse, time, json, copy, math, random, os
from datetime import datetime, timezone
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision import transforms

from src.utils.logging_utils import configure_logging, get_logger
from src.core.transforms import build_transforms
from src.core.env import (
    bootstrap_env, 
    log_env_once, 
    get_env_info
)
from src.core.mapping import (
    read_index_remap,
    expected_classes_from_remap,
    verify_dataset_classes,
    default_index_remap_path,
    copy_index_remap,
)
from src.utils.parser_utils import (
    add_common_logging_args,
    add_common_dataset_args,
    add_common_train_args,
    add_model_args,
)
from src.pipeline.evaluate import evaluate


log = get_logger(__name__)


def worker_init_fn(worker_id: int):
    """
    Deterministic DataLoader workers (seed NumPy & random per-worker).
    """
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

def write_training_summary(
    out_summary_dir: Path,
    args_namespace,
    class_names: list[str],
    class_to_idx: dict[str, int],
    per_class_counts: dict[str, int],
    best_f1: float,
    best_epoch: int,
    ckpt_path: Path,
    mapping_src: Path,
    *,
    run_id: str | None = None,
):
    """
    Write a self-contained JSON summary of the training run and copy `index_remap.json`
    next to the checkpoint for airtight traceability.

    Parameters
    ----------
    out_summary_dir : Path
        Destination directory for the summary JSON.
    args_namespace : argparse.Namespace
        CLI args (will be serialized).
    class_names : list[str]
        Ordered class names used by the model.
    class_to_idx : dict[str,int]
        Mapping from class name to label index.
    per_class_counts : dict[str,int]
        Counts of training samples per class (pre split into train/val).
    best_f1 : float
        Best validation macro F1 achieved.
    best_epoch : int
        Epoch where best F1 occurred.
    ckpt_path : Path
        Path to the saved best checkpoint (.pth).
    mapping_src : Path
        Path to the index_remap used for this run.
    run_id : str | None
        Optional run identifier to include in the manifest.
    """
    out_summary_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    fname = f"training_summary_{(run_id or 'no-runid')}_{ts}.json"
    summary_path = out_summary_dir / fname

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args_namespace).items()},
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "class_distribution": per_class_counts,
        "results": {
            "best_f1": round(float(best_f1), 6),
            "best_epoch": int(best_epoch),
        },
        "artifacts": {
            "checkpoint": str(ckpt_path.resolve()),
        },
        "env": get_env_info().to_dict(),
        "seed": args_namespace.seed,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    try:
        latest_path = out_summary_dir / "training_summary_latest.json"
        latest_path.write_text(Path(summary_path).read_text(encoding="utf-8"), encoding="utf-8")
        log.info("training_summary_latest_updated", extra={"path": str(latest_path)})
    except Exception as e:
        log.warning("training_summary_latest_update_failed", extra={"error": str(e)})

    # Copy mapping next to the checkpoint (shared helper)
    try:
        if mapping_src and Path(mapping_src).exists():
            copy_index_remap(mapping_src, ckpt_path.parent)
            log.info("mapping_copied", extra={
                "from": str(mapping_src),
                "to": str(ckpt_path.parent / "index_remap.json")
            })
    except Exception as e:
        log.warning("mapping_copy_failed", extra={"error": str(e)})

    log.info("training_summary_written", extra={"path": str(summary_path)})

def make_parser_train() -> argparse.ArgumentParser:
    """
    Create CLI parser for the training step.

    Examples
    --------
    # minimal (uses defaults)
    python -m src.pipeline.train

    # custom epochs / lr and save models to a custom dir
    python -m src.pipeline.train --epochs 20 --lr 3e-4 --out-models models/brain_tumor

    # point to specific resized folders
    python -m src.pipeline.train --train-in data/training_resized --test-in data/testing_resized
    """
    parser = argparse.ArgumentParser(description="Train a CNN on resized MRI images.")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square size after resize/pad (must match your preprocessing)")
    add_common_dataset_args(parser)   # roots, batch size, workers, val_frac, seed, out dirs, etc.
    add_common_train_args(parser)     # epochs, lr, weight_decay, scheduler, amp, etc.
    add_model_args(parser)            # model name + pretrained flag
    add_common_logging_args(parser)   # log level/file
    parser.add_argument("--index-remap", type=Path, default=None,
                        help="Path to index_remap.json (defaults to outputs/mappings/latest.json)")
    return parser


def make_stratified_subsets(
    training_dir: Path,
    tf_train,
    tf_val,
    val_frac: float,
    seed: int,
    index_remap_path: Path | None,
) -> tuple[Subset, Subset, list[str], dict[str, int], dict[str, int]]:
    """
    Create stratified train/val subsets from a class-structured image root,
    verifying that the class mapping matches `index_remap.json` and logging per-class counts.

    Args
    ----
    training_dir : Path
        Path to the resized training directory.
    tf_train : transforms.Compose
        Transform pipeline for training set.
    tf_val : transforms.Compose
        Transform pipeline for validation set.
    val_frac : float
        Fraction of total data to reserve for validation (per class).
    seed : int
        RNG seed for reproducibility.
    index_remap_path : Path | None
        Path to `index_remap.json` (if None, uses `outputs/mappings/latest.json`).

    Returns
    -------
    (train_data, val_data, class_names, class_to_idx, per_class_counts)
    """
    # Resolve mapping path and read/validate via shared core
    mapping_path = index_remap_path or default_index_remap_path()
    log.info("class_mapping.read", extra={"index_remap_path": str(mapping_path)})

    idx_to_class = read_index_remap(mapping_path)                                  # shared reader
    expected_classes = expected_classes_from_remap(idx_to_class)                   # ordered classes

    # Load full training set using the *train* transforms to compute labels
    full_train = ImageFolder(training_dir, transform=tf_train)

    # Strict verification for training
    verify_dataset_classes(full_train.classes, expected_classes, strict=True)      # raises on mismatch

    # Counts per class
    per_class_counts = {cls: 0 for cls in expected_classes}
    for _, y in full_train.samples:
        per_class_counts[expected_classes[y]] += 1

    empties = [c for c, n in per_class_counts.items() if n == 0]
    if empties:
        log.error("empty_classes_detected", extra={"empty_classes": empties, "counts": per_class_counts})
        raise RuntimeError(f"Empty classes found in training data: {empties}")

    log.info("class_mapping_verified", extra={"classes": expected_classes})
    log.info("class_distribution", extra={"per_class_counts": per_class_counts})

    # Stratified split
    y = np.array([label for _, label in full_train.samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))

    train_data = Subset(full_train, train_idx)
    # Re-create for val to apply val transforms
    val_full = ImageFolder(training_dir, transform=tf_val)
    val_data = Subset(val_full, val_idx)

    return train_data, val_data, full_train.classes, full_train.class_to_idx, per_class_counts


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    """
    Create a DataLoader with consistent, deterministic settings.

    Returns
    -------
    DataLoader
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    log.info("dataloader_created", extra={
        "num_samples": len(dataset),
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers
    })
    return loader


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet model with the final layer replaced for the given number of classes.

    Args
    ----
    model_name : {'resnet18','resnet34','resnet50'}
    num_classes : int
    pretrained : bool

    Returns
    -------
    nn.Module
        A PyTorch model ready for training.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    log.info("model_built", extra={
        "model_name": model_name,
        "pretrained": pretrained,
        "num_classes": num_classes
    })
    return model



def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
    amp: bool,
    out_dir: Path,
):
    """
    Full training loop for classification models. Saves only the single best
    weights file at the end of training (by validation F1).

    Returns
    -------
    (best_f1, best_epoch, ckpt_path)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and torch.cuda.is_available())

    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = -math.inf
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation ----
        val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, device)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_wts = copy.deepcopy(model.state_dict())
            log.info("best_updated", extra={"epoch": best_epoch, "best_f1": round(best_f1, 4)})

        dt = time.time() - t0
        log.info("epoch_end", extra={
            "epoch": epoch,
            "time_sec": round(dt, 1),
            "train_loss": round(train_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_prec": round(val_prec, 4),
            "val_rec": round(val_rec, 4),
            "val_f1": round(val_f1, 4),
            "best_f1": round(best_f1, 4),
            "best_epoch": best_epoch
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_valF1_{best_f1:.4f}_epoch{best_epoch}.pth"
    torch.save(best_wts, ckpt_path)
    log.info("checkpoint_saved", extra={"path": str(ckpt_path)})

    model.load_state_dict(best_wts)
    return best_f1, best_epoch, ckpt_path


def main(argv=None) -> int:
    """
    CLI entry point:
    - Parses args and configures logging (with RUN_ID + stage='train').
    - Verifies mapping, builds loaders, model, and trains.
    - Writes a summary and copies mapping next to the checkpoint.
    """
    parser = make_parser_train()
    args = parser.parse_args(argv)

    # Run-aware logging (ties logs across stages)
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="train")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="train")

    # Device & env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device_selected", extra={"device": str(device)})

    bootstrap_env(seed=args.seed)
    log_env_once()

    # Transforms
    tfs = build_transforms(args.image_size)

    # Mapping path (default to outputs/mappings/latest.json)
    mapping_path = args.index_remap or default_index_remap_path()

    # Train/Val subsets (strict mapping verification)
    train_data, val_data, class_names, class_to_idx, per_class_counts = make_stratified_subsets(
        training_dir=args.train_in,
        tf_train=tfs["train"],
        tf_val=tfs["val"],
        val_frac=args.val_frac,
        seed=args.seed,
        index_remap_path=mapping_path,
    )

    log.info("class_mapping", extra={
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names),
        "class_names": class_names
    })
    log.info("class_distribution", extra={"per_class_counts": per_class_counts})

    # Loaders
    train_loader = make_loader(train_data, args.batch_size, shuffle=True,  num_workers=args.num_workers, seed=args.seed)
    val_loader   = make_loader(val_data,   args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    # Model
    num_classes = len(class_names)
    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained).to(device)

    # Train
    args.out_models.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, ckpt_path = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        amp=args.amp,
        out_dir=args.out_models,
    )

    log.info("training_complete", extra={
        "best_f1": round(best_f1, 4),
        "best_epoch": best_epoch,
        "weights_dir": str(args.out_models)
    })
    print(f"✅ Training complete. Best F1={best_f1:.4f} at epoch {best_epoch}")

    write_training_summary(
        out_summary_dir=args.out_summary,
        args_namespace=args,
        class_names=class_names,
        class_to_idx=class_to_idx,
        per_class_counts=per_class_counts,
        best_f1=best_f1,
        best_epoch=best_epoch,
        ckpt_path=ckpt_path,
        mapping_src=mapping_path,
        run_id=run_id,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())