"""
Training runner: orchestrates data, model, loop, and artifacts.

This module wraps the end-to-end training flow for classification:
- Builds transforms and datasets with strict class-mapping verification.
- Creates deterministic DataLoaders.
- Trains with AMP, StepLR, and AdamW while tracking best macro-F1.
- Saves the best checkpoint and a standardized training summary.

Dependencies
------------
- src.core.transforms.build_transforms
- src.core.data.make_loader
- src.core.metrics.evaluate
- src.core.model.build_model, src.core.model.get_device
- src.core.mapping (strict class verification)
- src.core.artifacts.write_training_summary
- src.core.env.get_env_info (for manifest 'env' field)

Logging
-------
Emits structured logs:
- 'train.start', 'train.epoch_end', 'train.best_updated', 'checkpoint.saved',
- 'training.summary_written' (via artifacts module).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.utils.logging_utils import get_logger
from src.core.transforms import build_transforms
from src.core.data import make_loader
from src.core.metrics import evaluate
from src.core.model import build_model, get_device
from src.core.mapping import (
    read_index_remap,
    expected_classes_from_remap,
    verify_dataset_classes,
    default_index_remap_path,
    copy_index_remap,
)
from src.core.artifacts import write_training_summary
from src.core.env import get_env_info

log = get_logger(__name__)


@dataclass(frozen=True)
class TrainRunnerInputs:
    """
    Typed inputs for the training runner.

    Fields
    ------
    image_size : int
        Target square size after offline resize/pad (e.g., 224).
    train_in : Path
        Root directory with class subfolders for training/validation split.
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        Number of worker processes for DataLoaders.
    val_frac : float
        Fraction of the dataset reserved for validation (stratified).
    seed : int
        RNG seed for deterministic split/loader behavior.
    model_name : {'resnet18','resnet34','resnet50'}
        Backbone architecture name.
    pretrained : bool
        Whether to use ImageNet weights for the backbone.
    epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    step_size : int
        StepLR step size (in epochs).
    gamma : float
        StepLR gamma.
    amp : bool
        Enable automatic mixed precision if CUDA is available.
    out_models : Path
        Directory to save the best checkpoint.
    out_summary : Path
        Directory to write the training summary JSON.
    index_remap : Path | None
        Optional explicit path to index_remap.json; default is outputs/mappings/latest.json.
    run_id : str | None
        Optional run identifier used in logs and manifest filenames.
    args_dict : Dict[str, str | int | float | bool]
        A CLI/config snapshot for traceability in the summary.
    """
    image_size: int
    train_in: Path
    batch_size: int
    num_workers: int
    val_frac: float
    seed: int
    model_name: str
    pretrained: bool
    epochs: int
    lr: float
    weight_decay: float
    step_size: int
    gamma: float
    amp: bool
    out_models: Path
    out_summary: Path
    index_remap: Path | None
    run_id: str | None
    args_dict: Dict[str, object]


def _stratified_split(full_ds: ImageFolder, val_frac: float, seed: int) -> Tuple[Subset, Subset]:
    """
    Create stratified train/val subsets from an ImageFolder.

    Returns
    -------
    (train_subset, val_subset)
    """
    y = np.array([label for _, label in full_ds.samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))

    train_subset = Subset(full_ds, train_idx)
    # Rebuild a separate dataset for val to apply eval transforms later
    return train_subset, Subset(full_ds, val_idx)


def run(inputs: TrainRunnerInputs) -> tuple[float, int, Path]:
    """
    Execute the full training flow and return headline results.

    Returns
    -------
    (best_f1, best_epoch, checkpoint_path, class_names, class_to_idx, per_class_counts)
    """
    log.info("train.start", extra={"run_id": inputs.run_id, "args": inputs.args_dict})

    # Device
    device = get_device(prefer_cuda=True)

    # Transforms
    aug_cfg = inputs.args_dict.get("aug", {})  # dict from config
    tfs = build_transforms(
        inputs.image_size,
        rotate_deg=aug_cfg.get("rotate_deg", 15),
        hflip_prob=aug_cfg.get("hflip_prob", 0.5),
        jitter_brightness=aug_cfg.get("jitter_brightness", 0.1),
        jitter_contrast=aug_cfg.get("jitter_contrast", 0.1),
    )

    # Mapping (strict for training)
    mapping_path = inputs.index_remap or default_index_remap_path()
    idx_to_class = read_index_remap(mapping_path)
    expected_classes = expected_classes_from_remap(idx_to_class)

    # Load full training set (train transforms for label reading)
    full_train = ImageFolder(inputs.train_in, transform=tfs["train"])
    verify_dataset_classes(full_train.classes, expected_classes, strict=True)

    # Per-class counts
    per_class_counts: Dict[str, int] = {cls: 0 for cls in expected_classes}
    for _, y in full_train.samples:
        per_class_counts[expected_classes[y]] += 1
    empties = [c for c, n in per_class_counts.items() if n == 0]
    if empties:
        log.error("class.empty_detected", extra={"empty_classes": empties, "counts": per_class_counts})
        raise RuntimeError(f"Empty classes found in training data: {empties}")

    # Stratified split and rewrap val with eval transforms
    train_subset, val_subset_idx = _stratified_split(full_train, inputs.val_frac, inputs.seed)
    val_full = ImageFolder(inputs.train_in, transform=tfs["val"])
    val_subset = Subset(val_full, val_subset_idx.indices)  # reuse same indices

    # Loaders
    train_loader = make_loader(train_subset, inputs.batch_size, shuffle=True,  num_workers=inputs.num_workers, seed=inputs.seed)
    val_loader   = make_loader(val_subset,   inputs.batch_size, shuffle=False, num_workers=inputs.num_workers, seed=inputs.seed)

    # Model
    num_classes = len(expected_classes)
    model = build_model(inputs.model_name, num_classes=num_classes, pretrained=inputs.pretrained).to(device)

    # Optim/criterion/scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=inputs.lr,
        weight_decay=inputs.weight_decay,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=inputs.step_size, gamma=inputs.gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=inputs.amp and torch.cuda.is_available())

    # Train loop (best by macro-F1)
    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = -math.inf
    best_epoch = -1

    for epoch in range(1, inputs.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=inputs.amp and torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * xb.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, device)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_wts = copy.deepcopy(model.state_dict())
            log.info("train.best_updated", extra={"epoch": best_epoch, "best_f1": round(best_f1, 6)})

        dt = time.time() - t0
        log.info("train.epoch_end", extra={
            "epoch": epoch,
            "time_sec": round(dt, 2),
            "train_loss": round(train_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_prec": round(val_prec, 6),
            "val_rec": round(val_rec, 6),
            "val_f1": round(val_f1, 6),
            "best_f1": round(best_f1, 6),
            "best_epoch": int(best_epoch),
        })

    # Save best
    inputs.out_models.mkdir(parents=True, exist_ok=True)
    ckpt_path = inputs.out_models / f"best_valF1_{best_f1:.4f}_epoch{best_epoch}.pth"
    torch.save(best_wts, ckpt_path)
    log.info("checkpoint.saved", extra={"path": str(ckpt_path)})

    # Finalize model state
    model.load_state_dict(best_wts)

    # Copy mapping next to checkpoint for airtight traceability
    try:
        copy_index_remap(mapping_path, ckpt_path.parent)
    except Exception as e:
        log.warning("mapping.copy_failed", extra={"error": str(e)})

    # Summary manifest
    write_training_summary(
        out_dir=inputs.out_summary,
        run_id=inputs.run_id,
        args_dict={k: (str(v) if isinstance(v, Path) else v) for k, v in inputs.args_dict.items()},
        class_names=expected_classes,
        class_to_idx=full_train.class_to_idx,
        per_class_counts=per_class_counts,
        best_f1=best_f1,
        best_epoch=best_epoch,
        checkpoint_path=ckpt_path,
        env_info=get_env_info().to_dict(),
    )

    return best_f1, best_epoch, ckpt_path
