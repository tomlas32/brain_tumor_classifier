"""
Shared evaluation metrics and reporting utilities.

This module centralizes:
- Metric computation (accuracy, macro precision/recall/F1) for a model + DataLoader.
- Saving a scikit-learn classification report to disk.
- Saving confusion matrices (counts and row-normalized) as figures.


Logging
-------
- 'metrics.evaluated' on successful metric computation.
- 'report.saved' when the classification report is written.
- 'confusion.saved' for each confusion matrix image.

Usage
-----
from src.core.metrics import evaluate, save_classification_report, save_confusions

acc, prec, rec, f1, y_true, y_pred = evaluate(model, loader, device)
save_classification_report(y_true, y_pred, class_names, out_path)
save_confusions(y_true, y_pred, class_names, out_dir, title="Test")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Compute accuracy, macro precision, macro recall, and macro F1 for a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode (this function will set eval() within a no_grad block).
    loader : torch.utils.data.DataLoader
        DataLoader yielding (inputs, labels).
    device : torch.device
        Device on which inference should run.

    Returns
    -------
    (acc, prec, rec, f1, y_true, y_pred)
        acc, prec, rec, f1 are floats; y_true and y_pred are 1D numpy arrays.

    Notes
    -----
    - Uses macro averaging for multi-class performance stability.
    - Zero-division is set to 0 to avoid exceptions on edge cases.
    """
    model.eval()
    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    log.info("metrics.evaluated", extra={
        "acc": round(acc, 6),
        "precision_macro": round(float(prec), 6),
        "recall_macro": round(float(rec), 6),
        "f1_macro": round(float(f1), 6),
        "samples": int(len(y_true)),
    })
    return acc, float(prec), float(rec), float(f1), y_true, y_pred


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: Path) -> None:
    """
    Save a scikit-learn classification report to a UTF-8 text file.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    class_names : list[str]
        Ordered class names.
    out_path : Path
        Destination path (e.g., outputs/evaluation/test_classification_report.txt).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cr_txt = classification_report(
            y_true, y_pred, target_names=class_names, digits=3, zero_division=0
        )
        out_path.write_text(cr_txt, encoding="utf-8")
        log.info("report.saved", extra={"path": str(out_path)})
    except Exception as e:
        log.warning("report.save_failed", extra={"path": str(out_path), "error": str(e)})


def save_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_dir: Path, *, title: str = "Test") -> None:
    """
    Save confusion matrices (counts and row-normalized) as PNGs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list[str]
        Ordered class names.
    out_dir : Path
        Directory to save figures.
    title : str
        Plot title prefix (e.g., 'Validation', 'Test').

    Produces
    --------
    - <title>_confusion_counts.png
    - <title>_confusion_row-normalized.png
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mats = [
        (None, "Counts", f"{_slugify(title)}_confusion_counts.png"),
        ("true", "Row-normalized (Recall)", f"{_slugify(title)}_confusion_row-normalized.png"),
    ]
    ticks = np.arange(len(class_names))

    for normalize, name, fname in mats:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title(f"{title} — Confusion Matrix ({name})")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)
        # annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
                plt.text(j, i, txt, ha="center", va="center")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        save_path = out_dir / fname
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("confusion.saved", extra={"normalize": normalize or "none", "path": str(save_path)})


def _slugify(text: str) -> str:
    """Lowercase and replace non-alphanumeric with dashes."""
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")
