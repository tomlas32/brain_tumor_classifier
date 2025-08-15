"""
Evaluation runner: orchestrates dataset build, mapping alignment, model loading,
metrics, reports, optional galleries/Grad-CAM, and a standardized summary.

This module centralizes the end-to-end evaluation flow so the CLI script
(`evaluate.py`) can stay thin and focused on argument parsing + logging.

Dependencies
------------
- src.core.transforms.build_transforms
- src.core.data.make_eval_loader
- src.core.mapping.align_or_warn_for_eval
- src.core.model.build_model, src.core.model.load_weights, src.core.model.get_device
- src.core.metrics.evaluate, save_classification_report, save_confusions
- src.core.viz.show_calls_gallery, show_gradcam_gallery
- src.core.artifacts.write_evaluation_summary

Logging
-------
Structured INFO/WARNING logs with concise keys:
- 'evaluate.start', 'device.selected', 'metrics.evaluated',
- 'report.saved', 'confusion.saved' (from core.metrics),
- 'gallery.saved', 'gradcam.saved' (from core.viz),
- 'artifact.write', 'artifact.latest_updated' (from core.artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision.datasets import ImageFolder

from src.utils.logging_utils import get_logger
from src.core.transforms import build_transforms
from src.core.data import make_eval_loader
from src.core.mapping import align_or_warn_for_eval
from src.core.model import build_model, load_weights, get_device
from src.core.metrics import evaluate as eval_metrics
from src.core.metrics import save_classification_report, save_confusions
from src.core.viz import show_calls_gallery, show_gradcam_gallery
from src.core.artifacts import write_evaluation_summary

log = get_logger(__name__)


@dataclass(frozen=True)
class EvalRunnerInputs:
    """
    Typed inputs for the evaluation runner.

    Fields
    ------
    image_size : int
        Target square size after offline resize/pad (e.g., 224).
    eval_in : Path
        Root directory of the resized test set (class-structured).
    mapping_path : Path
        Path to `index_remap.json` used to align class encoding.
    model_name : {'resnet18','resnet34','resnet50'}
        Backbone architecture name (must match training).
    weights_path : Path
        Path to trained weights (.pth).
    batch_size : int
        Batch size for the evaluation DataLoader.
    num_workers : int
        Number of worker processes for the DataLoader.
    seed : int
        Seed for deterministic DataLoader sampling.
    eval_out : Path
        Directory to write evaluation artifacts (reports, matrices, galleries).
    run_id : str | None
        Optional run identifier for filenames/manifests.
    args_dict : dict
        A snapshot of CLI/config args for summary reproducibility.
    make_galleries : bool
        Whether to render misclassification/top-correct galleries.
    make_gradcam : bool
        Whether to render Grad-CAM overlays (requires pytorch_grad_cam).
    top_per_class : int
        Number of items per class for galleries/Grad-CAM.
    """
    image_size: int
    eval_in: Path
    mapping_path: Path
    model_name: str
    weights_path: Path
    batch_size: int
    num_workers: int
    seed: int
    eval_out: Path
    run_id: str | None
    args_dict: Dict[str, object]
    make_galleries: bool = True
    make_gradcam: bool = True
    top_per_class: int = 6


def _get_paths_for_dataset(ds) -> List[str]:
    """
    Return absolute file paths for each sample in an ImageFolder (or Subset).
    """
    # ImageFolder guarantees .samples [(path,label), ...]
    return [p for p, _ in getattr(ds, "samples", [])]


@torch.inference_mode()
def _infer_collect(model: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and collect y_true, y_pred, and per-class probabilities.

    Returns
    -------
    (y_true, y_pred, probs)
    """
    model.eval()
    ys, yhat, prob = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        ys.append(yb.numpy())
        yhat.append(probs.argmax(1).cpu().numpy())
        prob.append(probs.cpu().numpy())
    return np.concatenate(ys), np.concatenate(yhat), np.concatenate(prob)


def _select_examples_per_true_class(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, paths: List[str], *, max_per_class: int, n_classes: int
) -> Tuple[list[dict], list[dict]]:
    """
    Build lists of top misclassifications and top correct predictions per true class.

    Returns
    -------
    (mistakes, corrects)
    """
    mistakes, corrects = [], []
    for c in range(n_classes):
        # mistakes: true=c but predicted!=c, ranked by predicted confidence
        m_mask = (y_true == c) & (y_pred != c)
        if m_mask.any():
            conf = probs[m_mask, y_pred[m_mask]]
            idxs = np.argsort(-conf)[:max_per_class]
            mis_idx = np.flatnonzero(m_mask)[idxs]
            mistakes.extend([{"true": int(y_true[i]), "pred": int(y_pred[i]), "conf": float(conf[j]), "path": paths[i]}
                             for j, i in enumerate(mis_idx)])

        # corrects: true=c and predicted=c, ranked by class c confidence
        c_mask = (y_true == c) & (y_pred == c)
        if c_mask.any():
            conf = probs[c_mask, c]
            idxs = np.argsort(-conf)[:max_per_class]
            cor_idx = np.flatnonzero(c_mask)[idxs]
            corrects.extend([{"true": c, "pred": c, "conf": float(conf[j]), "path": paths[i]}
                             for j, i in enumerate(cor_idx)])
    return mistakes, corrects


def run(inputs: EvalRunnerInputs) -> Tuple[float, float, float, float]:
    """
    Execute the full evaluation flow and return headline metrics.

    Returns
    -------
    (acc, precision_macro, recall_macro, f1_macro)
    """
    log.info("evaluate.start", extra={"run_id": inputs.run_id, "args": inputs.args_dict})

    # Device & transforms
    device = get_device(prefer_cuda=True)
    tfs = build_transforms(inputs.image_size)

    # Dataset & class alignment
    test_root = inputs.eval_in
    if not test_root.exists():
        raise FileNotFoundError(f"--eval-in not found: {test_root}")
    test_ds = ImageFolder(test_root, transform=tfs["test"])

    class_names = align_or_warn_for_eval(test_ds, inputs.mapping_path)
    log.info("class.info", extra={"num_classes": len(class_names), "class_names": class_names})

    # DataLoader
    test_loader = make_eval_loader(test_ds, inputs.batch_size, num_workers=inputs.num_workers, seed=inputs.seed)

    # Model & weights
    model = build_model(inputs.model_name, num_classes=len(class_names), pretrained=False)
    load_weights(model, inputs.weights_path, device, strict=True)

    # Metrics
    acc, prec, rec, f1, y_true, y_pred = eval_metrics(model, test_loader, device)
    print(f"TEST — Acc={acc:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # Reports & confusion matrices
    out_dir = Path(inputs.eval_out); out_dir.mkdir(parents=True, exist_ok=True)
    save_classification_report(y_true, y_pred, class_names, out_dir / "test_classification_report.txt")
    save_confusions(y_true, y_pred, class_names, out_dir, title="Test")

    # Optional galleries / Grad-CAM
    if inputs.make_galleries or inputs.make_gradcam:
        y_true2, y_pred2, probs = _infer_collect(model, test_loader, device)
        # sanity check alignment
        if not (np.array_equal(y_true, y_true2) and np.array_equal(y_pred, y_pred2)):
            log.warning("evaluate.sanity_mismatch", extra={"note": "evaluate() vs inference outputs differ"})

        paths = _get_paths_for_dataset(test_ds)
        mistakes, corrects = _select_examples_per_true_class(
            y_true2, y_pred2, probs, paths, max_per_class=inputs.top_per_class, n_classes=len(class_names)
        )

        if inputs.make_galleries:
            gallery_dir = out_dir / "galleries"
            show_calls_gallery(mistakes, class_names, cols=6, title="misclassifications", save_dir=gallery_dir)
            show_calls_gallery(corrects, class_names, cols=6, title="top-correct", save_dir=gallery_dir)

        if inputs.make_gradcam:
            gradcam_dir = out_dir / "gradcam"
            show_gradcam_gallery(
                predictions=mistakes, class_names=class_names, model=model, device=device,
                ds_for_transform=test_ds, cols=4, title="misclass_gradcam", target_layer=None, save_dir=gradcam_dir
            )
            show_gradcam_gallery(
                predictions=corrects, class_names=class_names, model=model, device=device,
                ds_for_transform=test_ds, cols=4, title="correct_class_gradcam", target_layer=None, save_dir=gradcam_dir
            )

    # Summary manifest (+ latest pointer)
    metrics_payload = {
        "acc": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }
    write_evaluation_summary(
        out_dir=out_dir,
        run_id=inputs.run_id,
        args_dict={k: (str(v) if isinstance(v, Path) else v) for k, v in inputs.args_dict.items()},
        class_names=class_names,
        metrics=metrics_payload,
    )

    return float(acc), float(prec), float(rec), float(f1)
