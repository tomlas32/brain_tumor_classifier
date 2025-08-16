# src/core/viz.py
"""
Visualization utilities for evaluation artifacts.

Provides:
- show_calls_gallery: Render a tiled gallery of predictions (e.g., misclassifications / top-correct).
- show_gradcam_gallery: Render Grad-CAM overlays for selected predictions.

Both functions emit structured logs and save PNG artifacts to the given directory.

Author: Tomasz Lasota
Date: 2025-08-16
Version: 1.1
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import io
import math
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Try to import Grad-CAM backend (optional dependency)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _HAS_CAM = True
except Exception:
    _HAS_CAM = False


# --------------------------- Helpers -----------------------------------------


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _open_rgb(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        log.warning("viz.image_read_error", extra={"path": path, "error": str(e)})
        return None


def _pil_to_numpy_float01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def _make_title(item: Dict, class_names: List[str]) -> str:
    """Compose a concise tile title from a prediction record."""
    t = item.get("true"); p = item.get("pred"); c = item.get("conf")
    t_name = class_names[t] if isinstance(t, int) and 0 <= t < len(class_names) else str(t)
    p_name = class_names[p] if isinstance(p, int) and 0 <= p < len(class_names) else str(p)
    if c is None:
        return f"T:{t_name}  P:{p_name}"
    return f"T:{t_name}  P:{p_name}  conf:{c:.2f}"


def _default_transform_for_display(image_size: int = 224) -> T.Compose:
    """Fallback minimal eval transform for display (resize+center-crop → tensor in [0,1])."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),  # [0,1]
    ])


def _select_target_layer(model) -> torch.nn.Module:
    """
    Heuristic to pick a good last conv layer for ResNet-style models.
    If unavailable, returns None and Grad-CAM will skip.
    """
    # Common for torchvision ResNets: model.layer4[-1]
    try:
        layer4 = getattr(model, "layer4", None)
        if layer4 and hasattr(layer4, "__getitem__"):
            return layer4[-1]
        return None
    except Exception:
        return None


# --------------------------- Public API --------------------------------------


def show_calls_gallery(
    items: List[Dict],
    class_names: List[str],
    *,
    cols: int = 6,
    title: str = "gallery",
    save_dir: Path,
    image_size: int = 224,
) -> Optional[Path]:
    """
    Render a grid of images from `items` (each item has 'path', 'true', 'pred', 'conf').

    Parameters
    ----------
    items : list of dict
        Each dict should include: {'path': str, 'true': int, 'pred': int, 'conf': float}
    class_names : list[str]
        Class names indexed by label id.
    cols : int
        Number of columns in the grid.
    title : str
        Basename for the saved figure (PNG); also drawn atop the figure.
    save_dir : Path
        Directory to save the figure.
    image_size : int
        Optional resize for uniform preview.

    Returns
    -------
    saved_path : Path | None
        Path to the saved PNG, or None if nothing was rendered.
    """
    if not items:
        log.info("viz.warning.empty_gallery", extra={"title": title})
        return None

    _ensure_dir(save_dir)
    rows = math.ceil(len(items) / cols)
    fig = plt.figure(figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(title, fontsize=14, y=0.98)

    for i, item in enumerate(items):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = _open_rgb(item.get("path"))
        if img is None:
            ax.axis("off")
            continue
        if image_size:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        ax.imshow(img)
        ax.set_title(_make_title(item, class_names), fontsize=9)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = save_dir / f"{title}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    log.info("gallery.saved", extra={"path": str(out_path), "count": len(items), "title": title})
    return out_path


def show_gradcam_gallery(
    predictions: List[Dict],
    class_names: List[str],
    *,
    model: torch.nn.Module,
    device: torch.device,
    ds_for_transform,  # e.g., an ImageFolder from which to steal the eval transform
    cols: int = 4,
    title: str = "gradcam",
    target_layer: Optional[torch.nn.Module] = None,
    save_dir: Path = Path("."),
    image_size: int = 224,
) -> Optional[Path]:
    """
    Render Grad-CAM overlays as a grid for selected predictions.

    Parameters
    ----------
    predictions : list of dict
        Each dict should include: {'path': str, 'true': int, 'pred': int, 'conf': float}
    class_names : list[str]
        Class names indexed by label id.
    model : nn.Module
        Trained classification model.
    device : torch.device
        Inference device.
    ds_for_transform : Dataset
        Any dataset that defines the evaluation transform; if missing, a default
        display transform is used internally.
    cols : int
        Number of columns in the grid.
    title : str
        Basename for the saved figure (PNG); also drawn atop the figure.
    target_layer : nn.Module | None
        Specific conv layer for CAM; if None, will try to auto-select.
    save_dir : Path
        Directory to save the figure.
    image_size : int
        Preprocess/preview size.

    Returns
    -------
    saved_path : Path | None
        Path to the saved PNG, or None if no backend is available or inputs empty.
    """
    if not predictions:
        log.info("viz.warning.empty_gallery", extra={"title": title})
        return None

    if not _HAS_CAM:
        log.warning("viz.warning.no_gradcam_backend", extra={"title": title, "hint": "pip install pytorch-grad-cam"})
        return None

    _ensure_dir(save_dir)
    model.eval()

    # Reuse eval transform if possible; otherwise fallback
    eval_tf = getattr(ds_for_transform, "transform", None)
    if eval_tf is None:
        eval_tf = _default_transform_for_display(image_size)

    # Pick target layer if not provided
    tlayer = target_layer or _select_target_layer(model)
    if tlayer is None:
        log.warning("viz.warning.no_target_layer", extra={"title": title, "note": "Could not infer a conv layer"})
        return None

    cam = GradCAM(model=model, target_layers=[tlayer], use_cuda=(device.type == "cuda"))

    cols = max(1, int(cols))
    rows = math.ceil(len(predictions) / cols)
    fig = plt.figure(figsize=(cols * 3.0, rows * 3.0))
    fig.suptitle(title, fontsize=14, y=0.98)

    rendered = 0
    for i, item in enumerate(predictions):
        path = item.get("path")
        pil = _open_rgb(path)
        ax = fig.add_subplot(rows, cols, i + 1)
        if pil is None:
            ax.axis("off")
            continue

        # For CAM overlay, we need both the original RGB (float[0,1]) and the normalized tensor
        rgb_float = _pil_to_numpy_float01(pil.resize((image_size, image_size), Image.BILINEAR))

        # Run transform pipeline → 1xCxHxW tensor
        x = eval_tf(pil).unsqueeze(0).to(device)
        with torch.inference_mode():
            targets = None  # default: strongest class in forward pass
            grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # HxW

        # Create overlay
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # uint8
        overlay_img = Image.fromarray(overlay)

        ax.imshow(overlay_img)
        ax.set_title(_make_title(item, class_names), fontsize=9)
        ax.axis("off")
        rendered += 1

    if rendered == 0:
        plt.close(fig)
        log.info("viz.warning.empty_gallery", extra={"title": title})
        return None

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = save_dir / f"{title}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    log.info("gradcam.saved", extra={"path": str(out_path), "count": int(rendered), "title": title})
    return out_path
