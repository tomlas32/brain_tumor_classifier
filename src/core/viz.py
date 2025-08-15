"""
Visualization helpers: galleries and Grad-CAM overlays.

This module centralizes non-essential (but highly useful) visual outputs used
during model evaluation so they can be re-used and tested independently.

Contents
--------
- show_calls_gallery: render a tiled gallery of images (e.g., top-correct /
  misclassifications) with compact titles.
- show_gradcam_gallery: render Grad-CAM overlays for a selection of images.
- _pick_last_conv_layer: heuristic to choose a last conv layer for Grad-CAM.

Dependencies
------------
- matplotlib
- pillow
- pytorch_grad_cam (for Grad-CAM only)

Logging
-------
- 'gallery.saved' with {'path', 'n_items'}
- 'gradcam.saved' with {'path', 'n_items'}
- Warnings on I/O or CAM failures.

Usage
-----
from src.core.viz import show_calls_gallery, show_gradcam_gallery

show_calls_gallery(items, class_names, cols=6, title="misclassifications", save_dir=Path(...))
show_gradcam_gallery(items, class_names, model, device, ds_for_transform, cols=4,
                     title="misclass_gradcam", target_layer=None, save_dir=Path(...))
"""


from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ----- Small utilities --------------------------------------------------------

def _slugify(text: str) -> str:
    """Lowercase and replace non-alphanumeric with dashes."""
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


# ----- Galleries (plain images) -----------------------------------------------

def show_calls_gallery(
    predictions: List[dict],
    class_names: List[str],
    cols: int,
    title: str,
    save_dir: str | Path | None,
) -> None:
    """
    Render a tiled gallery for a list of prediction dicts.

    Parameters
    ----------
    predictions : list[dict]
        Each dict must contain: {'true': int, 'pred': int, 'conf': float, 'path': str}
    class_names : list[str]
        Ordered class names used for labels in titles.
    cols : int
        Number of columns in the grid.
    title : str
        Figure title and output filename prefix.
    save_dir : str | Path | None
        Directory to save the PNG; if None, only renders (still closed).

    Produces
    --------
    <save_dir>/<slug(title)>.png
    """
    n = len(predictions)
    rows = max(1, int(np.ceil(n / cols)))
    plt.figure(figsize=(3.0 * cols, 3.3 * rows))

    for i, item in enumerate(predictions, start=1):
        ax = plt.subplot(rows, cols, i)
        try:
            img = ImageOps.exif_transpose(Image.open(item["path"])).convert("L")
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"{class_names[item['true']]} → {class_names[item['pred']]}\nconf={item['conf']:.2f}",
                fontsize=9
            )
            ax.axis("off")
        except Exception as e:
            ax.axis("off")
            ax.set_title("I/O error", fontsize=9)
            log.warning("gallery.io_error", extra={"path": item.get("path"), "error": str(e)})

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{_slugify(title)}.png"
        try:
            plt.savefig(out_path, dpi=180, bbox_inches="tight")
            log.info("gallery.saved", extra={"path": str(out_path), "n_items": n})
        except Exception as e:
            log.warning("gallery.save_failed", extra={"path": str(out_path), "error": str(e)})
    plt.close()


# ----- Grad-CAM ---------------------------------------------------------------

def _pick_last_conv_layer(model):
    """
    Heuristic to select a convolutional layer for Grad-CAM in ResNet-like models.

    Returns
    -------
    torch.nn.Module
        The last encountered nn.Conv2d module.

    Raises
    ------
    RuntimeError if none is found.
    """
    import torch.nn as nn
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")


def show_gradcam_gallery(
    predictions: List[dict],
    class_names: List[str],
    model,
    device,
    ds_for_transform,  # e.g., an ImageFolder with the desired .transform
    cols: int,
    title: str,
    target_layer=None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Render Grad-CAM overlays for selected predictions.

    Parameters
    ----------
    predictions : list[dict]
        Each dict: {'true': int, 'pred': int, 'conf': float, 'path': str}
    class_names : list[str]
        Ordered class names for captioning.
    model : torch.nn.Module
        Model (should already be on device and set to eval by caller).
    device : torch.device
        CUDA or CPU device.
    ds_for_transform :
        Dataset providing the `.transform` used for evaluation (ensures
        the same normalization and resizing as evaluation).
    cols : int
        Grid columns.
    title : str
        Title and filename stem.
    target_layer :
        Optional layer to target; if None, a reasonable conv layer is chosen.
    save_dir : Path | None
        Directory to write the PNG.

    Produces
    --------
    <save_dir>/<slug(title)>.png

    Notes
    -----
    - Requires `pytorch_grad_cam` package.
    - Uses predicted class (item['pred']) as the target for the CAM.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layer = target_layer or _pick_last_conv_layer(model)

    n = len(predictions)
    rows = max(1, int(np.ceil(n / cols)))
    plt.figure(figsize=(3.5 * cols, 3.8 * rows))

    try:
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == "cuda"))
    except Exception as e:
        log.warning("gradcam.init_failed", extra={"error": str(e)})
        plt.close()
        return

    for i, item in enumerate(predictions, start=1):
        ax = plt.subplot(rows, cols, i)
        try:
            from PIL import Image
            img = ImageOps.exif_transpose(Image.open(item["path"])).convert("RGB")
            img_arr = np.array(img).astype(np.float32) / 255.0

            # Reuse exact eval transform (grayscale→3ch, tensor, normalize)
            x = ds_for_transform.transform(img).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(item["pred"])])[0]
            overlay = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)

            ax.imshow(overlay)
            ax.set_title(
                f"{class_names[item['true']]} → {class_names[item['pred']]}\nconf={item['conf']:.2f}",
                fontsize=9
            )
            ax.axis("off")
        except Exception as e:
            ax.axis("off")
            ax.set_title("Grad-CAM error", fontsize=9)
            log.warning("gradcam.item_failed", extra={"path": item.get("path"), "error": str(e)})

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{_slugify(title)}.png"
        try:
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            log.info("gradcam.saved", extra={"path": str(out_path), "n_items": n})
        except Exception as e:
            log.warning("gradcam.save_failed", extra={"path": str(out_path), "error": str(e)})
    plt.close()