"""
Transform builders for training/validation/testing.

This module centralizes torchvision preprocessing so that **train** and
**evaluate** use identical pipelines (and identical normalization stats).
Keeping transforms in one place prevents train/test drift.

Features
--------
- ImageNet mean/std constants (used for normalization).
- `build_transforms(image_size)` returning a dict with 'train' | 'val' | 'test'.
  * Train: grayscale→3ch, light aug (rotation/flip/jitter), tensor, normalize.
  * Val/Test: grayscale→3ch, tensor, normalize (no augmentation).
- Structured logging when transforms are constructed.

Usage
-----
from src.core.transforms import build_transforms

tfs = build_transforms(224)
train_tf = tfs["train"]; val_tf = tfs["val"]; test_tf = tfs["test"]
"""


from __future__ import annotations

from typing import Dict
from src.utils.logging_utils import get_logger
from torchvision import transforms

log = get_logger(__name__)

# Public constants so other modules (e.g., docs/notebooks) can import them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)



def build_transforms(
    image_size: int,
    *,
    rotate_deg: int = 15,
    hflip_prob: float = 0.5,
    jitter_brightness: float = 0.1,
    jitter_contrast: float = 0.1,
) -> Dict[str, transforms.Compose]:
    """
    Build torchvision transforms for train/val/test with configurable augmentation.

    Parameters
    ----------
    image_size : int
        Target square size after offline resize/pad (e.g., 224). Kept for
        future-proofing if you later add Resize/Crop here.
    rotate_deg : int
        Max absolute rotation for RandomRotation. 0 disables rotation.
    hflip_prob : float
        Probability for RandomHorizontalFlip. 0.0 disables flipping.
    jitter_brightness : float
        Brightness factor for ColorJitter. 0.0 disables brightness jitter.
    jitter_contrast : float
        Contrast factor for ColorJitter. 0.0 disables contrast jitter.

    Returns
    -------
    dict[str, torchvision.transforms.Compose]
        Keys: 'train', 'val', 'test'.

    Notes
    -----
    - Grayscale→3 channels to satisfy ImageNet-pretrained backbones.
    - Eval transforms intentionally exclude augmentation.
    """
    train_ops = [transforms.Grayscale(num_output_channels=3)]

    if rotate_deg and rotate_deg != 0:
        train_ops.append(transforms.RandomRotation(int(rotate_deg)))
    if hflip_prob and hflip_prob > 0.0:
        train_ops.append(transforms.RandomHorizontalFlip(p=float(hflip_prob)))
    if (jitter_brightness and jitter_brightness > 0.0) or (jitter_contrast and jitter_contrast > 0.0):
        train_ops.append(
            transforms.ColorJitter(
                brightness=float(jitter_brightness or 0.0),
                contrast=float(jitter_contrast or 0.0),
            )
        )

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tf_train = transforms.Compose(train_ops)
    tf_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tfs = {"train": tf_train, "val": tf_eval, "test": tf_eval}

    log.info("transforms.built", extra={
        "image_size": image_size,
        "aug": {
            "rotate_deg": int(rotate_deg),
            "hflip_prob": float(hflip_prob),
            "jitter_brightness": float(jitter_brightness),
            "jitter_contrast": float(jitter_contrast),
        },
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
    })
    return tfs