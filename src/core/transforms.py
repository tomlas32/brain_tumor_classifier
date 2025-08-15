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



def build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    """
    Build torchvision transforms for train/val/test sets.

    Parameters
    ----------
    image_size : int
        Target square size after your offline resize/pad step. This value
        should match what was used during preprocessing (e.g., 224).

    Returns
    -------
    dict[str, torchvision.transforms.Compose]
        Keys: 'train', 'val', 'test'.

    Notes
    -----
    - We explicitly convert to 3 channels via `Grayscale(num_output_channels=3)`
      to keep ResNet happy (expects 3-channel input).
    - Augmentations are intentionally mild (medical imagery).
    - Normalization uses ImageNet stats to match pretrained backbones.
    """
    tf_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tf_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tfs = {"train": tf_train, "val": tf_eval, "test": tf_eval}

    # Structured, minimal log for traceability
    log.info("transforms.built", extra={
        "image_size": image_size,
        "train_aug": ["RandomRotation(15)", "RandomHorizontalFlip", "ColorJitter(0.1,0.1)"],
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
    })
    return tfs