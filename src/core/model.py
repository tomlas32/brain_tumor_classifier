"""
Model factory and checkpoint utilities.

This module provides a single source of truth for:
- Building supported CNN backbones (currently ResNet 18/34/50).
- Loading checkpoints robustly (handles common 'module.' prefixes, nested dicts).
- Selecting the appropriate torch.device.

Logging
-------
All public functions emit structured logs (INFO/ERROR) with concise, queryable
keys, e.g., 'model.built', 'weights.loaded', 'device.selected'.
"""


from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from src.utils.logging_utils import get_logger

# torchvision imports are scoped to the functions to avoid import cost when unused
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

log = get_logger(__name__)

ArchName = Literal["resnet18", "resnet34", "resnet50"]

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Select a torch.device for computation.

    Parameters
    ----------
    prefer_cuda : bool
        If True (default), prefer CUDA when available; otherwise fall back to CPU.

    Returns
    -------
    torch.device
        Selected device.

    Logging
    -------
    Emits 'device.selected' with {'device': 'cuda'|'cpu'}.
    """
    device = torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")
    log.info("device.selected", extra={"device": str(device)})
    return device


def build_model(name: ArchName, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """
    Construct a ResNet backbone and replace the classification head.

    Parameters
    ----------
    name : {'resnet18','resnet34','resnet50'}
        Architecture identifier.
    num_classes : int
        Number of output classes for the final linear layer.
    pretrained : bool
        If True, load ImageNet weights for the backbone (recommended).

    Returns
    -------
    torch.nn.Module
        A model ready for training or inference.

    Notes
    -----
    - Uses torchvision's explicit Weights enums for clarity.
    - Final fully-connected layer is replaced with nn.Linear(in_features, num_classes).

    Logging
    -------
    Emits 'model.built' with {'arch': name, 'pretrained': bool, 'num_classes': int}.
    """
    lname = name.lower()
    if lname == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif lname == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    elif lname == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    log.info("model.built", extra={"arch": lname, "pretrained": pretrained, "num_classes": num_classes})
    return model


def load_weights(
    model: nn.Module,
    weights_path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> None:
    """
    Load a checkpoint into `model` robustly.

    Parameters
    ----------
    model : nn.Module
        Target model with matching architecture/head dimensions.
    weights_path : str | Path
        Path to a .pth or .pt file. Supports:
        - raw state_dict
        - dict with 'model_state_dict'
    device : torch.device
        Device to map tensors to during load.
    strict : bool
        Passed through to `load_state_dict`. Keep True unless you know why.

    Behavior
    --------
    - Handles DDP-trained checkpoints by stripping a leading 'module.' prefix.
    - Calls model.to(device) and model.eval() after loading.

    Logging
    -------
    Emits:
      - 'weights.loaded' on success with {'path': str(...), 'strict': bool}
      - 'weights.not_found' (ERROR) if file missing
      - 'weights.load_failed' (ERROR) on exceptions
    """
    p = Path(weights_path)
    if not p.exists():
        log.error("weights.not_found", extra={"path": str(p)})
        raise FileNotFoundError(f"Weights file not found: {p}")

    try:
        state = torch.load(p, map_location=device)

        # unwrap common wrapper dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # strip 'module.' prefixes from DDP-trained checkpoints
        if isinstance(state, dict) and any(isinstance(k, str) for k in state.keys()):
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=strict)
        model.to(device)
        model.eval()
        log.info("weights.loaded", extra={"path": str(p), "strict": strict})
    except Exception as e:
        log.error("weights.load_failed", extra={"path": str(p), "error": str(e)})
        raise