"""
Environment bootstrap utilities.

This module centralizes environment initialization, reproducibility settings,
and runtime environment introspection for consistent behavior across all stages
of the ML pipeline.

Features
--------
1) Base directory helper:
   - `bootstrap_env()` can ensure `data/`, `models/`, and `outputs/` exist
     before a stage reads/writes artifacts. (No directories are created unless
     you call `bootstrap_env()`.)

2) Deterministic seeding:
   - Seeds Python `random`, NumPy, and (optionally) PyTorch for reproducible runs.
   - Configures PyTorch cuDNN flags for deterministic execution.

3) Environment logging:
   - Captures and logs runtime library versions, CUDA/CPU availability, and other
     accelerator-related information.
   - Provides structured dictionaries for saving in run manifests.

Usage
-----
from src.core.env import bootstrap_env, log_env_once

bootstrap_env(seed=42)  # create dirs + set seed
log_env_once()          # log library/CUDA info once at run start

Notes
-----
- This module is designed to be idempotent; calling bootstrap_env multiple times
  will not cause errors but may log repeated "seed_set" entries.
- Torch and torchvision imports are optional to allow partial environments
  (e.g., fetch/resize stages without ML dependencies).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.paths import ensure_base_dirs

log = get_logger(__name__)

# Soft imports so this module works even if torch/torchvision aren’t installed yet.
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import torchvision  # type: ignore
except Exception:  # pragma: no cover
    torchvision = None  # type: ignore


@dataclass(frozen=True)
class EnvInfo:
    """
    Snapshot of key environment details relevant to ML reproducibility.

    Fields
    ------
    timestamp_utc : str
        Current UTC timestamp in ISO-8601 format.
    python_hash_seed : Optional[str]
        PYTHONHASHSEED environment variable, if set.
    torch : Optional[str]
        PyTorch version string (if available).
    torchvision : Optional[str]
        Torchvision version string (if available).
    cuda_available : Optional[bool]
        True if CUDA is available and usable by PyTorch.
    cuda_device_count : Optional[int]
        Number of visible CUDA devices.
    cudnn_enabled : Optional[bool]
        Whether cuDNN is enabled.
    cudnn_deterministic : Optional[bool]
        cuDNN deterministic flag state.
    cudnn_benchmark : Optional[bool]
        cuDNN benchmark flag state.
    """

    timestamp_utc: str
    python_hash_seed: Optional[str]
    torch: Optional[str]
    torchvision: Optional[str]
    cuda_available: Optional[bool]
    cuda_device_count: Optional[int]
    cudnn_enabled: Optional[bool]
    cudnn_deterministic: Optional[bool]
    cudnn_benchmark: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        """Return this environment snapshot as a plain dict."""
        return asdict(self)


def set_seed(seed: int) -> None:
    """
    Configure deterministic behavior across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The seed value to set for RNGs and hash functions.

    Notes
    -----
    - Also sets PYTHONHASHSEED for Python's hash randomization.
    - Configures cuDNN for deterministic execution (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    log.info("env.seed_set", extra={"seed": seed})


def get_env_info() -> EnvInfo:
    """
    Collect current runtime environment details.

    Returns
    -------
    EnvInfo
        Dataclass containing library versions and accelerator info.
    """
    return EnvInfo(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        torch=(getattr(torch, "__version__", None) if torch is not None else None),
        torchvision=(getattr(torchvision, "__version__", None) if torchvision is not None else None),
        cuda_available=(torch.cuda.is_available() if torch is not None else None),
        cuda_device_count=(torch.cuda.device_count() if torch is not None else None),
        cudnn_enabled=(getattr(torch.backends.cudnn, "enabled", None) if torch is not None else None),
        cudnn_deterministic=(getattr(torch.backends.cudnn, "deterministic", None) if torch is not None else None),
        cudnn_benchmark=(getattr(torch.backends.cudnn, "benchmark", None) if torch is not None else None),
    )


def log_env_once() -> None:
    """
    Log a one-time summary of the runtime environment.

    The log entry will contain:
    - Timestamps
    - Python hash seed
    - Library versions
    - CUDA/cuDNN availability and flags
    """
    info = get_env_info().to_dict()
    log.info("env.info", extra=info)


def bootstrap_env(*, seed: Optional[int] = None) -> None:
    """
    Prepare the environment for a reproducible ML run.

    Actions
    -------
    1) Ensures base directories (data/, models/, outputs/) exist.
    2) Sets deterministic seeds if `seed` is provided.

    Parameters
    ----------
    seed : int, optional
        RNG seed for reproducibility.
    """
    ensure_base_dirs()
    if seed is not None:
        set_seed(seed)
