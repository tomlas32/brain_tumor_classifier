"""
Data utilities: deterministic workers and unified DataLoader builders.

This module centralizes how DataLoaders are created so that **train** and
**evaluate** use identical, reproducible settings (worker seeding, pinned
memory, persistent workers, and a per-loader RNG).

Features
--------
- `worker_init_fn` that deterministically seeds NumPy & Python `random`
  for each worker based on PyTorch's initial seed.
- `make_loader(...)` for training/validation (shuffle toggle).
- `make_eval_loader(...)` convenience wrapper with `shuffle=False`.
- (Optional) `ImageFolderDataModule` to bundle datasets + loaders.

Logging
-------
Emits concise, structured logs:
- 'loader.created' with dataset size, batch size, shuffle, workers, and seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def worker_init_fn(worker_id: int) -> None:
    """
    Deterministically seed each DataLoader worker process.

    Parameters
    ----------
    worker_id : int
        The worker index provided by PyTorch.

    Notes
    -----
    - Uses the per-worker initial seed from PyTorch to derive NumPy/Python
      seeds, ensuring reproducible shuffling/augmentations that rely on NumPy
      or `random`.
    """
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a deterministic DataLoader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to wrap.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle each epoch (True for train).
    num_workers : int
        Number of worker processes.
    seed : int
        Seed for this loader's internal RNG (controls PyTorch's sampler).
    pin_memory : bool
        Whether to enable pinned memory for faster host→device transfer.

    Returns
    -------
    DataLoader
        Configured loader with deterministic RNG & worker seeding.

    Logging
    -------
    Emits 'loader.created' with {'num_samples', 'batch_size', 'shuffle',
    'num_workers', 'seed'}.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # Compute dataset length even for Subset
    num_samples = len(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    log.info("loader.created", extra={
        "num_samples": num_samples,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "seed": seed,
    })
    return loader


def make_eval_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience wrapper for evaluation loaders (no shuffling).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to wrap.
    batch_size : int
        Batch size.
    num_workers : int
        Number of worker processes.
    seed : int
        Seed for deterministic sampling.
    pin_memory : bool
        Whether to enable pinned memory.

    Returns
    -------
    DataLoader
        Deterministic, non-shuffled loader.
    """
    return make_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
    )


# --- Optional convenience wrapper for ImageFolder projects --------------------

@dataclass
class ImageFolderDataModule:
    """
    Thin helper to hold ImageFolder train/val/test datasets and build loaders.

    Fields
    ------
    train : Optional[Dataset]
        Training dataset (subset or full).
    val : Optional[Dataset]
        Validation dataset.
    test : Optional[Dataset]
        Test dataset.
    """
    train: Optional[Dataset] = None
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None

    def loaders(
        self,
        *,
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """
        Build train/val/test loaders with deterministic settings.

        Returns
        -------
        (train_loader, val_loader, test_loader)
        """
        train_loader = make_loader(self.train, batch_size, shuffle=True, num_workers=num_workers, seed=seed) if self.train else None
        val_loader   = make_eval_loader(self.val, batch_size, num_workers=num_workers, seed=seed) if self.val else None
        test_loader  = make_eval_loader(self.test, batch_size, num_workers=num_workers, seed=seed) if self.test else None
        return train_loader, val_loader, test_loader
