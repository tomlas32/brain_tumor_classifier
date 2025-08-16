"""
Callbacks module for training pipeline.

Implements lightweight extensible callbacks for:
- Early stopping
- Checkpointing
- Learning rate logging

Each callback follows a common API (`on_train_start`, `on_epoch_start`,
`on_epoch_end`, `on_train_end`) and integrates with the training runner.

Author: Tomasz Lasota
Date: 2025-08-16
Version: 1.0
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import math
import torch

from src.utils.logging_utils import get_logger
from src.core.artifacts import safe_json_dump, write_latest_pointer

log = get_logger(__name__)


class Callback:
    """Base interface for all callbacks."""
    def on_train_start(self, **kwargs): ...
    def on_epoch_start(self, **kwargs): ...
    def on_epoch_end(self, **kwargs): ...
    def on_train_end(self, **kwargs): ...


# --------------------------- Early Stopping ----------------------------------
@dataclass
class EarlyStoppingConfig:
    """Configuration for EarlyStoppingCallback."""
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.0
    monitor: str = "val_f1"


class EarlyStoppingCallback(Callback):
    """
    Stop training when a monitored metric stops improving.

    Logs: `callback.early_stop` when triggered.
    """
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best: float = -math.inf
        self.bad_epochs: int = 0
        self._stop: bool = False

    @property
    def should_stop(self) -> bool:
        """Return True if early stopping should trigger."""
        return bool(self.cfg.enabled and self._stop)

    def on_train_start(self, **_):
        self.best = -math.inf
        self.bad_epochs = 0
        self._stop = False

    def on_epoch_end(self, **kwargs):
        if not self.cfg.enabled:
            return
        metric = float(kwargs.get("metrics", {}).get(self.cfg.monitor, -math.inf))
        improved = metric > (self.best + float(self.cfg.min_delta))
        if improved:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= int(self.cfg.patience):
                self._stop = True
                log.info("callback.early_stop", extra={
                    "monitor": self.cfg.monitor,
                    "best": round(self.best, 6),
                    "patience": self.cfg.patience,
                })


# ------------------------------ Checkpointing --------------------------------
@dataclass
class CheckpointConfig:
    """Configuration for CheckpointCallback."""
    save_best: bool = True
    save_last: bool = False
    every_n_epochs: int = 0  # 0 disables periodic saves
    out_dir: Optional[Path] = None


class CheckpointCallback(Callback):
    """
    Save model checkpoints during training.

    - Saves best model by monitored metric (default: val_f1).
    - Optionally saves last model each epoch.
    - Optionally saves periodic snapshots.

    Logs: `checkpoint.saved` whenever a file is written.
    """
    def __init__(self, cfg: CheckpointConfig, *, out_models: Path):
        self.cfg = cfg
        self.out_models = Path(cfg.out_dir or out_models)
        self.best: float = -math.inf
        self.best_path: Optional[Path] = None

    def _save(self, state_dict: Dict[str, Any], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, path)
        log.info("checkpoint.saved", extra={"path": str(path)})

    def on_epoch_end(self, **kwargs):
        model: torch.nn.Module = kwargs["model"]
        epoch: int = int(kwargs["epoch"])
        metrics: Dict[str, float] = kwargs.get("metrics", {})
        val_f1 = float(metrics.get("val_f1", -math.inf))

        # Save-best
        if self.cfg.save_best and val_f1 > self.best:
            self.best = val_f1
            self.best_path = self.out_models / f"best_valF1_{val_f1:.4f}_epoch{epoch}.pth"
            self._save(model.state_dict(), self.best_path)
            write_latest_pointer(
                {"best_f1": float(val_f1), "epoch": epoch, "path": str(self.best_path)},
                self.out_models,
                stem="checkpoint",
            )

        # Save-last
        if self.cfg.save_last:
            last_path = self.out_models / "last.pth"
            self._save(model.state_dict(), last_path)

        # Periodic
        if self.cfg.every_n_epochs and epoch % int(self.cfg.every_n_epochs) == 0:
            periodic = self.out_models / f"ckpt_epoch{epoch}.pth"
            self._save(model.state_dict(), periodic)


# ------------------------------ LR Logger ------------------------------------
@dataclass
class LRLoggerConfig:
    """Configuration for LRLoggerCallback."""
    enabled: bool = True
    out_json: Optional[Path] = None


class LRLoggerCallback(Callback):
    """
    Track and log learning rate schedule.

    Collects per-epoch LR and writes JSON at end of training.

    Logs: `lr_logger.written` when file is saved.
    """
    def __init__(self, cfg: LRLoggerConfig, *, out_summary: Path, run_id: Optional[str]):
        self.cfg = cfg
        self.records = []  # list of {"epoch": int, "lr": float}
        self.out_summary = Path(cfg.out_json or out_summary)
        self.run_id = run_id

    def on_epoch_start(self, **kwargs):
        if not self.cfg.enabled:
            return
        epoch: int = int(kwargs["epoch"])
        optimizer = kwargs["optimizer"]
        lr = float(optimizer.param_groups[0]["lr"])
        self.records.append({"epoch": epoch, "lr": lr})

    def on_train_end(self, **_):
        if not self.cfg.enabled:
            return
        self.out_summary.mkdir(parents=True, exist_ok=True)
        path = self.out_summary / f"lr_history_{(self.run_id or 'no-runid')}.json"
        safe_json_dump({"run_id": self.run_id, "history": self.records}, path)
        log.info("lr_logger.written", extra={"path": str(path), "count": len(self.records)})
