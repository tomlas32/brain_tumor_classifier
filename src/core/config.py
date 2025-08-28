"""
Typed run configurations + YAML loader and override utilities.

This module provides dataclass-based configs for training/evaluation and
helpers to:
- load a YAML config file,
- apply CLI overrides like `model.name=resnet34 train.epochs=50`,
- convert configs to plain dicts for summaries/logs.

Logging
-------
Use your stage logger to record the final, resolved config before running.
(e.g., log.info("config.resolved", extra={"config": cfg.to_dict()}))
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import copy
import yaml 

from src.utils.paths import CONFIGS_DIR

# ----------------------- Stage Configs ---------------------------

### Fetch Config
@dataclass
class FetchConfig:
    """
    Config for the 'fetch' stage.

    Parameters
    ----------
    dataset : str | None
        Kaggle slug 'owner/dataset'. If None, the script's CLI default is used.
    cache_dir : Path | None
        Destination cache dir for KaggleHub (defaults to DATA_DIR in fetch.py if None).
    write_pointer : bool
        Whether to write latest/history handoff JSONs.
    pointer_dir : Path | None
        Directory to place the handoff pointers. If None, uses the default
        outputs/downloads_pointer/<owner>/<slug>/.
    """
    dataset: Optional[str] = None
    cache_dir: Optional[Path] = None
    write_pointer: bool = True
    pointer_dir: Optional[Path] = None
    dry_run: bool = False  # for testing

def build_fetch_config(yaml_path: Optional[Path], overrides: List[str]) -> FetchConfig:
    """
    Build a FetchConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = {"dataset": None, "cache_dir": None, "write_pointer": True, "pointer_dir": None}
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)
    # Normalize Path-like fields
    cache_dir = Path(base["cache_dir"]) if base.get("cache_dir") else None
    pointer_dir = Path(base["pointer_dir"]) if base.get("pointer_dir") else None
    return FetchConfig(
        dataset=base.get("dataset"),
        cache_dir=cache_dir,
        write_pointer=bool(base.get("write_pointer", True)),
        pointer_dir=pointer_dir,
        dry_run=bool(base.get("dry_run", False)),
    )


### Merge Config
@dataclass
class MergeConfig:
    """
    Config for the 'merge' stage.

    dataset : str | None
        Kaggle slug 'owner/dataset' used to auto-locate the latest fetch pointer.
        Ignored if 'pointer' is provided.
    pointer : Path | None
        Explicit path to the fetch pointer JSON; overrides 'dataset'.
    exts : list[str] | 'all' | None
        Allowed file extensions when scanning source roots; None => CLI/default behavior.
    clear_dest : bool
        If true, empties DATA_DIR/merged before writing.
    """
    dataset: Optional[str] = None
    pointer: Optional[Path] = None
    exts: Optional[object] = None
    clear_dest: bool = False
    dry_run: bool = False  # for testing

def build_merge_config(yaml_path: Optional[Path], overrides: List[str]) -> MergeConfig:
    """
    Build a MergeConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = {
        "dataset": None,
        "pointer": None,
        "exts": None,
        "clear_dest": False,
        "dry_run": False,
    }
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)

    pointer = Path(base["pointer"]) if base.get("pointer") else None
    return MergeConfig(
        dataset=base.get("dataset"),
        pointer=pointer,
        exts=base.get("exts"),
        clear_dest=bool(base.get("clear_dest", False)),
        dry_run=bool(base.get("dry_run", False)),
    )

### Validate Config
@dataclass
class ValidateConfig:
    """
    Config for the 'validate' stage.

    in_dir : Path
        Root directory to validate (e.g., data/training_resized).
    index_remap : Path
        Path to index_remap.json with allowed class names.
    size : int
        Expected square image size.
    exts : list[str] | str | null
        Allowed extensions: list like ['.jpg','.png'], or 'all', or null.
    dup_check : bool
        Enable SHA-1 duplicate detection (warning-level).
    warn_low_std : float
        Warn if per-image std < this threshold.
    min_file_bytes : int
        Warn if file size < this many bytes.
    fail_on : {'error','warning','never'}
        Exit policy.
    write_report : bool
        Write JSON report to outputs/validation_reports/.
    """
    in_dir: Optional[Path] = None
    index_remap: Optional[Path] = None
    size: int = 224
    exts: Optional[object] = None  # list[str] | "all" | None
    dup_check: bool = False
    phash: bool = True
    phash_thresh: int = 8
    ssim_thresh: float = 0.90
    warn_low_std: float = 3.0
    min_file_bytes: int = 1024
    fail_on: str = "error"
    mapping_pointer: Optional[Path] = None 
    write_report: bool = True
    dry_run: bool = False
    enforce_size: bool = True
    require_rgb: bool = True
    report_tag: Optional[str] = None  # "pre" | "post" | None
    # How to detect duplicates: "file" (bytes), "content" (RGB+resize), or "both"
    dup_mode: str = "file"

def build_validate_config(yaml_path: Optional[Path], overrides: List[str]) -> ValidateConfig:
    """
    Build a ValidateConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = {
        "in_dir": None,
        "index_remap": None,
        "size": 224,
        "exts": None,
        "dup_check": False,
        "phash": True,
        "phash_thresh": 8,
        "ssim_thresh": 0.90,
        "warn_low_std": 3.0,
        "min_file_bytes": 1024,
        "fail_on": "error",
        "write_report": True,
        "dry_run": False,
        "enforce_size": True,
        "require_rgb": True,
        "report_tag": None,
        "dup_mode": "file",           # "file" | "content" | "both"
    }

    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)

    def _p(x): return Path(x) if x is not None else None

    return ValidateConfig(
        in_dir=_p(base.get("in_dir")),
        index_remap=_p(base.get("index_remap")),
        size=int(base.get("size", 224)),
        exts=base.get("exts"),  
        dup_check=bool(base.get("dup_check", False)),
        phash=bool(base.get("phash", True)),
        phash_thresh=int(base.get("phash_thresh", 8)),
        ssim_thresh=float(base.get("ssim_thresh", 0.90)),
        warn_low_std=float(base.get("warn_low_std", 3.0)),
        min_file_bytes=int(base.get("min_file_bytes", 1024)),
        fail_on=str(base.get("fail_on", "error")),
        mapping_pointer=_p(base.get("mapping_pointer")),
        write_report=bool(base.get("write_report", True)),
        dry_run=bool(base.get("dry_run", False)),
        enforce_size=bool(base.get("enforce_size", True)),
        require_rgb=bool(base.get("require_rgb", True)),
        report_tag=base.get("report_tag"),
        dup_mode=str(base.get("dup_mode", "file")),
    )


### Cleanup Config
@dataclass
class CleanupConfig:
    stage: str = "cleanup"
    report: str = "latest"
    policy: str = "strict"       # strict | within_class | report_only
    why: str = "errors"          # errors | warnings | both
    dry_run: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None
    report_tag: Optional[str] = None

def build_cleanup_config(config_file: Path | None, overrides: list[str] | None = None) -> CleanupConfig:
    """
    Build a CleanupConfig from YAML + overrides, consistent with other builders.
    Priority: defaults < YAML < overrides.
    """
    base = _to_nested_dict(CleanupConfig())  # defaults

    if config_file:
        yaml_cfg = load_yaml_config(config_file)
        _deep_update(base, yaml_cfg or {})

    base = apply_overrides(base, overrides or [])

    return CleanupConfig(
        stage=str(base.get("stage", "cleanup")),
        report=str(base.get("report", "latest")),
        policy=str(base.get("policy", "strict")),
        why=str(base.get("why", "errors")),
        dry_run=bool(base.get("dry_run", False)),
        log_level=str(base.get("log_level", "INFO")),
        log_file=base.get("log_file"),
        report_tag=base.get("report_tag"),
    )


### Resize Config
@dataclass
class ResizeConfig:
    """
    Config for the 'resize' stage.

    train_in / train_out / test_in / test_out :
        I/O roots (class-structured) for source and destination.
    size : int
        Target square size in pixels (e.g., 224).
    exts : list[str] | str | null
        Allowed extensions. Accepts:
          - list (e.g., ['.jpg','.png'])
          - 'all' (accept any)
          - null  (use script default semantics)
    """
    train_in: Optional[Path] = None
    train_out: Optional[Path] = None
    test_in: Optional[Path] = None
    test_out: Optional[Path] = None
    size: int = 224
    exts: Optional[object] = None  # list[str] | 'all' | None
    dry_run: bool = False  # for testing

def build_resize_config(yaml_path: Optional[Path], overrides: List[str]) -> ResizeConfig:
    """
    Build a ResizeConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = {
        "train_in": None, "train_out": None,
        "test_in": None,  "test_out": None,
        "size": 224, "exts": None,
    }
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)

    # Normalize paths if provided
    def _p(x): return Path(x) if x is not None else None

    # Normalize exts: allow comma string → list
    exts_val = base.get("exts")
    if isinstance(exts_val, str) and exts_val.lower() != "all":
        # Split on commas and prepend dots if missing
        parts = [e.strip() for e in exts_val.split(",") if e.strip()]
        exts_val = [p if p.startswith(".") else f".{p}" for p in parts]

    return ResizeConfig(
        train_in=_p(base.get("train_in")),
        train_out=_p(base.get("train_out")),
        test_in=_p(base.get("test_in")),
        test_out=_p(base.get("test_out")),
        size=int(base.get("size", 224)),
        exts=exts_val,  
        dry_run=bool(base.get("dry_run", False)),
    )


### Split Config
@dataclass
class SplitConfig:
    """
    Config for the 'split' stage.

    dataset : str | None
        Kaggle slug 'owner/dataset' used to auto-locate the latest fetch pointer.
        Ignored if 'pointer' is provided.
    pointer : Path | None
        Explicit path to the fetch pointer JSON; overrides 'dataset'.
    test_frac : float
        Fraction per class for the final test set (0-1).
    seed : int
        RNG seed used for shuffling per-class pools.
    clear_dest : bool
        If true, empties DATA_DIR/training and DATA_DIR/testing before writing.
    exts : list[str] | None
        Allowed file extensions (lowercase, with leading dot). None means "any".
        Example: ['.jpg', '.jpeg', '.png'].
    save_remap_to_project_root : bool
        Also save index_remap.json to the project root (./index_remap.json).
    mapping_use_dataset_subdir : bool
        If true, writes mapping under outputs/mappings/<owner>/<slug>/…
    mapping_write_split_copy : bool
        If true, copies index_remap.json into the split root (DATA_DIR).
    """
    dataset: Optional[str] = None
    pointer: Optional[Path] = None
    test_frac: float = 0.20
    seed: int = 42
    clear_dest: bool = False
    exts: Optional[List[str]] = None
    save_remap_to_project_root: bool = False
    mapping_use_dataset_subdir: bool = False
    mapping_write_split_copy: bool = False
    dry_run: bool = False  # for testing
    val_frac: float = 0.10

def build_split_config(yaml_path: Optional[Path], overrides: List[str]) -> SplitConfig:
    """
    Build a SplitConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = {
        "dataset": None,
        "pointer": None,
        "test_frac": 0.20,
        "val_frac": 0.10,
        "seed": 42,
        "clear_dest": False,
        "exts": None,
        "save_remap_to_project_root": False,
        "mapping_use_dataset_subdir": False,
        "mapping_write_split_copy": False,
    }
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)

    # Normalize path-like fields
    pointer = Path(base["pointer"]) if base.get("pointer") else None

    # Normalize exts: allow comma string → list[str] with leading dots
    exts_val = base.get("exts")
    if isinstance(exts_val, str) and exts_val.lower() != "all":
        parts = [e.strip() for e in exts_val.split(",") if e.strip()]
        exts_val = [p if p.startswith(".") else f".{p}" for p in parts]

    return SplitConfig(
        dataset=base.get("dataset"),
        pointer=pointer,
        test_frac=float(base.get("test_frac", 0.20)),
        val_frac=float(base.get("val_frac", 0.10)),
        seed=int(base.get("seed", 42)),
        clear_dest=bool(base.get("clear_dest", False)),
        exts=exts_val,
        save_remap_to_project_root=bool(base.get("save_remap_to_project_root", False)),
        mapping_use_dataset_subdir=bool(base.get("mapping_use_dataset_subdir", False)),
        mapping_write_split_copy=bool(base.get("mapping_write_split_copy", False)),
        dry_run=bool(base.get("dry_run", False)),
    )



# ----------------------- Shared Configs ---------------------------
@dataclass
class DataConfig:
    image_size: int = 224
    train_in: Optional[Path] = None   # training root (class folders)
    eval_in: Optional[Path] = None    # test root (class folders)
    mapping_path: Optional[Path] = None
    mapping_pointer: Optional[Path] = None
    batch_size: int = 32
    num_workers: int = 4
    val_frac: float = 0.2
    seed: int = 42

@dataclass
class AugmentConfig:
    """
    Data augmentation knobs for training transforms.

    rotate_deg : int
        Max absolute degrees for RandomRotation (±rotate_deg). 0 disables.
    hflip_prob : float
        Probability for RandomHorizontalFlip. 0 disables.
    jitter_brightness : float
        Brightness factor for ColorJitter (range [0, 1]). 0 disables.
    jitter_contrast : float
        Contrast factor for ColorJitter (range [0, 1]). 0 disables.
    """
    rotate_deg: int = 15
    hflip_prob: float = 0.5
    jitter_brightness: float = 0.1
    jitter_contrast: float = 0.1

@dataclass
class EarlyStoppingCfg:
    """
    Early stopping configuration.

    Parameters
    ----------
    enabled : bool
        Whether early stopping is active.
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum improvement required to reset the patience counter.
    monitor : str
        Metric key to monitor (runner logs use 'val_f1').
    """
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.0
    monitor: str = "val_f1"

@dataclass
class CheckpointCfg:
    """
    Checkpointing configuration.

    Parameters
    ----------
    save_best : bool
        Save a checkpoint whenever the monitored metric improves.
    save_last : bool
        Save a 'last.pth' checkpoint at the end of each epoch.
    every_n_epochs : int
        Periodic checkpointing interval (0 disables).
    out_dir : Path | None
        Optional override of the models output directory.
    """
    save_best: bool = True
    save_last: bool = False
    every_n_epochs: int = 0
    out_dir: Optional[Path] = None

@dataclass
class LRLoggerCfg:
    """
    Learning rate logging configuration.

    Parameters
    ----------
    enabled : bool
        Record LR each epoch and write a JSON trace at train end.
    out_json : Path | None
        Optional custom JSON path; defaults under training summary dir.
    """
    enabled: bool = True
    out_json: Optional[Path] = None

@dataclass
class CallbacksConfig:
    """
    Aggregate callbacks configuration block attached to TrainConfig.

    Fields
    ------
    early_stopping : EarlyStoppingCfg
        Early stopping options.
    checkpoint : CheckpointCfg
        Checkpoint saving options.
    lr_logger : LRLoggerCfg
        Learning rate logging options.
    """
    early_stopping: EarlyStoppingCfg = field(default_factory=EarlyStoppingCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    lr_logger: LRLoggerCfg = field(default_factory=LRLoggerCfg)

@dataclass
class ModelConfig:
    name: str = "resnet18"            # resnet18|resnet34|resnet50
    pretrained: bool = True
    weights_path: Optional[Path] = None  # used for evaluation

@dataclass
class TrainLoopConfig:
    """Training loop settings."""
    epochs: int = 20

@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    step_size: int = 10
    gamma: float = 0.1
    amp: bool = True

@dataclass
class TrainIOConfig:
    out_models: Path = Path("models")
    out_summary: Path = Path("outputs/training")

@dataclass
class EvalIOConfig:
    eval_out: Path = Path("outputs/evaluation")
    make_galleries: bool = True
    make_gradcam: bool = True
    top_per_class: int = 6
    gallery_cols: int = 4        
    gradcam_cols: int = 4         


### Train Config
@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    io: TrainIOConfig = field(default_factory=TrainIOConfig)
    loop: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    aug: AugmentConfig = field(default_factory=AugmentConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    run_id: Optional[str] = None
    dry_run: bool = False
    val_in: Optional[Path] = None

def build_train_config(yaml_path: Optional[Path], overrides: List[str]) -> TrainConfig:
    """
    Build a TrainConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = _to_nested_dict(TrainConfig())  # defaults (now includes callbacks)
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)

    # Rebuild nested callbacks block explicitly to ensure typed dataclasses
    cb = base.get("callbacks", {}) or {}
    callbacks = CallbacksConfig(
        early_stopping=EarlyStoppingCfg(**(cb.get("early_stopping", {}) or {})),
        checkpoint=CheckpointCfg(**(cb.get("checkpoint", {}) or {})),
        lr_logger=LRLoggerCfg(**(cb.get("lr_logger", {}) or {})),
    )

    # Normalize path-like fields inside io block
    io_block = base.get("io", {}) or {}
    out_models = io_block.get("out_models")
    out_summary = io_block.get("out_summary")
    if isinstance(out_models, str):
        io_block["out_models"] = Path(out_models)
    if isinstance(out_summary, str):
        io_block["out_summary"] = Path(out_summary)
    
    val_in_path = base.get("val_in")
    if isinstance(val_in_path, str):
        base["val_in"] = Path(val_in_path)

    return TrainConfig(
        data=DataConfig(**base.get("data", {})),
        model=ModelConfig(**base.get("model", {})),
        optim=OptimConfig(**base.get("optim", {})),
        io=TrainIOConfig(**io_block),
        loop=TrainLoopConfig(**base.get("loop", {})),
        aug=AugmentConfig(**base.get("aug", {})),
        callbacks=callbacks,
        run_id=base.get("run_id"),
        dry_run=bool(base.get("dry_run", False)),
        val_in=val_in_path if val_in_path else None,
    )


### Eval Config
@dataclass
class EvalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    io: EvalIOConfig = field(default_factory=EvalIOConfig)
    run_id: Optional[str] = None
    dry_run: bool = False

def build_eval_config(yaml_path: Optional[Path], overrides: List[str]) -> EvalConfig:
    """
    Build an EvalConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = _to_nested_dict(EvalConfig())  # defaults
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)

    # Normalize path-like fields that may come in as strings
    model_block = base.get("model", {}) or {}
    io_block = base.get("io", {}) or {}
    data_block = base.get("data", {}) or {}

    wp = model_block.get("weights_path")
    if isinstance(wp, str):
        model_block["weights_path"] = Path(wp)

    eo = io_block.get("eval_out")
    if isinstance(eo, str):
        io_block["eval_out"] = Path(eo)

    mp = data_block.get("mapping_path")
    if isinstance(mp, str):
        data_block["mapping_path"] = Path(mp)

    mptr = data_block.get("mapping_pointer")
    if isinstance(mptr, str):
        data_block["mapping_pointer"] = Path(mptr)

    return EvalConfig(
        data=DataConfig(**data_block),
        model=ModelConfig(**model_block),
        io=EvalIOConfig(**io_block),
        run_id=base.get("run_id"),
        dry_run=bool(base.get("dry_run", False)),
    )


@dataclass
class EnvConfig:
    """
    Environment knobs applied once at pipeline start (orchestrator).

    seed : int
        Global RNG seed (propagated to stages via bootstrap_env()).
    prefer_cuda : bool
        If True, the orchestrator will prefer CUDA devices when available.
    cudnn_deterministic : bool
        cuDNN determinism flag (True for reproducibility).
    cudnn_benchmark : bool
        cuDNN autotuner (False for reproducibility).
    """
    seed: int = 42
    prefer_cuda: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False


@dataclass
class LoggingConfig:
    """
    Logging defaults applied by the orchestrator.

    level : str
        Log level (e.g., 'INFO', 'DEBUG').
    file : Optional[str]
        Optional fixed log file path. If None, each stage decides (your code
        already supports auto/fixed in configure_logging()).
    json : bool
        Future toggle for JSON logs if you add a JSON formatter.
    """
    level: str = "INFO"
    file: Optional[str] = None
    json: bool = False


@dataclass
class MasterConfig:
    """
    End-to-end pipeline configuration.

    This nests per-stage configs (fetch → split → resize → validate → train → evaluate)
    plus top-level run, environment, and logging defaults.

    Notes on logging
    ----------------
    The orchestrator should log a single 'config.resolved' entry with
    the fully merged MasterConfig (use `to_dict()`), then pass the
    relevant sub-configs to each stage. Stages can still log their own
    resolved configs for local provenance; redundancy is fine.

    Overriding behavior
    -------------------
    - CLI overrides apply to the master structure with dotted keys, e.g.:
      `train.data.image_size=256`, `resize.size=224`, `log.level=DEBUG`.
    - In Step 2 we’ll make the orchestrator optionally propagate
      `run_id` down into child configs if they don’t set one.
    """
    run_id: Optional[str] = None
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)

    # Per-stage blocks (you already defined these dataclasses & builders):
    fetch: "FetchConfig" = field(default_factory=lambda: FetchConfig())
    merge: "MergeConfig" = field(default_factory=lambda: MergeConfig())
    split: "SplitConfig" = field(default_factory=lambda: SplitConfig())
    resize: "ResizeConfig" = field(default_factory=lambda: ResizeConfig())
    validate: "ValidateConfig" = field(default_factory=lambda: ValidateConfig())
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    train: "TrainConfig" = field(default_factory=lambda: TrainConfig())
    evaluate: "EvalConfig" = field(default_factory=lambda: EvalConfig())

def build_master_config(yaml_path: Optional[Path], overrides: List[str]) -> MasterConfig:
    """
    Build a MasterConfig from optional YAML + overrides.

    Strategy:
      1) Merge defaults + YAML + CLI into dict `base`
      2) Build each stage by delegating to its build_*_config with flattened overrides
         (so we reuse per-stage normalization like Path casting and exts handling).
    """
    # 1) Start from defaults of the full master structure
    base = _to_nested_dict(MasterConfig())

    # 2) Merge YAML (if provided)
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg or {})

    # 3) Apply CLI overrides last
    base = apply_overrides(base, overrides or [])

    # 4) Build each stage using its dedicated builder (reuses all normalization)
    fetch_cfg    = build_fetch_config   (None, _flatten_to_overrides(base.get("fetch", {})))
    merge_cfg    = build_merge_config   (None, _flatten_to_overrides(base.get("merge", {})))
    split_cfg    = build_split_config   (None, _flatten_to_overrides(base.get("split", {})))
    resize_cfg   = build_resize_config  (None, _flatten_to_overrides(base.get("resize", {})))
    validate_cfg = build_validate_config(None, _flatten_to_overrides(base.get("validate", {})))
    cleanup_cfg  = build_cleanup_config (None, _flatten_to_overrides(base.get("cleanup", {})))
    train_cfg    = build_train_config   (None, _flatten_to_overrides(base.get("train", {})))
    eval_cfg     = build_eval_config    (None, _flatten_to_overrides(base.get("evaluate", {})))

    # 5) Top-level blocks (not stage-specific)
    env_cfg = EnvConfig(**base.get("env", {}))
    log_cfg = LoggingConfig(**base.get("log", {}))

    return MasterConfig(
        run_id=base.get("run_id"),
        env=env_cfg,
        log=log_cfg,
        fetch=fetch_cfg,
        merge=merge_cfg,
        split=split_cfg,
        resize=resize_cfg,
        validate=validate_cfg,
        cleanup=cleanup_cfg,
        train=train_cfg,
        evaluate=eval_cfg,
    )

# ====================== Utils ======================

def _deep_update(dst: dict, src: dict) -> dict:
    """Recursively merge dict `src` into dict `dst` (in place) and return dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _to_nested_dict(obj) -> dict:
    """Dataclass or simple object → dict suitable for deep update."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return copy.deepcopy(obj)
    raise TypeError(f"Unsupported config type: {type(obj)}")


def load_yaml_config(path: str | Path) -> dict:
    """
    Load YAML with minimal, reliable resolution:
    - absolute path
    - path relative to current working directory
    - path relative to CONFIGS_DIR (src/configs)
    Also tolerates Windows forms like "\\configs\\file.yaml" by stripping the root.
    """
    raw = str(path)
    p = Path(raw)

    # If the string looks rooted on Win/Unix, reinterpret as relative name
    if not Path(raw).is_absolute() and raw.startswith("\\"):
        p = Path(raw.lstrip("\\/"))

    candidates = [
        p if p.is_absolute() else None,
        Path.cwd() / p,
        CONFIGS_DIR / p,
    ]

    for c in filter(None, candidates):
        if c.exists():
            with c.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Top-level YAML must be a mapping (dict).")
            return data

    tried = "\n  - " + "\n  - ".join(str(c) for c in filter(None, candidates))
    raise FileNotFoundError(
        f"YAML not found for '{path}'. Tried:{tried}\n"
        f"CWD={Path.cwd()}\nCONFIGS_DIR={CONFIGS_DIR}"
    )


def apply_overrides(base: dict, overrides: List[str]) -> dict:
    """
    Apply CLI overrides of the form 'a.b.c=value' to a nested dict.

    Value parsing:
      - Prefer yaml.safe_load(raw) to support lists/dicts/bools/null & 1e-3
      - Fall back to the raw string if YAML parsing fails.
    """
    out = copy.deepcopy(base)

    def parse_value(raw: str):
        try:
            return yaml.safe_load(raw)
        except Exception:
            return raw  # last resort

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value: '{ov}'")
        key, raw = ov.split("=", 1)
        path = key.split(".")
        val = parse_value(raw)

        cursor = out
        for p in path[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = {}
            cursor = cursor[p]
        cursor[path[-1]] = val

    return out

def to_dict(dc) -> Dict[str, Any]:
    """Dataclass → plain JSON-serializable dict (converts Path → str)."""
    from dataclasses import is_dataclass, fields

    def _conv(obj):
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj):
            return {f.name: _conv(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, dict):
            return {k: _conv(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_conv(x) for x in obj]
        return obj

    return _conv(dc)


def _flatten_to_overrides(d: dict) -> List[str]:
    """
    Flatten a nested dict into dotted CLI-style overrides with stable, single-line values.
    - Scalars (None, bool, int, float, str) are encoded without using PyYAML to avoid '...'.
    - Collections (list/tuple/dict) use YAML flow style, then are cleaned to one line.
    """
    out: List[str] = []

    def encode_scalar(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            # repr preserves scientific notation if already present
            return repr(v)
        if isinstance(v, str):
            return v
        # Fallback for any other scalar-ish types
        return str(v)

    def encode_any(v):
        if isinstance(v, (list, tuple, dict)):
            s = yaml.safe_dump(v, default_flow_style=True).strip()
            # Ensure single-line, no YAML doc end markers
            return s.replace("\n", " ").replace(" ...", "").replace("...", "").strip()
        return encode_scalar(v)

    def rec(prefix: List[str], obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                rec(prefix + [k], v)
        else:
            key = ".".join(prefix)
            val = encode_any(obj)
            out.append(f"{key}={val}")

    rec([], d or {})
    return out