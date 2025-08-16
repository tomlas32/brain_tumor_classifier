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
import json
import yaml  # PyYAML

# ----------------------- Dataclasses (typed schema) ---------------------------

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
    warn_low_std: float = 3.0
    min_file_bytes: int = 1024
    fail_on: str = "error"
    mapping_pointer: Optional[Path] = None 
    write_report: bool = True


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


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    io: TrainIOConfig = field(default_factory=TrainIOConfig)
    loop: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    aug: AugmentConfig = field(default_factory=AugmentConfig)
    run_id: Optional[str] = None


@dataclass
class EvalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    io: EvalIOConfig = field(default_factory=EvalIOConfig)
    run_id: Optional[str] = None

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
    split: "SplitConfig" = field(default_factory=lambda: SplitConfig())
    resize: "ResizeConfig" = field(default_factory=lambda: ResizeConfig())
    validate: "ValidateConfig" = field(default_factory=lambda: ValidateConfig())
    train: "TrainConfig" = field(default_factory=lambda: TrainConfig())
    evaluate: "EvalConfig" = field(default_factory=lambda: EvalConfig())


# ----------------------- Loader / overrides / utils ---------------------------

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


def load_yaml_config(path: Path) -> dict:
    """
    Load a YAML file into a plain dict.

    Parameters
    ----------
    path : Path
        YAML path.

    Returns
    -------
    dict
        Parsed configuration as a nested dict.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping (dict).")
    return data


def apply_overrides(base: dict, overrides: List[str]) -> dict:
    """
    Apply CLI overrides of the form 'a.b.c=value' to a nested dict.

    Supported value parsing: bool, int, float, null, strings.
    Examples:
        model.name=resnet50
        optim.lr=3e-4
        io.eval_out=outputs/eval_x
        data.seed=1337
        io.make_galleries=false

    Returns
    -------
    dict
        Mutated copy of base with overrides applied.
    """
    out = copy.deepcopy(base)

    def parse_scalar(s: str):
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("null", "none"):
            return None
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value: '{ov}'")
        key, raw = ov.split("=", 1)
        path = key.split(".")
        val = parse_scalar(raw)

        cursor = out
        for p in path[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = {}
            cursor = cursor[p]
        cursor[path[-1]] = val

    return out


def build_train_config(yaml_path: Optional[Path], overrides: List[str]) -> TrainConfig:
    """
    Build a TrainConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = _to_nested_dict(TrainConfig())  # defaults
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg)
    base = apply_overrides(base, overrides)
    # Convert nested dict → dataclass
    return TrainConfig(
        data=DataConfig(**base.get("data", {})),
        model=ModelConfig(**base.get("model", {})),
        optim=OptimConfig(**base.get("optim", {})),
        io=TrainIOConfig(**base.get("io", {})),
        loop=TrainLoopConfig(**base.get("loop", {})),
        aug=AugmentConfig(**base.get("aug", {})),
        run_id=base.get("run_id"),
    )

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
    )


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
    return EvalConfig(
        data=DataConfig(**base.get("data", {})),
        model=ModelConfig(**base.get("model", {})),
        io=EvalIOConfig(**base.get("io", {})),
        run_id=base.get("run_id"),
    )

def build_split_config(yaml_path: Optional[Path], overrides: List[str]) -> SplitConfig:
    """
    Build a SplitConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.
    """
    base = {
        "dataset": None,
        "pointer": None,
        "test_frac": 0.20,
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

    # Normalize exts: accept list from YAML; if someone passed a comma string via override,
    # let split.py convert it with its existing parse_exts().
    exts = base.get("exts")
    if isinstance(exts, str):
        exts = [s.strip() for s in exts.split(",") if s.strip()]

    return SplitConfig(
        dataset=base.get("dataset"),
        pointer=pointer,
        test_frac=float(base.get("test_frac", 0.20)),
        seed=int(base.get("seed", 42)),
        clear_dest=bool(base.get("clear_dest", False)),
        exts=exts if exts is None or isinstance(exts, list) else None,
        save_remap_to_project_root=bool(base.get("save_remap_to_project_root", False)),
        mapping_use_dataset_subdir=bool(base.get("mapping_use_dataset_subdir", False)),
        mapping_write_split_copy=bool(base.get("mapping_write_split_copy", False)),
    )


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
    return ResizeConfig(
        train_in=_p(base.get("train_in")),
        train_out=_p(base.get("train_out")),
        test_in=_p(base.get("test_in")),
        test_out=_p(base.get("test_out")),
        size=int(base.get("size", 224)),
        exts=base.get("exts"),  # leave as-is; script will parse via parse_exts()
    )

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
        "warn_low_std": 3.0,
        "min_file_bytes": 1024,
        "fail_on": "error",
        "write_report": True,
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
        exts=base.get("exts"),  # leave parsing to validate.py via parse_exts()
        dup_check=bool(base.get("dup_check", False)),
        warn_low_std=float(base.get("warn_low_std", 3.0)),
        min_file_bytes=int(base.get("min_file_bytes", 1024)),
        fail_on=str(base.get("fail_on", "error")),
        mapping_pointer=_p(base.get("mapping_pointer")),
        write_report=bool(base.get("write_report", True)),
    )

def build_master_config(yaml_path: Optional[Path], overrides: List[str]) -> MasterConfig:
    """
    Build a MasterConfig from optional YAML + overrides.

    Priority: defaults < YAML < overrides.

    Parameters
    ----------
    yaml_path : Optional[Path]
        Path to a master YAML. If None, defaults are used and only overrides apply.
    overrides : list[str]
        Dotted key overrides, e.g.:
        - "train.data.image_size=256"
        - "resize.size=224"
        - "log.level=DEBUG"
        - "env.seed=1337"

    Returns
    -------
    MasterConfig
        Fully materialized master config object.

    Logging
    -------
    This function does not log; the caller (orchestrator) should log:
      log.info("config.resolved", extra={"config": to_dict(master_cfg)})
    """
    # Start from defaults of the full master structure
    base = _to_nested_dict(MasterConfig())  # uses asdict on all nested dataclasses

    # Merge YAML (if provided)
    if yaml_path:
        yaml_cfg = load_yaml_config(yaml_path)
        _deep_update(base, yaml_cfg or {})

    # Apply CLI overrides last
    base = apply_overrides(base, overrides or [])

    # Reconstruct nested dataclasses from the merged dict
    return MasterConfig(
        run_id=base.get("run_id"),
        env=EnvConfig(**base.get("env", {})),
        log=LoggingConfig(**base.get("log", {})),
        fetch=FetchConfig(**base.get("fetch", {})),
        split=SplitConfig(**base.get("split", {})),
        resize=ResizeConfig(**base.get("resize", {})),
        validate=ValidateConfig(**base.get("validate", {})),
        train=TrainConfig(
            data=DataConfig(**base.get("train", {}).get("data", {})),
            model=ModelConfig(**base.get("train", {}).get("model", {})),
            optim=OptimConfig(**base.get("train", {}).get("optim", {})),
            io=TrainIOConfig(**base.get("train", {}).get("io", {})),
            loop=TrainLoopConfig(**base.get("train", {}).get("loop", {})),
            aug=AugmentConfig(**base.get("train", {}).get("aug", {})),
            run_id=base.get("train", {}).get("run_id"),
        ),
        evaluate=EvalConfig(
            data=DataConfig(**base.get("evaluate", {}).get("data", {})),
            model=ModelConfig(**base.get("evaluate", {}).get("model", {})),
            io=EvalIOConfig(**base.get("evaluate", {}).get("io", {})),
            run_id=base.get("evaluate", {}).get("run_id"),
        ),
    )


def to_dict(dc) -> Dict[str, Any]:
    """Dataclass → plain dict (for logging/manifests)."""
    return asdict(dc)
