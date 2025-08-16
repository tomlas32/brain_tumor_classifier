"""
Pipeline orchestrator: run fetch → split → resize → validate → train → evaluate
from a single MasterConfig, with dry-run/skip/resume controls.

This module does **not** reimplement any stage logic. It simply:
  1) Loads a MasterConfig (YAML + overrides).
  2) Builds a per-stage execution plan.
  3) Optionally performs a dry-run (plan only).
  4) Dispatches each stage module's `main(argv)` with `--config` pointing to a
     temporary, stage-specific YAML file derived from the MasterConfig.
  5) Records per-stage exit codes and writes a run manifest.

Logging (structured)
--------------------
- 'orchestrator.start'   : run_id, stages, dry_run/skip/resume
- 'config.resolved'      : resolved MasterConfig (dict)
- 'orchestrator.plan'    : ordered plan (stage → config file + argv)
- 'cli.dispatch'         : before each stage call (stage + argv)
- 'orchestrator.stage_end' : stage, exit_code, duration_sec
- 'orchestrator.done'    : overall_exit_code, manifest_path

Environment
-----------
- Seeds and env flags are applied once via core.env.bootstrap_env(seed) and logged
  via core.env.log_env_once().
- RUN_ID is exported to the process environment for stages that pick it up.

Usage
-----
from src.pipeline.orchestrator import run_pipeline
exit_code = run_pipeline(
    master_yaml=Path("configs/pipeline.yaml"),
    overrides=["train.aug.rotate_deg=5"],
    dry_run=False,
    skip=["fetch","split"],            # or []
    resume_from=None,                  # or "resize"
)

CLI integration is added in Step 3 via src/cli.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger, configure_logging
from src.utils.paths import OUTPUTS_DIR

from src.core.config import (
    MasterConfig,
    build_master_config,
    to_dict,
)
from src.core.artifacts import safe_json_dump
from src.core.env import bootstrap_env, log_env_once

# Stage modules (each exposes main(argv))
from src.pipeline import fetch as fetch_mod
from src.pipeline import split as split_mod
from src.pipeline import resize as resize_mod
from src.pipeline import validate as validate_mod
from src.pipeline import train as train_mod
from src.pipeline import evaluate as evaluate_mod


log = get_logger(__name__)

DEFAULT_ORDER = ["fetch", "split", "resize", "validate", "train", "evaluate"]

VALID_STAGES = set(DEFAULT_ORDER)

def _validate_stage_names(skip: List[str], resume_from: Optional[str]) -> Optional[str]:
    """
    Return an error message if invalid stage names are supplied; otherwise None.
    """
    bad = [s for s in (skip or []) if s not in VALID_STAGES]
    if bad:
        return f"Invalid --skip stages: {bad}. Valid: {sorted(VALID_STAGES)}"
    if resume_from and resume_from not in VALID_STAGES:
        return f"Invalid --resume-from: '{resume_from}'. Valid: {sorted(VALID_STAGES)}"
    return None


def _dataset_owner_slug(dataset: Optional[str]) -> tuple[str, str]:
    """
    Split 'owner/slug' → (owner, slug). Returns ('_unknown_', '_unknown_') if not parseable.
    """
    if not dataset or "/" not in dataset:
        return "_unknown_", "_unknown_"
    owner, slug = dataset.split("/", 1)
    return owner or "_unknown_", slug or "_unknown_"


def _expected_pointer_paths(master: MasterConfig) -> Dict[str, Dict[str, str]]:
    """
    Derive expected pointer directories and latest.json paths for this run.

    Rules
    -----
    - Fetch:
        If fetch.pointer_dir provided → use that dir.
        Else, derive from fetch.dataset → outputs/pointers/fetch/<owner>/<slug>/
    - Mapping:
        Prefer an explicit mapping pointer from any stage (train/eval/validate), in this order:
          train.data.mapping_pointer → eval.data.mapping_pointer → validate.mapping_pointer
        Else, derive from fetch.dataset → outputs/pointers/mapping/<owner>/<slug>/

    Returns
    -------
    {
      "fetch":   {"dir": ".../pointers/fetch/<o>/<s>/",   "latest": ".../latest.json"},
      "mapping": {"dir": ".../pointers/mapping/<o>/<s>/", "latest": ".../latest.json"},
    }
    """
    # --- Fetch pointer
    if getattr(master.fetch, "pointer_dir", None):
        fetch_dir = Path(master.fetch.pointer_dir)
    else:
        o, s = _dataset_owner_slug(getattr(master.fetch, "dataset", None))
        fetch_dir = OUTPUTS_DIR / "pointers" / "fetch" / o / s
    fetch_latest = fetch_dir / "latest.json"

    # --- Mapping pointer (prefer explicit pointers if any)
    mp = (
        getattr(getattr(master.train, "data", None), "mapping_pointer", None)
        or getattr(getattr(master.evaluate, "data", None), "mapping_pointer", None)
        or getattr(master.validate, "mapping_pointer", None)
    )
    if mp:
        mapping_dir = Path(mp) if Path(mp).is_dir() else Path(mp).parent
    else:
        o, s = _dataset_owner_slug(getattr(master.fetch, "dataset", None))
        mapping_dir = OUTPUTS_DIR / "pointers" / "mapping" / o / s
    mapping_latest = mapping_dir / "latest.json"

    return {
        "fetch": {"dir": str(fetch_dir), "latest": str(fetch_latest)},
        "mapping": {"dir": str(mapping_dir), "latest": str(mapping_latest)},
    }


def _utc_ts_compact() -> str:
    """Return UTC timestamp as YYYYMMDD_HHMMSSZ."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _mk_run_dir(run_id: str) -> Path:
    """
    Create a working directory for this run's orchestrator files.

    Returns
    -------
    Path to: outputs/orchestrator/<run_id>/
    """
    out = Path("outputs/orchestrator") / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_stage_yaml(run_dir: Path, stage: str, block: object) -> Path:
    """
    Persist a stage sub-config to YAML (actually JSON via safe_json_dump for simplicity).

    Parameters
    ----------
    run_dir : Path
        Orchestrator run directory.
    stage : str
        Stage name (used in filename).
    block : dataclass or dict
        The stage sub-config to write.

    Returns
    -------
    Path
        Path to the written config file.

    Notes
    -----
    - Uses safe_json_dump for robust I/O and consistent logging.
    - Stages accept '--config' pointing at YAML, but our config loader uses
      yaml.safe_load; JSON is a subset of YAML and is valid input.
    """
    cfg_path = run_dir / f"{stage}.yaml"
    # Convert to plain dict regardless of dataclass
    payload = to_dict(block) if hasattr(block, "__dataclass_fields__") else block
    safe_json_dump(payload, cfg_path)
    return cfg_path


def _stage_module(stage: str):
    """Map stage name → imported module reference (with main(argv))."""
    return {
        "fetch": fetch_mod,
        "split": split_mod,
        "resize": resize_mod,
        "validate": validate_mod,
        "train": train_mod,
        "evaluate": evaluate_mod,
    }[stage]


def _stage_should_run(stage: str, *, skip: List[str], start_from: Optional[str]) -> bool:
    """
    Decide whether a stage should run given skip list and resume point.

    Rules
    -----
    - If start_from is set, run only stages at or after that stage (ignore skip).
    - Else, run all stages not present in 'skip'.
    """
    if start_from:
        idx_all = DEFAULT_ORDER.index(stage)
        idx_start = DEFAULT_ORDER.index(start_from)
        return idx_all >= idx_start
    return stage not in (skip or [])


def _build_plan(master: MasterConfig, *, dry_run: bool, skip: List[str], resume_from: Optional[str]) -> List[Dict]:
    """
    Build an ordered execution plan: per stage → config file + argv.

    Returns
    -------
    list[dict]
        Each item has: {'stage', 'config_path', 'argv'}.
    """
    run_id = master.run_id or _utc_ts_compact()
    run_dir = _mk_run_dir(run_id)

    # Ensure children inherit run_id if unset (non-destructive)
    if master.train.run_id is None:
        master.train.run_id = run_id  # type: ignore[attr-defined]
    if master.evaluate.run_id is None:
        master.evaluate.run_id = run_id  # type: ignore[attr-defined]

    # Serialize sub-configs
    cfg_paths = {
        "fetch": _write_stage_yaml(run_dir, "fetch", master.fetch),
        "split": _write_stage_yaml(run_dir, "split", master.split),
        "resize": _write_stage_yaml(run_dir, "resize", master.resize),
        "validate": _write_stage_yaml(run_dir, "validate", master.validate),
        "train": _write_stage_yaml(run_dir, "train", master.train),
        "evaluate": _write_stage_yaml(run_dir, "evaluate", master.evaluate),
    }

    plan: List[Dict] = []
    for stage in DEFAULT_ORDER:
        if not _stage_should_run(stage, skip=skip, start_from=resume_from):
            continue
        argv = ["--config", str(cfg_paths[stage])]
        plan.append({"stage": stage, "config_path": cfg_paths[stage], "argv": argv})

    log.info("orchestrator.plan", extra={
        "run_id": run_id,
        "stages": [p["stage"] for p in plan],
        "dry_run": bool(dry_run),
        "skip": skip,
        "resume_from": resume_from,
        "config_files": {k: str(v) for k, v in cfg_paths.items()},
    })
    return plan


def run_pipeline(
    master_yaml: Optional[Path],
    overrides: Optional[List[str]] = None,
    *,
    dry_run: bool = False,
    skip: Optional[List[str]] = None,
    resume_from: Optional[str] = None,
) -> int:
    """
    Execute the pipeline end-to-end using a MasterConfig.

    Parameters
    ----------
    master_yaml : Path | None
        Path to the master YAML. If None, defaults + overrides apply.
    overrides : list[str] | None
        Dotted key overrides (e.g., 'train.data.image_size=256').
    dry_run : bool
        If True, only logs the plan; does not call any stage.
    skip : list[str] | None
        Stages to skip entirely (ignored if resume_from is set).
    resume_from : str | None
        Stage name to start from; all prior stages are skipped.

    Returns
    -------
    int
        Overall exit code: 0 if all executed stages returned 0; otherwise the
        last non-zero exit code from a stage.

    Behavior
    --------
    - Applies environment bootstrap once (seed, determinism) and exports RUN_ID.
    - Writes per-stage config files into outputs/orchestrator/<run_id>/.
    - Writes a run manifest at the end with per-stage codes and durations.
    """
    # Resolve config
    master = build_master_config(master_yaml, overrides or [])
    run_id = master.run_id or _utc_ts_compact()

    # Top-level logging (orchestrator only)
    # Note: Each stage still configures its own logging per its script.
    if master.log.file:
        configure_logging(log_level=master.log.level, file_mode="fixed", log_file=master.log.file, run_id=run_id, stage="pipeline")
    else:
        configure_logging(log_level=master.log.level, file_mode="auto", run_id=run_id, stage="pipeline")

    log.info("orchestrator.start", extra={
        "run_id": run_id,
        "dry_run": bool(dry_run),
        "skip": skip or [],
        "resume_from": resume_from,
    })
    log.info("config.resolved", extra={"config": to_dict(master)})

    # Validate skip/resume names up-front
    _err = _validate_stage_names(skip or [], resume_from)
    if _err:
        log.error("orchestrator.args_invalid", extra={"error": _err})
        return 2

    # Resolve expected pointer locations and log once
    pointers = _expected_pointer_paths(master)
    log.info("orchestrator.pointers_resolved", extra=pointers)

    # Phase 0 (once)
    bootstrap_env(seed=master.env.seed)
    log_env_once()
    os.environ["RUN_ID"] = run_id

    # Build plan + ensure run dir
    run_dir = _mk_run_dir(run_id)
    plan = _build_plan(master, dry_run=dry_run, skip=skip or [], resume_from=resume_from)

    # Dump the plan to a file for auditability
    plan_payload = {
        "run_id": run_id,
        "dry_run": bool(dry_run),
        "skip": skip or [],
        "resume_from": resume_from,
        "stages": [
            {"stage": it["stage"], "config": str(it["config_path"]), "argv": it["argv"]}
            for it in plan
        ],
        "pointers": pointers,
    }
    plan_path = run_dir / "plan.json"
    safe_json_dump(plan_payload, plan_path)
    log.info("orchestrator.plan_written", extra={"path": str(plan_path)})

    if dry_run:
        log.info("orchestrator.done", extra={
            "run_id": run_id,
            "overall_exit_code": 0,
            "note": "dry_run",
            "plan": str(plan_path),
        })
        return 0


    # Execute stages
    results: List[Dict] = []
    overall_code = 0
    run_dir = _mk_run_dir(run_id)

    for item in plan:
        stage = item["stage"]
        argv = item["argv"]
        mod = _stage_module(stage)

        t0 = time.time()
        log.info("cli.dispatch", extra={"stage": stage, "argv": argv})
        try:
            code = int(mod.main(argv))  # each stage exposes main(argv)->int
        except SystemExit as e:
            # In case a stage uses raise SystemExit(code)
            code = int(e.code) if isinstance(e.code, int) else 1
        except Exception as e:
            log.error("orchestrator.stage_exception", extra={"stage": stage, "error": str(e)})
            code = 1
        dt = round(time.time() - t0, 2)

        results.append({"stage": stage, "exit_code": code, "duration_sec": dt, "argv": argv})
        log.info("orchestrator.stage_end", extra={"stage": stage, "exit_code": code, "duration_sec": dt})

        if code != 0:
            overall_code = code
            # Fail-fast: stop subsequent stages
            break

    # Manifest
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": asdict(master.env),
        "log": asdict(master.log),
        "stages_planned": [p["stage"] for p in plan],
        "stages_run": results,
        "overall_exit_code": overall_code,
        "pointers": pointers, 
    }
    manifest_path = run_dir / "run_manifest.json"
    safe_json_dump(manifest, manifest_path)

    log.info("orchestrator.done", extra={"run_id": run_id, "overall_exit_code": overall_code, "manifest": str(manifest_path)})
    return overall_code
