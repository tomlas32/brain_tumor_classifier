"""
Path configuration module for the project.

- Dynamically resolves the project root directory by navigating two levels up 
  from the current file (utils/paths.py).
- Defines absolute paths for key project directories: data, models, outputs, configs, and notebooks.
- Ensures that critical directories (data, models, outputs) exist by creating them if necessary.

Usage:
    from utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CONFIGS_DIR, NOTEBOOKS

This centralizes path handling to avoid hardcoding and ensures consistent directory structure.


Author: Tomasz Lasota
Date: 2025-0-13
Version: 1.0

"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # <repo>
SRC_DIR      = PROJECT_ROOT / "src"                  

DATA_DIR    = Path(os.getenv("DATA_DIR",    PROJECT_ROOT / "data"))
MODELS_DIR  = Path(os.getenv("MODELS_DIR",  PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
CONFIGS_DIR = SRC_DIR / "configs"  
NOTEBOOKS   = PROJECT_ROOT / "notebooks"

# NEW: canonical data roots introduced by the new pipeline order
MERGED_DIR     = DATA_DIR / "merged"      # output of merge stage (class folders)
PROCESSED_DIR  = DATA_DIR / "processed"   # output of resize stage (class folders)

# validation reports are written by `validate` and consumed by `cleanup`:
VALIDATION_REPORTS_DIR = OUTPUTS_DIR / "validation_reports"

# Class mappings written by `split` and used by others:
MAPPINGS_DIR           = OUTPUTS_DIR / "mappings"

# cleanup own outputs (plan/manifest):
CLEANUP_REPORTS_DIR    = OUTPUTS_DIR / "cleanup_reports"

# quarantine destination root:
QUARANTINE_ROOT        = DATA_DIR / "quarantine"


DEFAULT_INDEX_REMAP = OUTPUTS_DIR / "mappings" / "latest.json"

def ensure_base_dirs():
    for p in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        p.mkdir(parents=True, exist_ok=True)