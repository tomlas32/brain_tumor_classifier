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

# --- Canonical subdirectories (shared across stages) ---
# validation reports are written by `validate` and consumed by `cleanup`:
VALIDATION_REPORTS_DIR = OUTPUTS_DIR / "validation_reports"
# (optional but handy) class mappings written by `split` and used by others:
MAPPINGS_DIR           = OUTPUTS_DIR / "mappings"
# cleanup own outputs (plan/manifest). Okay to centralize as a path constant:
CLEANUP_REPORTS_DIR    = OUTPUTS_DIR / "cleanup_reports"
# quarantine destination root (kept here to avoid hardcoding in multiple places):
QUARANTINE_ROOT        = DATA_DIR / "quarantine"

def ensure_base_dirs():
    for p in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        p.mkdir(parents=True, exist_ok=True)