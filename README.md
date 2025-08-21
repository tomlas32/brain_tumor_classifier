# 🧠 Brain Tumor MRI Classification Pipeline

This repository implements a **modular, config-driven ML pipeline** for early detection of brain tumors from MRI scans.  
The design emphasizes **reproducibility**, **artifact consistency**, and **extensibility** in both research and production.

---

## ✨ Features

- **End-to-end pipeline**: `fetch → merge → validate(pre) → cleanup → resize → validate(post) → split → train → evaluate`
- **Artifact pointers**: standardized JSON pointers for fetch & mapping stages, ensuring consistent downstream consumption.
- **Config-first runs**: all stages run from YAML configs; CLI is a thin wrapper.
- **Reproducibility**: automatic run IDs, persisted `plan.json`, final run manifest, per-stage logs.
- **Extensible core**: modularized `core/` includes transforms, metrics, mapping, env bootstrap, artifact helpers, and visualization.
- **Structured logging**: consistent key/value logs across stages.
- **Callbacks**: early stopping, checkpoint saving (best/last), LR logging with JSON history.
- **Visualization**: Grad-CAM overlays and prediction galleries centralized in `core/viz.py`.
- **CI-ready**: Pytest suite with smoke tests, conftest log suppression, and GitHub Actions workflow.

---

## 📂 Project Structure

```
src/
  core/
    config.py         # typed dataclasses for all configs
    transforms.py     # training/validation/test transforms
    metrics.py        # accuracy, precision, recall, F1, confusion
    mapping.py        # index remap utils
    env.py            # bootstrap_env + log_env_once
    artifacts.py      # summary writing, pointers
    viz.py            # Grad-CAM + galleries
    callbacks.py      # early stopping, checkpoint, LR logging
    cleanup_policy.py # quarantine policy
    data.py           # deterministic workers, ImageFolderDataModule
    model.py          # model architecture, loading, saving
  pipeline/
    orchestrator.py   # run all stages from master config
    fetch.py
    merge.py
    cleanup.py
    split.py
    split_planner.py
    resize.py
    resize_planner.py
    train.py
    validate.py
    evaluate.py
  train/
    runner.py         # training loop w/ callbacks
  evaluate/
    runner.py         # evaluation loop
  utils/
    logging_utils.py
    paths.py
    configs.py
    parser_utils.py
  cli.py              # Typer CLI
configs/
  fetch.yaml
  merge.yaml
  validate_pre.yaml
  cleanup.yaml
  resize.yaml
  validate_post.yaml
  split.yaml
  train.yaml
  eval.yaml
  pipeline.yaml
  pipeline_minimal.yaml
outputs/
models/               # checkpoints (best/last, per naming template)
.github/workflows/
  ci.yml              # GitHub Actions workflow (pytest, artifacts)
tests/
  test_pipeline_smoke.py
  test_callback_and_runner.py
  test_viz.py
  pytest.ini
  conftest.py
```

---

## 🚀 Getting Started

### 1) Install dependencies
```bash
pip install -r requirements_dev.txt
```

### 2) Authentication (Kaggle)


### 3) Run the full pipeline
```bash
python -m src.cli pipeline --config src/configs/pipeline.yaml
```

### 4) Inspect outputs
- `outputs/orchestrator/<run_id>/plan.json` — planned stages, argv, pointer locations  
- `outputs/orchestrator/<run_id>/run_manifest.json` — per-stage exit codes, durations, pointers  
- `outputs/pointers/` — canonical **fetch** and **mapping** pointers  
- `models/` — saved checkpoints  
- `outputs/` — training/evaluation summaries, validation reports

---

## ⚙️ CLI Commands & Flags

Each stage is runnable on its own (use `--config` where available):

```bash
# Fetch dataset (e.g. Kaggle or local pointer)
python -m src.cli fetch --config src/configs/fetch.yaml

# Merge multiple sources into a unified dataset
python -m src.cli merge --config src/configs/merge.yaml

# Pre-validation (orchestrator sets report_tag=pre)
python -m src.cli validate --config src/configs/validate.yaml

# Cleanup (optional quarantine of duplicates or corrupt files)
python -m src.cli cleanup --config src/configs/cleanup.yaml

# Resize & normalize dataset
python -m src.cli resize --config src/configs/resize.yaml

# Post-validation (orchestrator sets report_tag=post)
python -m src.cli validate --config src/configs/validate.yaml

# Train model
python -m src.cli train --config src/configs/train.yaml

# Evaluate model + Grad-CAM visualizations
python -m src.cli evaluate --config src/configs/eval.yaml

# Full end-to-end pipeline
python -m src.cli pipeline --config src/configs/pipeline.yaml
```

**Common flags**

- `--config PATH` — YAML config for the stage (or master config for `pipeline`)
- `--override key=value` / `-o key=value` — override YAML fields (e.g., `train.loop.epochs=5`)
- `--dry-run` (pipeline) — write `plan.json` and exit
- `--skip STAGE` (pipeline) — skip one or more stages (valid: fetch, split, resize, validate, train, evaluate)
- `--resume-from STAGE` (pipeline) — start from a specific stage

---

## 📑 Artifacts & Pointers (Single Source of Truth)

### Fetch Pointer

**Path:** `outputs/pointers/fetch/<owner>/<slug>/latest.json`  
**Schema (minimal):**
```json
{
  "dataset": "owner/slug",
  "dataset_root": "path/to/dataset/root",
  "training_dir": "path/to/Training or null",
  "testing_dir": "path/to/Testing or null",
  "version": null,
  "run_id": "bt-exp-001",
  "fetched_at_utc": "2025-08-16T19:41:00Z"
}
```

### Mapping Pointer

**Path:** `outputs/pointers/mapping/<owner>/<slug>/latest.json`  
**Schema (minimal):**
```json
{
  "classes": ["glioma", "meningioma", "pituitary", "notumor"],
  "num_classes": 4,
  "index_remap": null,
  "path": "outputs/mappings/.../index_remap.json",
  "dataset": "owner/slug",
  "run_id": "bt-exp-001",
  "written_at_utc": "2025-08-16T19:45:00Z"
}
```

Both pointers also write a timestamped copy to a `history/` subfolder for auditability. Downstream stages **prefer pointers** and only fall back to raw paths if needed.

---

## 🧪 Testing & CI

Run smoke tests (includes orchestrator dry-run, mocked full run, and CLI checks):

```bash
pytest -q
```

Continuous Integration: GitHub Actions workflow runs tests across Python 3.10 and 3.11, uploads logs & outputs as artifacts.

---

## 📊 Example Workflow

1. **Fetch** dataset  
   ```bash
   python -m src.cli fetch --dataset sartajbhuvaji/brain-tumor-classification-mri
   ```
2. **Merge** into train/test (reads the **fetch** pointer)  
   ```bash
   python -m src.cli merge --config src/configs/merge.yaml
   ```
3. **Pre-validate** images (orchestrator sets **report_tag=pre**) 
   ```bash
   python -m src.cli validate --config src/configs/validate.yaml
   ```
4. **Cleanup** duplicates/corrupt files (quarantine as needed) 
   ```bash
   python -m src.cli cleanup --config src/configs/cleanup.yaml
   ```
5. **Resize** & normalize images  
   ```bash
   python -m src.cli resize --config src/configs/resize.yaml
   ```
6. **Post-validate** dataset (orchestrator sets **report_tag=post**)
   ```bash
   python -m src.cli evaluate --config src/configs/eval.yaml
   ```
7. **Split** into training & validation sets (creates mapping pointer)
   ```bash
   python -m src.cli evaluate --config src/configs/eval.yaml
   ```
8. **Train** (uses **mapping pointer**; callbacks active if enabled in config)
   ```bash
   python -m src.cli evaluate --config src/configs/eval.yaml
   ```
9. **Evaluate** model (uses **mapping pointer**; generates reports & Grad-CAM galleries via core/viz.py)
   ```bash
   python -m src.cli evaluate --config src/configs/eval.yaml
   ```

---

## 📚 Notes

- Configs are typed dataclasses (`src/core/config.py`).
- Callbacks configurable via `train.yaml` (`callbacks.early_stopping`, `callbacks.checkpoint`, `callbacks.lr_logger`).
- Visualization centralized in `core/viz.py`.
- Training & evaluation prefer **mapping pointers**; raw `index_remap.json` still supported.

---

## 📝 License

MIT License © 2025 — Contributions welcome.

Data was derived from [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)