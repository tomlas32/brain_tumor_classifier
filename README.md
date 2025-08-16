# 🧠 Brain Tumor MRI Classification Pipeline

This repository implements a **modular, config-driven ML pipeline** for early detection of brain tumors from MRI scans.  
The design emphasizes **reproducibility**, **artifact consistency**, and **extensibility** in both research and production.

---

## ✨ Features

- **End-to-end pipeline**: `fetch → split → resize → validate → train → evaluate`
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
  pipeline/
    orchestrator.py   # run all stages from master config
    fetch.py
    split.py
    resize.py
    validate.py
  train/
    runner.py         # training loop w/ callbacks
  evaluate/
    runner.py         # evaluation loop
  utils/
    logging_utils.py
    paths.py
    configs.py
    parser_utils.py
  cli.py              # Typer CLI: fetch/split/resize/validate/train/evaluate/pipeline
configs/
  train.yaml
  eval.yaml
  resize.yaml
  split.yaml
  validate.yaml
  pipeline.yaml
  pipeline_minimal.yaml
outputs/
  orchestrator/<run_id>/{plan.json, run_manifest.json, stage-sub-configs}
  pointers/
    fetch/<owner>/<slug>/{latest.json, history/...}
    mapping/<owner>/<slug>/{latest.json, history/...}
models/               # checkpoints (best/last, per naming template)
.github/workflows/
  ci.yml              # GitHub Actions workflow (pytest, artifacts)
tests/
  test_pipeline_smoke.py
  test_viz.py
pytest.ini
```

---

## 🚀 Getting Started

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Authentication (Kaggle)
Create `~/.kaggle/kaggle.json` with your Kaggle API credentials (chmod 600).

### 3) Run the full pipeline
```bash
python -m src.cli pipeline --config configs/pipeline.yaml
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
python -m src.cli fetch --dataset owner/slug
python -m src.cli split --config configs/split.yaml
python -m src.cli resize --config configs/resize.yaml
python -m src.cli validate --config configs/validate.yaml
python -m src.cli train --config configs/train.yaml
python -m src.cli evaluate --config configs/eval.yaml
python -m src.cli pipeline --config configs/pipeline.yaml
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
2. **Split** into train/test (reads the **fetch** pointer)  
   ```bash
   python -m src.cli split --config configs/split.yaml
   ```
3. **Resize** images  
   ```bash
   python -m src.cli resize --config configs/resize.yaml
   ```
4. **Validate** dataset (uses **mapping** pointer or `index_remap.json`)  
   ```bash
   python -m src.cli validate --config configs/validate.yaml
   ```
5. **Train** (uses **mapping** pointer; callbacks active if enabled in config)  
   ```bash
   python -m src.cli train --config configs/train.yaml
   ```
6. **Evaluate** (uses **mapping** pointer; galleries/Grad-CAM saved via `core/viz.py`)  
   ```bash
   python -m src.cli evaluate --config configs/eval.yaml
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