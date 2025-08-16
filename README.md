# 🧠 Brain Tumor MRI Classification Pipeline

This repository implements a **modular, config-driven ML pipeline** for early detection of brain tumors from MRI scans. The design emphasizes **reproducibility**, **artifact consistency**, and **ease of use** in both research and production settings.

---

## ✨ Features

- **End-to-end pipeline**: `fetch → split → resize → validate → train → evaluate`
- **Artifact pointers**: standardized JSON pointers for fetch & mapping stages, ensuring consistent downstream consumption.
- **Config-first runs**: each stage can be controlled via YAML configs; CLI is a thin wrapper.
- **Reproducibility**: automatic run IDs, a persisted plan (`plan.json`), a final run manifest, and per-stage logs.
- **Extensible core**: modularized `core/` includes transforms, metrics, mapping utilities, environment bootstrap, and artifact helpers.
- **Structured logging**: consistent keys across stages to make tracing easy in both console and files.
- **Optional visualization**: Grad-CAM overlays and prediction galleries during evaluation.

---

## 📂 Project Structure

```
src/
  core/               # config, transforms, metrics, mapping, env, artifacts, (viz optional)
  pipeline/
    orchestrator.py   # runs fetch→…→evaluate from one master config; writes plan & manifest
    fetch.py
    split.py
    resize.py
    validate.py
  train/
    runner.py         # training loop orchestration
  evaluate/
    runner.py         # evaluation orchestration
  utils/
    logging_utils.py  # structured logging
    paths.py          # project paths and helpers
  cli.py              # Typer CLI: fetch/split/resize/validate/train/evaluate/pipeline
configs/
  train.yaml
  eval.yaml
  validate.yaml
  pipeline.yaml
outputs/
  orchestrator/<run_id>/{plan.json, run_manifest.json, stage sub-configs}
  pointers/
    fetch/<owner>/<slug>/{latest.json, history/...}
    mapping/<owner>/<slug>/{latest.json, history/...}
models/               # checkpoints (e.g., best.pth)
```

---

## 🚀 Getting Started

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Authentication (Kaggle)
Create `~/.kaggle/kaggle.json` with your Kaggle API credentials and set permissions to `0600`.

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

Both pointers also write a timestamped copy to a `history/` subfolder for auditability. Downstream stages **prefer pointers** (dir or file) and fall back to raw paths only if needed.

---

## 🧪 Testing

Run smoke tests (includes orchestrator dry-run, mocked full run, and CLI checks):

```bash
pytest -q
```

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
5. **Train** (uses **mapping** pointer)  
   ```bash
   python -m src.cli train --config configs/train.yaml
   ```
6. **Evaluate** (uses **mapping** pointer; can create galleries/Grad-CAM)  
   ```bash
   python -m src.cli evaluate --config configs/eval.yaml
   ```

---

## 🧷 Example `configs/pipeline.yaml`

```yaml
run_id: bt-exp-001

env:
  seed: 42
  prefer_cuda: true
  cudnn_deterministic: true
  cudnn_benchmark: false

log:
  level: INFO
  file: null
  json: false

fetch:
  dataset: sartajbhuvaji/brain-tumor-classification-mri
  cache_dir: data
  write_pointer: true

split:
  dataset: sartajbhuvaji/brain-tumor-classification-mri
  test_frac: 0.25
  clear_dest: true
  exts: [".jpg", ".jpeg", ".png"]

resize:
  in_dir: data/combined_split_simple
  out_dir: data/training_resized
  size: 224
  exts: all

validate:
  in_dir: data/training_resized
  mapping_pointer: outputs/pointers/mapping/sartajbhuvaji/brain-tumor-classification-mri
  size: 224
  fail_on: warning
  warn_low_std: 3.0
  min_file_bytes: 1024
  dup_check: false

train:
  data:
    image_size: 224
    train_in: data/training_resized
    mapping_pointer: outputs/pointers/mapping/sartajbhuvaji/brain-tumor-classification-mri
    batch_size: 32
    num_workers: 4
    val_frac: 0.2
    seed: 42
  model:
    name: resnet18
    pretrained: true
  optim:
    lr: 0.001
    weight_decay: 0.0001
    step_size: 10
    gamma: 0.1
    amp: true
  io:
    out_models: models
    out_summary: outputs/training
  loop:
    epochs: 20
  aug:
    rotate_deg: 10
    hflip_prob: 0.5
    jitter_brightness: 0.05
    jitter_contrast: 0.05

evaluate:
  data:
    image_size: 224
    eval_in: data/testing_resized
    mapping_pointer: outputs/pointers/mapping/sartajbhuvaji/brain-tumor-classification-mri
    batch_size: 32
    num_workers: 4
    seed: 42
  model:
    name: resnet18
    weights_path: models/best.pth
  io:
    eval_out: outputs/evaluation
    make_galleries: true
    make_gradcam: true
    top_per_class: 6
```

---

## 📚 Notes

- Configs are typed dataclasses (see `src/core/config.py`).
- All runs log environment info at start (`bootstrap_env`, `log_env_once`) for reproducibility.
- Training & evaluation **prefer mapping pointers**; raw `index_remap.json` paths remain supported for back-compat.

---

## 📝 License

MIT License © 2025 — Contributions welcome.
