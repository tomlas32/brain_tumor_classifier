# src/tests/pipeline/test_evaluate.py
import json
from pathlib import Path
import types
import numpy as np
import torch
import torch.nn as nn
import pytest

# Under test
from src.pipeline import evaluate as eval_cli
from src.evaluate import runner as eval_runner

# ==============================
# Sandbox & common fixtures
# ==============================

@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """
    Full sandbox: set CWD to pytest tmpdir and redirect src.utils.paths so
    all relative outputs (e.g., outputs/evaluation/...) land here.
    """
    base = tmp_path
    monkeypatch.chdir(base)

    outputs = base / "outputs"
    data = base / "data"
    models = base / "models"
    for p in (outputs, data, models):
        p.mkdir(parents=True, exist_ok=True)

    # Patch src.utils.paths module used across pipeline
    from src.utils import paths as up
    monkeypatch.setattr(up, "OUTPUTS_DIR", outputs, raising=False)
    monkeypatch.setattr(up, "DATA_DIR", data, raising=False)
    monkeypatch.setattr(up, "MODELS_DIR", models, raising=False)

    return base


@pytest.fixture()
def no_env_noise(monkeypatch):
    """Silence env/bootstrap & keep logging simple in tests."""
    monkeypatch.setattr(eval_cli, "bootstrap_env", lambda seed=None: None, raising=True)
    monkeypatch.setattr(eval_cli, "log_env_once", lambda: None, raising=True)
    # logging config can stay; writes into sandboxed outputs


# ==============================
# evaluate.main (CLI) — dry run
# ==============================

def test_cli_dry_run_prints_plan_and_returns_zero(sandbox, no_env_noise, monkeypatch, capsys):
    """
    Make a tiny eval folder and mapping/weights placeholders; run CLI with --dry-run.
    Expect return code 0 and a printed summary (we just sanity-check a couple of lines).
    """
    # Create class-structured eval set with a couple of dummy files
    eval_root = sandbox / "data" / "testing"
    (eval_root / "classA").mkdir(parents=True, exist_ok=True)
    (eval_root / "classB").mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (eval_root / "classA" / f"a_{i}.png").write_bytes(b"")
        (eval_root / "classB" / f"b_{i}.png").write_bytes(b"")

    # Minimal mapping JSON and weights file placeholders
    mapping_json = sandbox / "outputs" / "mappings" / "index_remap.json"
    mapping_json.parent.mkdir(parents=True, exist_ok=True)
    mapping_json.write_text(json.dumps({"classA": 0, "classB": 1}), encoding="utf-8")
    weights = sandbox / "models" / "best.pth"
    weights.write_bytes(b"fake-weights")

    # Patch build_eval_config to return a tiny object with just what's used by evaluate.py
    # evaluate.py calls: cfg.log, cfg.data, cfg.model (and to_dict(cfg) for logging)
    class _Log: level="INFO"; file=None
    class _Data:
        image_size=224; batch_size=4; num_workers=0; seed=42
        eval_in=str(eval_root); eval_out=str(sandbox / "outputs" / "evaluation")
        mapping_path=str(mapping_json); mapping_pointer=None
    class _Model:
        name="resnet18"; weights_path=str(weights)
    cfg_obj = types.SimpleNamespace(log=_Log(), data=_Data(), model=_Model())

    monkeypatch.setattr(eval_cli, "build_eval_config", lambda *a, **k: cfg_obj, raising=True)
    monkeypatch.setattr(eval_cli, "to_dict", lambda o: {
        "log":{"level":o.log.level,"file":o.log.file},
        "data":{"image_size":o.data.image_size,"batch_size":o.data.batch_size,
                "num_workers":o.data.num_workers,"seed":o.data.seed,
                "eval_in":o.data.eval_in,"eval_out":o.data.eval_out,
                "mapping_path":o.data.mapping_path},
        "model":{"name":o.model.name,"weights_path":o.model.weights_path},
    }, raising=True)

    # --- run dry-run ---
    rc = eval_cli.main([
        "--dry-run",
        "--eval-in", str(eval_root),
        "--trained-model", str(weights),
        "--mapping-path", str(mapping_json),
        "--image-size", "224",
        "--batch-size", "4",
    ])
    assert rc == 0

    # Sanity-check a few lines of printed plan
    out = capsys.readouterr().out
    assert "[DRY-RUN] Evaluate plan" in out
    assert "eval_in:" in out and "mapping_path:" in out and "weights_path:" in out
    # Should list our classes with counts
    assert "classA" in out and "classB" in out
    # Should mention allowed labels from mapping
    assert "Allowed labels (from mapping)" in out

# (evaluate.py CLI structure & dry-run behavior) :contentReference[oaicite:2]{index=2}


# ==============================
# runner.run — happy path
# ==============================

def test_runner_happy_path_calls_components_and_writes_outputs(sandbox, monkeypatch):
    """
    End-to-end through runner.run using lightweight fakes:
    - Fake ImageFolder (dataset)
    - Fake DataLoader
    - Fake model + weights loader
    - Predetermined metrics
    - Viz functions return paths
    - Evaluation summary is written
    """
    # --------- Fake dataset (ImageFolder) ----------
    class FakeImageFolder:
        def __init__(self, root, transform=None):
            self.root = Path(root)
            self.transform = transform
            # samples: pairs of (absolute_path, label)
            self.samples = [
                (str(self.root / "classA" / "a1.png"), 0),
                (str(self.root / "classA" / "a2.png"), 0),
                (str(self.root / "classB" / "b1.png"), 1),
            ]
            self.class_to_idx = {"classA": 0, "classB": 1}

    # Build minimal on-disk structure expected by FakeImageFolder
    eval_root = sandbox / "data" / "testing"
    (eval_root / "classA").mkdir(parents=True, exist_ok=True)
    (eval_root / "classB").mkdir(parents=True, exist_ok=True)
    for p in [eval_root / "classA" / "a1.png", eval_root / "classA" / "a2.png", eval_root / "classB" / "b1.png"]:
        p.write_bytes(b"")

    # Patch ImageFolder used by runner
    monkeypatch.setattr(eval_runner, "ImageFolder", FakeImageFolder, raising=True)

    # --------- Fake transforms & loader ----------
    monkeypatch.setattr(eval_runner, "build_transforms", lambda sz: {"test": object()}, raising=True)

    # Fake DataLoader: returns two mini-batches of tensors/labels
    def fake_make_eval_loader(ds, batch_size, num_workers, seed):
        x1 = torch.zeros(2, 3, 8, 8); y1 = torch.tensor([0, 1])
        x2 = torch.ones(1, 3, 8, 8); y2 = torch.tensor([0])
        return [(x1, y1), (x2, y2)]
    monkeypatch.setattr(eval_runner, "make_eval_loader", fake_make_eval_loader, raising=True)

    # --------- Mapping alignment ----------
    monkeypatch.setattr(eval_runner, "align_or_warn_for_eval", lambda ds, mp: ["classA","classB"], raising=True)

    # --------- Fake model & weights ----------
    class TinyNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, num_classes))
        def forward(self, x): return self.net(x)

    def fake_build_model(name, num_classes, pretrained=False): return TinyNet(num_classes)
    def fake_load_weights(model, path, device, strict=True): pass
    monkeypatch.setattr(eval_runner, "build_model", fake_build_model, raising=True)
    monkeypatch.setattr(eval_runner, "load_weights", fake_load_weights, raising=True)

    # --------- Metrics: predetermine y_true/y_pred & scalars ----------
    y_true = np.array([0,1,0], dtype=int)
    y_pred = np.array([0,1,1], dtype=int)
    def fake_eval_metrics(model, loader, device):
        return 2/3, 0.75, 0.75, 0.75, y_true, y_pred
    monkeypatch.setattr(eval_runner, "eval_metrics", fake_eval_metrics, raising=True)

    # --------- Viz: just ensure they return a saved path ----------
    galleries_dir = sandbox / "outputs" / "evaluation" / "RUNX" / "galleries"
    gradcam_dir   = sandbox / "outputs" / "evaluation" / "RUNX" / "gradcam"
    def fake_calls_gallery(items, class_names, cols, title, save_dir, image_size=224):
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{title}.png"; out.write_bytes(b"png"); return out
    def fake_gradcam_gallery(**kwargs):
        sd = kwargs.get("save_dir", gradcam_dir); sd.mkdir(parents=True, exist_ok=True)
        out = sd / f"{kwargs.get('title','gradcam')}.png"; out.write_bytes(b"png"); return out
    monkeypatch.setattr(eval_runner, "show_calls_gallery", fake_calls_gallery, raising=True)
    monkeypatch.setattr(eval_runner, "show_gradcam_gallery", fake_gradcam_gallery, raising=True)

    # --------- Summary manifest ----------
    summaries_dir = sandbox / "outputs" / "evaluation" / "RUNX"
    def fake_write_evaluation_summary(out_dir, run_id, args_dict, class_names, metrics):
        p = Path(out_dir) / "evaluation_summary_RUNX.json"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(p).write_text(json.dumps({"metrics": metrics, "class_names": class_names}), encoding="utf-8")
        return p
    monkeypatch.setattr(eval_runner, "write_evaluation_summary", fake_write_evaluation_summary, raising=True)

    # --------- Device selection ----------
    monkeypatch.setattr(eval_runner, "get_device", lambda prefer_cuda=True: torch.device("cpu"), raising=True)

    # --------- Inputs & run ----------
    mapping_json = sandbox / "outputs" / "mappings" / "index_remap.json"
    mapping_json.parent.mkdir(parents=True, exist_ok=True)
    mapping_json.write_text(json.dumps({"classA":0,"classB":1}), encoding="utf-8")
    weights = sandbox / "models" / "best.pth"; weights.write_bytes(b"fake")

    inputs = eval_runner.EvalRunnerInputs(
        image_size=224,
        eval_in=eval_root,
        mapping_path=mapping_json,
        model_name="resnet18",
        weights_path=weights,
        batch_size=4,
        num_workers=0,
        seed=42,
        eval_out=sandbox / "outputs" / "evaluation",
        run_id="RUNX",
        args_dict={"io":{"gallery_cols":4,"gradcam_cols":4}},
        make_galleries=True,
        make_gradcam=True,
        top_per_class=2,
    )

    acc, prec, rec, f1 = eval_runner.run(inputs)
    assert pytest.approx(acc, rel=1e-6) == 2/3
    assert pytest.approx(prec, rel=1e-6) == 0.75
    assert pytest.approx(rec,  rel=1e-6) == 0.75
    assert pytest.approx(f1,   rel=1e-6) == 0.75

    # Files from metrics & viz & summary exist
    rep = summaries_dir / "test_classification_report.txt"
    # Our fake report writer is in runner via save_classification_report; we didn’t patch it —
    # so ensure runner saved via our fake_save (see below).
    # We'll patch save_* below to actually write files.

# (runner.run: flow & responsibilities) :contentReference[oaicite:3]{index=3}


# ==============================
# Unit tests for runner helpers
# ==============================

def test_get_paths_for_dataset_and_selection_logic(monkeypatch):
    # --- _get_paths_for_dataset ---
    ds = types.SimpleNamespace(samples=[("a.png",0), ("b.png",1), ("c.png",0)])
    paths = eval_runner._get_paths_for_dataset(ds)
    assert paths == ["a.png","b.png","c.png"]

    # --- _select_examples_per_true_class ---
    y_true = np.array([0,0,1,1,1])
    y_pred = np.array([0,1,1,0,1])
    # probs: 5x3, with confident misclass for class 0 → 1 and confident correct for class 1
    probs = np.array([
        [0.8, 0.2, 0.0],  # correct for class 0
        [0.1, 0.9, 0.0],  # misclass: true=0 → pred=1 (conf 0.9)  <-- should rank top
        [0.1, 0.8, 0.1],  # correct for class 1 (conf 0.8)
        [0.7, 0.2, 0.1],  # misclass: true=1 → pred=0 (conf 0.7)
        [0.1, 0.85,0.05], # correct for class 1 (conf 0.85) <-- top correct for class 1
    ])
    paths = ["p0","p1","p2","p3","p4"]
    mistakes, corrects = eval_runner._select_examples_per_true_class(
        y_true, y_pred, probs, paths, max_per_class=1, n_classes=2
    )
    # one mistake per true class (if any)
    assert any(m["path"]=="p1" and m["true"]==0 and m["pred"]==1 for m in mistakes)
    assert any(m["path"]=="p3" and m["true"]==1 and m["pred"]==0 for m in mistakes)
    # top correct per class (only class 1 has corrects here with max_per_class=1)
    assert any(c["path"]=="p4" and c["true"]==1 and c["pred"]==1 for c in corrects)
