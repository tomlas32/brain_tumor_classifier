# src/tests/pipeline/test_train.py
from __future__ import annotations

import contextlib
import importlib
import os
import re
from io import StringIO
from pathlib import Path
import types

import pytest


def _import_train():
    train = importlib.import_module("src.pipeline.train")
    importlib.reload(train)
    return train


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _base_cfg(tmp_path: Path, with_val: bool = False, mapping_pointer=None, mapping_path=None, dry_run: bool = False):
    data_ns = _ns(
        train_in=str(tmp_path / "data" / "training_resized"),
        val_frac=0.2,
        batch_size=4,
        num_workers=0,
        seed=42,
        image_size=224,
        mapping_pointer=mapping_pointer,
        mapping_path=mapping_path,
    )
    loop_ns = _ns(epochs=3)
    model_ns = _ns(name="resnet18", pretrained=False)
    optim_ns = _ns(lr=1e-3, weight_decay=1e-4, step_size=1, gamma=0.9, amp=False)
    io_ns = _ns(out_models=str(tmp_path / "models"), out_summary=str(tmp_path / "outputs" / "training"))
    log_ns = _ns(level="INFO", file=None)

    cfg = _ns(
        data=data_ns,
        loop=loop_ns,
        model=model_ns,
        optim=optim_ns,
        io=io_ns,
        log=log_ns,
        run_id=None,
        dry_run=dry_run,
    )
    if with_val:
        setattr(cfg, "val_in", str(tmp_path / "data" / "validation_resized"))
    return cfg


def _run_main(train_mod, argv):
    return int(train_mod.main(argv))


def test_train_dry_run_plan_counts(tmp_path, monkeypatch, capsys):
    """
    Dry-run should print a plan with per-class counts and NOT call the runner.
    """
    train = _import_train()

    # Layout: training_resized/{meningioma,glioma}
    tr = tmp_path / "data" / "training_resized"
    (tr / "meningioma").mkdir(parents=True, exist_ok=True)
    (tr / "glioma").mkdir(parents=True, exist_ok=True)
    for name in ["a.jpg", "b.jpg"]:
        (tr / "meningioma" / name).write_bytes(b"x")
    (tr / "glioma" / "g1.jpg").write_bytes(b"x")

    idx_map = tmp_path / "outputs" / "mappings" / "latest.json"
    idx_map.parent.mkdir(parents=True, exist_ok=True)
    idx_map.write_text('{"0":"meningioma","1":"glioma"}', encoding="utf-8")

    cfg = _base_cfg(tmp_path, with_val=False, mapping_path=str(idx_map), dry_run=True)
    monkeypatch.setattr(train, "build_train_config", lambda *a, **k: cfg, raising=True)
    monkeypatch.setattr(train, "to_dict", lambda c: {"data": {}, "loop": {}, "model": {}, "optim": {}}, raising=True)
    monkeypatch.setattr(train, "configure_logging", lambda **kw: None, raising=True)
    monkeypatch.setattr(train, "bootstrap_env", lambda seed=None: None, raising=True)
    monkeypatch.setattr(train, "log_env_once", lambda: None, raising=True)

    called = {"v": False}
    def _no_run(*a, **k):
        called["v"] = True
        return 0.0, 0, tmp_path / "models" / "best.pth"
    monkeypatch.setattr(train, "run_training", _no_run, raising=True)

    monkeypatch.setenv("RUN_ID", "TST_RUN")

    # No manual stdout redirection; rely on capsys (more robust with your logging)
    code = _run_main(train, ["--config", str(tmp_path / "cfg.yaml"), "--dry-run"])
    s = capsys.readouterr().out

    assert code == 0
    assert "[DRY-RUN] Train plan" in s
    men = re.search(r"meningioma\s*->\s*(\d+)", s)
    gli = re.search(r"glioma\s*->\s*(\d+)", s)
    assert men and gli
    assert int(men.group(1)) == 2
    assert int(gli.group(1)) == 1
    assert "No training will be started" in s
    assert called["v"] is False


def test_train_mapping_pointer_error_returns_exit2(tmp_path, monkeypatch):
    """
    When a mapping pointer is provided but cannot be read, exit code should be 2.
    """
    train = _import_train()

    cfg = _base_cfg(tmp_path, mapping_pointer=str(tmp_path / "pointer_dir"), dry_run=False)
    monkeypatch.setattr(train, "build_train_config", lambda *a, **k: cfg, raising=True)
    monkeypatch.setattr(train, "to_dict", lambda c: {}, raising=True)
    monkeypatch.setattr(train, "configure_logging", lambda **kw: None, raising=True)
    monkeypatch.setattr(train, "bootstrap_env", lambda seed=None: None, raising=True)
    monkeypatch.setattr(train, "log_env_once", lambda: None, raising=True)

    monkeypatch.setattr(train, "read_mapping_pointer", lambda p: (_ for _ in ()).throw(RuntimeError("bad ptr")), raising=True)
    monkeypatch.setattr(train, "run_training", lambda *a, **k: (_ for _ in ()).throw(AssertionError("runner should not be called")), raising=True)

    code = _run_main(train, ["--config", str(tmp_path / "cfg.yaml")])
    assert code == 2


def test_train_missing_mapping_path_returns_exit2(tmp_path, monkeypatch):
    """
    If neither mapping_pointer nor mapping_path are provided, return exit code 2.
    """
    train = _import_train()

    cfg = _base_cfg(tmp_path, mapping_pointer=None, mapping_path=None, dry_run=False)
    monkeypatch.setattr(train, "build_train_config", lambda *a, **k: cfg, raising=True)
    monkeypatch.setattr(train, "to_dict", lambda c: {}, raising=True)
    monkeypatch.setattr(train, "configure_logging", lambda **kw: None, raising=True)
    monkeypatch.setattr(train, "bootstrap_env", lambda seed=None: None, raising=True)
    monkeypatch.setattr(train, "log_env_once", lambda: None, raising=True)

    monkeypatch.setattr(train, "run_training", lambda *a, **k: (_ for _ in ()).throw(AssertionError("runner should not be called")), raising=True)

    code = _run_main(train, ["--config", str(tmp_path / "cfg.yaml")])
    assert code == 2


def test_train_success_invokes_runner_and_prints(tmp_path, monkeypatch, capsys):
    """
    Happy path: mapping pointer resolves to an index_remap path, runner invoked,
    and stdout contains completion lines.
    """
    train = _import_train()

    tr = tmp_path / "data" / "training_resized"
    tr.mkdir(parents=True, exist_ok=True)

    idx_map = tmp_path / "outputs" / "mappings" / "latest.json"
    idx_map.parent.mkdir(parents=True, exist_ok=True)
    idx_map.write_text('{"0":"class0","1":"class1"}', encoding="utf-8")

    cfg = _base_cfg(tmp_path, mapping_pointer=str(tmp_path / "ptr_dir"), mapping_path=None, dry_run=False)
    monkeypatch.setattr(train, "build_train_config", lambda *a, **k: cfg, raising=True)
    monkeypatch.setattr(train, "to_dict", lambda c: {"data": {}, "env": {"prefer_cuda": False}}, raising=True)
    monkeypatch.setattr(train, "configure_logging", lambda **kw: None, raising=True)
    monkeypatch.setattr(train, "bootstrap_env", lambda seed=None: None, raising=True)
    monkeypatch.setattr(train, "log_env_once", lambda: None, raising=True)

    monkeypatch.setattr(train, "read_mapping_pointer", lambda p: {"path": str(idx_map), "num_classes": 2}, raising=True)

    called = {"inputs": None}
    def _fake_run(inputs):
        called["inputs"] = inputs
        best = 0.9123
        ckpt = Path(cfg.io.out_models) / (os.getenv("RUN_ID") or "RUNX") / "best_valF1_0.9123_epoch3.pth"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"ckpt")
        return best, 3, ckpt

    monkeypatch.setattr(train, "run_training", _fake_run, raising=True)
    monkeypatch.setenv("RUN_ID", "TST_RUN")

    code = _run_main(train, ["--config", str(tmp_path / "cfg.yaml")])
    out = capsys.readouterr().out

    assert code == 0
    assert "✅ Training complete." in out
    assert "📦 Best checkpoint:" in out

    assert called["inputs"] is not None
    inp = called["inputs"]
    assert inp.image_size == 224
    assert inp.train_in == Path(cfg.data.train_in)
    assert str(inp.index_remap) == str(idx_map)

    run_models = Path(cfg.io.out_models) / "TST_RUN"
    assert run_models.exists()
