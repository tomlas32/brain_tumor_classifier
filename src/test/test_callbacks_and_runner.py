# tests/test_callbacks_and_runner.py
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logging_utils import get_logger
import pytest

# Import the training runner inputs + run()
from src.train.runner import TrainRunnerInputs, run as run_training


@pytest.fixture
def tiny_dataset():
    # 8 samples, 2 classes (labels 0/1)
    X = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    return TensorDataset(X, y)


@pytest.fixture
def monkeypatch_data(monkeypatch, tiny_dataset):
    """
    Patch ImageFolder → returns an object with .classes and .samples (like ImageFolder),
    but we won't use its internal image loading.
    Patch make_loader → DataLoader over our tiny tensor dataset (train/val both).
    """
    # Fake ImageFolder with 'classes' and 'samples'
    class FakeImageFolder:
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.samples = [(f"fake_{i}.png", int(l.item())) for i, (_, l) in enumerate(tiny_dataset)]
            self.classes = ["class0", "class1"]

    monkeypatch.setattr("src.training.runner.ImageFolder", FakeImageFolder)

    # make_loader: just wrap our tensor dataset regardless of subset
    def fake_make_loader(subset_or_ds, batch_size, shuffle, num_workers, seed):
        # We ignore subset indices here; tiny_dataset is fine for both
        return DataLoader(tiny_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    monkeypatch.setattr("src.training.runner.make_loader", fake_make_loader)


@pytest.fixture
def monkeypatch_mapping(monkeypatch, tmp_path):
    """
    Patch mapping utils so we don't rely on real mapping files.
    """
    # expected classes
    def fake_read_index_remap(_):
        return {"0": "class0", "1": "class1"}

    def fake_expected_classes_from_remap(d):
        return [d["0"], d["1"]]

    def fake_verify_dataset_classes(ds_classes, expected_classes, strict=True):
        assert ds_classes == expected_classes
        return True, None

    def fake_default_index_remap_path():
        return tmp_path / "outputs" / "mappings" / "latest.json"

    def fake_copy_index_remap(src, dst_dir):
        # write a small file to simulate copy
        d = Path(dst_dir); d.mkdir(parents=True, exist_ok=True)
        p = d / "index_remap.json"
        p.write_text('{"0":"class0","1":"class1"}', encoding="utf-8")
        return p

    monkeypatch.setattr("src.training.runner.read_index_remap", fake_read_index_remap)
    monkeypatch.setattr("src.training.runner.expected_classes_from_remap", fake_expected_classes_from_remap)
    monkeypatch.setattr("src.training.runner.verify_dataset_classes", fake_verify_dataset_classes)
    monkeypatch.setattr("src.training.runner.default_index_remap_path", fake_default_index_remap_path)
    monkeypatch.setattr("src.training.runner.copy_index_remap", fake_copy_index_remap)


@pytest.fixture
def monkeypatch_metrics(monkeypatch):
    """
    Patch evaluate(model, loader, device) to return controlled metrics that
    make early stopping and checkpointing deterministic.
    """
    # Sequence of F1 values per epoch we will simulate
    # test_early_stopping: we let it improve twice then plateau/decline
    f1_seq = {"values": [0.50, 0.62, 0.64, 0.64, 0.63, 0.62]}

    def fake_evaluate(model, loader, device):
        # pop left-style without mutation complexity:
        idx = min(len(f1_seq["values"]) - 1, fake_evaluate._epoch)
        f1 = float(f1_seq["values"][idx])
        fake_evaluate._epoch += 1
        # acc/prec/rec not used for control, keep coherent
        return f1, f1, f1, f1, torch.zeros(8, dtype=torch.long).numpy(), torch.zeros(8, dtype=torch.long).numpy()

    fake_evaluate._epoch = 0
    monkeypatch.setattr("src.training.runner.evaluate", fake_evaluate)
    return f1_seq


def _base_inputs(tmp_path, args_dict_overrides=None):
    args_dict = {
        "aug": {},
        "env": {"prefer_cuda": False},
        "sched": {"name": "steplr", "params": {"step_size": 2, "gamma": 0.5}},
        "callbacks": {
            "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0, "monitor": "val_f1"},
            "checkpoint": {"save_best": True, "save_last": True, "every_n_epochs": 0},
            "lr_logger": {"enabled": True},
        },
        "data": {},
    }
    if args_dict_overrides:
        # shallow update ok for tests
        for k, v in args_dict_overrides.items():
            args_dict[k] = v

    return TrainRunnerInputs(
        image_size=224,
        train_in=tmp_path / "data" / "training_resized",
        batch_size=4,
        num_workers=0,
        val_frac=0.25,
        seed=42,
        model_name="resnet18",
        pretrained=False,
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        step_size=2,
        gamma=0.5,
        amp=False,
        out_models=tmp_path / "models",
        out_summary=tmp_path / "outputs" / "training",
        index_remap=None,
        run_id="t-run",
        args_dict=args_dict,
    )


def test_early_stopping_triggers(tmp_path, caplog, monkeypatch_data, monkeypatch_mapping, monkeypatch_metrics):
    """
    Expect: training stops early and logs `callback.early_stop`.
    """
    log = get_logger("src.training.runner")
    caplog.set_level("INFO", logger=log.name)

    inputs = _base_inputs(tmp_path)
    best_f1, best_epoch, ckpt_path = run_training(inputs)

    # Filesystem effects
    assert best_epoch < inputs.epochs, "Expected early stop before max epochs"
    assert ckpt_path.exists(), "Best checkpoint should exist"

    # Log assertions
    records = [r for r in caplog.records if r.name == log.name]
    assert any(r.message == "callback.early_stop" for r in records), \
        "Expected 'callback.early_stop' in logs"


def test_checkpoint_best_and_last(tmp_path, caplog, monkeypatch_data, monkeypatch_mapping, monkeypatch_metrics):
    """
    Expect: best checkpoint saved (dynamic name), last.pth saved each epoch,
    and both actions logged with `checkpoint.saved`.
    """
    log = get_logger("src.training.runner")
    caplog.set_level("INFO", logger=log.name)

    inputs = _base_inputs(tmp_path)
    _, _, ckpt_path = run_training(inputs)

    # Filesystem effects
    assert ckpt_path.exists(), "Best checkpoint missing"
    last = inputs.out_models / "last.pth"
    assert last.exists(), "last.pth should be saved (save_last=true)"
    assert (ckpt_path.parent / "index_remap.json").exists(), "index_remap.json should be copied next to checkpoint"

    # Log assertions
    records = [r for r in caplog.records if r.name == log.name and r.message == "checkpoint.saved"]
    assert len(records) >= 2, "Expected at least two 'checkpoint.saved' logs (best + last)"


def test_lr_logger_json_and_log(tmp_path, caplog, monkeypatch_data, monkeypatch_mapping, monkeypatch_metrics):
    """
    Expect: LR history JSON written and `lr_logger.written` logged.
    """
    log = get_logger("src.training.runner")
    caplog.set_level("INFO", logger=log.name)

    inputs = _base_inputs(tmp_path)
    _, best_epoch, _ = run_training(inputs)

    # Filesystem effects
    lr_hist_files = list((inputs.out_summary).glob("lr_history_*.json"))
    assert lr_hist_files, "LR history JSON not found"
    payload = json.loads(lr_hist_files[0].read_text(encoding="utf-8"))
    hist = payload["history"]
    assert len(hist) == best_epoch, "LR history length should match epochs actually run"
    assert {"epoch", "lr"}.issubset(hist[0].keys()), "LR history entries must include 'epoch' and 'lr'"

    # Log assertions
    assert any(r.name == log.name and r.message == "lr_logger.written" for r in caplog.records), \
        "Expected 'lr_logger.written' log"


def test_checkpoint_periodic(tmp_path, caplog, monkeypatch_data, monkeypatch_mapping, monkeypatch_metrics):
    """
    Expect: periodic checkpoints at epochs 2 and 4, and logs for each save.
    """
    log = get_logger("src.training.runner")
    caplog.set_level("INFO", logger=log.name)

    # Enable periodic snapshots every 2 epochs; disable ES to run full loop
    args_overrides = {
        "callbacks": {
            "early_stopping": {"enabled": False},
            "checkpoint": {"save_best": True, "save_last": False, "every_n_epochs": 2},
            "lr_logger": {"enabled": False},
        }
    }
    inputs = _base_inputs(tmp_path, args_overrides)
    inputs.epochs = 5  # → expect ckpt_epoch2 & ckpt_epoch4
    _ = run_training(inputs)

    # Filesystem effects
    assert (inputs.out_models / "ckpt_epoch2.pth").exists(), "Missing periodic checkpoint @2"
    assert (inputs.out_models / "ckpt_epoch4.pth").exists(), "Missing periodic checkpoint @4"

    # Log assertions (optional)
    saves = [r for r in caplog.records if r.name == log.name and r.message == "checkpoint.saved"]
    assert any("ckpt_epoch2.pth" in str(getattr(r, "__dict__", {})) for r in saves), "Expected log for ckpt_epoch2.pth"
    assert any("ckpt_epoch4.pth" in str(getattr(r, "__dict__", {})) for r in saves), "Expected log for ckpt_epoch4.pth"


def test_scheduler_selected_logged(tmp_path, caplog, monkeypatch_data, monkeypatch_mapping, monkeypatch_metrics):
    # Get the runner logger (same as used in training runner)
    log = get_logger("src.training.runner")

    args_overrides = {
        "sched": {"name": "steplr", "params": {"step_size": 7, "gamma": 0.3}},
        "callbacks": {
            "early_stopping": {"enabled": True, "patience": 1},
            "checkpoint": {"save_best": True, "save_last": False, "every_n_epochs": 0},
            "lr_logger": {"enabled": False},
        },
    }
    inputs = _base_inputs(tmp_path, args_overrides)
    inputs.epochs = 2

    # Capture logs emitted by our project logger
    caplog.set_level("INFO", logger=log.name)
    _ = run_training(inputs)

    # Verify a structured log entry with key 'scheduler.selected' was emitted
    records = [r for r in caplog.records if r.name == log.name]
    assert any(r.message == "scheduler.selected" for r in records), "Expected 'scheduler.selected' log"

    # Ensure extras (params) were attached
    found = False
    for r in records:
        if r.message == "scheduler.selected":
            assert "step_size" in r.__dict__["extra"]
            assert "gamma" in r.__dict__["extra"]
            found = True
    assert found, "Expected scheduler parameters in log extra"

