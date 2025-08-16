"""
Tests for src.core.viz (galleries + Grad-CAM).

Covers:
- show_calls_gallery: file written + 'gallery.saved' log
- show_calls_gallery: empty items → 'viz.warning.empty_gallery' log
- show_gradcam_gallery: backend missing → 'viz.warning.no_gradcam_backend' log
- show_gradcam_gallery: mocked backend → file written + 'gradcam.saved' log
"""
from pathlib import Path
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pytest

from src.core.viz import show_calls_gallery, show_gradcam_gallery
from src.utils.logging_utils import get_logger


# ----------------------- helpers -----------------------

def _make_dummy_png(path: Path, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.full((size, size, 3), 180, dtype=np.uint8))
    img.save(path)


def _items_from_paths(paths: List[Path]) -> List[Dict]:
    # Alternate true/pred labels and conf
    items = []
    for i, p in enumerate(paths):
        items.append({
            "path": str(p),
            "true": i % 2,
            "pred": (i + 1) % 2,
            "conf": 0.75 if i % 2 == 0 else 0.33,
        })
    return items


# -------------------- show_calls_gallery --------------------

def test_show_calls_gallery_writes_and_logs(tmp_path, caplog):
    log = get_logger("src.core.viz")
    caplog.set_level("INFO", logger=log.name)

    # create 3 dummy images
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    p3 = tmp_path / "img3.png"
    for p in [p1, p2, p3]:
        _make_dummy_png(p)

    items = _items_from_paths([p1, p2, p3])
    cls = ["class0", "class1"]
    out = show_calls_gallery(items, cls, cols=2, title="misclassifications", save_dir=tmp_path, image_size=64)
    assert out is not None and out.exists(), "Expected gallery PNG to be saved"

    # structured log
    recs = [r for r in caplog.records if r.name == log.name and r.message == "gallery.saved"]
    assert recs, "Expected 'gallery.saved' log"
    # best-effort: check count present in extras
    blob = "\n".join(str(r.__dict__) for r in recs)
    assert "count" in blob and "3" in blob


def test_show_calls_gallery_empty_logs_warning(tmp_path, caplog):
    log = get_logger("src.core.viz")
    caplog.set_level("INFO", logger=log.name)

    out = show_calls_gallery([], ["class0", "class1"], cols=2, title="empty", save_dir=tmp_path, image_size=64)
    assert out is None, "Empty items should return None (no file)"

    # structured log
    assert any(r.name == log.name and r.message == "viz.warning.empty_gallery" for r in caplog.records)


# -------------------- show_gradcam_gallery --------------------

def test_show_gradcam_no_backend_logs_warning(tmp_path, caplog, monkeypatch):
    log = get_logger("src.core.viz")
    caplog.set_level("INFO", logger=log.name)

    # Force backend missing
    monkeypatch.setattr("src.core.viz._HAS_CAM", False, raising=False)

    # dummy image and items
    p = tmp_path / "img.png"
    _make_dummy_png(p)
    items = _items_from_paths([p])

    # dummy dataset without transform (will trigger default display transform)
    class DummyDS:
        transform = None

    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))
    device = torch.device("cpu")

    out = show_gradcam_gallery(
        predictions=items,
        class_names=["class0", "class1"],
        model=model,
        device=device,
        ds_for_transform=DummyDS(),
        cols=1,
        title="gradcam_missing",
        target_layer=None,  # will attempt to auto-select, but backend is disabled anyway
        save_dir=tmp_path,
        image_size=64,
    )
    assert out is None, "No backend → should return None"

    assert any(r.name == log.name and r.message == "viz.warning.no_gradcam_backend" for r in caplog.records)


def test_show_gradcam_with_mocked_backend(tmp_path, caplog, monkeypatch):
    log = get_logger("src.core.viz")
    caplog.set_level("INFO", logger=log.name)

    # Enable backend and mock GradCAM + show_cam_on_image to lightweight fakes
    monkeypatch.setattr("src.core.viz._HAS_CAM", True, raising=False)

    class FakeGradCAM:
        def __init__(self, model, target_layers, use_cuda=False):
            self.sz = 64
        def __call__(self, input_tensor, targets=None):
            # Return a single grayscale map HxW per input (batch size 1 in this test)
            b, _, h, w = input_tensor.shape
            return np.ones((b, h, w), dtype=np.float32)

    def fake_show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True):
        # Return a uint8 overlay of same size
        arr = (np.clip(rgb_float, 0, 1) * 255.0).astype(np.uint8)
        return arr

    monkeypatch.setattr("src.core.viz.GradCAM", FakeGradCAM, raising=False)
    monkeypatch.setattr("src.core.viz.show_cam_on_image", fake_show_cam_on_image, raising=False)

    # Create tiny image and items
    p = tmp_path / "img.png"
    _make_dummy_png(p, size=64)
    items = _items_from_paths([p, p])  # two tiles

    # Minimal model with a recognizable last conv block so _select_target_layer() finds it
    class SmallResNetLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 8, 3, padding=1)
            self.layer4 = nn.ModuleList([nn.Conv2d(8, 8, 3, padding=1)])  # last conv layer
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 2)
        def forward(self, x):
            x = torch.relu(self.stem(x))
            x = torch.relu(self.layer4[-1](x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    model = SmallResNetLike()
    device = torch.device("cpu")

    class DummyDS:
        transform = None  # use default display transform

    out = show_gradcam_gallery(
        predictions=items,
        class_names=["class0", "class1"],
        model=model,
        device=device,
        ds_for_transform=DummyDS(),
        cols=2,
        title="gradcam_mocked",
        target_layer=None,  # let viz auto-pick from layer4[-1]
        save_dir=tmp_path,
        image_size=64,
    )
    assert out is not None and out.exists(), "Expected Grad-CAM gallery PNG to be saved"

    recs = [r for r in caplog.records if r.name == log.name and r.message == "gradcam.saved"]
    assert recs, "Expected 'gradcam.saved' log"
    blob = "\n".join(str(r.__dict__) for r in recs)
    assert "count" in blob and "2" in blob
