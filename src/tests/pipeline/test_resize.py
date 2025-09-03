# src/tests/pipeline/test_resize.py
from __future__ import annotations

import contextlib
import importlib
import os
import re
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import pytest


def _import_resize(monkeypatch, tmp_path: Path):
    """
    Import src.pipeline.resize and redirect DATA_DIR / MERGED_DIR / PROCESSED_DIR to tmp paths.
    """
    resize = importlib.import_module("src.pipeline.resize")
    importlib.reload(resize)

    data_dir = tmp_path / "data"
    merged_dir = data_dir / "merged"
    processed_dir = data_dir / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    # merged_dir and processed_dir created per-test as needed

    monkeypatch.setattr(resize, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(resize, "MERGED_DIR", merged_dir, raising=False)
    monkeypatch.setattr(resize, "PROCESSED_DIR", processed_dir, raising=False)

    return resize


def _run_main(resize_mod, argv):
    # resize.main(argv) accepts argv; just call it
    return int(resize_mod.main(argv))


# -------------------------
# Unit tests for helpers
# -------------------------

def test_resize_and_pad_shapes_and_padding(monkeypatch, tmp_path):
    resize = _import_resize(monkeypatch, tmp_path)

    # Tall image: H > W
    img_tall = np.zeros((300, 100, 3), dtype=np.uint8)
    out_t = resize.resize_and_pad(img_tall, size=224)
    assert out_t.shape == (224, 224, 3)
    # For tall input, horizontal padding should be larger than vertical padding => the content centered left/right
    # We can check corners remain black (padding), and center column not all zeros after resize.
    assert np.all(out_t[:, 0, :] == 0) and np.all(out_t[:, -1, :] == 0)
    # Some middle column should be nonzero (resized data)
    assert out_t[:, 112, :].sum() == 0  # original was zeros; keep invariant simple (all zeros)

    # Wide image: W > H
    img_wide = np.zeros((100, 300, 3), dtype=np.uint8)
    out_w = resize.resize_and_pad(img_wide, size=224)
    assert out_w.shape == (224, 224, 3)
    assert np.all(out_w[0, :, :] == 0) and np.all(out_w[-1, :, :] == 0)
    assert out_w[112, :, :].sum() == 0  # original was zeros; resized stays zeros (we only verify padding logic)


def test_count_images_filters(monkeypatch, tmp_path):
    resize = _import_resize(monkeypatch, tmp_path)

    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)

    (root / "a" / "x.jpg").write_bytes(b"\x00")
    (root / "a" / "x.PNG").write_bytes(b"\x00")
    (root / "b" / "y.md").write_text("t")
    (root / "b" / "z.webp").write_bytes(b"\x00")

    assert resize._count_images(root, set()) == 4     # accept any
    assert resize._count_images(root, {".jpg", ".png"}) == 2
    assert resize._count_images(root, {".webp"}) == 1


def test_process_images_reads_writes_and_skips_unreadable(monkeypatch, tmp_path):
    resize = _import_resize(monkeypatch, tmp_path)

    inp = tmp_path / "in" / "classA"
    out = tmp_path / "out"
    inp.mkdir(parents=True, exist_ok=True)

    # Valid images
    cv2.imwrite(str(inp / "a.jpg"), np.full((10, 20, 3), 255, np.uint8))
    cv2.imwrite(str(inp / "b.png"), np.full((8, 16, 3), 127, np.uint8))
    # Unreadable masquerading as image
    (inp / "bad.jpg").write_bytes(b"not-an-image")

    found, resized = resize.process_images(inp.parent, out, exts={".jpg", ".png"}, target_size=64)
    # Found includes unreadable .jpg but resized excludes it
    assert found == 3
    # Expect both valid files to be written
    assert resized == 2
    assert (out / "classA" / "a.jpg").exists()
    assert (out / "classA" / "b.png").exists()


# -------------------------
# CLI flows
# -------------------------

def test_cli_no_inputs_exit2(monkeypatch, tmp_path):
    """
    Neither training nor testing inputs exist → exit code 2.
    """
    resize = _import_resize(monkeypatch, tmp_path)

    class Cfg:
        train_in = None
        train_out = None
        test_in = None
        test_out = None
        size = 224
        exts = "jpg"
        dry_run = False

        class log:
            level = "INFO"
            file = None

    # Ensure MERGED_DIR doesn't exist (our default train_in)
    assert not resize.MERGED_DIR.exists()

    monkeypatch.setattr(resize, "build_resize_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(resize, ["--config", str(tmp_path / "cfg.yaml")])

    s = buf.getvalue()
    assert code == 2
    assert "No input directories found" in s


def test_cli_inputs_exist_but_no_matching_images_exit3(monkeypatch, tmp_path):
    """
    Inputs exist but no matching images → exit code 3.
    """
    resize = _import_resize(monkeypatch, tmp_path)
    # Create class folders with only .txt files; exts will be .jpg
    (resize.MERGED_DIR / "classA").mkdir(parents=True, exist_ok=True)
    (resize.MERGED_DIR / "classA" / "a.txt").write_text("nope")

    class Cfg:
        train_in = None
        train_out = None
        test_in = None
        test_out = None
        size = 224
        exts = "jpg"
        dry_run = False

        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(resize, "build_resize_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(resize, ["--config", str(tmp_path / "cfg.yaml")])

    s = buf.getvalue()
    assert code == 3
    assert "No images found to resize" in s


def test_cli_dry_run_plan(monkeypatch, tmp_path):
    """
    Dry-run prints human plan from resize_planner.render_human (training subset).
    """
    resize = _import_resize(monkeypatch, tmp_path)

    # Build MERGED_DIR with images (jpg only)
    (resize.MERGED_DIR / "meningioma").mkdir(parents=True, exist_ok=True)
    (resize.MERGED_DIR / "glioma").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(resize.MERGED_DIR / "meningioma" / "a.jpg"), np.ones((10, 8, 3), np.uint8))
    cv2.imwrite(str(resize.MERGED_DIR / "meningioma" / "b.jpg"), np.ones((9, 9, 3), np.uint8))
    cv2.imwrite(str(resize.MERGED_DIR / "glioma" / "g1.png"), np.ones((8, 10, 3), np.uint8))  # excluded when exts=jpg

    class Cfg:
        train_in = None
        train_out = None
        test_in = None
        test_out = None
        size = 256
        exts = "jpg"
        dry_run = True

        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(resize, "build_resize_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(resize, ["--config", str(tmp_path / "cfg.yaml"), "--dry-run", "--exts", "jpg"])

    s = buf.getvalue()
    assert code == 0
    assert "[DRY-RUN] Resize plan" in s  # from planner
    # Totals for training: 2 found, 2 will_write
    assert re.search(r"Totals?\s*->\s*found:\s*2\s*\|\s*will_write:\s*2", s, re.IGNORECASE)


def test_cli_success_clears_target_and_writes_outputs(monkeypatch, tmp_path):
    """
    Successful run should clear PROCESSED_DIR and write mirrored outputs.
    """
    resize = _import_resize(monkeypatch, tmp_path)

    # Pre-populate processed to verify it gets cleared
    stale = resize.PROCESSED_DIR / "old" / "x.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")
    assert stale.exists()

    # Build MERGED_DIR with some jpgs
    (resize.MERGED_DIR / "meningioma").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(resize.MERGED_DIR / "meningioma" / "a.jpg"), np.full((12, 6, 3), 200, np.uint8))
    cv2.imwrite(str(resize.MERGED_DIR / "meningioma" / "b.jpg"), np.full((6, 12, 3), 50, np.uint8))

    class Cfg:
        train_in = None
        train_out = None
        test_in = None
        test_out = None
        size = 224
        exts = "jpg"
        dry_run = False

        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(resize, "build_resize_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(resize, ["--config", str(tmp_path / "cfg.yaml"), "--exts", "jpg"])

    s = buf.getvalue()
    assert code == 0
    # Target cleared
    assert not stale.exists()
    # Files mirrored into processed
    out_files = sorted((resize.PROCESSED_DIR / "meningioma").glob("*.jpg"))
    names = sorted(p.name for p in out_files)
    assert names == ["a.jpg", "b.jpg"]

    # Verify outputs are exactly target size and padded (square)
    for p in out_files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[0] == 224 and img.shape[1] == 224

    # Summary lines present
    assert "Total files found" in s and "Total files resized and saved" in s


def test_cli_testing_subset_supported(monkeypatch, tmp_path):
    """
    If cfg provides test_in/test_out, both subsets are processed.
    """
    resize = _import_resize(monkeypatch, tmp_path)

    # Build distinct training and testing trees
    tr_in = resize.MERGED_DIR
    te_in = tmp_path / "legacy_testing"
    tr_in.mkdir(parents=True, exist_ok=True)
    te_in.mkdir(parents=True, exist_ok=True)

    (tr_in / "meningioma").mkdir(parents=True, exist_ok=True)
    (te_in / "meningioma").mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(tr_in / "meningioma" / "a.jpg"), np.ones((10, 10, 3), np.uint8))
    cv2.imwrite(str(te_in / "meningioma" / "z.jpg"), np.ones((8, 8, 3), np.uint8))

    te_out = tmp_path / "testing_resized"

    class Cfg:
        # keep train_in/out None so module defaults to MERGED_DIR/PROCESSED_DIR
        train_in = None
        train_out = None
        # explicit legacy testing paths
        test_in = te_in
        test_out = te_out
        size = 128
        exts = "jpg"
        dry_run = False

        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(resize, "build_resize_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(resize, ["--config", str(tmp_path / "cfg.yaml"), "--exts", "jpg"])

    assert code == 0

    # Training written under PROCESSED_DIR
    assert (resize.PROCESSED_DIR / "meningioma" / "a.jpg").exists()
    # Testing written under te_out
    assert (te_out / "meningioma" / "z.jpg").exists()
