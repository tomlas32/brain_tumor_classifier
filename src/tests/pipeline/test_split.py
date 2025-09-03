# src/tests/pipeline/test_split.py
from __future__ import annotations

import importlib
import os
import re
import sys
from io import StringIO
from pathlib import Path
import contextlib

import pytest


def _import_split(monkeypatch, tmp_path: Path):
    """
    Import src.pipeline.split and redirect DATA_DIR / PROCESSED_DIR to tmp paths.
    """
    split = importlib.import_module("src.pipeline.split")
    importlib.reload(split)

    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Patch constants used by split.py
    monkeypatch.setattr(split, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(split, "PROCESSED_DIR", processed_dir, raising=False)

    return split


def _run_main(split_mod, argv):
    # split.main(argv) already accepts argv; just call it
    return int(split_mod.main(argv))


# -------------------------
# Unit tests for helpers
# -------------------------

def test_gather_by_class_exts_filter_and_accept_any(tmp_path, monkeypatch):
    split = _import_split(monkeypatch, tmp_path)

    root = tmp_path / "root"
    (root / "meningioma").mkdir(parents=True)
    (root / "glioma").mkdir(parents=True)

    (root / "meningioma" / "a.jpg").write_bytes(b"x")
    (root / "meningioma" / "b.png").write_bytes(b"x")
    (root / "meningioma" / "c.txt").write_text("x")

    (root / "glioma" / "g1.jpg").write_bytes(b"x")
    (root / "glioma" / "g2.webp").write_bytes(b"x")
    (root / "glioma" / "readme.md").write_text("md")

    # accept_any (exts empty) -> include everything
    out_any = split.gather_by_class(root, exts=set())
    assert set(out_any.keys()) == {"meningioma", "glioma"}
    assert len(out_any["meningioma"]) == 3
    assert len(out_any["glioma"]) == 3

    # filter .jpg only
    out_jpg = split.gather_by_class(root, exts={".jpg"})
    assert sorted(p.name for p in out_jpg["meningioma"]) == ["a.jpg"]
    assert sorted(p.name for p in out_jpg["glioma"]) == ["g1.jpg"]


def test_safe_copy_collision_suffix(tmp_path, monkeypatch):
    split = _import_split(monkeypatch, tmp_path)

    src1 = tmp_path / "src1.jpg"
    src2 = tmp_path / "src2.jpg"
    dst = tmp_path / "dst.jpg"
    src1.write_bytes(b"one")
    src2.write_bytes(b"two")

    split.safe_copy(src1, dst)
    assert dst.exists() and dst.read_bytes() == b"one"

    split.safe_copy(src2, dst)
    alt = dst.with_name("dst__1.jpg")
    assert alt.exists() and alt.read_bytes() == b"two"


# -------------------------
# CLI / planner flows
# -------------------------

def test_split_missing_fracs(tmp_path, monkeypatch):
    """
    If neither config nor CLI provides test/val fractions -> exit 2.
    """
    split = _import_split(monkeypatch, tmp_path)

    class Cfg:
        dataset = "owner/name"
        test_frac = None
        val_frac = None
        seed = 123
        clear_dest = False
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(split, "build_split_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(split, ["--config", str(tmp_path / "cfg.yaml")])

    s = buf.getvalue()
    assert code == 2
    assert "Missing fractions" in s


def test_split_invalid_fracs(tmp_path, monkeypatch):
    """
    Invalid fractions (e.g., test+val >= 1) -> exit 2.
    """
    split = _import_split(monkeypatch, tmp_path)

    class Cfg:
        dataset = "owner/name"
        test_frac = 0.7
        val_frac = 0.5
        seed = 123
        clear_dest = False
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(split, "build_split_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(split, ["--config", str(tmp_path / "cfg.yaml")])

    s = buf.getvalue()
    assert code == 2
    assert "Invalid fractions" in s


def test_split_processed_missing(tmp_path, monkeypatch):
    """
    With valid fracs but missing data/processed -> exit 2.
    """
    split = _import_split(monkeypatch, tmp_path)

    # Remove processed dir that _import_split created
    (split.PROCESSED_DIR).rmdir()

    class Cfg:
        dataset = "owner/name"
        test_frac = 0.2
        val_frac = 0.2
        seed = 42
        clear_dest = False
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(split, "build_split_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(split, ["--config", str(tmp_path / "cfg.yaml")])

    s = buf.getvalue()
    assert code == 2
    assert "data/processed not found" in s


def test_split_dry_run_plan_counts(tmp_path, monkeypatch):
    """
    Dry-run prints a human summary via split_planner.render_human.
    """
    split = _import_split(monkeypatch, tmp_path)

    # Build processed set
    tr = split.PROCESSED_DIR
    for d in [tr / "meningioma", tr / "glioma"]:
        d.mkdir(parents=True, exist_ok=True)
    # files
    (tr / "meningioma" / "a.jpg").write_bytes(b"x")
    (tr / "meningioma" / "b.jpg").write_bytes(b"x")
    (tr / "glioma" / "g1.jpg").write_bytes(b"x")
    (tr / "glioma" / "g2.png").write_bytes(b"x")  # excluded when exts=jpg

    class Cfg:
        dataset = "owner/name"
        test_frac = 0.2
        val_frac = 0.2
        seed = 123
        clear_dest = False
        class log:
            level = "INFO"
            file = None
        dry_run = True

    monkeypatch.setattr(split, "build_split_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(split, ["--config", str(tmp_path / "cfg.yaml"), "--dry-run", "--exts", "jpg"])

    s = buf.getvalue()
    assert code == 0
    assert "Split plan (dry run)" in s  # from split_planner.render_human :contentReference[oaicite:3]{index=3}
    # totals line exists
    assert re.search(r"Totals -> source:\s*3\s*\|\s*train:\s*\d+\s*\|\s*val:\s*\d+\s*\|\s*test:\s*\d+", s)


def test_split_success_copies_and_clear_dest(tmp_path, monkeypatch):
    """
    Successful split: clears targets when --clear-dest, copies files into train/val/test.
    """
    split = _import_split(monkeypatch, tmp_path)

    # Prepopulate to ensure clear_dest removes them
    pre_train = split.DATA_DIR / "training" / "old" / "x.png"
    pre_val = split.DATA_DIR / "validation" / "old" / "y.png"
    pre_test = split.DATA_DIR / "testing" / "old" / "z.png"
    for p in [pre_train, pre_val, pre_test]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"old")

    # Build processed content
    tr = split.PROCESSED_DIR
    for d in [tr / "meningioma", tr / "glioma"]:
        d.mkdir(parents=True, exist_ok=True)
    (tr / "meningioma" / "a.jpg").write_bytes(b"x")
    (tr / "meningioma" / "b.jpg").write_bytes(b"x")
    (tr / "glioma" / "g1.jpg").write_bytes(b"x")

    class Cfg:
        dataset = "owner/name"
        test_frac = 0.34   # rounded via int(); at least 1 test ensured by code :contentReference[oaicite:4]{index=4}
        val_frac = 0.33
        seed = 7
        clear_dest = True
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(split, "build_split_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(split, ["--config", str(tmp_path / "cfg.yaml"), "--clear-dest", "--exts", "jpg"])

    assert code == 0

    # cleared
    assert not pre_train.exists()
    assert not pre_val.exists()
    assert not pre_test.exists()

    # all inputs accounted for across splits (3 files total)
    all_out = list((split.DATA_DIR / "training").rglob("*")) + \
              list((split.DATA_DIR / "validation").rglob("*")) + \
              list((split.DATA_DIR / "testing").rglob("*"))
    leaf_files = [p for p in all_out if p.is_file()]
    assert len(leaf_files) == 3  # each input exactly once

    # class folders exist and contain their files (possibly in any split)
    expected = {"a.jpg", "b.jpg", "g1.jpg"}
    seen = set(p.name for p in leaf_files)
    assert seen == expected


def test_split_balance_equalize(tmp_path, monkeypatch):
    """
    With balance='equalize', classes are capped to the smallest class size before splitting. 
    """
    split = _import_split(monkeypatch, tmp_path)

    # Build uneven processed content
    tr = split.PROCESSED_DIR
    (tr / "meningioma").mkdir(parents=True, exist_ok=True)
    (tr / "glioma").mkdir(parents=True, exist_ok=True)

    # meningioma: 5 files, glioma: 2 files
    for i in range(5):
        (tr / "meningioma" / f"m{i}.jpg").write_bytes(b"x")
    for i in range(2):
        (tr / "glioma" / f"g{i}.jpg").write_bytes(b"x")

    class Cfg:
        dataset = "owner/name"
        test_frac = 0.2
        val_frac = 0.2
        seed = 123
        clear_dest = True
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(split, "build_split_config", lambda *a, **k: Cfg(), raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(split, [
            "--config", str(tmp_path / "cfg.yaml"),
            "--clear-dest",
            "--exts", "jpg",
            "--balance", "equalize"
        ])

    assert code == 0

    # After equalization cap, each class should contribute 2 source files total
    # Verify total output across splits equals 4 files
    all_out = list((split.DATA_DIR / "training").rglob("*")) + \
              list((split.DATA_DIR / "validation").rglob("*")) + \
              list((split.DATA_DIR / "testing").rglob("*"))
    leaf_files = [p for p in all_out if p.is_file()]
    assert len(leaf_files) == 4

    # Per-class presence
    men = list((split.DATA_DIR / "training" / "meningioma").glob("*")) + \
          list((split.DATA_DIR / "validation" / "meningioma").glob("*")) + \
          list((split.DATA_DIR / "testing" / "meningioma").glob("*"))
    gli = list((split.DATA_DIR / "training" / "glioma").glob("*")) + \
          list((split.DATA_DIR / "validation" / "glioma").glob("*")) + \
          list((split.DATA_DIR / "testing" / "glioma").glob("*"))
    assert len([p for p in men if p.is_file()]) == 2
    assert len([p for p in gli if p.is_file()]) == 2
