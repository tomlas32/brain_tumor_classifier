# src/tests/pipeline/test_merge.py
from __future__ import annotations

import importlib
import os
from io import StringIO
from pathlib import Path
import contextlib
import types
import json

import pytest


def _import_merge(monkeypatch, tmp_path: Path):
    """
    Import src.pipeline.merge and redirect its key path constants to tmp_path.
    We also keep handles to functions we plan to patch per test.
    """
    merge = importlib.import_module("src.pipeline.merge")
    importlib.reload(merge)

    # Redirect path constants to temp dirs so we don't touch the repo
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    merged_dir = data_dir / "merged"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(merge, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(merge, "OUTPUTS_DIR", outputs_dir, raising=False)
    monkeypatch.setattr(merge, "MERGED_DIR", merged_dir, raising=False)

    return merge


def _run_main(merge_mod, argv):
    # Support main(argv) style
    return int(merge_mod.main(argv))


# -------------------------
# Unit tests for helpers
# -------------------------

def test_gather_by_class_exts_filter_and_accept_any(tmp_path, monkeypatch):
    merge = _import_merge(monkeypatch, tmp_path)

    root = tmp_path / "root"
    (root / "meningioma").mkdir(parents=True)
    (root / "glioma").mkdir(parents=True)

    # files
    (root / "meningioma" / "a.jpg").write_bytes(b"x")
    (root / "meningioma" / "b.png").write_bytes(b"x")
    (root / "meningioma" / "c.txt").write_text("x")  # will be included when accept_any

    (root / "glioma" / "g1.jpg").write_bytes(b"x")
    (root / "glioma" / "g2.webp").write_bytes(b"x")
    (root / "glioma" / "readme.md").write_text("md")  # included only when accept_any

    # accept_any (exts = empty set) → includes EVERYTHING that is a file
    out_any = merge.gather_by_class(root, exts=set())
    assert set(out_any.keys()) == {"meningioma", "glioma"}
    assert len(out_any["meningioma"]) == 3
    assert len(out_any["glioma"]) == 3

    # filter only .jpg
    out_jpg = merge.gather_by_class(root, exts={".jpg"})
    men_jpg = sorted(p.name for p in out_jpg["meningioma"])
    gli_jpg = sorted(p.name for p in out_jpg["glioma"])
    assert men_jpg == ["a.jpg"]
    assert gli_jpg == ["g1.jpg"]


def test_safe_copy_collision_suffix(tmp_path, monkeypatch):
    merge = _import_merge(monkeypatch, tmp_path)

    src1 = tmp_path / "src1.jpg"
    src2 = tmp_path / "src2.jpg"
    src1.write_bytes(b"one")
    src2.write_bytes(b"two")

    dst = tmp_path / "dst.jpg"

    # First copy: to dst
    p1 = merge.safe_copy(src1, dst)
    assert p1 == dst and p1.exists() and p1.read_bytes() == b"one"

    # Second copy: should create dst__1.jpg
    p2 = merge.safe_copy(src2, dst)
    assert p2.name == "dst__1.jpg" and p2.exists() and p2.read_bytes() == b"two"


# -------------------------
# CLI / integration tests
# -------------------------

def test_merge_dry_run_no_pointer(tmp_path, monkeypatch):
    merge = _import_merge(monkeypatch, tmp_path)

    # Config returns dry_run True, dataset slug present, no pointer provided
    class Cfg:
        dataset = "owner/name"
        pointer = None
        exts = None
        clear_dest = False
        dry_run = True
        class log:  # noqa: N801
            level = "INFO"
            file = None

    monkeypatch.setattr(merge, "build_merge_config", lambda *a, **k: Cfg(), raising=True)

    # Ensure computed pointer path doesn't exist
    # (merge._pointer_path_for uses OUTPUTS_DIR/... which we redirected)
    # Do not create it to simulate "no pointer"
    argv = ["--dataset", "owner/name", "--dry-run", "--config", str(tmp_path / "dummy.yaml")]

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(merge, argv)

    s = buf.getvalue()
    assert code == 0
    assert "[DRY-RUN] Merge plan" in s
    assert "No fetch pointer found" in s


def test_merge_pointer_read_error(tmp_path, monkeypatch):
    merge = _import_merge(monkeypatch, tmp_path)

    class Cfg:
        dataset = "owner/name"
        pointer = tmp_path / "ptr.json"  # we will raise on read
        exts = None
        clear_dest = False
        dry_run = False
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(merge, "build_merge_config", lambda *a, **k: Cfg(), raising=True)

    def _raise_read(path: Path):
        raise RuntimeError("bad-pointer")
    monkeypatch.setattr(merge, "read_fetch_pointer", _raise_read, raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(merge, ["--config", str(tmp_path / "cfg.yaml")])

    assert code == 2  # as per merge.pointer_read_error branch


def test_merge_missing_source_dirs(tmp_path, monkeypatch):
    merge = _import_merge(monkeypatch, tmp_path)

    class Cfg:
        dataset = "owner/name"
        pointer = tmp_path / "ptr.json"
        exts = None
        clear_dest = False
        dry_run = False
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(merge, "build_merge_config", lambda *a, **k: Cfg(), raising=True)

    # read_fetch_pointer returns a dataset_root that exists but has no Training/Testing
    ds = tmp_path / "dataset"
    ds.mkdir()
    monkeypatch.setattr(merge, "read_fetch_pointer",
                        lambda p: {"dataset_root": str(ds), "dataset": "owner/name"},
                        raising=True)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(merge, ["--config", str(tmp_path / "cfg.yaml")])

    assert code == 2  # merge.source_dirs_missing


def test_merge_dry_run_plan_counts(tmp_path, monkeypatch):
    import re
    merge = _import_merge(monkeypatch, tmp_path)

    # Build a fake dataset with Training/Testing and classes+files
    ds = tmp_path / "dataset"
    tr = ds / "Training"
    te = ds / "Testing"
    for d in [tr / "meningioma", tr / "glioma", te / "meningioma", te / "glioma"]:
        d.mkdir(parents=True, exist_ok=True)
    # files
    (tr / "meningioma" / "a.jpg").write_bytes(b"x")
    (tr / "glioma" / "g1.jpg").write_bytes(b"x")
    (te / "meningioma" / "b.jpg").write_bytes(b"x")
    (te / "glioma" / "g2.png").write_bytes(b"x")  # should be excluded if exts=jpg

    # Create a real pointer file (your merge checks existence before read)
    ptr = tmp_path / "ptr.json"
    ptr.write_text("{}")

    class Cfg:
        dataset = "owner/name"
        pointer = ptr
        exts = "jpg"    # limit to JPG only
        clear_dest = False
        dry_run = True
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(merge, "build_merge_config", lambda *a, **k: Cfg(), raising=True)
    monkeypatch.setattr(
        merge,
        "read_fetch_pointer",
        lambda p: {"dataset_root": str(ds), "dataset": "owner/name"},
        raising=True,
    )

    from io import StringIO
    import contextlib
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(merge, ["--config", str(tmp_path / "cfg.yaml"), "--dry-run", "--exts", "jpg"])

    s = buf.getvalue()
    assert code == 0
    assert "[DRY-RUN] Merge plan" in s
    assert "Total files to copy: 3" in s

    # Be robust to spacing/format changes: extract counts via regex
    men_match = re.search(r"meningioma.*?->\s*(\d+)", s)
    gli_match = re.search(r"glioma.*?->\s*(\d+)", s)
    assert men_match and gli_match, f"Class count lines not found in output:\n{s}"
    assert int(men_match.group(1)) == 2
    assert int(gli_match.group(1)) == 1



def test_merge_success_copies_clears_and_writes_mapping_and_manifest(tmp_path, monkeypatch):
    merge = _import_merge(monkeypatch, tmp_path)

    # Prepopulate MERGED_DIR with a file to ensure --clear-dest removes it
    pre = merge.MERGED_DIR / "old_class" / "old.png"
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.write_bytes(b"old")

    # Fake dataset
    ds = tmp_path / "dataset"
    tr = ds / "Training"
    te = ds / "Testing"
    for d in [tr / "meningioma", tr / "glioma", te / "meningioma", te / "glioma"]:
        d.mkdir(parents=True, exist_ok=True)
    # files
    (tr / "meningioma" / "a.jpg").write_bytes(b"x")
    (te / "meningioma" / "b.jpg").write_bytes(b"x")
    (tr / "glioma" / "g1.jpg").write_bytes(b"x")

    class Cfg:
        dataset = "owner/name"
        pointer = tmp_path / "ptr.json"
        exts = "jpg"
        clear_dest = True
        dry_run = False
        save_remap_to_project_root = False
        mapping_use_dataset_subdir = True
        mapping_write_split_copy = False
        class log:
            level = "INFO"
            file = None

    monkeypatch.setattr(merge, "build_merge_config", lambda *a, **k: Cfg(), raising=True)
    monkeypatch.setattr(merge, "read_fetch_pointer",
                        lambda p: {"dataset_root": str(ds), "dataset": "owner/name"},
                        raising=True)

    # Capture mapping calls
    calls = {"write_index": None, "write_pointer": None}

    def fake_write_index(classes, dataset, use_dataset_subdir, run_id):
        calls["write_index"] = dict(classes=list(classes), dataset=dataset,
                                    use_dataset_subdir=use_dataset_subdir, run_id=run_id)
        # Return path to a pretend latest mapping file
        path = tmp_path / "latest_map.json"
        path.write_text(json.dumps({"classes": classes}))
        return path

    def fake_write_pointer(**kwargs):
        calls["write_pointer"] = kwargs
        return {"latest": "L"}

    # Patch the names imported into merge.py
    monkeypatch.setattr(merge, "mapping_write_index_remap", fake_write_index, raising=True)
    monkeypatch.setattr(merge, "write_mapping_pointer", fake_write_pointer, raising=True)
    monkeypatch.setattr(merge, "copy_index_remap", lambda *a, **k: None, raising=True)

    # Stable RUN_ID so we can assert manifest paths
    monkeypatch.setenv("RUN_ID", "TST123")

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(merge, ["--config", str(tmp_path / "cfg.yaml"), "--clear-dest", "--exts", "jpg"])

    s = buf.getvalue()
    assert code == 0
    assert "Merge complete →" in s

    # Old merged content should be cleared
    assert not pre.exists()

    # New files should exist under MERGED_DIR/<class>/
    men_files = sorted(p.name for p in (merge.MERGED_DIR / "meningioma").glob("*"))
    gli_files = sorted(p.name for p in (merge.MERGED_DIR / "glioma").glob("*"))
    assert men_files == ["a.jpg", "b.jpg"]
    assert gli_files == ["g1.jpg"]

    # Mapping calls were made with detected classes
    assert calls["write_index"] is not None
    assert sorted(calls["write_index"]["classes"]) == ["glioma", "meningioma"]
    assert calls["write_index"]["dataset"] == "owner/name"
    assert calls["write_index"]["use_dataset_subdir"] is True
    assert calls["write_index"]["run_id"] == "TST123"

    assert calls["write_pointer"] is not None
    assert calls["write_pointer"]["dataset"] == "owner/name"
    assert calls["write_pointer"]["run_id"] == "TST123"

    # Manifest written in outputs/merge/TST123/
    out_dir = merge.OUTPUTS_DIR / "merge" / "TST123"
    latest = out_dir / "latest.json"
    # latest.json exists and points to a manifest file
    assert latest.exists()
    latest_obj = json.loads(latest.read_text())
    manifest_path = Path(latest_obj["latest"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dest_merged"] == str(merge.MERGED_DIR)
    assert manifest["total_copied"] == 3
