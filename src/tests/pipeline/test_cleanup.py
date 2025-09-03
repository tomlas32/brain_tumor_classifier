# src/tests/pipeline/test_cleanup.py
from __future__ import annotations

import json
import os
from pathlib import Path
import importlib

import pytest


def _import_cleanup(monkeypatch, tmp_path: Path):
    """
    Import src.pipeline.cleanup and redirect path constants to temp dirs.
    """
    cleanup = importlib.import_module("src.pipeline.cleanup")
    importlib.reload(cleanup)

    vdir = tmp_path / "validation_reports"
    cdir = tmp_path / "cleanup_reports"
    qroot = tmp_path / "quarantine"

    for p in (vdir, cdir, qroot):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cleanup, "VALIDATION_REPORTS_DIR", vdir, raising=False)
    monkeypatch.setattr(cleanup, "CLEANUP_REPORTS_DIR", cdir, raising=False)
    monkeypatch.setattr(cleanup, "QUARANTINE_ROOT", qroot, raising=False)
    return cleanup


def _run_main(cleanup_mod, argv):
    return int(cleanup_mod.main(argv))


# --------------------
# Unit-level helpers
# --------------------

def test_safe_move_collision_suffix(tmp_path, monkeypatch):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # Create two sources with same name; move to same destination path
    src1 = tmp_path / "src1.jpg"
    src2 = tmp_path / "src2.jpg"
    dst  = tmp_path / "out" / "dst.jpg"
    src1.write_bytes(b"one")
    src2.write_bytes(b"two")

    p1 = cleanup._safe_move(src1, dst)
    assert p1 == dst and p1.exists() and p1.read_bytes() == b"one"

    p2 = cleanup._safe_move(src2, dst)
    assert p2.name == "dst__1.jpg" and p2.exists() and p2.read_bytes() == b"two"


def test_plan_moves_strict_and_report_only_filters(tmp_path, monkeypatch):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # Findings: one error in STRICT set, one warning not in STRICT set
    f_err = cleanup.Finding(
        path=str(tmp_path / "data" / "train" / "cls" / "a.jpg"),
        label="cls",
        subset="train",
        kind="error",
        code="UNREADABLE",
    )
    f_warn = cleanup.Finding(
        path=str(tmp_path / "data" / "train" / "cls" / "b.jpg"),
        label="cls",
        subset="train",
        kind="warning",
        code="SUSPICIOUS_METADATA",
    )

    planned, counts = cleanup._plan_moves([f_err, f_warn], policy="strict", act_on="both", run_id="RUNX")
    # strict: act on errors + STRICT_ERROR_CODES, not generic warnings
    assert len(planned) == 1
    assert list(counts.keys()) == ["UNREADABLE"]

    planned2, counts2 = cleanup._plan_moves([f_err, f_warn], policy="report_only", act_on="both", run_id="RUNX")
    assert len(planned2) == 0
    assert counts2 == {}


def test_plan_moves_within_class_duplicates(tmp_path, monkeypatch):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # Two duplicate findings sharing sha1; one within same subset/label (should move),
    # one cross-class (should be ignored in within_class policy).
    first_path = tmp_path / "data" / "train" / "glioma" / "x.jpg"

    dup_same = cleanup.Finding(
        path=str(tmp_path / "data" / "train" / "glioma" / "y.jpg"),
        label="glioma",
        subset="train",
        kind="error",
        code="DUPLICATE",
        sha1="abc123",
        duplicate_of=str(first_path),
    )
    dup_cross = cleanup.Finding(
        path=str(tmp_path / "data" / "train" / "meningioma" / "m1.jpg"),
        label="meningioma",
        subset="train",
        kind="error",
        code="DUPLICATE",
        sha1="abc123",
        duplicate_of=str(first_path),
    )

    planned, counts = cleanup._plan_moves([dup_same, dup_cross], policy="within_class", act_on="both", run_id="RID1")
    # within_class: only quarantine duplicates inside the same subset+label
    assert len(planned) == 1
    assert planned[0][2].code == "DUPLICATE"
    assert counts["DUPLICATE"] == 1

    # In strict, both duplicates are planned
    planned_s, counts_s = cleanup._plan_moves([dup_same, dup_cross], policy="strict", act_on="both", run_id="RID1")
    assert len(planned_s) == 2
    assert counts_s["DUPLICATE"] == 2


# --------------------
# CLI flows
# --------------------

def test_cli_dry_run_writes_plan_and_prints_counts(tmp_path, monkeypatch, capsys):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # Create on-disk files referenced in findings (not required for dry-run, but realistic)
    f1 = tmp_path / "data" / "train" / "glioma" / "a.jpg"
    f2 = tmp_path / "data" / "train" / "glioma" / "b.jpg"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"a")
    f2.write_bytes(b"b")

    report_path = tmp_path / "validation_report.json"
    report = {
        "findings": [
            {"path": str(f1), "label": "glioma", "subset": "train", "kind": "error", "code": "UNREADABLE"},
            {"path": str(f2), "label": "glioma", "subset": "train", "kind": "error", "code": "DUPLICATE", "sha1": "s1", "duplicate_of": str(f1)},
        ]
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    # Stable run id for predictable file names
    os.environ["RUN_ID"] = "TST123"

    code = _run_main(
        cleanup,
        ["--report", str(report_path), "--policy", "strict", "--why", "both", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert code == 0
    # 2 planned moves for strict: UNREADABLE + DUPLICATE
    assert "[DRY-RUN] Planned moves: 2" in out
    assert "By code" in out

    # Plan file exists and is json with correct counts
    plan_file = (tmp_path / "cleanup_reports" / "cleanup_plan_TST123.json")
    assert plan_file.exists()
    plan = json.loads(plan_file.read_text())
    assert plan["run_id"] == "TST123"
    assert plan["policy"] == "strict"
    assert plan["planned_count"] == 2
    assert len(plan["items"]) == 2
    assert plan["source_report"] == str(report_path)


def test_cli_execute_moves_and_manifest(tmp_path, monkeypatch, capsys):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # One existing file to move + one missing source to trigger 'skipped'
    good = tmp_path / "data" / "train" / "meningioma" / "keep1.jpg"
    missing = tmp_path / "data" / "train" / "meningioma" / "missing.jpg"
    good.parent.mkdir(parents=True, exist_ok=True)
    good.write_bytes(b"img")

    report_path = tmp_path / "vr.json"
    data = {
        "findings": [
            {"path": str(good),    "label": "meningioma", "subset": "train", "kind": "error", "code": "BAD_SIZE"},
            {"path": str(missing), "label": "meningioma", "subset": "train", "kind": "error", "code": "UNREADABLE"},
        ]
    }
    report_path.write_text(json.dumps(data), encoding="utf-8")

    os.environ["RUN_ID"] = "RUNMOVE"

    code = _run_main(
        cleanup,
        ["--report", str(report_path), "--policy", "strict", "--why", "both"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Cleanup summary: moved=1 | skipped=1" in out
    assert "Manifest:" in out

    # File moved into quarantine under RUN_ID/subset/label/
    qdst = tmp_path / "quarantine" / "RUNMOVE" / "train" / "meningioma" / good.name
    assert qdst.exists() and not good.exists()

    # Manifest exists and contains moved+skipped counts and by_code
    manifests = sorted((tmp_path / "cleanup_reports").glob("quarantine_RUNMOVE_*.json"))
    assert manifests, "No manifest written"
    manifest = json.loads(manifests[-1].read_text())
    assert manifest["run_id"] == "RUNMOVE"
    assert manifest["policy"] == "strict"
    assert manifest["acted_on"] == "both"
    assert manifest["totals"]["planned"] == 2
    assert manifest["totals"]["moved"] == 1
    assert manifest["totals"]["skipped"] == 1
    assert set(manifest["totals"]["by_code"].keys()) == {"BAD_SIZE", "UNREADABLE"}
    assert manifest["quarantine_root"].endswith(str((tmp_path / "quarantine" / "RUNMOVE").as_posix()).split("/")[-1])  # basic sanity


def test_cli_report_latest_with_tag_resolution(tmp_path, monkeypatch, capsys):
    """
    Ensure --report latest --report-tag <tag> selects the most recent per RUN_ID,
    falling back to legacy dir pattern if needed.
    """
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # Create per-run dir with two matching reports, pick newest
    os.environ["RUN_ID"] = "RIDTAG"
    run_dir = (tmp_path / "validation_reports" / "RIDTAG")
    run_dir.mkdir(parents=True, exist_ok=True)
    r1 = run_dir / "20240101_aaa_tag.json"
    r2 = run_dir / "20250101_bbb_tag.json"
    r1.write_text('{"findings":[]}', encoding="utf-8")
    r2.write_text('{"findings":[]}', encoding="utf-8")

    # Make r2 the newer one (mtime higher)
    os.utime(r2, (r2.stat().st_atime + 10, r2.stat().st_mtime + 10))

    code = _run_main(
        cleanup,
        ["--report", "latest", "--report-tag", "tag", "--policy", "report_only", "--why", "both"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "[REPORT-ONLY] Planned moves:" in out  # empty findings → 0 planned
