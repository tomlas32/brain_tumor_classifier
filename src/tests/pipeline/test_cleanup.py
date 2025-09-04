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
    assert len(planned) == 1
    assert list(counts.keys()) == ["UNREADABLE"]

    planned2, counts2 = cleanup._plan_moves([f_err, f_warn], policy="report_only", act_on="both", run_id="RUNX")
    assert len(planned2) == 0
    assert counts2 == {}


def test_plan_moves_within_class_duplicates(tmp_path, monkeypatch):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

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
    assert len(planned) == 1
    assert planned[0][2].code == "DUPLICATE"
    assert counts["DUPLICATE"] == 1

    planned_s, counts_s = cleanup._plan_moves([dup_same, dup_cross], policy="strict", act_on="both", run_id="RID1")
    assert len(planned_s) == 2
    assert counts_s["DUPLICATE"] == 2


# --------------------
# CLI flows (use real src.core.config via --override)
# --------------------

def test_cli_dry_run_writes_plan_and_prints_counts(tmp_path, monkeypatch, capsys):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # Create source files referenced in the report
    f1 = tmp_path / "data" / "train" / "glioma" / "a.jpg"
    f2 = tmp_path / "data" / "train" / "glioma" / "b.jpg"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"a")
    f2.write_bytes(b"b")

    report_path = tmp_path / "validation_report.json"
    report = {
        "findings": [
            {"path": str(f1), "label": "glioma", "subset": "train", "kind": "error", "code": "UNREADABLE"},
            {"path": str(f2), "label": "glioma", "subset": "train", "kind": "error",
             "code": "DUPLICATE", "sha1": "s1", "duplicate_of": str(f1)},
        ]
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    os.environ["RUN_ID"] = "TST123"

    code = _run_main(
        cleanup,
        [
            "--override", f"report={report_path}",
            "--override", "policy=strict",
            "--override", "why=both",
            "--override", "dry_run=true",
        ]
    )
    captured = capsys.readouterr()
    if code != 0:
        print("STDOUT:\n", captured.out)
        print("STDERR:\n", captured.err)
    out = captured.out
    assert code == 0
    assert "[DRY-RUN] Planned moves: 2" in out
    assert "By code" in out

    plan_file = (tmp_path / "cleanup_reports" / "cleanup_plan_TST123.json")
    assert plan_file.exists()
    plan = json.loads(plan_file.read_text())
    assert plan["run_id"] == "TST123"
    assert plan["policy"] == "strict"
    assert plan["acted_on"] == "both"
    assert plan["planned_count"] == 2
    assert len(plan["items"]) == 2
    assert plan["source_report"] == str(report_path)


def test_cli_execute_moves_and_manifest(tmp_path, monkeypatch, capsys):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

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
        [
            "--override", f"report={report_path}",
            "--override", "policy=strict",
            "--override", "why=both",
            # not a dry-run
        ]
    )
    captured = capsys.readouterr()
    if code != 0:
        print("STDOUT:\n", captured.out)
        print("STDERR:\n", captured.err)
    out = captured.out
    assert code == 0
    assert "Cleanup summary: moved=1 | skipped=1" in out
    assert "Manifest:" in out

    qdst = tmp_path / "quarantine" / "RUNMOVE" / "train" / "meningioma" / good.name
    assert qdst.exists() and not good.exists()

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


def test_cli_report_latest_with_tag_resolution(tmp_path, monkeypatch, capsys):
    """
    Ensure --report latest with a tag selects the most recent and prints the
    'clean dataset' message when no findings match the policy/severity.
    """
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    os.environ["RUN_ID"] = "RIDTAG"
    run_dir = (tmp_path / "validation_reports" / "RIDTAG")
    run_dir.mkdir(parents=True, exist_ok=True)
    r1 = run_dir / "20240101_aaa_pre.json"
    r2 = run_dir / "20250101_bbb_pre.json"
    r1.write_text('{"findings":[]}', encoding="utf-8")
    r2.write_text('{"findings":[]}', encoding="utf-8")
    os.utime(r2, (r2.stat().st_atime + 10, r2.stat().st_mtime + 10))

    code = _run_main(
        cleanup,
        [
            "--report", "latest",
            "--report-tag", "pre",
            # ensure policy/severity through overrides (report_only + both)
            "--override", "policy=report_only",
            "--override", "why=both",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (
        "Nothing to quarantine. Dataset already clean per selected policy/severity." in out
        or "[REPORT-ONLY] Planned moves: 0" in out
    )


def test_plan_moves_crossclass_duplicate_removes_both(tmp_path, monkeypatch):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # First-occurrence path (in glioma)
    first_path = tmp_path / "data" / "train" / "glioma" / "x.jpg"
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_bytes(b"imgx")

    # Cross-class duplicate finding (in meningioma), pointing to first_path
    cross = cleanup.Finding(
        path=str(tmp_path / "data" / "train" / "meningioma" / "y.jpg"),
        label="meningioma",
        subset="train",
        kind="error",
        code="CROSSCLASS_DUP",
        sha1="abc123",
        duplicate_of=str(first_path),
    )
    # ensure counterpart file exists too
    yfile = tmp_path / "data" / "train" / "meningioma" / "y.jpg"
    yfile.parent.mkdir(parents=True, exist_ok=True)
    yfile.write_bytes(b"imgy")

    planned, counts = cleanup._plan_moves([cross], policy="strict", act_on="both", run_id="RIDX")

    # Should plan to move BOTH: the current path and its duplicate_of counterpart
    assert len(planned) == 2
    # Planned sources (unordered)
    planned_srcs = {str(p[0]) for p in planned}
    assert planned_srcs == {str(yfile), str(first_path)}
    # Count both under the same code
    assert counts["CROSSCLASS_DUP"] == 2


def test_cli_dry_run_crossclass_near_dup_plans_both(tmp_path, monkeypatch, capsys):
    cleanup = _import_cleanup(monkeypatch, tmp_path)

    # Files referenced by the report
    a = tmp_path / "data" / "train" / "glioma" / "a.jpg"
    b = tmp_path / "data" / "train" / "meningioma" / "b.jpg"
    a.parent.mkdir(parents=True, exist_ok=True)
    b.parent.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"A")
    b.write_bytes(b"B")

    # Pre-validation report with a single CROSSCLASS_NEAR_DUP finding
    report_path = tmp_path / "vr_cross.json"
    report = {
        "findings": [
            {
                "path": str(b),
                "label": "meningioma",
                "subset": "train",
                "kind": "error",
                "code": "CROSSCLASS_NEAR_DUP",
                "duplicate_of": str(a),
                "details": {"hamming": 4, "ssim": 0.95, "cur_label": "meningioma", "other_label": "glioma"},
            }
        ]
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    os.environ["RUN_ID"] = "RIDNEAR"

    code = _run_main(
        cleanup,
        [
            "--override", f"report={report_path}",
            "--override", "policy=strict",
            "--override", "why=both",
            "--override", "dry_run=true",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    # Plan file should list two items (both sides)
    plan_file = (tmp_path / "cleanup_reports" / "cleanup_plan_RIDNEAR.json")
    assert plan_file.exists()
    plan = json.loads(plan_file.read_text())
    assert plan["planned_count"] == 2
    srcs = {item["src"] for item in plan["items"]}
    assert srcs == {str(a), str(b)}
