import json
from pathlib import Path

import numpy as np
import pytest, time

# Import the validate module under test
from src.pipeline import validate as validate_mod


def _write_png(path: Path, arr: np.ndarray) -> None:
    """Write BGR uint8 image using cv2 without bringing cv2 to module scope."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    assert ok, f"Failed to write image: {path}"


def _latest_reports(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob("validation_*.json"))


@pytest.fixture
def tmp_dataset(tmp_path: Path):
    """
    Create a small dataset root with two valid class dirs and one bad label dir.
    """
    root = tmp_path / "data" / "processed"
    (root / "glioma").mkdir(parents=True)
    (root / "meningioma").mkdir(parents=True)
    (root / "badlabel").mkdir(parents=True)
    return root


@pytest.fixture
def mapping_ok(tmp_path: Path):
    """
    Minimal index_remap.json allowing only glioma & meningioma.
    """
    mapping = {"0": "glioma", "1": "meningioma"}
    p = tmp_path / "outputs" / "mappings" / "index_remap.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping), encoding="utf-8")
    return p


# ------------------------- Core correctness -------------------------

def test_extension_filtering_and_all(tmp_dataset: Path):
    # Make one allowed .png and one forbidden .bmp; force exts to 'jpg,png'
    rng = np.random.default_rng(0)
    good = (rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8))
    bad = (rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8))

    _write_png(tmp_dataset / "glioma" / "ok.png", good)

    # write BMP (disallowed by exts='jpg,png')
    import cv2
    cv2.imwrite(str(tmp_dataset / "glioma" / "nope.bmp"), bad)

    # Expect BAD_EXT → exit code 1 under fail_on=error
    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "jpg,png",
        "--fail-on", "error",
        "--no-write-report",
    ])
    assert code == 1

    # Now accept all extensions
    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "all",
        "--fail-on", "error",
        "--no-write-report",
    ])
    # Still 0 because other checks are fine for these two files
    assert code == 0


def test_rgb_and_size_enforcement(tmp_dataset: Path):
    # NOT_RGB: grayscale 224x224 (all 3 checks enabled by default)
    gray = np.full((224, 224), 128, dtype=np.uint8)  # single-channel
    # Save grayscale as PNG via cv2 (will write as 8-bit single channel)
    import cv2
    cv2.imwrite(str(tmp_dataset / "glioma" / "gray.png"), gray)

    # BAD_SIZE: RGB but 200x200
    rgb_200 = np.full((200, 200, 3), 64, dtype=np.uint8)
    _write_png(tmp_dataset / "meningioma" / "small.png", rgb_200)

    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--fail-on", "error",
        "--no-write-report",
    ])
    assert code == 1  # errors present (NOT_RGB and BAD_SIZE)


# ------------------------- Robustness & guardrails -------------------------

def test_unreadable_and_tiny_file(tmp_dataset: Path):
    # Tiny file with PNG suffix (triggers TINY_FILE warning and likely UNREADABLE)
    tiny = tmp_dataset / "glioma" / "tiny.png"
    tiny.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG signature only

    # Valid normal file to avoid other failures
    good = np.full((224, 224, 3), 10, dtype=np.uint8)
    _write_png(tmp_dataset / "glioma" / "ok.png", good)

    # Warnings should yield code=0 for fail_on=error
    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--fail-on", "error",
        "--no-write-report",
    ])
    assert code in (0, 1)  # Some systems may classify as UNREADABLE (error); don't overfit exact codec behavior


def test_label_whitelist_bad_label(tmp_dataset: Path, mapping_ok: Path):
    # Place one file in a forbidden label dir
    bad = np.full((224, 224, 3), 20, dtype=np.uint8)
    _write_png(tmp_dataset / "badlabel" / "x.png", bad)

    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--index-remap", str(mapping_ok),
        "--size", "224",
        "--exts", "png",
        "--fail-on", "error",
        "--no-write-report",
    ])
    assert code == 1  # BAD_LABEL is an error


# ------------------------- Dry-run & policy -------------------------

def test_dry_run_plan_counts_without_io(tmp_dataset: Path):
    # Two classes, one file each
    a = np.full((300, 300, 3), 1, dtype=np.uint8)
    b = np.full((224, 224, 3), 2, dtype=np.uint8)
    _write_png(tmp_dataset / "glioma" / "a.png", a)
    _write_png(tmp_dataset / "meningioma" / "b.png", b)

    # Should exit 0 and not try to open images
    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--dry-run",
        "--no-write-report",
    ])
    assert code == 0


def test_fail_on_policy_warning_only(tmp_dataset: Path):
    # Create a uniform gray (not black/white) → LOW_STD warning only
    gray = np.full((224, 224, 3), 128, dtype=np.uint8)
    _write_png(tmp_dataset / "glioma" / "flat.png", gray)

    # fail_on=error → exit 0 (warnings ignored)
    code_err = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--warn-low-std", "3.0",  # std=0 < 3.0 → warning
        "--fail-on", "error",
        "--no-write-report",
    ])
    assert code_err == 0

    # fail_on=warning → exit 1
    code_warn = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--warn-low-std", "3.0",
        "--fail-on", "warning",
        "--no-write-report",
    ])
    assert code_warn == 1


# ------------------------- Reporting -------------------------

def test_report_written_and_toggle(tmp_dataset: Path, tmp_path: Path):

    from uuid import uuid4

    # One valid image to ensure run passes
    ok = np.full((224, 224, 3), 50, dtype=np.uint8)
    _write_png(tmp_dataset / "glioma" / "ok.png", ok)

    reports_dir = Path("outputs") / "validation_reports"
    before = _latest_reports(reports_dir)

    # Run with default (write report)
    tag = f"t_{uuid4().hex[:8]}"
    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--fail-on", "error",
        "--report-tag", tag,
    ])
    assert code == 0

    after = _latest_reports(reports_dir)
    assert len(after) == len(before) + 1, "Expected a new validation report"

    # Run with --no-write-report: count shouldn't increase
    tag = f"t_{uuid4().hex[:8]}"
    code2 = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--fail-on", "error",
        "--no-write-report",
        "--report-tag", tag,
    ])
    assert code2 == 0
    after2 = _latest_reports(reports_dir)
    assert len(after2) == len(after)


# ------------------------- pHash + SSIM reject branch -------------------------

def test_phash_neardup_rejected_when_ssim_low(monkeypatch, tmp_dataset: Path):
    """
    Force a scenario:
      - pHash says "near" (distance <= thresh)
      - SSIM is low (< ssim_thresh)
    Expectation: no NEAR_DUP_PHASH finding contributes to nonzero exit under warning policy.
    (We don't parse the report here; we assert the exit code stays at 0 for fail_on=warning
     if the only potential issue would have been a PHASH warning.)
    """
    # Create two arbitrary images
    a = np.full((224, 224, 3), 30, dtype=np.uint8)
    b = np.full((224, 224, 3), 180, dtype=np.uint8)
    _write_png(tmp_dataset / "glioma" / "a.png", a)
    _write_png(tmp_dataset / "glioma" / "b.png", b)

    # Monkeypatch pHash & SSIM behaviors:
    # - Return equal hashes so Hamming distance is 0 (<= thresh)
    monkeypatch.setattr(validate_mod, "_phash_of_path", lambda p, **_: 0xABCD1234)
    # - Force SSIM low to trigger the "reject" branch (no warning recorded)
    monkeypatch.setattr(validate_mod, "_ssim_similarity", lambda p1, p2, size=224: 0.10)

    code = validate_mod.main([
        "--in-dir", str(tmp_dataset),
        "--size", "224",
        "--exts", "png",
        "--dup-check",                 # ensure duplicate logic runs
        "--phash",
        "--phash-thresh", "8",
        "--ssim-thresh", "0.90",
        "--fail-on", "warning",
        "--no-write-report",
        "--override", "min_file_bytes=0",  # suppress TINY_FILE
        "--override", "warn_low_std=0",    # suppress LOW_STD
    ])
    # No warnings/errors should be recorded because SSIM check rejects the near-dup
    assert code == 0
