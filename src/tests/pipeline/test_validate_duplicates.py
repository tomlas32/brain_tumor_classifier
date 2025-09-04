"""
Unit tests for duplicate detection in validate_dataset.

These tests exercise the three duplicate detection pathways:
1. Exact duplicates caught via file SHA-1 (byte-for-byte matches).
2. Flip-aware duplicates caught via normalized content hash of original + horizontal mirror.
3. Near-duplicates (JPEG re-saves, small tweaks) caught via perceptual hash (pHash).

Each test creates a temporary synthetic dataset with a small set of images
(base image, exact copy, mirrored copy, JPEG variant, unrelated class).
The tests assert:
- Counts in errors_by_type / warnings_by_type are incremented correctly.
- Findings include the expected codes (DUPLICATE, NEAR_DUP_PHASH).
- duplicate_of paths and details (e.g. Hamming distance) are recorded.

By design:
- pHash tests run both with pHash enabled and disabled to confirm toggling.
- Label checking is skipped (index_remap_path=None) so only duplicate logic is exercised.
"""


import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps, ImageEnhance

from src.pipeline.validate import validate_dataset


# ---------- helpers ----------
def _mk_rgb(size=(224, 224), color=(120, 60, 200)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr, mode="RGB")

def _save(im: Image.Image, path: Path, fmt=None, **savekw):
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or path.suffix.lstrip(".").upper()
    im.save(path, format=fmt, **savekw)
    return path

def _findings(summary, code=None, path_endswith=None):
    def _norm(s: str) -> str:
        return s.replace("\\", "/").lower()
    out = []
    want_suffix = _norm(path_endswith) if path_endswith else None
    for rec in summary.get("findings", []):
        if code is not None and rec.get("code") != code:
            continue
        if want_suffix is not None:
            rec_path = _norm(str(rec.get("path", "")))
            if not rec_path.endswith(want_suffix):
                continue
        out.append(rec)
    return out



# ---------- fixtures ----------
@pytest.fixture
def tiny_ds(tmp_path: Path):
    """
    Creates:
      root/
        glioma/
          a.png                (base)
          a_copy.png           (exact duplicate of a.png)
          a_flip.png           (horizontally flipped of a.png)
          a_jpeg.jpg           (JPEG resave / slight tweak)
        meningioma/
          b.png                (unrelated)
    """
    root = tmp_path / "root"
    g = root / "glioma"
    m = root / "meningioma"

    base = _mk_rgb()
    _save(base, g / "a.png", fmt="PNG")                          # base
    _save(base, g / "a_copy.png", fmt="PNG")                      # exact duplicate

    a_flip = ImageOps.mirror(base)
    _save(a_flip, g / "a_flip.png", fmt="PNG")                    # flip-duplicate

    # near-duplicate via JPEG re-encode + slight brightness tweak
    a_jpeg = ImageEnhance.Brightness(base).enhance(1.02)
    _save(a_jpeg, g / "a_jpeg.jpg", fmt="JPEG", quality=92)       # pHash should catch

    b = _mk_rgb(color=(10, 10, 10))
    _save(b, m / "b.png", fmt="PNG")

    return root


# ---------- tests ----------
def test_crossclass_exact_duplicate_flags_error(tiny_ds: Path):
    """
    Exact byte-for-byte duplicate across labels must emit CROSSCLASS_DUP (error)
    and increment errors_by_type accordingly.
    """
    # Make an exact copy of glioma/a.png into meningioma/
    src = tiny_ds / "glioma" / "a.png"
    dst = tiny_ds / "meningioma" / "a_copy_from_glioma.png"
    dst.write_bytes(src.read_bytes())

    summary = validate_dataset(
        in_dir=tiny_ds,
        index_remap_path=None,
        size=224,
        exts=".png,.jpg,.jpeg",
        dup_check=True,
        warn_low_std=0.0,
        min_file_bytes=1,
        enforce_size=True,
        require_rgb=True,
        phash=False,   # not needed for exact dup
    )

    # New code present and counted as error
    assert summary["errors_by_type"].get("CROSSCLASS_DUP", 0) >= 1

    # Finding includes duplicate_of and both labels in details
    recs = _findings(summary, code="CROSSCLASS_DUP", path_endswith="meningioma/a_copy_from_glioma.png")
    assert len(recs) == 1
    r = recs[0]
    assert r.get("duplicate_of")
    assert r["details"]["cur_label"] == "meningioma"
    assert r["details"]["first_label"] == "glioma"


def test_crossclass_near_duplicate_phash_flags_error(tiny_ds: Path):
    """
    Near-duplicate across labels (pHash within threshold + SSIM >= thresh)
    must emit CROSSCLASS_NEAR_DUP (error).
    """
    # Move the JPEG near-dup to the other class to force cross-class detection
    src = tiny_ds / "glioma" / "a_jpeg.jpg"
    dst = tiny_ds / "meningioma" / "a_jpeg_cross.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        dst.write_bytes(src.read_bytes())
        src.unlink()

    summary = validate_dataset(
        in_dir=tiny_ds,
        index_remap_path=None,
        size=224,
        exts=".png,.jpg,.jpeg",
        dup_check=True,
        warn_low_std=0.0,
        min_file_bytes=1,
        enforce_size=True,
        require_rgb=True,
        phash=True,           # enable perceptual hash
        phash_thresh=8,
        ssim_thresh=0.90,     # uses current default; adjust if you changed it
    )

    # Should be counted as error (not warning)
    assert summary["errors_by_type"].get("CROSSCLASS_NEAR_DUP", 0) >= 1
    assert summary["warnings_by_type"].get("NEAR_DUP_PHASH", 0) >= 0  # may still exist for same-class cases

    recs = _findings(summary, code="CROSSCLASS_NEAR_DUP", path_endswith="meningioma/a_jpeg_cross.jpg")
    assert len(recs) == 1
    r = recs[0]
    assert r.get("duplicate_of")
    assert "hamming" in r["details"]
    assert "ssim" in r["details"]
    assert 0.0 <= r["details"]["ssim"] <= 1.0
    assert r["details"]["cur_label"] == "meningioma"
    assert r["details"]["other_label"] == "glioma"


def test_sameclass_near_duplicate_stays_warning(tiny_ds: Path):
    """
    Same-class near-duplicates must remain NEAR_DUP_PHASH (warning) and not be
    promoted to error codes.
    """
    summary = validate_dataset(
        in_dir=tiny_ds,
        index_remap_path=None,
        size=224,
        exts=".png,.jpg,.jpeg",
        dup_check=True,
        warn_low_std=0.0,
        min_file_bytes=1,
        enforce_size=True,
        require_rgb=True,
        phash=True,
        phash_thresh=8,
        ssim_thresh=0.90,
    )
    # Expect at least one same-class near-dup warning (from glioma/a_jpeg.jpg vs a.png)
    assert summary["warnings_by_type"].get("NEAR_DUP_PHASH", 0) >= 1
    # No cross-class errors should appear in this test (dataset as-is is single-class for near-dup)
    assert summary["errors_by_type"].get("CROSSCLASS_NEAR_DUP", 0) == 0

    

def test_validate_exact_duplicate_filehash(tiny_ds: Path):
    """Flags exact byte-for-byte duplicates via file SHA-1."""
    summary = validate_dataset(
        in_dir=tiny_ds,
        index_remap_path=None,
        size=224,
        exts=".png,.jpg,.jpeg",
        dup_check=True,
        warn_low_std=0.0,
        min_file_bytes=1,
        enforce_size=True,
        require_rgb=True,
        phash=False,   # off
    )
    # Count check
    assert summary["errors_by_type"].get("DUPLICATE", 0) >= 1

    # Finding details: ensure a_copy was flagged, and duplicate_of points to something in glioma/
    recs = _findings(summary, code="DUPLICATE", path_endswith="glioma/a_copy.png")
    assert len(recs) >= 1
    assert recs[0].get("duplicate_of") is not None
    # It should be the base 'a.png' (most likely) — allow any glioma/ path if ordering changes
    assert "/glioma/" in str(recs[0]["duplicate_of"]).replace("\\", "/")
    # SHA-1 content for cleanup key should be present
    assert "sha1" in recs[0]


def test_validate_flip_duplicate_contenthash(tiny_ds: Path):
    """Flags mirrored images via flip-aware normalized content hash."""
    summary = validate_dataset(
        in_dir=tiny_ds,
        index_remap_path=None,
        size=224,
        exts=".png,.jpg,.jpeg",
        dup_check=True,
        warn_low_std=0.0,
        min_file_bytes=1,
        enforce_size=True,
        require_rgb=True,
        phash=False,   # keep off to isolate flip-aware path
    )
    # Count check
    assert summary["errors_by_type"].get("DUPLICATE", 0) >= 1

    # Finding details: specifically verify the flipped image got flagged
    recs = _findings(summary, code="DUPLICATE", path_endswith="glioma/a_flip.png")
    assert len(recs) >= 1
    # Should reference a glioma/ image (likely a.png) as the source it duplicates
    assert recs[0].get("duplicate_of") is not None
    assert recs[0]["duplicate_of"].endswith(".png") or recs[0]["duplicate_of"].endswith(".jpg")


def test_validate_phash_near_duplicate_enabled(tiny_ds: Path):
    """Flags near-duplicates when perceptual hash is enabled."""
    summary = validate_dataset(
        in_dir=tiny_ds,
        index_remap_path=None,
        size=224,
        exts=".png,.jpg,.jpeg",
        dup_check=True,
        warn_low_std=0.0,
        min_file_bytes=1,
        enforce_size=True,
        require_rgb=True,
        phash=True,            # enable perceptual hash
        phash_thresh=8,
    )
    # Count check
    assert summary["warnings_by_type"].get("NEAR_DUP_PHASH", 0) >= 1

    # Finding details: ensure the JPEG near-dup is recorded with hamming distance
    recs = _findings(summary, code="NEAR_DUP_PHASH", path_endswith="glioma/a_jpeg.jpg")
    assert len(recs) >= 1
    r = recs[0]
    assert r.get("duplicate_of") is not None
    assert "ssim" in r["details"]
    assert 0.9 <= r["details"]["ssim"] <= 1.0


def test_validate_phash_near_duplicate_disabled(tiny_ds: Path):
    """Does not flag near-duplicates when perceptual hash is disabled."""
    summary = validate_dataset(
        in_dir=tiny_ds,
        index_remap_path=None,
        size=224,
        exts=".png,.jpg,.jpeg",
        dup_check=True,
        warn_low_std=0.0,
        min_file_bytes=1,
        enforce_size=True,
        require_rgb=True,
        phash=False,          # disabled
    )
    assert summary["warnings_by_type"].get("NEAR_DUP_PHASH", 0) == 0
    # But regular dups should still be detected
    assert summary["errors_by_type"].get("DUPLICATE", 0) >= 1
