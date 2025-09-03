# src/tests/pipeline/test_fetch.py
from __future__ import annotations

import importlib
import os
import sys
import types
from io import StringIO
from pathlib import Path
from typing import Callable, Optional

import contextlib
import pytest


# ---------------------------
# Import + monkeypatch helper
# ---------------------------

def _import_fetch(monkeypatch, kaggle_impl, artifacts_impl: Optional[Callable] | Exception | None = None):
    """
    Import your real fetch module, then monkeypatch:
      - fetch.kagglehub.dataset_download
      - src.core.artifacts.write_fetch_pointer
      - (and also fetch.write_fetch_pointer if it exists)
    """
    fetch = importlib.import_module("src.pipeline.fetch")
    importlib.reload(fetch)

    # Patch kagglehub symbol used by fetch
    if isinstance(kaggle_impl, Exception):
        def _raise(*a, **k):
            raise kaggle_impl
        dummy_kaggle = types.SimpleNamespace(dataset_download=_raise)
    else:
        dummy_kaggle = types.SimpleNamespace(dataset_download=kaggle_impl)
    monkeypatch.setattr(fetch, "kagglehub", dummy_kaggle, raising=False)

    # Patch write_fetch_pointer on the real artifacts module
    import src.core.artifacts as artifacts
    if artifacts_impl is None:
        def write_fetch_pointer(*, dataset: str, dataset_root: str | Path, run_id: str, dst_dir: Path | None = None):
            latest = Path(dst_dir or Path("outputs/pointers")) / "latest.json"
            hist = Path(dst_dir or Path("outputs/pointers")) / "history" / f"fetch_{run_id}.json"
            return {"latest": str(latest), "history": str(hist)}
    elif isinstance(artifacts_impl, Exception):
        def write_fetch_pointer(*a, **k):
            raise artifacts_impl
    else:
        write_fetch_pointer = artifacts_impl

    # Patch on both the artifacts module and (if present) the fetch module alias
    monkeypatch.setattr(artifacts, "write_fetch_pointer", write_fetch_pointer, raising=True)
    if hasattr(fetch, "write_fetch_pointer"):
        monkeypatch.setattr(fetch, "write_fetch_pointer", write_fetch_pointer, raising=False)

    return fetch


def _run_main(fetch_mod, argv):
    """
    Run fetch.main and return its int exit code.
    Supports main(argv) or main() using sys.argv.
    """
    if "argv" in fetch_mod.main.__code__.co_varnames:
        return int(fetch_mod.main(argv))
    orig = sys.argv
    try:
        sys.argv = ["fetch"] + list(argv)
        return int(fetch_mod.main())
    finally:
        sys.argv = orig


# ---------------------------
# Unit tests: fetch_kaggle()
# ---------------------------

def test_fetch_kaggle_sets_cache_and_downloads(tmp_path, monkeypatch, caplog):
    """
    Unit: ensure KAGGLEHUB_CACHE is set and dataset_download is called; path returned.
    """
    cache_dir = tmp_path / "data"

    # Fake download returns a materialized "downloaded" path
    def fake_download(slug: str):
        p = tmp_path / "kagglehub_cache" / slug.replace("/", "_") / "1.0"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    fetch = _import_fetch(monkeypatch, fake_download)

    caplog.set_level("INFO")
    out = fetch.fetch_kaggle(dataset="owner/name", cache_dir=cache_dir)

    assert Path(out).exists()
    assert cache_dir.exists()
    assert os.environ.get("KAGGLEHUB_CACHE") == str(cache_dir.resolve())
    # sanity: download logs present
    assert any("download" in r.getMessage().lower() for r in caplog.records)


def test_fetch_kaggle_download_failure_raises(tmp_path, monkeypatch, caplog):
    """
    Unit: propagate underlying download exceptions.
    """
    fetch = _import_fetch(monkeypatch, RuntimeError("boom"))

    caplog.set_level("INFO")
    with pytest.raises(RuntimeError):
        fetch.fetch_kaggle(dataset="owner/name", cache_dir=tmp_path / "data")
    # either error record exists or we have some captured logs
    assert any("fail" in r.getMessage().lower() for r in caplog.records) or caplog.records


# ---------------------------
# CLI tests: assert on stdout
# ---------------------------

def test_cli_dry_run_skips_network(tmp_path, monkeypatch):
    """
    Dry-run: should not call kagglehub.dataset_download and should print plan.
    """
    calls = {"n": 0}
    def fake_download(slug: str):
        calls["n"] += 1
        return str(tmp_path / "dummy")

    fetch = _import_fetch(monkeypatch, fake_download)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, ["--dataset", "owner/name", "--cache-dir", str(tmp_path / "data"), "--dry-run"])

    s = buf.getvalue()
    assert code == 0
    assert calls["n"] == 0
    assert "[DRY-RUN]" in s
    assert "dry_run.fetch.plan" in s


def test_cli_success_writes_pointer_by_default(tmp_path, monkeypatch):
    """
    Success path: pointer writing on; should call write_fetch_pointer and print path.
    """
    downloaded = tmp_path / "kcache" / "owner" / "name" / "1.0"
    def fake_download(slug: str):
        downloaded.mkdir(parents=True, exist_ok=True)
        return str(downloaded)

    write_calls = {"kwargs": None}
    def fake_write_pointer(**kwargs):
        write_calls["kwargs"] = kwargs
        return {"latest": "L", "history": "H"}

    fetch = _import_fetch(monkeypatch, fake_download, artifacts_impl=fake_write_pointer)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, ["--dataset", "owner/name", "--cache-dir", str(tmp_path / "data")])

    out = buf.getvalue().strip()
    assert code == 0
    assert write_calls["kwargs"] is not None
    assert write_calls["kwargs"]["dataset"] == "owner/name"
    assert Path(write_calls["kwargs"]["dataset_root"]) == downloaded
    assert "run_id" in write_calls["kwargs"]
    assert out.endswith(str(downloaded))


def test_cli_pointer_can_be_disabled(tmp_path, monkeypatch):
    """
    --no-pointer disables pointer writing.
    """
    downloaded = tmp_path / "kcache" / "o" / "n" / "1.0"
    def fake_download(slug: str):
        downloaded.mkdir(parents=True, exist_ok=True)
        return str(downloaded)

    called = {"v": False}
    def fake_write_pointer(**kwargs):
        called["v"] = True
        return {}

    fetch = _import_fetch(monkeypatch, fake_download, artifacts_impl=fake_write_pointer)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, ["--dataset", "owner/name", "--no-pointer"])

    assert code == 0
    assert called["v"] is False


def test_cli_pointer_dir_override(tmp_path, monkeypatch):
    """
    --pointer-dir should be passed through to write_fetch_pointer(dst_dir=...).
    """
    downloaded = tmp_path / "kcache" / "o" / "n" / "1.0"
    def fake_download(slug: str):
        downloaded.mkdir(parents=True, exist_ok=True)
        return str(downloaded)

    got = {"dst_dir": None}
    def fake_write_pointer(**kwargs):
        got["dst_dir"] = kwargs.get("dst_dir")
        return {"latest": "L", "history": "H"}

    fetch = _import_fetch(monkeypatch, fake_download, artifacts_impl=fake_write_pointer)
    ptr_dir = tmp_path / "custom_ptrs"

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, ["--dataset", "owner/name", "--pointer-dir", str(ptr_dir)])

    assert code == 0
    assert Path(got["dst_dir"]) == ptr_dir


def test_cli_download_error_returns_exit1(tmp_path, monkeypatch):
    """
    If download blows up, main() should not crash; it should return exit code 1
    and print failure log lines to stdout.
    """
    fetch = _import_fetch(monkeypatch, RuntimeError("net-down"))

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, ["--dataset", "owner/name"])

    s = buf.getvalue()
    assert code == 1
    assert ("download.failed" in s) or ("fetch.failed" in s)


def test_cli_pointer_write_failure_is_soft_warning(tmp_path, monkeypatch):
    """
    Pointer write failure shouldn't fail the fetch; log a warning and exit 0.
    """
    downloaded = tmp_path / "kcache" / "o" / "n" / "1.0"
    def fake_download(slug: str):
        downloaded.mkdir(parents=True, exist_ok=True)
        return str(downloaded)

    fetch = _import_fetch(monkeypatch, fake_download, artifacts_impl=ValueError("pointer-err"))

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, ["--dataset", "owner/name"])

    s = buf.getvalue()
    assert code == 0
    assert "fetch.pointer_write_failed" in s


def test_cli_sets_kagglehub_cache_from_cache_dir(tmp_path, monkeypatch):
    """
    Ensure env var honors --cache-dir at CLI level too.
    """
    downloaded = tmp_path / "kcache" / "o" / "n" / "1.0"
    def fake_download(slug: str):
        downloaded.mkdir(parents=True, exist_ok=True)
        return str(downloaded)

    fetch = _import_fetch(monkeypatch, fake_download)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, ["--dataset", "owner/name", "--cache-dir", str(tmp_path / "DATA")])

    assert code == 0
    assert os.environ.get("KAGGLEHUB_CACHE") == str((tmp_path / "DATA").resolve())


# ---------------------------
# Config behavior (aligned with defaults)
# ---------------------------

@pytest.mark.parametrize(
    "cli_args, expect_cache_env_suffix, expect_uses_default",
    [
        ([], None, True),  # config with nulls -> default to DATA_DIR
        (["--dataset", "owner/cli", "--cache-dir", "cli_data"], "cli_data", False),
    ],
)
def test_cli_config_and_overrides(tmp_path, monkeypatch, cli_args, expect_cache_env_suffix, expect_uses_default):
    """
    - When config has nulls and no CLI overrides, fetch defaults cache_dir to DATA_DIR.
    - With CLI overrides, env should reflect the CLI cache dir.
    """
    downloaded = tmp_path / "kcache" / "o" / "n" / "1.0"
    def fake_download(slug: str):
        downloaded.mkdir(parents=True, exist_ok=True)
        return str(downloaded)

    fetch = _import_fetch(monkeypatch, fake_download)

    # Dummy config path (fetch prints config.resolved regardless)
    cfg_file = tmp_path / "fetch.yaml"
    cfg_file.write_text("dummy: true")
    argv = ["--config", str(cfg_file)] + list(cli_args)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(fetch, argv)

    s = buf.getvalue()
    assert code == 0
    assert "config.resolved" in s

    if expect_uses_default:
        from src.utils.paths import DATA_DIR
        assert os.environ.get("KAGGLEHUB_CACHE") == str(DATA_DIR.resolve())
    else:
        assert os.environ.get("KAGGLEHUB_CACHE", "").endswith(expect_cache_env_suffix)
