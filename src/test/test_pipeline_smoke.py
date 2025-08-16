"""
Smoke tests for the end-to-end pipeline orchestrator and CLI.

These tests do NOT download data or train models. Instead they:
- run a DRY-RUN through the orchestrator (plan-only),
- run a mocked full pipeline by monkeypatching each stage's `main(argv)` to succeed,
- invoke the Typer CLI `pipeline` command with DRY-RUN.

Best practices:
- Each test chdirs into a temporary directory so artifacts (outputs/orchestrator/*)
  do not pollute the repo.
- We use a fixed run_id in the master config to make assertions deterministic.

EXAMPLES:

To test the entire pipeline, run:
pytest -q

To run a specific test file
pytest -q test/test_pipeline_smoke.py

To run a specific test function
pytest -q tests/test_pipeline_smoke.py::test_orchestrator_dry_run_plan


"""

from __future__ import annotations

from pathlib import Path
import json

import pytest
from typer.testing import CliRunner

# Import orchestrator + CLI app
from src.pipeline.orchestrator import run_pipeline
from src.pipeline import orchestrator as orch_mod  # to access imported stage modules
from src.cli import app as cli_app


@pytest.fixture
def master_yaml_min(tmp_path: Path) -> Path:
    """
    Create a minimal master YAML that is valid for DRY-RUN planning.

    Notes:
    - We keep stage blocks present but lean; orchestrator only needs them to serialize
      per-stage configs. Real execution is covered by the mocked test below.
    """
    content = """\
run_id: test-smoke-000
env:
  seed: 1
  prefer_cuda: false
  cudnn_deterministic: true
  cudnn_benchmark: false
log:
  level: INFO
  file: null
  json: false

fetch:
  dataset: owner/dataset
  cache_dir: data
  write_pointer: true

split:
  dataset: owner/dataset
  test_frac: 0.2
  clear_dest: false
  exts: all

resize:
  size: 64
  exts: all

validate:
  in_dir: data/training_resized
  index_remap: outputs/mappings/latest.json
  size: 64
  fail_on: warning
  dup_check: false

train:
  data:
    image_size: 64
    train_in: data/training_resized
    mapping_path: outputs/mappings/latest.json
    batch_size: 2
    num_workers: 0
    val_frac: 0.5
    seed: 1
  model:
    name: resnet18
    pretrained: false
  optim:
    lr: 0.01
    weight_decay: 0.0
    step_size: 1
    gamma: 0.5
    amp: false
  io:
    out_models: models
    out_summary: outputs/training
  loop:
    epochs: 1
  aug:
    rotate_deg: 0
    hflip_prob: 0.0
    jitter_brightness: 0.0
    jitter_contrast: 0.0

evaluate:
  data:
    image_size: 64
    eval_in: data/testing_resized
    mapping_path: outputs/mappings/latest.json
    batch_size: 2
    num_workers: 0
    seed: 1
  model:
    name: resnet18
    weights_path: models/best.pth
  io:
    eval_out: outputs/evaluation
    make_galleries: false
    make_gradcam: false
    top_per_class: 1
"""
    path = tmp_path / "pipeline.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_orchestrator_dry_run_plan(tmp_path: Path, master_yaml_min: Path, monkeypatch: pytest.MonkeyPatch):
    """
    DRY-RUN: The orchestrator should build a plan and exit(0) without running stages.
    Also verifies it writes per-stage sub-configs into outputs/orchestrator/<run_id>/.
    """
    # Run in a temp working dir so artifacts land under tmp_path/outputs/...
    monkeypatch.chdir(tmp_path)

    code = run_pipeline(master_yaml=master_yaml_min, overrides=None, dry_run=True, skip=[], resume_from=None)
    assert code == 0

    # Verify the orchestrator output directory exists and contains sub-configs
    run_dir = Path("outputs/orchestrator/test-smoke-000")
    assert run_dir.exists(), "orchestrator run directory not created"
    for stage in ["fetch", "split", "resize", "validate", "train", "evaluate"]:
        assert (run_dir / f"{stage}.yaml").exists(), f"missing sub-config for stage: {stage}"


def test_orchestrator_full_run_with_mocks(tmp_path: Path, master_yaml_min: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Full pipeline with all stages mocked to return success (0).

    We monkeypatch each imported stage module's main(argv) to avoid network, IO,
    and training work. The orchestrator should:
      - run through all stages,
      - write a run manifest with exit_code 0,
      - and record per-stage durations.
    """
    monkeypatch.chdir(tmp_path)

    # Mock each stage's main(argv) to return 0 quickly
    def _ok_main(argv):  # pragma: no cover - trivial mock
        return 0

    monkeypatch.setattr(orch_mod.fetch_mod, "main", _ok_main, raising=True)
    monkeypatch.setattr(orch_mod.split_mod, "main", _ok_main, raising=True)
    monkeypatch.setattr(orch_mod.resize_mod, "main", _ok_main, raising=True)
    monkeypatch.setattr(orch_mod.validate_mod, "main", _ok_main, raising=True)
    monkeypatch.setattr(orch_mod.train_mod, "main", _ok_main, raising=True)
    monkeypatch.setattr(orch_mod.evaluate_mod, "main", _ok_main, raising=True)

    code = run_pipeline(master_yaml=master_yaml_min, overrides=None, dry_run=False, skip=[], resume_from=None)
    assert code == 0

    manifest_path = Path("outputs/orchestrator/test-smoke-000/run_manifest.json")
    assert manifest_path.exists(), "run manifest not written"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["overall_exit_code"] == 0
    stages_run = {entry["stage"] for entry in manifest["stages_run"]}
    assert stages_run == {"fetch", "split", "resize", "validate", "train", "evaluate"}


def test_cli_pipeline_dry_run(tmp_path: Path, master_yaml_min: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Typer CLI smoke test: invoke `pipeline` in DRY-RUN mode and expect exit code 0.
    """
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["pipeline", "--config", str(master_yaml_min), "--dry-run"])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
