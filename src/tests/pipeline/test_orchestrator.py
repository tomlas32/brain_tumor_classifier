# src/tests/pipeline/test_orchestrator.py
import json
from pathlib import Path
import pytest

from src.pipeline import orchestrator as orch
from src.core import config as cfg


# ==============================
# Fixtures
# ==============================
@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """
    Full sandbox: set pytest temp dir as CWD and redirect both src.utils.paths and
    orchestrator path constants so all artifacts land under tmp_path.
    """
    base = tmp_path
    monkeypatch.chdir(base)  # critical: relative 'outputs/...' land here

    outputs = base / "outputs"
    data = base / "data"
    models = base / "models"
    merged = data / "merged"
    processed = data / "processed"
    for p in (outputs, data, models, merged, processed):
        p.mkdir(parents=True, exist_ok=True)

    # Patch src.utils.paths so any module using it writes into the sandbox
    from src.utils import paths as up
    monkeypatch.setattr(up, "OUTPUTS_DIR", outputs, raising=False)
    monkeypatch.setattr(up, "DATA_DIR", data, raising=False)
    monkeypatch.setattr(up, "MODELS_DIR", models, raising=False)
    monkeypatch.setattr(up, "MERGED_DIR", merged, raising=False)
    monkeypatch.setattr(up, "PROCESSED_DIR", processed, raising=False)

    # Patch orchestrator path constants (in case it imported names directly)
    monkeypatch.setattr(orch, "OUTPUTS_DIR", outputs, raising=False)
    monkeypatch.setattr(orch, "MERGED_DIR", merged, raising=False)
    monkeypatch.setattr(orch, "PROCESSED_DIR", processed, raising=False)

    # Hermetic ensure_base_dirs
    def _ensure_base_dirs():
        for p in (outputs, data, models, merged, processed):
            p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(orch, "ensure_base_dirs", _ensure_base_dirs, raising=True)

    return base


@pytest.fixture()
def safe_io(monkeypatch):
    """
    Make safe_json_dump write plain JSON; use your real to_dict for dataclasses.
    """
    def _safe_json_dump(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use cfg.to_dict for dataclasses if available
        if hasattr(payload, "__dataclass_fields__"):
            payload = cfg.to_dict(payload)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    monkeypatch.setattr(orch, "safe_json_dump", _safe_json_dump, raising=True)
    monkeypatch.setattr(orch, "to_dict", cfg.to_dict, raising=True)


@pytest.fixture()
def no_env_bootstrap(monkeypatch):
    """
    Avoid side effects from env/bootstrap/logging.
    """
    monkeypatch.setattr(orch, "bootstrap_env", lambda seed=None: None, raising=True)
    monkeypatch.setattr(orch, "log_env_once", lambda: None, raising=True)
    # configure_logging can run; logs will go under sandboxed outputs


class FakeStageModule:
    """Records argv; returns a configured code or raises the configured exception."""
    def __init__(self, behavior=0):
        self.behavior = behavior
        self.calls = []

    def main(self, argv):
        self.calls.append(list(argv))
        b = self.behavior
        if isinstance(b, BaseException):
            raise b
        return b


@pytest.fixture()
def fake_stage_set(monkeypatch):
    """
    Plug fake stage modules into the orchestrator import slots.
    """
    mods = {
        "fetch": FakeStageModule(0),
        "merge": FakeStageModule(0),
        "validate": FakeStageModule(0),  # shared by validate_pre/post
        "cleanup": FakeStageModule(3),   # 3 should be treated as OK by cleanup
        "resize": FakeStageModule(0),
        "split": FakeStageModule(0),
        "train": FakeStageModule(0),
        "evaluate": FakeStageModule(0),
    }
    monkeypatch.setattr(orch, "fetch_mod", mods["fetch"], raising=True)
    monkeypatch.setattr(orch, "merge_mod", mods["merge"], raising=True)
    monkeypatch.setattr(orch, "validate_mod", mods["validate"], raising=True)
    monkeypatch.setattr(orch, "cleanup_mod", mods["cleanup"], raising=True)
    monkeypatch.setattr(orch, "resize_mod", mods["resize"], raising=True)
    monkeypatch.setattr(orch, "split_mod", mods["split"], raising=True)
    monkeypatch.setattr(orch, "train_mod", mods["train"], raising=True)
    monkeypatch.setattr(orch, "evaluate_mod", mods["evaluate"], raising=True)
    return mods


def _make_master(run_id="TST001", dataset="ownerX/slugY", *, fetch_pointer_dir=None, mapping_pointer=None):
    """
    Build a real MasterConfig using your dataclasses.
    """
    return cfg.MasterConfig(
        run_id=run_id,
        env=cfg.EnvConfig(seed=42),
        log=cfg.LoggingConfig(level="INFO", file=None),
        fetch=cfg.FetchConfig(
            dataset=dataset,
            pointer_dir=Path(fetch_pointer_dir) if fetch_pointer_dir else None,
        ),
        merge=cfg.MergeConfig(dataset=dataset),
        validate=cfg.ValidateConfig(
            mapping_pointer=Path(mapping_pointer) if mapping_pointer else None,
            write_report=True,
            fail_on="never",
        ),
        cleanup=cfg.CleanupConfig(policy="strict"),
        resize=cfg.ResizeConfig(size=224),
        split=cfg.SplitConfig(dataset=dataset, val_frac=0.1, test_frac=0.2),
        train=cfg.TrainConfig(
            run_id=None,
            data=cfg.DataConfig(mapping_pointer=Path(mapping_pointer) if mapping_pointer else None),
        ),
        evaluate=cfg.EvalConfig(
            run_id=None,
            data=cfg.DataConfig(mapping_pointer=Path(mapping_pointer) if mapping_pointer else None),
        ),
    )


@pytest.fixture()
def stub_build_master(monkeypatch):
    """
    Patch orchestrator.build_master_config to return a real MasterConfig we control.
    """
    holder = {"master": _make_master()}
    monkeypatch.setattr(orch, "build_master_config", lambda y, o: holder["master"], raising=True)
    return holder


# ==============================
# Unit-ish helpers
# ==============================
def test_dataset_owner_slug_parsing():
    assert orch._dataset_owner_slug("a/b") == ("a", "b")
    assert orch._dataset_owner_slug("x/y/z") == ("x", "y/z")
    assert orch._dataset_owner_slug("bad") == ("_unknown_", "_unknown_")
    assert orch._dataset_owner_slug(None) == ("_unknown_", "_unknown_")


def test_validate_stage_names_ok_and_bad():
    assert orch._validate_stage_names(["fetch", "cleanup"], None) is None
    assert "Invalid --skip" in orch._validate_stage_names(["nope"], None)
    assert "Invalid --resume-from" in orch._validate_stage_names([], "nope")


def test_expected_pointer_paths_precedence(sandbox, safe_io, no_env_bootstrap, stub_build_master):
    # default derived from dataset
    stub_build_master["master"] = _make_master(run_id="R1", dataset="o/s")
    p1 = orch._expected_pointer_paths(stub_build_master["master"])
    assert "pointers/fetch/o/s" in p1["fetch"]["dir"].replace("\\", "/")
    assert "pointers/mapping/o/s" in p1["mapping"]["dir"].replace("\\", "/")

    # explicit fetch pointer_dir wins
    custom_dir = sandbox / "custom_fetch_ptr"
    custom_dir.mkdir(parents=True, exist_ok=True)
    stub_build_master["master"] = _make_master(run_id="R2", dataset="o/s", fetch_pointer_dir=str(custom_dir))
    p2 = orch._expected_pointer_paths(stub_build_master["master"])
    assert p2["fetch"]["dir"].endswith("custom_fetch_ptr")

    # explicit mapping_pointer wins
    mp = sandbox / "explicit_map_ptr"
    mp.mkdir(parents=True, exist_ok=True)
    stub_build_master["master"] = _make_master(run_id="R3", dataset="o/s", mapping_pointer=str(mp))
    p3 = orch._expected_pointer_paths(stub_build_master["master"])
    assert p3["mapping"]["dir"].endswith("explicit_map_ptr")


# ==============================
# Planner + Runner integration-ish
# ==============================
def test_dry_run_writes_plan_and_per_stage_configs(sandbox, safe_io, no_env_bootstrap, stub_build_master, fake_stage_set):
    stub_build_master["master"] = _make_master(run_id="DRY001")
    code = orch.run_pipeline(master_yaml=None, overrides=None, dry_run=True)
    assert code == 0

    run_dir = sandbox / "outputs" / "orchestrator" / "DRY001"
    assert run_dir.exists()

    plan = json.loads((run_dir / "plan.json").read_text())
    stages = [s["stage"] for s in plan["stages"]]
    assert stages == orch.DEFAULT_ORDER  # validate_pre/post included

    for st in orch.DEFAULT_ORDER:
        assert (run_dir / f"{st}.yaml").exists()  # emitted via _write_stage_yaml

    # no stage executed in dry-run
    for m in fake_stage_set.values():
        assert m.calls == []


def test_success_path_accepts_cleanup_code3(sandbox, safe_io, no_env_bootstrap, stub_build_master, fake_stage_set):
    stub_build_master["master"] = _make_master(run_id="OK777")
    code = orch.run_pipeline(master_yaml=None, overrides=None, dry_run=False)
    assert code == 0

    manifest = json.loads((sandbox / "outputs" / "orchestrator" / "OK777" / "run_manifest.json").read_text())
    assert manifest["overall_exit_code"] == 0
    ran = [r["stage"] for r in manifest["stages_run"]]
    assert ran == orch.DEFAULT_ORDER

    cleanup_row = next(r for r in manifest["stages_run"] if r["stage"] == "cleanup")
    assert cleanup_row["exit_code"] == 3
    assert 3 in orch.STAGE_OK["cleanup"]


def test_failure_short_circuits_subsequent_stages(sandbox, safe_io, no_env_bootstrap, stub_build_master, fake_stage_set):
    fake_stage_set["resize"].behavior = SystemExit(2)  # stage failure via SystemExit
    stub_build_master["master"] = _make_master(run_id="FAIL002")

    code = orch.run_pipeline(master_yaml=None, overrides=None, dry_run=False)
    assert code == 2

    manifest = json.loads((sandbox / "outputs" / "orchestrator" / "FAIL002" / "run_manifest.json").read_text())
    assert manifest["overall_exit_code"] == 2
    ran = [r["stage"] for r in manifest["stages_run"]]
    assert ran == ["fetch", "merge", "validate_pre", "cleanup", "resize"]


def test_skip_and_resume_change_plan(sandbox, safe_io, no_env_bootstrap, stub_build_master, fake_stage_set):
    stub_build_master["master"] = _make_master(run_id="PLAN100")
    code = orch.run_pipeline(master_yaml=None, overrides=None, dry_run=True, skip=["fetch", "merge"])
    assert code == 0
    plan = json.loads((sandbox / "outputs" / "orchestrator" / "PLAN100" / "plan.json").read_text())
    stages = [s["stage"] for s in plan["stages"]]
    assert "fetch" not in stages and "merge" not in stages
    assert stages[0] == "validate_pre"   # first remaining stage

    stub_build_master["master"] = _make_master(run_id="PLAN200")
    code2 = orch.run_pipeline(master_yaml=None, overrides=None, dry_run=True, skip=["fetch", "merge"], resume_from="split")
    assert code2 == 0
    plan2 = json.loads((sandbox / "outputs" / "orchestrator" / "PLAN200" / "plan.json").read_text())
    stages2 = [s["stage"] for s in plan2["stages"]]
    assert stages2 == ["split", "train", "evaluate"]  # resume ignores skip list


def test_invalid_stage_names_return_code_2(sandbox, safe_io, no_env_bootstrap, stub_build_master, fake_stage_set):
    stub_build_master["master"] = _make_master(run_id="BAD001")
    code = orch.run_pipeline(master_yaml=None, overrides=None, dry_run=True, skip=["not_a_stage"])
    assert code == 2  # args_invalid handled early
