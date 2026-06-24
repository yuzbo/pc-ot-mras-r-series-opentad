from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_bh_sdc_full_train_gate.py"
LOCAL_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "bh_sdc_boundary_hazard_sparse_dense_local_precheck.py"
FULL_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py"
)
BH_SDC_ROUTE_LABEL = "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_bh_sdc_full_train_gate", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bh_sdc_configs_remain_fail_closed_and_use_sparse_dense_chain():
    validator = _load_validator()

    local = validator.validate_locked_config(LOCAL_CONFIG)
    full = validator.validate_locked_config(FULL_CONFIG)

    for result in (local, full):
        assert result["route"] == "bh_sdc_boundary_hazard_sparse_dense"
        assert result["route_label"] == BH_SDC_ROUTE_LABEL
        assert result["selector"] == "PCOTMRASBoundaryHazardSparseDenseFrameSelector"
        assert result["completion_bridge"] == "PCOTMRASBoundaryHazardSparseToDenseBridge"
        assert result["dense_window_size"] == 768
        assert result["min_budget"] < result["target_budget"] < result["max_budget"]
        assert result["launch_gate_passed"] is False
        assert result["allow_long_training"] is False
        assert result["base_chain_forbidden_tokens"] == []
        assert result["allowed_entrypoints"] == []

    local_text = LOCAL_CONFIG.read_text(encoding="utf-8")
    assert "pc_ot_mras_prebackbone_c3_hybrid_reader_candidate_n16r4.py" not in local_text
    assert "e2e_thumos_videomae_s_768x1_160_adapter.py" in local_text


def test_bh_sdc_resolved_config_builds_real_selector_and_completion_bridge():
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if torch_probe.returncode != 0:
        pytest.skip("torch unavailable")

    cfg = Config.fromfile(FULL_CONFIG)
    pretty = cfg.pretty_text
    assert cfg.experiment_scope.route_label == BH_SDC_ROUTE_LABEL
    assert cfg.experiment_scope.combo_status == "NO_COMBO_ROUTE_APPROVED"
    assert "PCOTMRASBoundaryHazardSparseDenseFrameSelector" in pretty
    assert "PCOTMRASBoundaryHazardSparseToDenseBridge" in pretty
    assert "PCOTMRASPreBackboneFrameSelector" not in pretty
    assert "PCOTMRASHybridFrameScout" not in pretty
    assert cfg.model.backbone.custom.pretrain is None

    for name in list(sys.modules):
        if name == "opentad.models.builder" or name.startswith("opentad.models.selectors"):
            sys.modules.pop(name, None)

    for package, path in (
        ("opentad", ROOT / "opentad"),
        ("opentad.models", ROOT / "opentad" / "models"),
        ("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors"),
    ):
        module = sys.modules.get(package)
        if module is None:
            module = types.ModuleType(package)
            module.__path__ = [str(path)]
            sys.modules[package] = module

    backbones = types.ModuleType("opentad.models.backbones")
    backbones.BackboneWrapper = lambda cfg: None
    sys.modules["opentad.models.backbones"] = backbones

    builder_spec = importlib.util.spec_from_file_location(
        "opentad.models.builder",
        ROOT / "opentad" / "models" / "builder.py",
    )
    builder = importlib.util.module_from_spec(builder_spec)
    sys.modules[builder_spec.name] = builder
    builder_spec.loader.exec_module(builder)

    module_spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.bh_sdc_frame_selector",
        ROOT / "opentad" / "models" / "selectors" / "bh_sdc_frame_selector.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)

    selector = builder.build_selector(cfg.model.frame_selector)
    completion = builder.build_token_compressor(cfg.model.token_compressor)

    assert type(selector).__name__ == "PCOTMRASBoundaryHazardSparseDenseFrameSelector"
    assert type(completion).__name__ == "PCOTMRASBoundaryHazardSparseToDenseBridge"
    assert selector.max_budget == cfg.model.backbone.backbone.total_frames
    assert completion.target_len == cfg.model.projection.max_seq_len == 768


def test_bh_sdc_validator_rejects_forbidden_base_chain_tokens(tmp_path):
    validator = _load_validator()
    base = tmp_path / "pc_ot_mras_prebackbone_c3_hybrid_reader_candidate_n16r4.py"
    base.write_text("model = dict(type='ActionFormer')\n", encoding="utf-8")
    child = tmp_path / "bh_sdc_bad_base.py"
    child.write_text(
        f"_base_ = ['{base.as_posix()}']\n"
        "experiment_scope = dict(route='bh_sdc_boundary_hazard_sparse_dense', "
        "route_label='DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3', "
        "route_family='BH_SDC_DIVERGENT_INNOVATION_ROUTE', combo_status='NO_COMBO_ROUTE_APPROVED')\n"
        "bh_sdc_gate = dict(route='bh_sdc_boundary_hazard_sparse_dense', "
        "route_label='DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3', "
        "requires_launch_gate=True, launch_gate_passed=False, allow_precheck_only=True, allowed_entrypoints=())\n"
        "model = dict(frame_selector=dict(type='PCOTMRASBoundaryHazardSparseDenseFrameSelector', "
        "dense_window_size=16, min_budget=4, target_budget=8, max_budget=12), "
        "token_compressor=dict(type='PCOTMRASBoundaryHazardSparseToDenseBridge', dense_window_size=16, target_len=16), "
        "backbone=dict(backbone=dict(total_frames=12), custom=dict(pretrain=None)), "
        "projection=dict(max_seq_len=16))\n"
        "inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden route/base token"):
        validator.validate_locked_config(child)


def test_bh_sdc_validator_rejects_non_strict_budget_bounds(tmp_path):
    validator = _load_validator()
    bad = tmp_path / "bh_sdc_bad_budget.py"
    bad.write_text(
        f"_base_ = ['{FULL_CONFIG.as_posix()}']\n"
        "model = dict(frame_selector=dict(target_budget=256))\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_budget < target_budget < max_budget <= dense_window_size"):
        validator.validate_locked_config(bad)


def test_bh_sdc_validator_cli_emits_locked_json_contract():
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            str(FULL_CONFIG),
            "--json",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["route_label"] == BH_SDC_ROUTE_LABEL
    assert payload["launch_gate_passed"] is False
    assert payload["allowed_entrypoints"] == []
    assert payload["base_chain_forbidden_tokens"] == []
    assert payload["min_budget"] < payload["target_budget"] < payload["max_budget"] <= payload["dense_window_size"]
