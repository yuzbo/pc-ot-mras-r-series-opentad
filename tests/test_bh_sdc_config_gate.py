from __future__ import annotations

import importlib.util
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
