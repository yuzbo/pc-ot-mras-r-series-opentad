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
LAUNCH_DECISION = "ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1"
REVIEWED_IMPL_COMMIT = "ae4354307d903f537e2be78723c39a3e19787f9b"
EXPECTED_REMOTE_WORKSPACE = "~/run/yuzibo/OpenTAD_Back_check"
EXPECTED_SYNC_COMMAND = f"REMOTE_SYNC_TO_N16R4:{EXPECTED_REMOTE_WORKSPACE}"
EXPECTED_SLURM_COMMAND = "sbatch scripts/run_bh_sdc_full_train_n16r4.sbatch"
EXPECTED_TRAIN_COMMAND = (
    "python tools/train.py "
    "configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --id 0"
)
EXPECTED_COMMAND_WHITELIST = [
    EXPECTED_SYNC_COMMAND,
    EXPECTED_SLURM_COMMAND,
    EXPECTED_TRAIN_COMMAND,
]


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_bh_sdc_full_train_gate", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_launch_payload(resolved_config_sha256: str, active_manifest_sha256: str | None = None):
    if active_manifest_sha256 is None:
        active_manifest_sha256 = "a" * 64
    return {
        "schema_version": 1,
        "decision": LAUNCH_DECISION,
        "explicit_user_pro_launch_decision": LAUNCH_DECISION,
        "pro_launch_gate_verdict": LAUNCH_DECISION,
        "pro_launch_gate_session": "synthetic-pro-launch-gate-session",
        "route": "bh_sdc_boundary_hazard_sparse_dense",
        "route_label": BH_SDC_ROUTE_LABEL,
        "stage": "bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4",
        "reviewed_impl_commit": REVIEWED_IMPL_COMMIT,
        "launch_gate_commit": "b" * 40,
        "pro_implementation_verdict": "PASS_SECOND_GPT_5_5_PRO_REVIEW_NO_BLOCKERS",
        "pro_implementation_session": "synthetic-pro-implementation-session",
        "final_read_only_review_verdict": "PASS_SUBAGENT_FINAL_REVIEW_ONLY",
        "final_read_only_review_id": "synthetic-final-read-only-review",
        "resolved_config_sha256": resolved_config_sha256,
        "active_sha256_manifest_sha256": active_manifest_sha256,
        "command_whitelist": list(EXPECTED_COMMAND_WHITELIST),
        "remote_workspace": EXPECTED_REMOTE_WORKSPACE,
        "remote_workspace_policy": "N16R4_YUZIBO_ONLY",
        "slurm_script": "scripts/run_bh_sdc_full_train_n16r4.sbatch",
        "slurm_partition": "gpu",
        "max_gpus": 1,
        "max_nodes": 1,
        "max_time_hours": 48,
        "max_epochs": 60,
        "train_command": EXPECTED_TRAIN_COMMAND,
        "allow_remote_sync": True,
        "allow_slurm": True,
        "allow_full_train": True,
        "allow_tools_train": True,
        "allow_tools_test": False,
        "allow_detector_map": False,
        "allow_metric_claim": False,
        "allow_paper_claim": False,
        "allow_runtime_flops_claim": False,
        "allow_deploy_claim": False,
        "no_gt_test_leakage_assertion": True,
        "no_teacher_or_oracle_assertion": True,
        "no_raw_prediction_cache_assertion": True,
        "uses_test_gt": False,
        "uses_val_test_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "allow_checkpoint_load": False,
        "allow_pretrained_initialization": False,
        "allow_resume": False,
        "allow_checkpoint_write": True,
        "checkpoint_write_policy": "route_work_dir_only",
        "checkpoint_load_policy": "none",
        "pretrained_initialization_policy": "none",
        "resume_policy": "none",
        "dataset_scope": "THUMOS14_TAD_ONLY_TRAIN200_VALTEST211",
    }


def test_bh_sdc_configs_use_sparse_dense_chain_and_distinct_gate_modes():
    validator = _load_validator()

    local = validator.validate_static_config(LOCAL_CONFIG)
    full = validator.validate_static_config(FULL_CONFIG)

    for result in (local, full):
        assert result["route"] == "bh_sdc_boundary_hazard_sparse_dense"
        assert result["route_label"] == BH_SDC_ROUTE_LABEL
        assert result["selector"] == "PCOTMRASBoundaryHazardSparseDenseFrameSelector"
        assert result["completion_bridge"] == "PCOTMRASBoundaryHazardSparseToDenseBridge"
        assert result["dense_window_size"] == 768
        assert result["min_budget"] < result["target_budget"] < result["max_budget"]
        assert result["base_chain_forbidden_tokens"] == []

    assert local["launch_gate_passed"] is False
    assert local["allow_long_training"] is False
    assert local["allowed_entrypoints"] == []
    assert local["launch_decision"] is None
    assert full["launch_gate_passed"] is True
    assert full["allow_long_training"] is True
    assert full["allow_tools_train"] is True
    assert full["allow_tools_test"] is False
    assert full["allowed_entrypoints"] == ["tools/train.py"]
    assert full["launch_decision"] == LAUNCH_DECISION
    assert full["static_authorization"] is False
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
        validator.validate_static_config(child)


def test_bh_sdc_validator_rejects_non_strict_budget_bounds(tmp_path):
    validator = _load_validator()
    bad = tmp_path / "bh_sdc_bad_budget.py"
    bad.write_text(
        f"_base_ = ['{FULL_CONFIG.as_posix()}']\n"
        "model = dict(frame_selector=dict(target_budget=256))\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_budget < target_budget < max_budget <= dense_window_size"):
        validator.validate_static_config(bad)


def test_bh_sdc_validator_cli_without_payload_remains_fail_closed():
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
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["ok"] is False
    assert payload["authorized"] is False
    assert "gate payload" in payload["reason"]
    assert payload["route_label"] == BH_SDC_ROUTE_LABEL
    assert payload["launch_decision"] == LAUNCH_DECISION


def test_bh_sdc_old_precheck_payload_does_not_authorize_training():
    validator = _load_validator()
    static = validator.validate_static_config(FULL_CONFIG)
    old_payload = {
        "decision": "GO_WITH_CONSTRAINTS_IMPLEMENT_BH_SDC_LOCAL_PROTOTYPE_ONLY",
        "route": "bh_sdc_boundary_hazard_sparse_dense",
        "route_label": BH_SDC_ROUTE_LABEL,
        "resolved_config_sha256": static["resolved_config_sha256"],
        "active_sha256_manifest_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="decision"):
        validator.validate_launch_gate_payload(
            FULL_CONFIG,
            old_payload,
            requested_action="slurm_full_train_candidate",
            requested_command=EXPECTED_TRAIN_COMMAND,
            active_manifest_sha256="a" * 64,
            resolved_config_sha256=static["resolved_config_sha256"],
        )


def test_bh_sdc_launch_payload_rejects_unknown_keys():
    validator = _load_validator()
    static = validator.validate_static_config(FULL_CONFIG)
    payload = _synthetic_launch_payload(static["resolved_config_sha256"])
    payload["surprise_unlock"] = True

    with pytest.raises(ValueError, match="unknown"):
        validator.validate_launch_gate_payload(
            FULL_CONFIG,
            payload,
            requested_action="slurm_full_train_candidate",
            requested_command=EXPECTED_TRAIN_COMMAND,
            active_manifest_sha256="a" * 64,
            resolved_config_sha256=static["resolved_config_sha256"],
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "pro_implementation_session",
        "pro_launch_gate_verdict",
        "final_read_only_review_verdict",
        "explicit_user_pro_launch_decision",
    ],
)
def test_bh_sdc_launch_payload_requires_review_and_launch_decision_evidence(missing_key):
    validator = _load_validator()
    static = validator.validate_static_config(FULL_CONFIG)
    payload = _synthetic_launch_payload(static["resolved_config_sha256"])
    payload.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        validator.validate_launch_gate_payload(
            FULL_CONFIG,
            payload,
            requested_action="slurm_full_train_candidate",
            requested_command=EXPECTED_TRAIN_COMMAND,
            active_manifest_sha256="a" * 64,
            resolved_config_sha256=static["resolved_config_sha256"],
        )


def test_bh_sdc_launch_payload_requires_exact_command_whitelist_and_rejects_tools_test():
    validator = _load_validator()
    static = validator.validate_static_config(FULL_CONFIG)
    payload = _synthetic_launch_payload(static["resolved_config_sha256"])

    extra_command_payload = dict(payload)
    extra_command_payload["command_whitelist"] = list(EXPECTED_COMMAND_WHITELIST) + ["python tools/test.py anything"]
    with pytest.raises(ValueError, match="command_whitelist"):
        validator.validate_launch_gate_payload(
            FULL_CONFIG,
            extra_command_payload,
            requested_action="slurm_full_train_candidate",
            requested_command=EXPECTED_TRAIN_COMMAND,
            active_manifest_sha256="a" * 64,
            resolved_config_sha256=static["resolved_config_sha256"],
        )

    with pytest.raises(ValueError, match="requested_command"):
        validator.validate_launch_gate_payload(
            FULL_CONFIG,
            payload,
            requested_action="slurm_full_train_candidate",
            requested_command="python tools/test.py configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py best.pth",
            active_manifest_sha256="a" * 64,
            resolved_config_sha256=static["resolved_config_sha256"],
        )


@pytest.mark.parametrize(
    ("requested_action", "requested_command"),
    [
        ("remote_sync_to_n16r4_workspace", EXPECTED_SYNC_COMMAND),
        ("slurm_submit_bh_sdc_n16r4", EXPECTED_SLURM_COMMAND),
        ("slurm_full_train_candidate", EXPECTED_TRAIN_COMMAND),
    ],
)
def test_bh_sdc_valid_synthetic_launch_payload_passes(requested_action, requested_command):
    validator = _load_validator()
    static = validator.validate_static_config(FULL_CONFIG)
    payload = _synthetic_launch_payload(static["resolved_config_sha256"])

    result = validator.validate_launch_gate_payload(
        FULL_CONFIG,
        payload,
        requested_action=requested_action,
        requested_command=requested_command,
        active_manifest_sha256="a" * 64,
        resolved_config_sha256=static["resolved_config_sha256"],
    )

    assert result["ok"] is True
    assert result["authorized"] is True
    assert result["decision"] == LAUNCH_DECISION
    assert result["requested_action"] == requested_action
    assert result["requested_command"] == requested_command
    assert result["command_whitelist"] == EXPECTED_COMMAND_WHITELIST
