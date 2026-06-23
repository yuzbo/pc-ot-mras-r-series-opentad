import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_e2e_frame_acquisition_actionformer_adapter_fixed50_candidate_n16r4.py"
)
LAUNCHER = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_e2e_frame_acquisition_actionformer_adapter_fixed50_n16r4.sbatch"
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_pc_ot_mras_prebackbone_e2e_acquisition_gate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
BASE_DETECTOR_PATH = ROOT / "opentad" / "models" / "detectors" / "base.py"
SELECTOR_INIT_PATH = ROOT / "opentad" / "models" / "selectors" / "__init__.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_text(path, text):
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _good_gate_payload(manifest="manifest-sha", resolved="resolved-sha", pretrained_sha="pretrained-sha"):
    return {
        "decision": "ALLOW_PC_OT_MRAS_PREBACKBONE_E2E_FRAME_ACQUISITION_TRAIN_FIXED50",
        "route": "pc_ot_mras_prebackbone_original_adatad",
        "execution_mode": "train",
        "selection_surface": "pre_backbone_raw_frame",
        "selection_timing": "online_before_backbone",
        "acquisition_unit": "frame_or_snippet",
        "claim_tier": "no_deploy_until_runtime_audit",
        "active_sha256_manifest_sha256": manifest,
        "resolved_config_sha256": resolved,
        "pretrained_sha256": pretrained_sha,
        "budget": 384,
        "dense_window_size": 768,
        "max_epochs": 60,
        "val_start_epoch": 40,
        "val_eval_interval": 2,
        "allow_slurm": True,
        "allow_gpu": True,
        "single_gpu": True,
        "allow_prebackbone_frame_selector": True,
        "allow_joint_selector_detector_training": True,
        "allow_tools_train": True,
        "allow_dataset_access": True,
        "allow_pretrained_initialization": True,
        "allow_checkpoint_write": True,
        "allow_train_validation_map": True,
        "allow_long_training": True,
        "reader_trainable": True,
        "uses_offline_ledger": False,
        "tools_test": False,
        "allow_tools_test": False,
        "direct_tools_test": False,
        "detector_map": False,
        "allow_detector_map": False,
        "checkpoint_load": False,
        "allow_checkpoint_load": False,
        "resume": False,
        "allow_resume": False,
        "offline_ledger": False,
        "frozen_reader_ledger": False,
        "post_projection_bridge": False,
        "raw_prediction_cache": False,
        "allow_raw_prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction": False,
        "metric_claim": False,
        "metric_claim_allowed": False,
        "paper_claim": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim": False,
        "deploy_claim_allowed": False,
    }


def test_prebackbone_selector_source_declares_pre_backbone_contract():
    text = SELECTOR_PATH.read_text(encoding="utf-8")
    init_text = SELECTOR_INIT_PATH.read_text(encoding="utf-8")
    base_text = BASE_DETECTOR_PATH.read_text(encoding="utf-8")

    assert "@SELECTORS.register_module()" in text
    assert "class PCOTMRASPreBackboneFrameSelector" in text
    assert "forward_train" in text
    assert "forward_test" in text
    assert "irregular_selected_positions" in text
    assert "remap_gt_to_selected_axis" in text
    assert "selector_gt_acquisition_loss" in text
    assert "PCOTMRASPreBackboneFrameSelector" in init_text
    assert "forbid_raw_prediction_cache" in base_text


def test_prebackbone_e2e_config_uses_frame_selector_not_post_projection_bridge(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_module(GUARD_PATH, "training_guard_for_prebackbone_e2e_test")
    pretrained = tmp_path / "videomae_pretrained.pth"
    pretrained_sha = _sha_text(pretrained, "pretrained bytes")
    monkeypatch.setenv("PC_OT_MRAS_PREBACKBONE_E2E_PRETRAINED_PATH", str(pretrained))

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.window_size == 384
    assert cfg.dense_window_size == 768
    assert cfg.experiment_scope.stage == "prebackbone_e2e_frame_acquisition_actionformer_adapter_train_fixed50"
    assert cfg.experiment_scope.selection_surface == "pre_backbone_raw_frame"
    assert cfg.experiment_scope.selection_timing == "online_before_backbone"
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.first_version_forward_contract == "hard_top1_train_eval"
    assert cfg.experiment_scope.budget_protocol == "fixed384_frame_slot_candidate"
    assert cfg.experiment_scope.s80r16_cell96x4_enabled is False
    assert "future route" in cfg.experiment_scope.s80r16_cell96x4_note.lower()
    assert "not" in cfg.experiment_scope.s80r16_cell96x4_note.lower()
    assert cfg.experiment_scope.changes_input_sampling is True
    assert cfg.experiment_scope.changes_detector_head is False
    assert cfg.experiment_scope.changes_neck is False
    assert cfg.experiment_scope.changes_post_processing is True
    assert cfg.experiment_scope.uses_offline_ledger is False

    assert "frame_selector" in cfg.model
    assert cfg.model.frame_selector.type == "PCOTMRASPreBackboneFrameSelector"
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.descriptor_dim == 12
    assert cfg.model.frame_selector.protected_uniform_count == 64
    assert cfg.model.frame_selector.transport_topk == 1
    assert cfg.model.frame_selector.eval_transport_topk == 1
    assert cfg.model.frame_selector.transport_topk == cfg.model.frame_selector.eval_transport_topk
    assert cfg.model.frame_selector.reader.type == "PCOTMRASReader"
    assert cfg.model.frame_selector.reader.in_dim == 12
    assert cfg.model.frame_selector.reader.enable_value_heads is True
    assert cfg.model.frame_selector.reader.emit_pair_distribution is False
    assert "pc_ot_mras_reader" not in cfg.model
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.neck.type != "PCOTMRASDetectorBridge"
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384

    for split in ("train", "val", "test"):
        dataset = cfg.dataset[split]
        assert dataset.window_size == 768
        pipeline_text = repr(dataset.pipeline).lower()
        assert "bata_value_transport_ledger_subsample" not in pipeline_text
        assert "hard_positions" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text

    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    assert gate.entrypoint_gate_context.allowed_decisions == (
        "ALLOW_PC_OT_MRAS_PREBACKBONE_E2E_FRAME_ACQUISITION_TRAIN_FIXED50",
    )
    assert gate.entrypoint_gate_context.required_exact_values.selection_surface == "pre_backbone_raw_frame"
    assert gate.entrypoint_gate_context.required_exact_values.selection_timing == "online_before_backbone"
    assert tuple(gate.entrypoint_gate_context.required_false_keys) == ("uses_offline_ledger",)
    assert gate.entrypoint_gate_context.sha256_file_bindings[0].gate_key == "pretrained_sha256"
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "prebackbone_gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload(pretrained_sha=pretrained_sha)), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_E2E_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_E2E_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_E2E_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_E2E_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")

    for forbidden_key in ("uses_offline_ledger", "raw_prediction_cache"):
        bad_payload = _good_gate_payload(pretrained_sha=pretrained_sha)
        bad_payload[forbidden_key] = True
        bad_gate_json = tmp_path / f"prebackbone_gate_{forbidden_key}.json"
        bad_gate_json.write_text(json.dumps(bad_payload), encoding="utf-8")
        bad_gate_sha = hashlib.sha256(bad_gate_json.read_bytes()).hexdigest()
        monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_E2E_GATE_JSON", str(bad_gate_json))
        monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_E2E_GATE_SHA256", bad_gate_sha)
        with pytest.raises(RuntimeError, match=forbidden_key):
            training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_prebackbone_e2e_gate_validator_accepts_bound_payload(tmp_path):
    validator = _load_module(VALIDATOR_PATH, "validate_prebackbone_e2e_gate_test")
    gate_json = tmp_path / "gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    payload = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        pretrained_sha256="pretrained-sha",
        budget=384,
        dense_window_size=768,
    )
    assert payload["selection_surface"] == "pre_backbone_raw_frame"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("decision", "ALLOW_PC_OT_MRAS_FRONTEND_JOINT_E2E_SOFT_TRANSPORT_TRAIN", "decision is not allowed"),
        ("route", "pc_ot_mras_frontend_original_adatad", "route mismatch"),
        ("selection_surface", "post_projection_feature_token", "selection_surface=pre_backbone_raw_frame"),
        ("selection_timing", "after_projection", "selection_timing=online_before_backbone"),
        ("execution_mode", "eval", "execution_mode=train"),
        ("allow_tools_train", False, "allow_tools_train=true"),
        ("allow_prebackbone_frame_selector", False, "allow_prebackbone_frame_selector=true"),
        ("allow_joint_selector_detector_training", False, "allow_joint_selector_detector_training=true"),
        ("uses_offline_ledger", True, "uses_offline_ledger=false"),
        ("offline_ledger", True, "offline_ledger=false/absent"),
        ("post_projection_bridge", True, "post_projection_bridge=false/absent"),
        ("tools_test", True, "tools_test=false/absent"),
        ("detector_map", True, "detector_map=false/absent"),
        ("raw_prediction_cache", True, "raw_prediction_cache=false/absent"),
        ("checkpoint_load", True, "checkpoint_load=false/absent"),
        ("uses_teacher", True, "uses_teacher=false/absent"),
        ("uses_oracle", True, "uses_oracle=false/absent"),
        ("metric_claim", True, "metric_claim=false/absent"),
        ("budget", 192, "budget=384"),
        ("dense_window_size", 384, "dense_window_size=768"),
    ],
)
def test_prebackbone_e2e_gate_validator_rejects_unsafe_payloads(key, value, match):
    validator = _load_module(VALIDATOR_PATH, "validate_prebackbone_e2e_gate_negative_test")
    payload = _good_gate_payload()
    payload[key] = value
    with pytest.raises(ValueError, match=match):
        validator.validate_gate_payload(
            payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
            budget=384,
            dense_window_size=768,
        )


def test_prebackbone_e2e_launcher_uses_dedicated_namespace_and_no_ledger_dump():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH -J pcot_prebb_e2e" in text
    assert "pc_ot_mras_prebackbone_e2e_frame_acquisition_actionformer_adapter_fixed50_candidate_n16r4.py" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_PC_OT_MRAS_PREBACKBONE_E2E=1" in text
    assert "validate_pc_ot_mras_prebackbone_e2e_acquisition_gate.py" in text
    assert "OPENTAD_PCOTMRAS_PREBACKBONE_E2E_GATE_JSON" in text
    assert "PC_OT_MRAS_PREBACKBONE_E2E_PRETRAINED_PATH" in text
    assert "tools/train.py \"$CONFIG\"" in text
    assert "codex/prebackbone-pcotmras-scout-selector" in text
    assert "ALLOW_LOGIN_NODE_DEBUG" not in text
    assert "login-node debug" not in text
    assert "tools/test.py \"$CONFIG\"" not in text
    assert "FRONTEND_JOINT_E2E" not in text
    assert "dump_pc_ot_mras_reader_snapshots.py" not in text
    assert "convert_pc_ot_mras_hard_positions_to_value_transport_ledger.py" not in text
