import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
SCRIPT_DIR = ROOT / "scripts"
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_pc_ot_mras_prebackbone_c3_full_train_gate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"

FORMAL_VARIANT = "C3-RS-Hybrid-ST-OriginalAdaTAD"
FORMAL_ROUTE = "pc_ot_mras_prebackbone_c3_rs_hybrid_st_original_adatad"
FORMAL_DECISION = "ALLOW_PC_OT_MRAS_PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_FIXED50"
FORMAL_READER = "PCOTMRASRSeriesHybridFrameScout"
FORMAL_SELECTOR_GRADIENT = "st_hard_real_frames_with_full_flat_soft_transport_surrogate"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_text(path, text):
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_config_paths():
    paths = []
    for path in CONFIG_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if FORMAL_VARIANT in text and "full_train" in path.stem:
            paths.append(path)
    return sorted(paths)


def _formal_config_path():
    paths = _formal_config_paths()
    assert paths, f"missing formal full-train config for {FORMAL_VARIANT}"
    assert len(paths) == 1, f"expected one formal {FORMAL_VARIANT} config, got {[path.name for path in paths]}"
    return paths[0]


def _formal_launcher_path():
    matches = []
    for path in SCRIPT_DIR.glob("*.sbatch"):
        text = path.read_text(encoding="utf-8")
        if FORMAL_VARIANT in text and FORMAL_DECISION in text and "tools/train.py" in text:
            matches.append(path)
    assert matches, f"missing dedicated formal launcher for {FORMAL_VARIANT}"
    assert len(matches) == 1, f"expected one formal launcher for {FORMAL_VARIANT}, got {[path.name for path in matches]}"
    return matches[0]


def _good_gate_payload(manifest="manifest-sha", resolved="resolved-sha", pretrained_sha="pretrained-sha"):
    return {
        "decision": FORMAL_DECISION,
        "route": FORMAL_ROUTE,
        "variant_id": FORMAL_VARIANT,
        "execution_mode": "train",
        "selection_surface": "pre_backbone_raw_frame",
        "selection_timing": "online_before_backbone",
        "acquisition_unit": "frame",
        "claim_tier": "no_deploy_until_runtime_audit",
        "active_sha256_manifest_sha256": manifest,
        "resolved_config_sha256": resolved,
        "pretrained_sha256": pretrained_sha,
        "budget": 384,
        "dense_window_size": 768,
        "selection_unit": 1,
        "selector_reader": FORMAL_READER,
        "reader_family": "RSeriesHybrid",
        "selector_gradient": FORMAL_SELECTOR_GRADIENT,
        "robust_aux_objective": "gt_duplicate_value_risk_uncertainty_redundancy_role",
        "st_surrogate_mode": "full_flat",
        "scout_feature_source": "compressed_pixels",
        "scout_spatial_size": 32,
        "scout_pixel_clamp": 5.0,
        "aux_gt_acquisition_loss_weight": 0.05,
        "aux_duplicate_cap_loss_weight": 0.001,
        "aux_duplicate_column_cap": 1.25,
        "aux_value_loss_weight": 0.02,
        "aux_risk_loss_weight": 0.02,
        "aux_uncertainty_loss_weight": 0.01,
        "aux_redundancy_loss_weight": 0.01,
        "aux_role_entropy_loss_weight": 0.001,
        "reader_regularizer_loss_weight": 0.01,
        "max_epochs": 60,
        "checkpoint_interval": 60,
        "val_start_epoch": 40,
        "val_eval_interval": 2,
        "allow_remote_sync": True,
        "allow_precheck_only": True,
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
        "st_hard_real_frames": True,
        "train_loop_finite_fail_fast": True,
        "nan_fail_fast": True,
        "robust_aux_enabled": True,
        "scout_pixel_normalize": True,
        "uses_p2": False,
        "uses_offline_ledger": False,
        "uses_teacher": False,
        "uses_test_gt": False,
        "uses_raw_prediction_cache": False,
        "st_off": False,
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
        "post_projection_bridge": False,
        "raw_prediction_cache": False,
        "allow_raw_prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
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


def test_c3_full_train_config_is_parseable_and_unlocks_only_tools_train(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_module(GUARD_PATH, "training_guard_for_c3_full_train_test")
    config_path = _formal_config_path()
    pretrained = tmp_path / "videomae_pretrained.pth"
    pretrained_sha = _sha_text(pretrained, "pretrained bytes")
    monkeypatch.setenv("PC_OT_MRAS_PREBACKBONE_C3_PRETRAINED_PATH", str(pretrained))

    cfg = mmengine_config.Config.fromfile(str(config_path))

    assert cfg.variant_id == FORMAL_VARIANT
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.checkpoint_interval == 60
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.experiment_scope.route == FORMAL_ROUTE
    assert "full_train" in cfg.experiment_scope.stage
    assert cfg.experiment_scope.free_frame_level_selector is True
    assert cfg.experiment_scope.protected_scaffold is False
    assert cfg.experiment_scope.uses_p2 is False
    assert cfg.experiment_scope.uses_offline_ledger is False
    assert cfg.experiment_scope.uses_teacher is False
    assert cfg.experiment_scope.uses_test_gt is False
    assert cfg.experiment_scope.uses_raw_prediction_cache is False
    assert cfg.experiment_scope.backend == "OriginalAdaTAD"
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.selector_gradient == FORMAL_SELECTOR_GRADIENT
    assert "st_off" not in repr(cfg.experiment_scope).lower()

    frame_selector = cfg.model.frame_selector
    assert frame_selector.type == "PCOTMRASPreBackboneFrameSelector"
    assert frame_selector.target_len == 384
    assert frame_selector.dense_window_size == 768
    assert frame_selector.selection_unit == 1
    assert frame_selector.protected_uniform_count == 0
    assert frame_selector.coverage_guard_count == 0
    assert frame_selector.scout_feature_source == "compressed_pixels"
    assert frame_selector.scout_spatial_size == 32
    assert frame_selector.descriptor_dim == 3 * 32 * 32
    assert frame_selector.transport_topk == 1
    assert frame_selector.eval_transport_topk == 1
    assert frame_selector.straight_through_downstream is True
    assert getattr(frame_selector, "straight_through_detector_loss", True) is not False
    assert frame_selector.remap_gt_to_selected_axis is True
    assert frame_selector.reader.type == FORMAL_READER
    assert frame_selector.reader.in_dim == 3 * 32 * 32
    assert frame_selector.reader.num_slots == 384
    assert frame_selector.reader.local_global_fusion == "rseries_temporal_geometry_slot_attention"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.neck.type != "PCOTMRASDetectorBridge"
    assert "PCOTMRASDetectorBridge" not in repr(cfg.model)
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    assert "window_size" not in cfg.dataset.train
    train_loadframes = next(step for step in cfg.dataset.train.pipeline if step.type == "LoadFrames")
    assert train_loadframes.trunc_len == 768
    for split in ("val", "test"):
        assert cfg.dataset[split].window_size == 768
    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        for forbidden in ("bata_value_transport_ledger_subsample", "hard_positions", "teacher", "oracle", "raw_prediction"):
            assert forbidden not in pipeline_text

    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    assert gate.route == FORMAL_ROUTE
    assert gate.stage == cfg.experiment_scope.stage
    assert gate.formal_train_candidate is True
    assert gate.allow_detector_training is True
    assert gate.launch_gate_passed is True
    assert gate.allow_remote_sync is True
    assert gate.allow_precheck_only is True
    assert gate.allow_slurm is True
    assert gate.allow_gpu is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_train_validation_map is True
    assert gate.allow_long_training is True
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)
    assert gate.entrypoint_gate_context.allowed_decisions == (
        FORMAL_DECISION,
    )
    assert gate.entrypoint_gate_context.required_exact_values.route == FORMAL_ROUTE
    assert gate.entrypoint_gate_context.required_exact_values.variant_id == FORMAL_VARIANT
    assert gate.entrypoint_gate_context.required_exact_values.max_epochs == 60
    assert gate.entrypoint_gate_context.required_exact_values.checkpoint_interval == 60
    assert gate.entrypoint_gate_context.required_exact_values.val_start_epoch == 40
    assert gate.entrypoint_gate_context.required_exact_values.val_eval_interval == 2
    assert gate.entrypoint_gate_context.sha256_file_bindings[0].gate_key == "pretrained_sha256"

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "c3_full_gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload(pretrained_sha=pretrained_sha)), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_C3_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_C3_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_C3_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_PREBACKBONE_C3_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_c3_full_train_gate_validator_accepts_and_rejects_bound_payloads(tmp_path):
    validator = _load_module(VALIDATOR_PATH, "validate_c3_full_train_gate_test")
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
    assert payload["decision"] == FORMAL_DECISION

    bad_payload = _good_gate_payload()
    bad_payload["max_epochs"] = 8
    with pytest.raises(ValueError, match="max_epochs=60"):
        validator.validate_gate_payload(
            bad_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
            budget=384,
            dense_window_size=768,
        )

    bad_payload = _good_gate_payload()
    bad_payload["uses_p2"] = True
    with pytest.raises(ValueError, match="uses_p2=false"):
        validator.validate_gate_payload(
            bad_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
            budget=384,
            dense_window_size=768,
        )

    bad_payload = _good_gate_payload()
    bad_payload["variant_id"] = "C3-F1-LR-TinyTransformer-ST-OriginalAdaTAD"
    with pytest.raises(ValueError, match=FORMAL_VARIANT):
        validator.validate_gate_payload(
            bad_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
            budget=384,
            dense_window_size=768,
        )

    bad_payload = _good_gate_payload()
    bad_payload["st_off"] = True
    with pytest.raises(ValueError, match="st_off=false"):
        validator.validate_gate_payload(
            bad_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
            budget=384,
            dense_window_size=768,
        )


def test_c3_full_train_launcher_is_dedicated_to_c3_and_fail_closed():
    launcher = _formal_launcher_path()
    text = launcher.read_text(encoding="utf-8")

    assert "#SBATCH -J" in text
    assert FORMAL_VARIANT in text
    assert FORMAL_ROUTE in text
    assert FORMAL_DECISION in text
    assert "validate_pc_ot_mras_prebackbone_c3_full_train_gate.py" in text
    assert "descriptor_dim) == 3072" in text
    assert "protected_uniform_count) == 0" in text
    assert FORMAL_READER in text
    assert "num_slots) == 384" in text
    assert "straight_through_downstream" in text
    assert "PRECHECK_ONLY=0" in text
    assert "OPENTAD_PCOTMRAS_PREBACKBONE_C3_GATE_JSON" in text
    assert "tools/train.py \"$CONFIG\"" in text
    assert "tools/test.py \"$CONFIG\"" not in text
    assert "raw prediction/cache is forbidden" in text
    assert "offline ledger is forbidden" in text
