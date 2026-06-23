import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke.py"
)
LAUNCHER = (
    ROOT
    / "scripts"
    / "run_ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke_n16r4.sbatch"
)
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
TRAIN_PATH = ROOT / "tools" / "train.py"
TRAIN_ENGINE_PATH = ROOT / "opentad" / "cores" / "train_engine.py"
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_p2qr_short_smoke_gate.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mmengine_config():
    mmengine_config = pytest.importorskip("mmengine.config")
    assert CONFIG.exists()
    return mmengine_config.Config.fromfile(str(CONFIG))


def test_p2qr_short_smoke_config_is_parseable_and_fail_closed():
    cfg = _load_mmengine_config()

    assert cfg.p2_quality_rank_calibrator_v0_gate is None
    gate = cfg.p2_quality_rank_calibrator_v0_short_smoke_gate
    assert gate.stage == "P2_NIIQ_QualityRank_Calibrator_v0_short_smoke_scaffold"
    assert gate.launch_scaffold_only is True
    assert gate.smoke_only is True
    assert gate.launch_gate_passed is False
    assert gate.allow_detector_training is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_long_training is False
    assert gate.max_epochs == 1
    assert gate.max_train_iters == 2

    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 2
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.workflow.val_loss_interval == -1
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.model.rpn_head.area_head.quality_calibration.enable is True


def test_p2qr_short_smoke_config_still_blocks_train_entrypoint():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_p2qr_launch_scaffold_test", GUARD_PATH)

    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_p2qr_short_smoke_config_rejects_tools_test_even_if_tampered():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_p2qr_tools_test_test", GUARD_PATH)
    cfg.p2_quality_rank_calibrator_v0_short_smoke_gate.allow_detector_training = True
    cfg.p2_quality_rank_calibrator_v0_short_smoke_gate.launch_gate_passed = True
    cfg.p2_quality_rank_calibrator_v0_short_smoke_gate.allow_tools_train = True
    cfg.p2_quality_rank_calibrator_v0_short_smoke_gate.allow_tools_test = False

    with pytest.raises(RuntimeError, match="allow_tools_test"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_p2qr_future_gate_schema_is_default_safe():
    cfg = _load_mmengine_config()
    validator = _load_module("p2qr_gate_validator_schema_test", VALIDATOR_PATH)
    context = cfg.p2_quality_rank_calibrator_v0_short_smoke_gate.future_execution_gate_context

    assert context.required_before_any_training is True
    assert context.gate_json_env == "P2QR_SHORT_SMOKE_GATE_JSON"
    assert context.gate_sha256_env == "P2QR_SHORT_SMOKE_GATE_SHA256"
    assert context.active_manifest_sha256_env == "P2QR_ACTIVE_MANIFEST_SHA256"
    assert context.resolved_config_sha256_env == "P2QR_RESOLVED_CONFIG_SHA256"
    assert context.require_resolved_config_sha256 is True
    assert tuple(context.allowed_decisions) == (
        "PASS_ALLOW_BOUNDED_P2QR_SLURM_SHORT_SMOKE_ONLY",
    )
    assert tuple(context.required_true_keys) == (
        "allow_slurm",
        "allow_gpu",
        "allow_tools_train",
        "single_gpu",
    )
    assert context.unknown_key_policy == "reject_unknown_except_explicit_harmless_metadata"
    assert tuple(context.harmless_metadata_keys) == validator.HARMLESS_METADATA_KEYS
    assert tuple(context.forbidden_true_keys) == validator.FORBIDDEN_TRUE_KEYS
    assert "detector_map" in tuple(context.forbidden_true_keys)
    assert "allow_detector_map" in tuple(context.forbidden_true_keys)
    assert "formal_eval" in tuple(context.forbidden_true_keys)
    assert "dataset" in tuple(context.forbidden_true_keys)
    assert "allow_dataset" in tuple(context.forbidden_true_keys)
    assert "allow_dataset_access" in tuple(context.forbidden_true_keys)
    assert "checkpoint" in tuple(context.forbidden_true_keys)
    assert "allow_checkpoint" in tuple(context.forbidden_true_keys)
    assert "allow_checkpoint_access" in tuple(context.forbidden_true_keys)
    assert "resume_from" in tuple(context.forbidden_true_keys)
    assert "allow_resume_from" in tuple(context.forbidden_true_keys)
    assert "raw_predictions" in tuple(context.forbidden_true_keys)
    assert "allow_raw_predictions" in tuple(context.forbidden_true_keys)
    assert "allow_raw_prediction_cache" in tuple(context.forbidden_true_keys)
    assert "prediction_cache" in tuple(context.forbidden_true_keys)
    assert "load_from_raw_predictions" in tuple(context.forbidden_true_keys)
    assert "allow_load_from_raw_predictions" in tuple(context.forbidden_true_keys)
    assert "save_raw_predictions" in tuple(context.forbidden_true_keys)
    assert "allow_save_raw_predictions" in tuple(context.forbidden_true_keys)
    assert "metric_claim_allowed" in tuple(context.forbidden_true_keys)
    assert "paper_claim" in tuple(context.forbidden_true_keys)
    assert "paper_claim_allowed" in tuple(context.forbidden_true_keys)
    assert "runtime_flops_claim_allowed" in tuple(context.forbidden_true_keys)
    assert "runtime_or_flops_claim_allowed" in tuple(context.forbidden_true_keys)
    assert "deploy_claim_allowed" in tuple(context.forbidden_true_keys)
    assert "raw_prediction_cache" in tuple(context.forbidden_true_keys)


def test_p2qr_launcher_scaffold_is_fail_closed_textually():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_P2QR_SHORT_SMOKE="${ALLOW_P2QR_SHORT_SMOKE:-0}"' in text
    assert 'ALLOW_P2QR_SLURM="${ALLOW_P2QR_SLURM:-0}"' in text
    assert 'ALLOW_P2QR_GPU="${ALLOW_P2QR_GPU:-0}"' in text
    assert "ALLOW_P2QR_TOOLS_TEST" in text
    assert "ALLOW_P2QR_DETECTOR_MAP" in text
    assert "ALLOW_P2QR_LONG_TRAINING" in text
    assert "P2QR_SHORT_SMOKE_GATE_JSON" in text
    assert "P2QR_SHORT_SMOKE_GATE_SHA256" in text
    assert "P2QR_ACTIVE_MANIFEST_SHA256" in text
    assert "P2QR_RESOLVED_CONFIG_SHA256" in text
    assert "resolved_config_sha256=$RESOLVED_CONFIG_SHA256" in text
    assert "validate_p2qr_short_smoke_gate.py" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_P2QR_SLURM=1" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_P2QR_GPU=1" in text
    assert "P2QR short-smoke execution is still denied in this scaffold" in text
    assert "tools/test.py is not approved" in text
    assert "detector mAP is not approved" in text
    assert "raw prediction/cache shortcuts are forbidden" in text
    assert "PRECHECK_ONLY=1; exiting before any P2QR tools/train execution" in text


def test_p2qr_future_gate_payload_example_keeps_forbidden_flags_false():
    payload = {
        "decision": "PASS_ALLOW_BOUNDED_P2QR_SLURM_SHORT_SMOKE_ONLY",
        "active_sha256_manifest_sha256": "abc",
        "max_train_iters": 2,
        "allow_slurm": True,
        "allow_gpu": True,
        "allow_tools_train": True,
        "single_gpu": True,
        "tools_test": False,
        "allow_tools_test": False,
        "detector_map": False,
        "allow_detector_map": False,
        "long_training": False,
        "allow_long_training": False,
        "metric_claim": False,
        "paper_claim": False,
        "runtime_flops_claim": False,
        "deploy_claim": False,
    }

    assert json.dumps(payload)
    assert all(
        payload[key] is False
        for key in (
            "tools_test",
            "detector_map",
            "long_training",
            "metric_claim",
            "paper_claim",
            "runtime_flops_claim",
            "deploy_claim",
        )
    )


def _valid_gate_payload(active="a" * 64, resolved="b" * 64):
    return {
        "decision": "PASS_ALLOW_BOUNDED_P2QR_SLURM_SHORT_SMOKE_ONLY",
        "route": "CTF-BDI/PC-OT-MRAS",
        "active_sha256_manifest_sha256": active,
        "resolved_config_sha256": resolved,
        "max_train_iters": 2,
        "allow_slurm": True,
        "allow_gpu": True,
        "allow_tools_train": True,
        "single_gpu": True,
        "allow_tools_test": False,
        "allow_detector_map": False,
        "allow_long_training": False,
        "allow_metric_claim": False,
        "allow_paper_claim": False,
        "allow_runtime_flops_claim": False,
        "allow_deploy_claim": False,
        "dataset_access": False,
        "checkpoint_access": False,
        "raw_prediction_cache": False,
    }


def test_p2qr_future_gate_validator_accepts_bound_payload():
    validator = _load_module("p2qr_gate_validator_accept_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()

    assert validator.validate_gate_payload(payload, "a" * 64, "b" * 64) is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p.update(decision="PASS_ALLOW_REMOTE_PRECHECK_ONLY_ENV_CHECK"), "decision"),
        (lambda p: p.pop("resolved_config_sha256"), "resolved_config_sha256"),
        (lambda p: p.update(resolved_config_sha256="c" * 64), "resolved config sha256 mismatch"),
        (lambda p: p.update(max_train_iters=3), "max_train_iters=2"),
        (lambda p: p.update(allow_slurm=False), "allow_slurm=true"),
        (lambda p: p.update(allow_gpu=False), "allow_gpu=true"),
        (lambda p: p.update(allow_tools_train=False), "allow_tools_train=true"),
        (lambda p: p.update(single_gpu=False), "single_gpu=true"),
        (lambda p: p.update(allow_tools_test=True), "allow_tools_test=false/absent"),
        (lambda p: p.update(allow_detector_map=True), "allow_detector_map=false/absent"),
        (lambda p: p.update(raw_prediction_cache=True), "raw_prediction_cache=false/absent"),
    ],
)
def test_p2qr_future_gate_validator_rejects_unsafe_payloads(mutator, message):
    validator = _load_module("p2qr_gate_validator_reject_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    mutator(payload)

    with pytest.raises(ValueError, match=message):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "key",
    [
        "eval",
        "allow_eval",
        "formal_eval",
        "allow_formal_eval",
        "train_validation_map",
        "allow_train_validation_map",
        "tools_test_map",
        "allow_tools_test_map",
        "allow_dataset_access",
        "checkpoint_access",
        "allow_checkpoint_access",
        "checkpoint_load",
        "allow_checkpoint_load",
        "resume",
        "allow_resume",
        "load_from",
        "allow_load_from",
        "raw_prediction",
        "allow_raw_prediction",
        "allow_raw_prediction_cache",
        "prediction_cache",
        "allow_prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "runtime_claim",
        "allow_runtime_claim",
        "flops_claim",
        "allow_flops_claim",
        "allow_runtime_flops_claim",
        "allow_deploy_claim",
    ],
)
@pytest.mark.parametrize("unsafe_value", [True, "true", "1", "yes", 1])
def test_p2qr_future_gate_validator_rejects_forbidden_alias_truthy_values(key, unsafe_value):
    validator = _load_module("p2qr_gate_validator_alias_truthy_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload[key] = unsafe_value

    with pytest.raises(ValueError, match=f"{key}=false/absent"):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


def test_p2qr_future_gate_validator_rejects_all_forbidden_keys_when_true():
    validator = _load_module("p2qr_gate_validator_all_forbidden_test", VALIDATOR_PATH)

    for key in validator.FORBIDDEN_TRUE_KEYS:
        payload = _valid_gate_payload()
        payload[key] = True

        with pytest.raises(ValueError, match=f"{key}=false/absent"):
            validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "key",
    [
        "metric_claim_allowed",
        "paper_claim_allowed",
        "runtime_flops_claim_allowed",
        "runtime_or_flops_claim_allowed",
        "deploy_claim_allowed",
        "allow_load_from_raw_predictions",
        "allow_save_raw_prediction",
        "allow_save_raw_predictions",
        "save_raw_predictions",
        "raw_predictions",
        "allow_raw_predictions",
        "allow_dataset",
        "dataset",
        "allow_checkpoint",
        "checkpoint",
        "resume_from",
        "allow_resume_from",
    ],
)
@pytest.mark.parametrize("unsafe_value", [True, "true", "1", "yes", 1])
def test_p2qr_future_gate_validator_rejects_pro_reported_alias_values(key, unsafe_value):
    validator = _load_module("p2qr_gate_validator_pro_alias_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload[key] = unsafe_value

    with pytest.raises(ValueError, match=f"{key}=false/absent"):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


@pytest.mark.parametrize("false_like", [False, "false", 0, None])
def test_p2qr_future_gate_validator_rejects_non_bool_false_for_claim_aliases(false_like):
    validator = _load_module("p2qr_gate_validator_claim_false_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload["metric_claim_allowed"] = false_like

    if false_like is False:
        assert validator.validate_gate_payload(payload, "a" * 64, "b" * 64) is True
    else:
        with pytest.raises(ValueError, match="metric_claim_allowed=false/absent"):
            validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "key",
    [
        "mystery_metric_claim_allowed",
        "future_allow_dataset_access",
        "custom_prediction_cache_route",
        "unknown_resume_from_checkpoint",
        "shadow_runtime_flops_claim",
        "hidden_deploy_permission",
    ],
)
def test_p2qr_future_gate_validator_rejects_unknown_dangerous_permission_keys(key):
    validator = _load_module("p2qr_gate_validator_unknown_dangerous_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload[key] = False

    with pytest.raises(ValueError, match=f"unknown or unallowlisted key: {key}"):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "key",
    [
        "ALLOW_DETECTOR_MAP",
        "ALLOW_TOOLS_TEST",
        "CHECKPOINT",
        "RAW_PREDICTIONS",
        "Metric_Claim_Allowed",
        "PAPER_CLAIM_ALLOWED",
        "allowDetectorMap",
        "checkpointAccess",
    ],
)
@pytest.mark.parametrize("unsafe_value", [True, False])
def test_p2qr_future_gate_validator_rejects_case_variant_permission_keys(key, unsafe_value):
    validator = _load_module("p2qr_gate_validator_case_variant_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload[key] = unsafe_value

    with pytest.raises(ValueError, match=f"unknown or unallowlisted key: {key}"):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "key",
    [
        "use_cache",
        "cache",
        "inference_cache",
        "raw_pred",
        "raw_preds",
        "save_predictions",
        "load_predictions",
        "data_root",
        "data_path",
        "ann_file",
        "annotation",
        "annotations",
        "ckpt",
        "ckpt_path",
        "model_weights",
        "weights",
        "model_path",
        "result_detection",
        "training_permission",
        "execution_permission",
    ],
)
@pytest.mark.parametrize("unsafe_value", [True, False])
def test_p2qr_future_gate_validator_rejects_unallowlisted_permission_like_keys(key, unsafe_value):
    validator = _load_module("p2qr_gate_validator_permission_like_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload[key] = unsafe_value

    with pytest.raises(ValueError, match=f"unknown or unallowlisted key: {key}"):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


def test_p2qr_future_gate_validator_rejects_unknown_non_metadata_key_by_default():
    validator = _load_module("p2qr_gate_validator_unknown_default_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload["safe_note"] = "not explicitly allowed"

    with pytest.raises(ValueError, match="unknown or unallowlisted key: safe_note"):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


@pytest.mark.parametrize("key", ["note", "review_id"])
def test_p2qr_future_gate_validator_allows_harmless_metadata_key(key):
    validator = _load_module("p2qr_gate_validator_harmless_unknown_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload[key] = "manual review metadata only"

    assert validator.validate_gate_payload(payload, "a" * 64, "b" * 64) is True


def test_p2qr_future_gate_validator_rejects_non_bool_forbidden_false_alias():
    validator = _load_module("p2qr_gate_validator_non_bool_false_test", VALIDATOR_PATH)
    payload = _valid_gate_payload()
    payload["allow_dataset_access"] = "false"

    with pytest.raises(ValueError, match="allow_dataset_access=false/absent"):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


def test_p2qr_future_gate_validator_checks_file_hash(tmp_path):
    validator = _load_module("p2qr_gate_validator_file_test", VALIDATOR_PATH)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(_valid_gate_payload()), encoding="utf-8")
    good_sha = validator.sha256_file(gate_path)

    assert validator.validate_gate_file(gate_path, good_sha, "a" * 64, "b" * 64)["max_train_iters"] == 2

    with pytest.raises(ValueError, match="sha256 mismatch"):
        validator.validate_gate_file(gate_path, "0" * 64, "a" * 64, "b" * 64)


def test_p2qr_future_gate_validator_cli_accepts_and_rejects_hash(tmp_path):
    validator = _load_module("p2qr_gate_validator_cli_test", VALIDATOR_PATH)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(_valid_gate_payload()), encoding="utf-8")
    good_sha = validator.sha256_file(gate_path)

    ok = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--gate-json",
            str(gate_path),
            "--gate-sha256",
            good_sha,
            "--active-manifest-sha256",
            "a" * 64,
            "--resolved-config-sha256",
            "b" * 64,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert "P2QR_SHORT_SMOKE_GATE_VALIDATION_PASS" in ok.stdout

    bad = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--gate-json",
            str(gate_path),
            "--gate-sha256",
            "0" * 64,
            "--active-manifest-sha256",
            "a" * 64,
            "--resolved-config-sha256",
            "b" * 64,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert bad.returncode != 0
    assert "sha256 mismatch" in bad.stderr


def test_p2qr_train_entrypoint_passes_max_train_iters_to_train_loop_textually():
    train_text = TRAIN_PATH.read_text(encoding="utf-8")
    engine_text = TRAIN_ENGINE_PATH.read_text(encoding="utf-8")

    assert "max_train_iters=cfg.workflow.get(\"max_train_iters\", None)" in train_text
    assert "if max_train_iters is not None and (iter_idx + 1) >= max_train_iters" in engine_text
    assert "max_train_iters=%d reached; ending smoke epoch early" in engine_text
