from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "boundary_microscope_acquisition_local_precheck.py"
FULL_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "boundary_microscope_acquisition_full_train_candidate_n16r4.py"
)
VALIDATOR = ROOT / "tools" / "bata" / "validate_boundary_microscope_gate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
DOC_CONTEXT = ROOT / "docs" / "en" / "boundary_microscope_acquisition_route_review_context_20260624.md"
N16R4_PRECHECK_LAUNCHER = ROOT / "scripts" / "run_boundary_microscope_precheck_n16r4.sbatch"
N16R4_FULL_TRAIN_LAUNCHER = ROOT / "scripts" / "run_boundary_microscope_full_train_n16r4.sbatch"
FULL_TRAIN_USER_OVERRIDE_STATEMENT = "USER_EXPLICITLY_REQUESTED_BOUNDARY_MICROSCOPE_FULL_TRAIN_CANDIDATE_ON_2026-06-24"
EXPECTED_PRO_FIX_BRANCH = "codex/boundary-microscope-pro-fix-20260625"


def _load_validator():
    spec = importlib.util.spec_from_file_location("boundary_microscope_gate_for_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_guard():
    spec = importlib.util.spec_from_file_location("boundary_microscope_training_guard_for_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config_or_skip(path: Path):
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(path))


def _sha_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_train_payload(
    manifest="manifest-sha",
    resolved="resolved-sha",
    run_tag="boundary_microscope_full_train_fixed_run_tag",
):
    return {
        "decision": "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN",
        "route": "boundary_microscope_acquisition",
        "route_label": "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": manifest,
        "resolved_config_sha256": resolved,
        "run_tag": run_tag,
        "budget": 384,
        "dense_window_size": 768,
        "user_override_statement": FULL_TRAIN_USER_OVERRIDE_STATEMENT,
        "allow_precheck_only": False,
        "allow_tools_train": True,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": True,
        "allow_gpu": True,
        "allow_full_train": True,
        "allow_raw_prediction": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "test_time_gt_allowed": False,
        "teacher_allowed": False,
        "raw_prediction_cache_allowed": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def test_boundary_microscope_local_config_is_parseable_and_fail_closed():
    cfg = _load_config_or_skip(LOCAL_CONFIG)

    assert cfg.route_label == "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
    assert cfg.route_id == "boundary_microscope_acquisition"
    assert cfg.model.frame_selector.type == "BoundaryMicroscopeAcquisitionRoute"
    assert cfg.model.frame_selector.meta_key == "boundary_microscope_acquisition_plan"
    assert cfg.model.rpn_head.physical_grid_actionformer is True
    assert cfg.model.rpn_head.temporal_grid.mode == "physical"
    assert cfg.model.rpn_head.temporal_grid.decode_axis == "dense"
    assert cfg.model.rpn_head.temporal_grid.positions_key == "irregular_selected_positions"
    assert cfg.model.rpn_head.temporal_grid.valid_len_key == "irregular_selected_valid_len"
    assert cfg.model.rpn_head.temporal_grid.required is True
    assert cfg.model.rpn_head.temporal_grid.strict is True
    assert cfg.boundary_microscope_gate.allow_precheck_only is True
    assert cfg.boundary_microscope_gate.allow_tools_train is False
    assert cfg.boundary_microscope_gate.allow_tools_test is False
    assert cfg.boundary_microscope_gate.allow_remote_sync is False
    assert cfg.boundary_microscope_gate.allow_slurm is False
    assert cfg.boundary_microscope_gate.allow_gpu is False
    assert cfg.boundary_microscope_gate.allow_full_train is False
    assert cfg.boundary_microscope_gate.allow_raw_prediction is False
    assert cfg.boundary_microscope_gate.metric_claim_allowed is False
    assert cfg.boundary_microscope_gate.paper_claim_allowed is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


def test_boundary_microscope_full_train_candidate_config_is_gate_bound_train_only():
    cfg = _load_config_or_skip(FULL_CONFIG)

    gate = cfg.boundary_microscope_gate
    assert gate.stage == "full_train_candidate_n16r4"
    assert gate.requires_gate_json is True
    assert gate.allow_precheck_only is False
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.allow_full_train is False
    assert gate.allow_raw_prediction is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)
    assert gate.entrypoint_gate_context.required is True
    assert gate.entrypoint_gate_context.gate_json_env == "OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_JSON"
    assert gate.entrypoint_gate_context.gate_sha256_env == "OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_SHA256"
    assert tuple(gate.entrypoint_gate_context.allowed_decisions) == ("ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN",)
    assert gate.entrypoint_gate_context.required_exact_values.route == "boundary_microscope_acquisition"
    assert gate.entrypoint_gate_context.required_exact_values.budget == 384
    assert gate.entrypoint_gate_context.required_exact_values.dense_window_size == 768
    assert gate.entrypoint_gate_context.required_exact_values.user_override_statement == FULL_TRAIN_USER_OVERRIDE_STATEMENT
    assert "allow_tools_train" in tuple(gate.entrypoint_gate_context.required_true_keys)
    assert "allow_tools_test" in tuple(gate.entrypoint_gate_context.required_false_keys)
    assert "metric_claim_allowed" in tuple(gate.entrypoint_gate_context.required_false_keys)
    assert "paper_claim_allowed" in tuple(gate.entrypoint_gate_context.required_false_keys)
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.model.frame_selector.max_dense_gap <= 8
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.rpn_head.physical_grid_actionformer is True
    assert cfg.model.rpn_head.temporal_grid.mode == "physical"
    assert cfg.model.rpn_head.temporal_grid.decode_axis == "dense"
    assert cfg.model.rpn_head.temporal_grid.positions_key == "irregular_selected_positions"
    assert cfg.model.rpn_head.temporal_grid.valid_len_key == "irregular_selected_valid_len"


def test_boundary_microscope_training_guard_requires_external_full_train_gate(tmp_path, monkeypatch):
    cfg = _load_config_or_skip(FULL_CONFIG)
    training_guard = _load_guard()

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "boundary_full_train_gate.json"
    gate_json.write_text(json.dumps(_full_train_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_boundary_microscope_training_guard_rejects_invalid_full_train_gate(tmp_path, monkeypatch):
    cfg = _load_config_or_skip(FULL_CONFIG)
    training_guard = _load_guard()

    gate_json = tmp_path / "boundary_bad_full_train_gate.json"
    gate_json.write_text(json.dumps(_full_train_payload(resolved="other-resolved-sha")), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_BOUNDARY_MICROSCOPE_RESOLVED_CONFIG_SHA256", "resolved-sha")

    with pytest.raises(RuntimeError, match="resolved config sha256 mismatch"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_boundary_microscope_gate_validator_rejects_open_train_or_claims(tmp_path):
    validator = _load_validator()
    payload = {
        "decision": "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY",
        "route": "boundary_microscope_acquisition",
        "route_label": "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "allow_raw_prediction": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "test_time_gt_allowed": False,
        "teacher_allowed": False,
        "raw_prediction_cache_allowed": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }

    assert validator.validate_gate_payload(
        payload,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        budget=384,
        dense_window_size=768,
    )

    for key in ("allow_tools_train", "allow_slurm", "allow_full_train", "metric_claim_allowed"):
        bad = dict(payload)
        bad[key] = True
        with pytest.raises(ValueError, match=key):
            validator.validate_gate_payload(
                bad,
                active_manifest_sha256="manifest-sha",
                resolved_config_sha256="resolved-sha",
                budget=384,
                dense_window_size=768,
            )

    gate_json = tmp_path / "boundary_gate.json"
    gate_json.write_text(json.dumps(payload), encoding="utf-8-sig")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    loaded = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        budget=384,
        dense_window_size=768,
    )
    assert loaded["decision"] == "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY"


def test_boundary_microscope_gate_rejects_full_train_decision_and_route_attribution_drift():
    validator = _load_validator()
    payload = {
        "decision": "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY",
        "route": "boundary_microscope_acquisition",
        "route_label": "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "allow_raw_prediction": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "test_time_gt_allowed": False,
        "teacher_allowed": False,
        "raw_prediction_cache_allowed": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }

    for decision in (
        "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN",
        "ALLOW_BOUNDARY_MICROSCOPE_REMOTE_SYNC",
        "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY_AND_FULL_TRAIN",
    ):
        bad = dict(payload)
        bad["decision"] = decision
        with pytest.raises(ValueError, match="decision"):
            validator.validate_gate_payload(
                bad,
                active_manifest_sha256="manifest-sha",
                resolved_config_sha256="resolved-sha",
                budget=384,
                dense_window_size=768,
            )

    for token in ("BH-SDC", "Event Surprise", "event-surprise", "frame-token", "frame_token", "combo"):
        bad = dict(payload)
        bad["note"] = f"silent attribution drift to {token}"
        with pytest.raises(ValueError, match="attribution"):
            validator.validate_gate_payload(
                bad,
                active_manifest_sha256="manifest-sha",
                resolved_config_sha256="resolved-sha",
                budget=384,
                dense_window_size=768,
            )


def test_boundary_microscope_validator_cli_passes_for_precheck_payload(tmp_path):
    payload = {
        "decision": "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY",
        "route": "boundary_microscope_acquisition",
        "route_label": "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "allow_raw_prediction": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "test_time_gt_allowed": False,
        "teacher_allowed": False,
        "raw_prediction_cache_allowed": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    gate_json = tmp_path / "boundary_gate.json"
    gate_json.write_text(json.dumps(payload), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--gate-json",
            str(gate_json),
            "--gate-sha256",
            gate_sha,
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--budget",
            "384",
            "--dense-window-size",
            "768",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "BOUNDARY_MICROSCOPE_GATE_VALIDATION_PASS" in result.stdout


def test_boundary_microscope_full_train_gate_json_is_fail_closed_when_missing_or_invalid(tmp_path):
    gate_json = tmp_path / "boundary_full_train_gate.json"
    payload = _full_train_payload()
    gate_json.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    missing_result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--action",
            "full-train",
            "--gate-json",
            str(tmp_path / "missing_gate.json"),
            "--gate-sha256",
            gate_sha,
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--run-tag",
            "boundary_microscope_full_train_fixed_run_tag",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert missing_result.returncode != 0
    assert "missing Boundary microscope gate JSON" in missing_result.stderr

    invalid_result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--action",
            "full-train",
            "--gate-json",
            str(gate_json),
            "--gate-sha256",
            "bad-sha",
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--run-tag",
            "boundary_microscope_full_train_fixed_run_tag",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert invalid_result.returncode != 0
    assert "sha256 mismatch" in invalid_result.stderr


def test_boundary_microscope_full_train_gate_json_can_authorize_tools_train(tmp_path):
    gate_json = tmp_path / "boundary_full_train_gate.json"
    payload = _full_train_payload()
    gate_json.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--action",
            "full-train",
            "--gate-json",
            str(gate_json),
            "--gate-sha256",
            gate_sha,
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--run-tag",
            "boundary_microscope_full_train_fixed_run_tag",
            "--json",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_VALIDATION_PASS"
    assert report["decision"] == "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN"
    assert report["allows_tools_train"] is True
    assert report["allows_full_train"] is True
    assert report["run_tag"] == "boundary_microscope_full_train_fixed_run_tag"


def test_boundary_microscope_validator_cli_reports_config_is_precheck_only_json():
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(FULL_CONFIG),
            "--json",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS"
    assert report["route_label"] == "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
    assert report["allowed_decision"] == "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN"
    assert report["allows_tools_train"] is True
    assert report["requires_entrypoint_gate"] is True
    assert report["allows_full_train"] is False
    assert report["allows_remote_sync"] is False
    assert report["allows_slurm"] is False
    assert report["future_full_train_requires_separate_decision"] is True


def test_boundary_microscope_configs_do_not_reference_old_c3_selector_tokens():
    forbidden = (
        "pc_ot_mras_prebackbone_frame_selector",
        "PCOTMRASPreBackboneFrameSelector",
        "pc_ot_mras_prebackbone_c3",
        "C3-Pro",
        "C3_RS",
    )
    for path in (LOCAL_CONFIG, FULL_CONFIG):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_boundary_microscope_configs_docs_and_gate_keep_negative_route_attribution_explicit():
    config_text = "\n".join(path.read_text(encoding="utf-8") for path in (LOCAL_CONFIG, FULL_CONFIG)).lower()
    doc_text = DOC_CONTEXT.read_text(encoding="utf-8").lower()
    validator_text = VALIDATOR.read_text(encoding="utf-8").lower()

    for forbidden in ("bh-sdc", "event-surprise", "frame-token", "combo"):
        assert forbidden not in config_text
        assert forbidden in doc_text
    assert "must not be mixed" in doc_text
    assert "forbidden_attribution_tokens" in validator_text


def test_boundary_microscope_selector_is_exported_for_registry_discovery():
    selectors_init = ROOT / "opentad" / "models" / "selectors" / "__init__.py"
    text = selectors_init.read_text(encoding="utf-8")

    assert "boundary_microscope_acquisition_route" in text
    assert "BoundaryMicroscopeAcquisitionRoute" in text
    assert "BOUNDARY_MICROSCOPE_ROUTE_LABEL" in text


def test_boundary_microscope_n16r4_precheck_launcher_is_fail_closed():
    text = N16R4_PRECHECK_LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert f'EXPECTED_GIT_BRANCH="${{EXPECTED_GIT_BRANCH:-{EXPECTED_PRO_FIX_BRANCH}}}"' in text
    assert "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN" in text
    assert "PRECHECK_ONLY=1 must not set ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1" in text
    assert "full train remains locked" in text
    assert "BOUNDARY_MICROSCOPE_PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "tools/train.py" in text
    assert text.index('if [ "$PRECHECK_ONLY" = "1" ]') < text.index("tools/train.py")
    assert "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3" in text


def test_boundary_microscope_n16r4_full_train_launcher_is_locked_by_default_and_runs_train_after_gate():
    text = N16R4_FULL_TRAIN_LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert f'EXPECTED_GIT_BRANCH="${{EXPECTED_GIT_BRANCH:-{EXPECTED_PRO_FIX_BRANCH}}}"' in text
    assert "PRECHECK_ONLY=0 requires ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1" in text
    assert "BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON" in text
    assert "--action full-train" in text
    assert "--run-tag \"$RUN_TAG\"" in text
    assert "BOUNDARY_MICROSCOPE_PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "BOUNDARY_MICROSCOPE_FULL_TRAIN_TOOLS_TRAIN_STARTED" in text
    assert "OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_JSON" in text
    assert "OPENTAD_BOUNDARY_MICROSCOPE_RESOLVED_CONFIG_SHA256" in text
    assert "python tools/train.py \"$CONFIG\" --id \"$TRAIN_ID\"" in text
    assert "full train remains locked" not in text
    assert text.index("--action full-train") < text.index("python tools/train.py")
    assert "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3" in text
