import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py"
LAUNCHER = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.sbatch"
VALIDATOR = ROOT / "tools" / "bata" / "validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py"
GUARD = ROOT / "opentad" / "utils" / "training_guard.py"
PREBACKBONE_SELECTOR = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
SELECTOR_DIAGNOSTIC = ROOT / "tools" / "bata" / "analyze_pc_ot_mras_selector_posttrain_diagnostics.py"
C3_DIAGNOSTIC_GATE = ROOT / "tools" / "bata" / "validate_pc_ot_mras_c3_diagnostic_gate.py"

VARIANT_ID = "C3-PhysicalGridActionFormer-PreBackbone-OriginalAdaTAD"
ROUTE_ID = "pc_ot_mras_prebackbone_c3_physical_grid_actionformer"
STAGE_ID = "c3_physical_grid_actionformer_full_train_n16r4"
ALLOW_DECISION = "ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _class_init_params(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return {arg.arg for arg in item.args.args if arg.arg != "self"}
    raise AssertionError(f"missing {class_name}.__init__ in {path}")


def _cfg_dict(config_dict):
    return {key: value for key, value in dict(config_dict).items()}


def _gate_payload(**updates):
    payload = {
        "decision": ALLOW_DECISION,
        "route": ROUTE_ID,
        "variant_id": VARIANT_ID,
        "stage": STAGE_ID,
        "execution_mode": "train",
        "selection_surface": "pre_backbone_raw_frame",
        "selection_timing": "online_before_backbone",
        "acquisition_unit": "frame",
        "budget": 384,
        "dense_window_size": 768,
        "allow_tools_train": True,
        "allow_tools_test": False,
        "direct_tools_test": False,
        "allow_slurm": True,
        "allow_gpu": True,
        "single_gpu": True,
        "allow_precheck_only": True,
        "allow_prebackbone_frame_selector": True,
        "allow_train_validation_map": True,
        "allow_long_training": True,
        "allow_pretrained_initialization": True,
        "allow_checkpoint_write": True,
        "allow_detector_training": True,
        "uses_p2": False,
        "uses_offline_ledger": False,
        "uses_teacher": False,
        "uses_test_gt": False,
        "uses_raw_prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "metric_claim": False,
        "metric_claim_allowed": False,
        "paper_claim": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim": False,
        "deploy_claim_allowed": False,
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "pretrained_sha256": "pretrained-sha",
    }
    payload.update(updates)
    return payload


def test_physical_grid_full_train_selector_config_matches_runtime_signatures():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    selector_cfg = _cfg_dict(cfg.model.frame_selector)
    reader_cfg = _cfg_dict(selector_cfg.pop("reader"))
    selector_cfg.pop("type")
    reader_cfg.pop("type")

    selector_params = _class_init_params(PREBACKBONE_SELECTOR, "PCOTMRASPreBackboneFrameSelector")
    reader_params = _class_init_params(PREBACKBONE_SELECTOR, "PCOTMRASBoundaryDifficultyTemporalFrameScout")

    assert not (set(selector_cfg) - selector_params)
    assert not (set(reader_cfg) - reader_params)
    assert int(selector_cfg["descriptor_dim"]) == int(reader_cfg["in_dim"])
    assert len(tuple(reader_cfg["dilations"])) == int(reader_cfg["temporal_layers"])
    assert int(cfg.model.projection.max_seq_len) == int(cfg.model.frame_selector.target_len)
    assert int(cfg.model.backbone.backbone.total_frames) == int(cfg.model.frame_selector.target_len)
    assert int(cfg.model.backbone.custom.pre_processing_pipeline[0].t1) == 24
    assert int(cfg.model.backbone.custom.post_processing_pipeline[1].t1) == 24
    assert int(cfg.model.backbone.custom.post_processing_pipeline[2].size) == int(cfg.model.frame_selector.target_len)

    script = LAUNCHER.read_text(encoding="utf-8")
    assert "PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL" in script
    assert "PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_MAX_ROWS" in script


def test_physical_grid_full_train_selector_runtime_builds_when_torch_is_available():
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if torch_probe.returncode != 0:
        detail = torch_probe.stderr.strip() or torch_probe.stdout.strip() or f"exit {torch_probe.returncode}"
        pytest.skip(f"torch unavailable for runtime selector build check: {detail}")

    mmengine_config = pytest.importorskip("mmengine.config")
    helper = _load_module(
        ROOT / "tests" / "test_c3_physical_grid_actionformer_candidate.py",
        "c3_physical_grid_candidate_runtime_build_helper",
    )
    PCOTMRASPreBackboneFrameSelector = helper._install_prebackbone_selector_or_skip()
    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    selector_cfg = _cfg_dict(cfg.model.frame_selector)
    selector_cfg.pop("type")
    selector = PCOTMRASPreBackboneFrameSelector(**selector_cfg)

    assert selector.target_len == 384
    assert selector.dense_window_size == 768
    assert selector.remap_gt_to_selected_axis is False
    assert selector.selection_strategy == "frame_score_topk"
    assert selector.scout_feature_source == "compressed_pixels"
    assert selector.reader.__class__.__name__ == "PCOTMRASBoundaryDifficultyTemporalFrameScout"


def test_physical_grid_full_train_config_identity_and_guard_contract(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    guard = _load_module(GUARD, "training_guard_for_c3_physical_grid_full_train_test")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.variant_id == VARIANT_ID
    assert cfg.route_id == ROUTE_ID
    assert cfg.stage_id == STAGE_ID
    assert cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.route_family == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.experiment_scope.selection_surface == "pre_backbone_raw_frame"
    assert cfg.experiment_scope.backend == "OriginalAdaTAD_ActionFormerPhysicalGrid"
    assert cfg.experiment_scope.uses_p2 is False
    assert cfg.experiment_scope.uses_raw_prediction_cache is False
    assert cfg.experiment_scope.uses_teacher is False
    assert cfg.experiment_scope.uses_test_gt is False
    assert cfg.experiment_scope.uses_offline_ledger is False

    selector = cfg.model.frame_selector
    assert selector.type == "PCOTMRASPreBackboneFrameSelector"
    assert selector.target_len == 384
    assert selector.dense_window_size == 768
    assert selector.descriptor_dim == selector.reader.in_dim == 3 * 32 * 32
    assert len(tuple(selector.reader.dilations)) == selector.reader.temporal_layers
    assert selector.remap_gt_to_selected_axis is False
    assert cfg.model.projection.max_seq_len == selector.target_len
    assert cfg.model.backbone.backbone.total_frames == selector.target_len
    assert cfg.model.backbone.custom.pre_processing_pipeline[0].t1 == 24
    assert cfg.model.backbone.custom.post_processing_pipeline[1].t1 == 24
    assert cfg.model.backbone.custom.post_processing_pipeline[2].size == selector.target_len
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled is True
    assert cfg.model.rpn_head.physical_grid_actionformer.required is True
    assert cfg.model.rpn_head.physical_grid_actionformer.strict is True
    assert cfg.post_processing.save_dict is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    for split in ("train", "val", "test"):
        load_steps = [step for step in cfg.dataset[split].pipeline if step.get("type") == "LoadFrames"]
        assert len(load_steps) == 1
        assert load_steps[0].remap_gt_to_selected_axis is False
        assert cfg.dataset[split].window_size == selector.dense_window_size
        if split == "train":
            assert load_steps[0].method == "random_trunc"
            assert load_steps[0].trunc_len == selector.dense_window_size
        else:
            assert load_steps[0].method == "sliding_window"

    assert selector.target_len == 384

    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    assert gate.formal_train_candidate is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_slurm is True
    assert gate.allow_gpu is True
    assert gate.allow_detector_training is True
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)
    assert gate.entrypoint_gate_context.allowed_decisions == (ALLOW_DECISION,)

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "physical_grid_gate.json"
    gate_json.write_text(json.dumps(_gate_payload()), encoding="utf-8")
    gate_sha = subprocess.check_output(
        [sys.executable, "-c", "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())", str(gate_json)],
        text=True,
    ).strip()
    monkeypatch.setenv("OPENTAD_C3_PHYSICAL_GRID_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_C3_PHYSICAL_GRID_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_C3_PHYSICAL_GRID_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_C3_PHYSICAL_GRID_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_physical_grid_full_train_validator_accepts_config_and_rejects_leakage(tmp_path):
    validator = _load_module(VALIDATOR, "validate_c3_physical_grid_full_train_gate_test")
    assert validator.validate_config(CONFIG) is True

    bad_config = tmp_path / "bad_physical_grid_config.py"
    bad_config.write_text(
        f'_base_ = ["{CONFIG.as_posix()}"]\n'
        "model = dict(frame_selector=dict(remap_gt_to_selected_axis=True))\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="remap_gt_to_selected_axis"):
        validator.validate_config(bad_config)

    bad_payload = _gate_payload(uses_teacher=True)
    with pytest.raises(ValueError, match="uses_teacher=false"):
        validator.validate_gate_payload(
            bad_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
        )


def test_physical_grid_full_train_launcher_is_single_gpu_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert VARIANT_ID in text
    assert ROUTE_ID in text
    assert "PRECHECK_ONLY=\"${PRECHECK_ONLY:-1}\"" in text
    assert "ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN=1" in text
    assert "expected branch codex/c3-physical-grid-head-20260625" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py" in text
    assert "tools/train.py \"$CONFIG\"" in text
    assert "python tools/test.py" not in text
    assert "torchrun tools/test.py" not in text
    assert "raw prediction/cache is forbidden" in text
    assert "offline ledger is forbidden" in text
    assert "P2/teacher/test-GT shortcuts are forbidden" in text
    assert "PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "Training Over" in text
    assert "validate_pc_ot_mras_c3_diagnostic_gate.py" in text
    assert "POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED" in text
    assert 'grep -Eiq "load_from_raw_predictions|RAW_PREDICTION_CACHE|PREDICTION_CACHE"' not in text
    assert "load_from_raw_predictions[\\\"']?[[:space:]]*[:=][[:space:]]*True" in text


def test_physical_grid_full_train_launcher_and_validator_static_no_forbidden_routes():
    combined = CONFIG.read_text(encoding="utf-8") + "\n" + LAUNCHER.read_text(encoding="utf-8")
    lower = combined.lower()

    for token in (
        "divergent_innovation",
        "bh_sdc",
        "event-surprise",
        "boundary microscope",
        "frame/token hybrid",
        "load_from_raw_predictions=True",
        "save_raw_prediction=True",
    ):
        assert token not in lower


def test_physical_grid_full_train_runtime_dependencies_are_present_and_c3_mainline_only():
    selector_text = PREBACKBONE_SELECTOR.read_text(encoding="utf-8")
    selector_init = (ROOT / "opentad" / "models" / "selectors" / "__init__.py").read_text(encoding="utf-8")
    selector_diag_text = SELECTOR_DIAGNOSTIC.read_text(encoding="utf-8")
    gate_text = C3_DIAGNOSTIC_GATE.read_text(encoding="utf-8")

    assert "class PCOTMRASPreBackboneFrameSelector" in selector_text
    assert "class PCOTMRASBoundaryDifficultyTemporalFrameScout" in selector_text
    assert "remap_gt_to_selected_axis" in selector_text
    assert "PCOTMRASPreBackboneFrameSelector" in selector_init
    assert "PCOTMRASBoundaryDifficultyTemporalFrameScout" in selector_init
    assert "PC_OT_MRAS_SELECTOR_POSTTRAIN_DIAGNOSTIC_READY" in selector_diag_text
    assert "PC_OT_MRAS_C3_DIAGNOSTIC_GATE_READY" in gate_text

    combined = "\n".join((selector_text, selector_diag_text, gate_text)).lower()
    for token in (
        "bh_sdc",
        "divergent_innovation",
        "event-surprise",
        "boundary microscope",
        "frame/token hybrid",
    ):
        assert token not in combined
