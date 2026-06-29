import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py"
COARSE_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_coarse_actionness_uncertainty_c3_physical_grid_actionformer_n16r4.py"
)
EXACT_UNIFORM_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py"
)
UNIFORM_BIAS_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer_n16r4.py"
)
A_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py"
)
LAUNCHER = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.sbatch"
VALIDATOR = ROOT / "tools" / "bata" / "validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py"
GUARD = ROOT / "opentad" / "utils" / "training_guard.py"
PREBACKBONE_SELECTOR = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
SELECTOR_DIAGNOSTIC = ROOT / "tools" / "bata" / "analyze_pc_ot_mras_selector_posttrain_diagnostics.py"
C3_DIAGNOSTIC_GATE = ROOT / "tools" / "bata" / "validate_pc_ot_mras_c3_diagnostic_gate.py"

VARIANT_ID = "C3-PhysicalGridActionFormer-PreBackbone-OriginalAdaTAD"
ROUTE_ID = "pc_ot_mras_prebackbone_c3_physical_grid_actionformer"
STAGE_ID = "c3_physical_grid_actionformer_full_train_n16r4"
COARSE_VARIANT_ID = "C3-CoarseActionnessUncertainty-PreBackbone-OriginalAdaTAD"
COARSE_ROUTE_ID = "pc_ot_mras_coarse_actionness_uncertainty_c3_physical_grid_actionformer"
COARSE_STAGE_ID = "c3_coarse_actionness_uncertainty_fixed384_n16r4"
EXACT_UNIFORM_VARIANT_ID = "C3-ExactUniformPhysicalGrid-PreBackbone-OriginalAdaTAD"
EXACT_UNIFORM_ROUTE_ID = "pc_ot_mras_exact_uniform_c3_physical_grid_actionformer"
EXACT_UNIFORM_STAGE_ID = "c3_exact_uniform_physical_grid_fixed384_n16r4"
UNIFORM_BIAS_VARIANT_ID = "C3-UniformBiasedCoarseActionness-PreBackbone-OriginalAdaTAD"
UNIFORM_BIAS_ROUTE_ID = "pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer"
UNIFORM_BIAS_STAGE_ID = "c3_uniform_biased_coarse_actionness_fixed384_n16r4"
A_VARIANT_ID = "C3-A-UniformScaffoldSmallActionnessStrictMaxGap-PreBackbone-OriginalAdaTAD"
A_ROUTE_ID = "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer"
A_STAGE_ID = "c3_a_uniform_scaffold_small_actionness_strict_maxgap_fixed384_n16r4"
A_POLICY_KIND = "uniform_scaffold_small_actionness_strict_maxgap"
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
        "allow_checkpoint_load": False,
        "allow_resume": False,
        "allow_offline_ledger": False,
        "allow_raw_prediction_cache": False,
        "allow_detector_training": True,
        "uses_p2": False,
        "uses_offline_ledger": False,
        "uses_teacher": False,
        "uses_test_gt": False,
        "uses_raw_prediction_cache": False,
        "checkpoint_load": False,
        "resume": False,
        "offline_ledger": False,
        "raw_prediction_cache": False,
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
        "pretrained_resolved_path": "/remote/repo/pretrained/model.pth",
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


def test_coarse_actionness_full_train_selector_config_matches_runtime_signatures():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(COARSE_CONFIG))
    selector_cfg = _cfg_dict(cfg.model.frame_selector)
    reader_cfg = _cfg_dict(selector_cfg.pop("reader"))
    selector_cfg.pop("type")
    reader_cfg.pop("type")

    selector_params = _class_init_params(PREBACKBONE_SELECTOR, "PCOTMRASPreBackboneFrameSelector")
    reader_params = _class_init_params(PREBACKBONE_SELECTOR, "PCOTMRASCoarseActionnessFrameScout")

    assert cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.route_family == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.route_id == COARSE_ROUTE_ID
    assert cfg.variant_id == COARSE_VARIANT_ID
    assert cfg.stage_id == COARSE_STAGE_ID
    assert not (set(selector_cfg) - selector_params)
    assert not (set(reader_cfg) - reader_params)
    assert selector_cfg["selection_strategy"] == "coarse_actionness_uncertainty"
    assert reader_cfg["in_dim"] == selector_cfg["descriptor_dim"] == 3 * 32 * 32
    assert len(tuple(reader_cfg["dilations"])) == int(reader_cfg["temporal_layers"])
    assert cfg.model.frame_selector.aux_frame_score_boundary_loss_weight == 0.0
    assert cfg.model.frame_selector.aux_uncertainty_loss_weight == 0.0
    assert cfg.protocol_flags.remote_sync_allowed is True
    assert cfg.protocol_flags.slurm_allowed is True
    assert cfg.protocol_flags.tools_test_allowed is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.route == cfg.route_id
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.stage == cfg.stage_id


@pytest.mark.parametrize(
    ("config_path", "route_id", "variant_id", "stage_id"),
    (
        (EXACT_UNIFORM_CONFIG, EXACT_UNIFORM_ROUTE_ID, EXACT_UNIFORM_VARIANT_ID, EXACT_UNIFORM_STAGE_ID),
        (UNIFORM_BIAS_CONFIG, UNIFORM_BIAS_ROUTE_ID, UNIFORM_BIAS_VARIANT_ID, UNIFORM_BIAS_STAGE_ID),
    ),
)
def test_c3_coarse_next_step_configs_match_runtime_signatures(config_path, route_id, variant_id, stage_id):
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(config_path))
    selector_cfg = _cfg_dict(cfg.model.frame_selector)
    reader_cfg = _cfg_dict(selector_cfg.pop("reader"))
    selector_cfg.pop("type")
    reader_cfg.pop("type")

    selector_params = _class_init_params(PREBACKBONE_SELECTOR, "PCOTMRASPreBackboneFrameSelector")
    reader_params = _class_init_params(PREBACKBONE_SELECTOR, "PCOTMRASCoarseActionnessFrameScout")

    assert cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.route_family == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.route_id == route_id
    assert cfg.variant_id == variant_id
    assert cfg.stage_id == stage_id
    assert not (set(selector_cfg) - selector_params)
    assert not (set(reader_cfg) - reader_params)
    assert cfg.model.frame_selector.selection_strategy == "coarse_actionness_uncertainty"
    assert cfg.model.frame_selector.reader.type == "PCOTMRASCoarseActionnessFrameScout"
    assert cfg.model.frame_selector.reader.in_dim == cfg.model.frame_selector.descriptor_dim == 3 * 32 * 32
    assert len(tuple(cfg.model.frame_selector.reader.dilations)) == cfg.model.frame_selector.reader.temporal_layers
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.route == cfg.route_id
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.stage == cfg.stage_id


def test_c3_exact_uniform_config_is_reader_influence_free_uniform_control():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(EXACT_UNIFORM_CONFIG))
    selector = cfg.model.frame_selector

    assert cfg.experiment_scope.selection_strategy == "exact_uniform_physical_grid_control"
    assert cfg.experiment_scope.budget_protocol == "fixed384_over_dense768_exact_uniform_physical_grid_control"
    assert selector.coarse_uniform_count == 384
    assert selector.coarse_action_count == 0
    assert selector.coarse_uncertainty_count == 0
    assert selector.coarse_change_count == 0
    assert selector.coarse_background_count == 0
    assert selector.aux_gt_acquisition_loss_weight == 0.0
    assert selector.aux_duplicate_cap_loss_weight == 0.0
    assert selector.straight_through_detector_loss is False
    assert selector.straight_through_downstream is False
    assert selector.max_dense_gap == 0
    assert selector.max_gap_guard_count == 0
    assert selector.meta_source == "c3_exact_uniform_physical_grid_prebackbone_selector_control"


def test_c3_uniform_biased_config_keeps_uniform_scaffold_and_small_actionness_bias():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(UNIFORM_BIAS_CONFIG))
    selector = cfg.model.frame_selector

    assert cfg.experiment_scope.selection_strategy == "uniform_scaffold_small_actionness_bias_maxgap3"
    assert cfg.experiment_scope.budget_protocol == "fixed384_over_dense768_uniform288_action72_uncertainty24_guard12_maxgap3"
    assert selector.coarse_uniform_count == 288
    assert selector.coarse_action_count == 72
    assert selector.coarse_uncertainty_count == 24
    assert selector.coarse_change_count == 0
    assert selector.coarse_background_count == 0
    assert selector.max_dense_gap == 3
    assert selector.max_gap_guard_count == 12
    assert selector.coarse_action_weight == 1.0
    assert selector.coarse_uncertainty_weight == 0.35
    assert selector.coarse_change_weight == 0.0
    assert selector.meta_source == "c3_uniform_biased_coarse_actionness_guard12_maxgap3_prebackbone_selector"


def test_c3_a_config_keeps_uniform_scaffold_but_uses_new_identity_and_policy():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(A_CONFIG))
    selector = cfg.model.frame_selector

    assert cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.route_family == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.route_id == A_ROUTE_ID
    assert cfg.variant_id == A_VARIANT_ID
    assert cfg.stage_id == A_STAGE_ID
    assert cfg.experiment_scope.selection_strategy == A_POLICY_KIND
    assert cfg.experiment_scope.budget_protocol == "fixed384_over_dense768_uniform_scaffold_small_actionness_strict_maxgap_guard12"
    assert selector.coarse_uniform_count == 288
    assert selector.coarse_action_count == 72
    assert selector.coarse_uncertainty_count == 24
    assert selector.coarse_change_count == 0
    assert selector.coarse_background_count == 0
    assert selector.max_dense_gap == 3
    assert selector.max_gap_guard_count == 12
    assert selector.meta_source == "c3_a_uniform_scaffold_small_actionness_strict_maxgap_prebackbone_selector"


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
    monkeypatch.delenv("PC_OT_MRAS_PREBACKBONE_C3_PRETRAINED_PATH", raising=False)
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
        if split == "train":
            assert "window_size" not in cfg.dataset[split]
            assert load_steps[0].method == "random_trunc"
            assert load_steps[0].trunc_len == selector.dense_window_size
        else:
            assert cfg.dataset[split].window_size == selector.dense_window_size
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


def test_physical_grid_full_train_resolved_dataset_kwargs_match_runtime_constructors():
    mmengine_config = pytest.importorskip("mmengine.config")
    validator = _load_module(VALIDATOR, "validate_c3_physical_grid_full_train_dataset_kwargs_test")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    padding_params = _class_init_params(ROOT / "opentad" / "datasets" / "base" / "padding_dataset.py", "PaddingDataset")
    sliding_params = _class_init_params(ROOT / "opentad" / "datasets" / "base" / "sliding_dataset.py", "SlidingWindowDataset")
    expected_params = {
        "train": padding_params,
        "val": sliding_params,
        "test": sliding_params,
    }

    for split in ("train", "val", "test"):
        dataset_kwargs = validator.resolve_dataset_constructor_kwargs(cfg, split)
        assert "type" not in dataset_kwargs
        assert set(dataset_kwargs) <= expected_params[split]

    assert "window_size" not in validator.resolve_dataset_constructor_kwargs(cfg, "train")
    assert validator.resolve_dataset_constructor_kwargs(cfg, "val")["window_size"] == 768
    assert validator.resolve_dataset_constructor_kwargs(cfg, "test")["window_size"] == 768


def test_physical_grid_full_train_validator_accepts_config_and_rejects_leakage(tmp_path):
    validator = _load_module(VALIDATOR, "validate_c3_physical_grid_full_train_gate_test")
    assert validator.validate_config(CONFIG) is True
    assert validator.validate_config(COARSE_CONFIG) is True
    assert validator.validate_config(EXACT_UNIFORM_CONFIG) is True
    assert validator.validate_config(UNIFORM_BIAS_CONFIG) is True
    assert validator.validate_config(A_CONFIG) is True

    bad_dataset_config = tmp_path / "bad_physical_grid_dataset_config.py"
    bad_dataset_config.write_text(
        f'_base_ = ["{CONFIG.as_posix()}"]\n'
        "dataset = dict(train=dict(window_size=768))\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported constructor fields: .*window_size"):
        validator.validate_config(bad_dataset_config)

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

    missing_pretrained_path_payload = _gate_payload(pretrained_resolved_path="")
    with pytest.raises(ValueError, match="pretrained_resolved_path"):
        validator.validate_gate_payload(
            missing_pretrained_path_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
        )

    coarse_payload = _gate_payload(route=COARSE_ROUTE_ID, variant_id=COARSE_VARIANT_ID, stage=COARSE_STAGE_ID)
    assert (
        validator.validate_gate_payload(
            coarse_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
        )
        is coarse_payload
    )

    exact_uniform_payload = _gate_payload(
        route=EXACT_UNIFORM_ROUTE_ID,
        variant_id=EXACT_UNIFORM_VARIANT_ID,
        stage=EXACT_UNIFORM_STAGE_ID,
    )
    assert (
        validator.validate_gate_payload(
            exact_uniform_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
        )
        is exact_uniform_payload
    )

    uniform_bias_payload = _gate_payload(
        route=UNIFORM_BIAS_ROUTE_ID,
        variant_id=UNIFORM_BIAS_VARIANT_ID,
        stage=UNIFORM_BIAS_STAGE_ID,
    )
    assert (
        validator.validate_gate_payload(
            uniform_bias_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
        )
        is uniform_bias_payload
    )

    a_payload = _gate_payload(route=A_ROUTE_ID, variant_id=A_VARIANT_ID, stage=A_STAGE_ID)
    assert (
        validator.validate_gate_payload(
            a_payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
        )
        is a_payload
    )


def test_physical_grid_full_train_launcher_is_single_gpu_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert VARIANT_ID in text
    assert ROUTE_ID in text
    assert COARSE_CONFIG.name in text
    assert EXACT_UNIFORM_CONFIG.name in text
    assert UNIFORM_BIAS_CONFIG.name in text
    assert A_CONFIG.name in text
    assert 'CONFIG="${CONFIG:-$CONFIG_REVIEWED_A}"' in text
    assert '"$CONFIG_REVIEWED_A") ;;' in text
    assert "resolved_identity route=$ROUTE_ID variant=$VARIANT_ID stage=$STAGE_ID" in text
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
    assert '"allow_checkpoint_load": False' in text
    assert '"allow_resume": False' in text
    assert '"offline_ledger": False' in text
    assert '"raw_prediction_cache": False' in text
    assert "PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "Training Over" in text
    assert "validate_pc_ot_mras_c3_diagnostic_gate.py" in text
    assert "POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED" in text
    assert 'grep -Eiq "load_from_raw_predictions|RAW_PREDICTION_CACHE|PREDICTION_CACHE"' not in text
    assert "load_from_raw_predictions[\\\"']?[[:space:]]*[:=][[:space:]]*True" in text
    assert "(^|[[:space:]])PREDICTION_CACHE[[:space:]]*=" in text


def test_physical_grid_full_train_launcher_pretrained_gate_resolves_overrides_before_train():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "C3_PRETRAINED_PATH" in text
    assert 'PRETRAINED_PATH="${C3_PRETRAINED_PATH:-${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}}"' in text
    assert "PRETRAINED_RESOLVED_PATH=" in text
    assert "readlink -f" in text
    assert 'test -f "$PRETRAINED_RESOLVED_PATH" || fail "missing pretrained file' in text
    assert "resolved_pretrained_path=$PRETRAINED_RESOLVED_PATH" in text
    assert "PC_OT_MRAS_PREBACKBONE_C3_PRETRAINED_PATH=\"$PRETRAINED_RESOLVED_PATH\"" in text
    assert "tools/train.py \"$CONFIG\"" in text
    assert text.index('test -f "$PRETRAINED_RESOLVED_PATH"') < text.index("tools/train.py \"$CONFIG\"")
    assert '"pretrained_resolved_path": pretrained_resolved' in text
    assert "PRETRAINED_SHA256=\"$(sha256sum \"$PRETRAINED_RESOLVED_PATH\"" in text
    assert "--cfg-options" in text
    assert 'model.backbone.custom.pretrain="$PRETRAINED_RESOLVED_PATH"' in text


def test_physical_grid_full_train_stdout_raw_cache_marker_regression():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "if grep -Eq " in text
    pattern = (
        r"""load_from_raw_predictions["']?\s*[:=]\s*True"""
        r"""|save_raw_prediction["']?\s*[:=]\s*True"""
        r"""|(^|\s)RAW_PREDICTION_CACHE\s*="""
        r"""|(^|\s)PREDICTION_CACHE\s*="""
    )

    clean_stdout = "\n".join(
        [
            "inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)",
            "    allow_raw_prediction_cache=False,",
            "    uses_raw_prediction_cache=False",
            "2026-06-26 12:10:43 Train INFO: Training Over...",
        ]
    )
    assert re.search(pattern, clean_stdout) is None

    assert re.search(pattern, "load_from_raw_predictions=True")
    assert re.search(pattern, "save_raw_prediction=True")
    assert re.search(pattern, "PREDICTION_CACHE=/tmp/preds.json")
    assert re.search(pattern, "RAW_PREDICTION_CACHE=/tmp/preds.json")


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


def test_actionformer_optimizer_groups_cover_prebackbone_slot_queries():
    text = (ROOT / "opentad" / "models" / "detectors" / "actionformer.py").read_text(encoding="utf-8")

    assert 'pn.endswith("query_embed")' in text
    assert 'pn.endswith("slot_queries")' in text
    assert "pre-backbone selector reader slot queries" in text


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
