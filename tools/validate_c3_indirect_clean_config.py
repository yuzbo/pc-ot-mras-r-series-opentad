import argparse
from pathlib import Path

from mmengine.config import Config


COMMON_FORBIDDEN_CONFIG_TOKENS = [
    "physicalgrid",
    "physical_grid",
    "temporal_grid",
    "rf50",
    "bh_sdc",
    "bhsdc",
    "divergent",
    "pvr_qc",
    "pvr-qc",
    "pvrqc",
    "pqr",
    "pqr_ranking",
    "bvr",
    "p2head",
    "teacher",
    "oracle",
    "raw_prediction_cache",
    "start_head",
    "end_head",
]

COARSE_ONLY_FORBIDDEN_CONFIG_TOKENS = ["boundary_head"]


def _assert_no_forbidden_tokens(cfg_text, forbidden_tokens):
    found = [token for token in forbidden_tokens if token in cfg_text]
    if found:
        raise AssertionError(f"Forbidden old-route tokens in config: {found}")


def _validate_common_contract(cfg):
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.frame_selector.type == "PCOTMRASIndirectPreBackboneFrameSelector"
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.scout_spatial_size in (32, 64)
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    train_load = next(step for step in cfg.dataset.train.pipeline if step["type"] == "LoadFrames")
    assert train_load["trunc_len"] == 768
    assert cfg.dataset.val.window_size == 768
    assert cfg.dataset.test.window_size == 768


def _validate_coarse_actionness_route(cfg, cfg_text):
    assert cfg.model.frame_selector.scout.type == "PCOTMRASCoarseActionnessFrameScout"
    assert cfg.model.frame_selector.strategy == "coarse_actionness_uncertainty"
    _assert_no_forbidden_tokens(cfg_text, COMMON_FORBIDDEN_CONFIG_TOKENS + COARSE_ONLY_FORBIDDEN_CONFIG_TOKENS)


def _validate_cadf_densitymesh_route(cfg, cfg_text):
    selector = cfg.model.frame_selector
    assert selector.scout.type == "PCOTMRASCADFDensityFrameScout"
    assert selector.strategy == "cadf_density_mesh_st"
    assert cfg.get("c3_route_label", None) == "C3_MAINLINE_OPTIMIZATION"
    assert "C3_ORIGINAL_OPTIMIZATION_ROUTE" in cfg.get("c3_route_labels", [])
    if bool(cfg.get("c3_full_train_claim_unlocked", False)):
        raise AssertionError("CADF full-train mAP claim remains locked; do not unlock formal claim in config")
    claim_status = cfg.get("c3_claim_status", None)
    if claim_status not in {
        "precheck_only",
        "diagnostic_only",
        "backend_control",
        "formal_selector_candidate_locked",
        "formal_selector_candidate_fast_safe",
    }:
        raise AssertionError(f"Unsupported C3 claim status: {claim_status}")
    if bool(cfg.get("c3_physical_time_postprocess_enabled", False)):
        raise AssertionError("CADF physical-time postprocess must remain default-off diagnostic contract")
    _assert_no_forbidden_tokens(cfg_text, COMMON_FORBIDDEN_CONFIG_TOKENS)

    density_weights = selector.get("density_weights", {})
    boundary_weight = float(density_weights.get("boundary", 0.0))
    with_boundary_head = bool(selector.scout.get("with_boundary_head", False))
    if boundary_weight > 0.0 and not with_boundary_head:
        raise AssertionError("CADF boundary density weight requires scout.with_boundary_head=True")
    action_weight = float(density_weights.get("action", 0.0))
    uncertainty_weight = float(density_weights.get("uncertainty", 0.0))
    change_weight = float(density_weights.get("change", 0.0))
    if action_weight > 0.05:
        raise AssertionError("CADF direct action density weight must be zero or tiny")
    if uncertainty_weight <= action_weight or change_weight <= action_weight:
        raise AssertionError("CADF density must prioritize uncertainty/transition over direct action")
    if bool(selector.get("physical_time_postprocess_enabled", False)):
        raise AssertionError("CADF physical-time selector path must remain default-off diagnostic contract")
    if bool(selector.get("selected_index_aware_postprocess_enabled", False)):
        raise AssertionError("CADF selected-index-aware postprocess is metadata-only/default-off for this route gate")
    density_loss_v2_weights = {
        "density_window_mass_loss_weight": float(selector.get("density_window_mass_loss_weight", 0.0)),
        "density_max_gap_loss_weight": float(selector.get("density_max_gap_loss_weight", 0.0)),
        "density_blue_noise_loss_weight": float(selector.get("density_blue_noise_loss_weight", 0.0)),
        "density_weak_target_loss_weight": float(selector.get("density_weak_target_loss_weight", 0.0)),
    }
    for name, value in density_loss_v2_weights.items():
        if value < 0.0:
            raise AssertionError(f"CADF Density-Loss V2 weight must be non-negative: {name}={value}")
    density_loss_v2_enabled = any(value > 0.0 for value in density_loss_v2_weights.values())
    if density_loss_v2_enabled:
        if claim_status == "formal_selector_candidate_locked" or bool(cfg.get("c3_formal_selector_candidate", False)):
            raise AssertionError("Density-Loss V2 must remain default-off for formal selector candidates")
        if cfg.get("c3_claim_status", None) != "diagnostic_only":
            raise AssertionError("Density-Loss V2 may only be enabled by diagnostic_only configs")
        if cfg.get("c3_density_loss_v2_diagnostic_only", None) is not True:
            raise AssertionError("Density-Loss V2 enabled config must be marked c3_density_loss_v2_diagnostic_only=True")
        if cfg.get("c3_density_loss_v2_claim_unlocked", None) is not False:
            raise AssertionError("Density-Loss V2 diagnostic config must keep c3_density_loss_v2_claim_unlocked=False")
        if bool(cfg.get("c3_full_train_claim_unlocked", False)):
            raise AssertionError("Density-Loss V2 diagnostic config must not unlock full-train or paper claims")
    distribution_loss_weights = selector.get("density_distribution_loss_weights", {})
    distribution_loss_enabled = float(selector.get("density_distribution_loss_weight", 0.0)) > 0.0
    if distribution_loss_enabled:
        if cfg.get("c3_loss_select_v2", None) is not True:
            raise AssertionError("CADF loss-select V2 must be marked c3_loss_select_v2=True")
        if cfg.get("c3_loss_select_v2_deploy_time_inputs", None) != "scout_actionness_uncertainty_change_only":
            raise AssertionError("CADF loss-select V2 deploy-time inputs must be scout/actionness/uncertainty/change only")
        if cfg.get("c3_loss_select_v2_test_aux_source_leakage", None) != "forbidden":
            raise AssertionError("CADF loss-select V2 must explicitly forbid test-time auxiliary-source leakage")
        if bool(selector.get("density_distribution_loss_nan_guard", False)) is not True:
            raise AssertionError("CADF loss-select V2 must enable distribution NaN guard")
        if float(selector.get("density_distribution_logit_clamp", 0.0)) <= 0.0:
            raise AssertionError("CADF loss-select V2 must set a positive finite logit clamp")
        if float(selector.get("density_distribution_train_gt_target_weight", 0.0)) < 0.0:
            raise AssertionError("CADF loss-select V2 train-only GT target weight must be non-negative")
        required_distribution_parts = {"smooth", "local_cap", "large_gap", "collapse", "target_kl"}
        if set(distribution_loss_weights.keys()) != required_distribution_parts:
            raise AssertionError("CADF loss-select V2 must configure all distribution loss components")
        for name in required_distribution_parts:
            if float(distribution_loss_weights[name]) < 0.0:
                raise AssertionError(f"CADF loss-select V2 distribution weight must be non-negative: {name}")
        if selector.get("fast_cpu_selection", None) is not True:
            raise AssertionError("CADF loss-select V2 must preserve fast_cpu_selection")
        if bool(cfg.solver.get("amp", False)) or bool(cfg.solver.get("fp16_compress", False)) or bool(cfg.solver.get("ema", False)):
            fast_safe_speed_fix = cfg.get("c3_speed_fix", None) == "loss_select_v2_fast_safe_amp_fp16_withcp_off_diag_off"
            if not fast_safe_speed_fix:
                raise AssertionError("CADF loss-select V2 keeps AMP/fp16/EMA disabled unless the fast-safe gate is explicit")
            if bool(cfg.solver.get("amp", False)) is not True or bool(cfg.solver.get("fp16_compress", False)) is not True:
                raise AssertionError("CADF loss-select V2 fast-safe gate must enable both AMP and fp16_compress")
            if bool(cfg.solver.get("ema", False)):
                raise AssertionError("CADF loss-select V2 fast-safe gate must keep EMA disabled")
            if selector.get("emit_selection_diagnostics", None) is not False:
                raise AssertionError("CADF loss-select V2 fast-safe gate must disable per-iteration selection diagnostics")
            if int(selector.get("selection_diagnostics_interval", -1)) != 0:
                raise AssertionError("CADF loss-select V2 fast-safe gate must set selection_diagnostics_interval=0")
            backbone_cfg = cfg.model.get("backbone", {})
            backbone_inner_cfg = backbone_cfg.get("backbone", {}) if hasattr(backbone_cfg, "get") else {}
            if backbone_inner_cfg.get("with_cp", None) is not False:
                raise AssertionError("CADF loss-select V2 fast-safe gate must set model.backbone.backbone.with_cp=False")
        guard = cfg.solver.get("nonfinite_loss_guard", {})
        if not bool(guard.get("enabled", False)):
            raise AssertionError("CADF loss-select V2 must enable train-engine nonfinite_loss_guard")
        if int(guard.get("max_skips", -1)) != 0 or int(guard.get("max_consecutive_skips", -1)) != 0:
            raise AssertionError("CADF loss-select V2 nonfinite_loss_guard must fail closed on the first non-finite cost")
    if cfg.get("c3_alpha0_combo_gate", None) == "st_soft_path_plus_actionness_fp32_no_amp_no_ema":
        if bool(cfg.solver.get("amp", False)) or bool(cfg.solver.get("fp16_compress", False)) or bool(cfg.solver.get("ema", False)):
            raise AssertionError("CADF ST+actionness combo gate must keep AMP/fp16/EMA disabled")
        if float(selector.get("density_alpha", -1.0)) != 0.0:
            raise AssertionError("CADF ST+actionness combo gate must keep density_alpha=0.0")
        if float(selector.get("st_scale", 0.0)) <= 0.0 or int(selector.get("st_local_radius", 0)) <= 0:
            raise AssertionError("CADF ST+actionness combo gate must enable ST soft path")
        if float(selector.get("actionness_loss_weight", 0.0)) <= 0.0:
            raise AssertionError("CADF ST+actionness combo gate must enable actionness aux loss")
    if cfg.get("c3_speed_profile", None) == "train_iter_only_no_eval_no_checkpoint":
        if claim_status != "diagnostic_only":
            raise AssertionError("CADF train-iter speed profile must remain diagnostic_only")
        if int(cfg.workflow.get("max_train_iters", 0)) <= 0:
            raise AssertionError("CADF train-iter speed profile must bound max_train_iters")
        if int(cfg.workflow.get("val_eval_interval", 0)) != -1:
            raise AssertionError("CADF train-iter speed profile must disable eval")
        if int(cfg.workflow.get("val_loss_interval", 0)) != -1:
            raise AssertionError("CADF train-iter speed profile must disable val loss")
        if bool(cfg.workflow.get("disable_checkpoint", False)) is not True:
            raise AssertionError("CADF train-iter speed profile must disable checkpoints")
        profile_timing = cfg.workflow.get("profile_train_iter_timing", {})
        if not bool(profile_timing.get("enabled", False)):
            raise AssertionError("CADF train-iter speed profile must enable timing logs")
    if claim_status == "formal_selector_candidate_locked" and bool(cfg.get("c3_loss_select_v2_formal_candidate", False)):
        if cfg.get("launch_locked_until_user_unlock", None) is not True:
            raise AssertionError("CADF loss-select V2 formal candidate must remain locked until user unlock")
        if cfg.get("c3_loss_select_v2_user_unlock_evidence", None) != "PENDING":
            raise AssertionError("CADF loss-select V2 formal candidate evidence must remain PENDING before launch")
        if cfg.get("c3_speed_fix", None) != "selector_cpu_once_repair_diag_off_amp_withcp_probe":
            raise AssertionError("CADF loss-select V2 formal candidate must preserve the CADF fastfix marker")
        if selector.get("fast_cpu_selection", None) is not True:
            raise AssertionError("CADF loss-select V2 formal candidate must preserve fast_cpu_selection")
        if int(selector.get("selection_diagnostics_interval", -1)) != 1:
            raise AssertionError("CADF loss-select V2 formal candidate must keep bounded diagnostics interval=1")
        if "density_alpha_schedule" in selector:
            raise AssertionError("CADF loss-select V2 formal candidate must not use diagnostic alpha schedule")
        if float(selector.get("density_alpha", 0.0)) <= 0.0:
            raise AssertionError("CADF loss-select V2 formal candidate must use positive density_alpha")
    elif claim_status == "formal_selector_candidate_fast_safe" and bool(cfg.get("c3_loss_select_v2_formal_candidate", False)):
        if cfg.get("launch_locked_until_user_unlock", None) is not True:
            raise AssertionError("CADF fast-safe formal candidate must remain locked until user unlock")
        if cfg.get("c3_loss_select_v2_user_unlock_evidence", None) != "USER_AUTHORIZED_FAST_SAFE_REDEPLOY_AFTER_SLOW_PATH_STOP":
            raise AssertionError("CADF fast-safe formal candidate must record explicit redeploy authorization evidence")
        if cfg.get("c3_speed_fix", None) != "loss_select_v2_fast_safe_amp_fp16_withcp_off_diag_off":
            raise AssertionError("CADF fast-safe formal candidate must use the fast-safe speed-fix marker")
        if selector.get("fast_cpu_selection", None) is not True:
            raise AssertionError("CADF fast-safe formal candidate must enable fast_cpu_selection")
        if selector.get("emit_selection_diagnostics", None) is not False:
            raise AssertionError("CADF fast-safe formal candidate must disable per-iteration diagnostics")
        if int(selector.get("selection_diagnostics_interval", -1)) != 0:
            raise AssertionError("CADF fast-safe formal candidate must set selection_diagnostics_interval=0")
        if bool(cfg.solver.get("amp", False)) is not True or bool(cfg.solver.get("fp16_compress", False)) is not True:
            raise AssertionError("CADF fast-safe formal candidate must enable AMP/fp16_compress for speed")
        if bool(cfg.solver.get("ema", False)):
            raise AssertionError("CADF fast-safe formal candidate must keep EMA disabled")
        if int(cfg.scheduler.get("max_epoch", -1)) != int(cfg.workflow.get("end_epoch", -2)):
            raise AssertionError("CADF fast-safe formal candidate scheduler.max_epoch must match workflow.end_epoch")
        if int(cfg.scheduler.get("max_epoch", 0)) <= int(cfg.scheduler.get("warmup_epoch", 0)):
            raise AssertionError("CADF fast-safe formal candidate scheduler.max_epoch must exceed warmup_epoch")
        if int(cfg.workflow.get("val_eval_interval", -1)) != 5:
            raise AssertionError("CADF fast-safe formal candidate must avoid every-epoch eval; expected val_eval_interval=5")
        val_eval_epochs = [int(epoch) for epoch in cfg.workflow.get("val_eval_epochs", [])]
        if 2 not in val_eval_epochs:
            raise AssertionError("CADF fast-safe formal candidate must preserve the epoch2 first scheduled eval")
        if int(cfg.workflow.get("val_eval_interval_anchor_epoch", -1)) != 2:
            raise AssertionError("CADF fast-safe formal candidate must anchor later evals to epoch2")
        backbone_cfg = cfg.model.get("backbone", {})
        backbone_inner_cfg = backbone_cfg.get("backbone", {}) if hasattr(backbone_cfg, "get") else {}
        if backbone_inner_cfg.get("with_cp", None) is not False:
            raise AssertionError("CADF fast-safe formal candidate must set model.backbone.backbone.with_cp=False")
        if "density_alpha_schedule" in selector:
            raise AssertionError("CADF fast-safe formal candidate must not use diagnostic alpha schedule")
        if float(selector.get("density_alpha", 0.0)) <= 0.0:
            raise AssertionError("CADF fast-safe formal candidate must use positive density_alpha")
    elif claim_status == "formal_selector_candidate_locked":
        speed_fix = cfg.get("c3_speed_fix", None)
        speed_fix_enabled = speed_fix == "selector_cpu_once_repair_diag_off_amp_withcp_probe"
        if cfg.get("c3_formal_selector_candidate", None) is not True:
            raise AssertionError("CADF formal selector candidate must be explicitly marked")
        if cfg.get("launch_locked_until_combo_pass", None) is not True:
            raise AssertionError("CADF formal selector candidate must remain locked until combo pass evidence")
        required_config = "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py"
        if cfg.get("c3_combo_gate_required_config", None) != required_config:
            raise AssertionError("CADF formal selector candidate must point at the ST+actionness combo gate config")
        if cfg.get("c3_combo_gate_required_status", None) != "old_nan_window_pass_pending":
            raise AssertionError("CADF formal selector candidate must record pending old-window combo status")
        if cfg.get("c3_combo_old_window_pass_evidence", None) != "PENDING":
            raise AssertionError("CADF formal selector candidate evidence must remain PENDING before launch")
        if speed_fix_enabled:
            if selector.get("fast_cpu_selection", None) is not True:
                raise AssertionError("CADF speed-fix formal candidate must enable fast_cpu_selection")
            if selector.get("emit_selection_diagnostics", None) is not False:
                raise AssertionError("CADF speed-fix formal candidate must disable per-iteration selection diagnostics")
            if int(selector.get("selection_diagnostics_interval", -1)) != 0:
                raise AssertionError("CADF speed-fix formal candidate must set selection_diagnostics_interval=0")
            if bool(cfg.solver.get("ema", False)):
                raise AssertionError("CADF speed-fix formal candidate must keep EMA disabled")
            if bool(cfg.solver.get("amp", False)) is not True or bool(cfg.solver.get("fp16_compress", False)) is not True:
                raise AssertionError("CADF speed-fix formal candidate must explicitly gate AMP/fp16 for speed smoke")
        elif bool(cfg.solver.get("amp", False)) or bool(cfg.solver.get("fp16_compress", False)) or bool(cfg.solver.get("ema", False)):
            raise AssertionError("CADF formal selector candidate must keep AMP/fp16/EMA disabled")
        if float(selector.get("density_alpha", 0.0)) <= 0.0:
            raise AssertionError("CADF formal selector candidate must use positive density_alpha, not alpha0 backend control")
        if "density_alpha_schedule" in selector:
            raise AssertionError("CADF formal selector candidate must not use diagnostic alpha schedule")
        if int(selector.get("st_local_radius", 0)) != 2 or float(selector.get("st_scale", 0.0)) != 0.5:
            raise AssertionError("CADF formal selector candidate must preserve ST soft-path semantics")
        if float(selector.get("actionness_loss_weight", 0.0)) != 0.05:
            raise AssertionError("CADF formal selector candidate must preserve actionness auxiliary loss")
    if "density_alpha_schedule" in selector:
        schedule = selector.density_alpha_schedule
        assert float(schedule.train_start_alpha) <= float(schedule.train_target_alpha)
        assert int(schedule.warmup_iters) > 0
        assert schedule.test_alpha in {"target", "start", "base"} or isinstance(schedule.test_alpha, (int, float))
        if cfg.get("c3_alpha_schedule_recoverable", None) is not False:
            raise AssertionError("CADF alpha schedule is diagnostic and not checkpoint-resumable")
        if cfg.get("c3_alpha_schedule_scope", None) != "diagnostic_short_smoke_not_resumable":
            raise AssertionError("CADF alpha schedule scope must be diagnostic_short_smoke_not_resumable")
        if cfg.get("c3_long_train_resume_claim_locked", None) is not True:
            raise AssertionError("CADF long-train resume claim must remain locked for alpha schedule configs")


def validate_config(config_path):
    cfg = Config.fromfile(config_path)
    cfg_text = cfg.pretty_text.lower()
    _validate_common_contract(cfg)

    strategy = cfg.model.frame_selector.strategy
    scout_type = cfg.model.frame_selector.scout.type
    if strategy == "coarse_actionness_uncertainty" and scout_type == "PCOTMRASCoarseActionnessFrameScout":
        _validate_coarse_actionness_route(cfg, cfg_text)
        route = "coarse_actionness"
    elif strategy == "cadf_density_mesh_st" and scout_type == "PCOTMRASCADFDensityFrameScout":
        _validate_cadf_densitymesh_route(cfg, cfg_text)
        route = "cadf_density_mesh"
    else:
        raise AssertionError(f"Unsupported C3 selector route: strategy={strategy}, scout={scout_type}")

    print(f"PASS_C3_INDIRECT_CLEAN_CONFIG route={route} {Path(config_path).as_posix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="C3 indirect clean config path")
    args = parser.parse_args()
    validate_config(args.config)


if __name__ == "__main__":
    main()
