from pathlib import Path
import math

from mmengine.config import Config
import pytest

from opentad.models.builder import build_selector
from tools.validate_c3_indirect_clean_config import validate_config


ROOT = Path(__file__).resolve().parents[1]
SMOKE32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_short_smoke.py"
FULL32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_full_train.py"
ALPHA0_32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_backend_control.py"
ALPHA0_STABILITY_32 = (
    ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_pure_uniform_stability_probe.py"
)
ALPHA0_ONLY_ACTIONNESS_32 = (
    ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_only_actionness_stability_probe.py"
)
ALPHA0_ONLY_ST_32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_only_st_stability_probe.py"
ALPHA0_ST_ACTIONNESS_32 = (
    ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py"
)
ALPHA0_ONLY_AMP_32 = (
    ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_only_amp_fp16_ema_stability_probe.py"
)
STAGED32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_staged_diagnostic.py"
CANDIDATE32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_full_train_candidate_diagnostic.py"
FORMAL32 = (
    ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py"
)
FORMAL_FASTFIX32 = (
    ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_fastfix.py"
)
DENSITY_LOSS_V2_DIAG32 = (
    ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_density_loss_v2_diagnostic.py"
)
FORMAL_LAUNCHER = ROOT / "logs/run_c3_cadf_formal_selector_candidate_locked_n16r4.sh"
SMOKE64 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_64px_short_smoke.py"
FULL64 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_64px_full_train.py"


def _assert_cadf_config(cfg, scout_size):
    cfg_text = cfg.pretty_text.lower()
    selector_text = repr(cfg.model.frame_selector).lower()

    assert cfg.c3_route_label == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.c3_route_labels == ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
    assert cfg.c3_method == "C3-CADF-DensityMesh-ST-OriginalAdaTAD"
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.frame_selector.type == "PCOTMRASIndirectPreBackboneFrameSelector"
    assert cfg.model.frame_selector.strategy == "cadf_density_mesh_st"
    assert "quotas" not in cfg.model.frame_selector
    assert "quota" not in selector_text
    assert "category" not in selector_text
    assert cfg.model.frame_selector.density_repulsion_loss_weight == 0.0
    if bool(cfg.get("c3_density_loss_v2_diagnostic_only", False)):
        assert cfg.c3_claim_status == "diagnostic_only"
        assert cfg.get("c3_density_loss_v2_claim_unlocked", None) is False
    else:
        assert cfg.model.frame_selector.get("density_blue_noise_loss_weight", 0.0) == 0.0
        assert cfg.model.frame_selector.get("density_window_mass_loss_weight", 0.0) == 0.0
        assert cfg.model.frame_selector.get("density_max_gap_loss_weight", 0.0) == 0.0
        assert cfg.model.frame_selector.get("density_weak_target_loss_weight", 0.0) == 0.0
    assert cfg.model.frame_selector.scout.type == "PCOTMRASCADFDensityFrameScout"
    assert cfg.model.frame_selector.scout_spatial_size == scout_size
    assert cfg.model.frame_selector.scout.in_channels == 3 * scout_size * scout_size
    assert cfg.model.frame_selector.max_gap_guard_count >= 0
    assert cfg.model.frame_selector.boundary_loss_weight == 0.0
    assert cfg.c3_claim_status in (
        "precheck_only",
        "diagnostic_only",
        "backend_control",
        "formal_selector_candidate_locked",
    )
    assert cfg.c3_full_train_claim_unlocked is False
    assert cfg.c3_physical_time_postprocess_enabled is False
    assert cfg.model.frame_selector.density_weights.action <= 0.05
    assert cfg.model.frame_selector.density_weights.uncertainty > cfg.model.frame_selector.density_weights.action
    assert cfg.model.frame_selector.density_weights.change > cfg.model.frame_selector.density_weights.action

    assert cfg.c3_dense_window_size == 768
    assert cfg.window_size == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    for forbidden in [
        "physicalgrid",
        "physical_grid",
        "rf50",
        "bh_sdc",
        "p2head",
        "teacher",
        "raw_prediction_cache",
        "start_head",
        "end_head",
        "coarse_actionness_uncertainty",
    ]:
        assert forbidden not in cfg_text


def test_cadf_densitymesh_32px_smoke_config_is_original_adatad_and_fail_closed():
    cfg = Config.fromfile(SMOKE32)

    _assert_cadf_config(cfg, 32)
    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 2
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.solver.train.batch_size == 1


def test_cadf_densitymesh_32px_full_config_unlocks_only_training_schedule():
    cfg = Config.fromfile(FULL32)

    _assert_cadf_config(cfg, 32)
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.disable_checkpoint is False
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.solver.train.batch_size == 2
    assert cfg.solver.ema is True
    assert cfg.c3_claim_status == "diagnostic_only"
    assert cfg.c3_original_adatad_average_stride_backend_risk == "high"


def test_cadf_densitymesh_alpha0_backend_control_config_is_exact_uniform_like():
    cfg = Config.fromfile(ALPHA0_32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "backend_control"
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.disable_checkpoint is False
    assert cfg.workflow.val_start_epoch <= 2
    assert cfg.workflow.val_eval_interval == 1
    assert cfg.model.frame_selector.density_alpha == 0.0
    assert "density_alpha_schedule" not in cfg.model.frame_selector
    assert cfg.model.frame_selector.density_entropy_loss_weight == 0.0
    assert cfg.model.frame_selector.density_repulsion_loss_weight == 0.0
    assert cfg.model.frame_selector.max_gap_guard_count == 0


def test_cadf_densitymesh_alpha0_stability_probe_disables_unstable_training_paths():
    cfg = Config.fromfile(ALPHA0_STABILITY_32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "diagnostic_only"
    assert cfg.workflow.end_epoch == 4
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.solver.amp is False
    assert cfg.solver.fp16_compress is False
    assert cfg.solver.ema is False
    assert cfg.optimizer.lr == pytest.approx(5e-5)
    assert cfg.optimizer.backbone.custom[0].lr == pytest.approx(1e-4)
    assert cfg.model.frame_selector.density_alpha == 0.0
    assert cfg.model.frame_selector.actionness_loss_weight == 0.0
    assert cfg.model.frame_selector.st_scale == 0.0
    assert cfg.model.frame_selector.st_local_radius == 0
    assert cfg.model.frame_selector.density_entropy_loss_weight == 0.0
    assert cfg.model.frame_selector.density_repulsion_loss_weight == 0.0
    assert cfg.model.frame_selector.max_gap_guard_count == 0


def test_cadf_densitymesh_alpha0_single_factor_probe_matrix_is_isolated():
    actionness = Config.fromfile(ALPHA0_ONLY_ACTIONNESS_32)
    only_st = Config.fromfile(ALPHA0_ONLY_ST_32)
    only_amp = Config.fromfile(ALPHA0_ONLY_AMP_32)

    for cfg in (actionness, only_st, only_amp):
        _assert_cadf_config(cfg, 32)
        assert cfg.c3_claim_status == "diagnostic_only"
        assert cfg.workflow.end_epoch == 4
        assert cfg.workflow.disable_checkpoint is True
        assert cfg.workflow.val_eval_interval == -1
        assert cfg.optimizer.lr == pytest.approx(5e-5)
        assert cfg.optimizer.backbone.custom[0].lr == pytest.approx(1e-4)
        assert cfg.model.frame_selector.density_alpha == 0.0
        assert cfg.model.frame_selector.density_entropy_loss_weight == 0.0
        assert cfg.model.frame_selector.density_repulsion_loss_weight == 0.0
        assert cfg.model.frame_selector.max_gap_guard_count == 0

    assert actionness.solver.amp is False
    assert actionness.solver.fp16_compress is False
    assert actionness.solver.ema is False
    assert actionness.model.frame_selector.actionness_loss_weight == pytest.approx(0.05)
    assert actionness.model.frame_selector.st_scale == 0.0
    assert actionness.model.frame_selector.st_local_radius == 0

    assert only_st.solver.amp is False
    assert only_st.solver.fp16_compress is False
    assert only_st.solver.ema is False
    assert only_st.model.frame_selector.actionness_loss_weight == 0.0
    assert only_st.model.frame_selector.st_scale == pytest.approx(0.5)
    assert only_st.model.frame_selector.st_local_radius == 2

    assert only_amp.solver.amp is True
    assert only_amp.solver.fp16_compress is True
    assert only_amp.solver.ema is True
    assert only_amp.model.frame_selector.actionness_loss_weight == 0.0
    assert only_amp.model.frame_selector.st_scale == 0.0
    assert only_amp.model.frame_selector.st_local_radius == 0


def test_cadf_densitymesh_alpha0_st_actionness_combo_gate_is_fp32_no_amp_no_ema():
    cfg = Config.fromfile(ALPHA0_ST_ACTIONNESS_32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "diagnostic_only"
    assert cfg.c3_alpha0_combo_gate == "st_soft_path_plus_actionness_fp32_no_amp_no_ema"
    assert cfg.workflow.end_epoch == 4
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.solver.amp is False
    assert cfg.solver.fp16_compress is False
    assert cfg.solver.ema is False
    assert cfg.model.frame_selector.density_alpha == 0.0
    assert "density_alpha_schedule" not in cfg.model.frame_selector
    assert cfg.model.frame_selector.density_entropy_loss_weight == 0.0
    assert cfg.model.frame_selector.density_repulsion_loss_weight == 0.0
    assert cfg.model.frame_selector.max_gap_guard_count == 0
    assert cfg.model.frame_selector.st_scale == pytest.approx(0.5)
    assert cfg.model.frame_selector.st_local_radius == 2
    assert cfg.model.frame_selector.actionness_loss_weight == pytest.approx(0.05)


def test_shared_precheck_validator_rejects_combo_gate_with_amp_or_ema_enabled(tmp_path):
    cfg = Config.fromfile(ALPHA0_ST_ACTIONNESS_32)
    cfg.solver.amp = True
    cfg.solver.fp16_compress = True
    cfg.solver.ema = True
    bad_config = tmp_path / "bad_cadf_combo_amp_ema.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="combo gate.*AMP/fp16/EMA"):
        validate_config(bad_config)


def test_cadf_densitymesh_staged_diagnostic_config_has_early_validation_and_schedule():
    cfg = Config.fromfile(STAGED32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "diagnostic_only"
    assert cfg.c3_alpha_schedule_recoverable is False
    assert cfg.c3_long_train_resume_claim_locked is True
    assert cfg.workflow.val_start_epoch <= 5
    assert cfg.workflow.val_eval_interval == 1
    schedule = cfg.model.frame_selector.density_alpha_schedule
    assert schedule.train_start_alpha == 0.0
    assert schedule.train_target_alpha == pytest.approx(cfg.model.frame_selector.density_alpha)
    assert schedule.warmup_iters > 0
    assert schedule.test_alpha == "target"


def test_cadf_densitymesh_full_train_candidate_remains_fail_closed():
    cfg = Config.fromfile(CANDIDATE32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "diagnostic_only"
    assert cfg.c3_full_train_claim_unlocked is False
    assert cfg.c3_selected_index_aware_postprocess_contract == "metadata_only_default_off"
    assert cfg.c3_alpha_schedule_recoverable is False
    assert cfg.c3_long_train_resume_claim_locked is True


def test_cadf_densitymesh_formal_selector_candidate_is_locked_and_matches_combo_gate():
    cfg = Config.fromfile(FORMAL32)
    combo = Config.fromfile(ALPHA0_ST_ACTIONNESS_32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "formal_selector_candidate_locked"
    assert cfg.c3_formal_selector_candidate is True
    assert cfg.launch_locked_until_combo_pass is True
    assert cfg.c3_combo_gate_required_config == ALPHA0_ST_ACTIONNESS_32.relative_to(ROOT).as_posix()
    assert cfg.c3_combo_gate_required_status == "old_nan_window_pass_pending"
    assert cfg.c3_combo_old_window_pass_evidence == "PENDING"
    assert cfg.c3_combo_gate_remote_child == "1118197.376"
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.disable_checkpoint is False
    assert cfg.solver.amp is False
    assert cfg.solver.fp16_compress is False
    assert cfg.solver.ema is False
    assert cfg.model.frame_selector.density_alpha > 0.0
    assert "density_alpha_schedule" not in cfg.model.frame_selector
    assert cfg.model.frame_selector.st_scale == pytest.approx(combo.model.frame_selector.st_scale)
    assert cfg.model.frame_selector.st_local_radius == combo.model.frame_selector.st_local_radius
    assert cfg.model.frame_selector.actionness_loss_weight == pytest.approx(
        combo.model.frame_selector.actionness_loss_weight
    )
    assert cfg.model.frame_selector.density_entropy_loss_weight == pytest.approx(0.005)
    assert cfg.model.frame_selector.max_gap_guard_count == 12


def test_cadf_density_loss_v2_diagnostic_config_is_default_closed_for_claims():
    cfg = Config.fromfile(DENSITY_LOSS_V2_DIAG32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "diagnostic_only"
    assert cfg.c3_density_loss_v2_diagnostic_only is True
    assert cfg.c3_density_loss_v2_claim_unlocked is False
    assert cfg.c3_full_train_claim_unlocked is False
    assert cfg.model.frame_selector.density_window_mass_loss_weight > 0.0
    assert cfg.model.frame_selector.density_max_gap_loss_weight > 0.0
    assert cfg.model.frame_selector.density_blue_noise_loss_weight > 0.0
    assert cfg.model.frame_selector.density_weak_target_loss_weight == 0.0
    assert cfg.model.frame_selector.density_repulsion_loss_weight == 0.0
    assert cfg.model.frame_selector.physical_time_postprocess_enabled is False
    validate_config(DENSITY_LOSS_V2_DIAG32)


def test_shared_precheck_validator_rejects_formal_selector_candidate_with_density_loss_v2(tmp_path):
    cfg = Config.fromfile(FORMAL32)
    cfg.model.frame_selector.density_window_mass_loss_weight = 0.01
    cfg.model.frame_selector.density_max_gap_loss_weight = 0.01
    cfg.model.frame_selector.density_blue_noise_loss_weight = 0.01
    cfg.c3_density_loss_v2_diagnostic_only = True
    cfg.c3_density_loss_v2_claim_unlocked = False
    bad_config = tmp_path / "bad_cadf_formal_density_loss_v2.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="Density-Loss V2.*formal"):
        validate_config(bad_config)


def test_cadf_densitymesh_formal_selector_launcher_is_fail_closed_until_combo_pass():
    launcher = FORMAL_LAUNCHER.read_text(encoding="utf-8")

    assert "c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py" in launcher
    assert "CADF_COMBO_OLD_WINDOW_PASS" in launcher
    assert "CONFIRMED" in launcher
    assert "CADF_COMBO_OLD_WINDOW_EVIDENCE" in launcher
    assert "exit 2" in launcher
    assert "tools/train.py" in launcher
    assert "alpha0_single_factor" not in launcher


def test_cadf_densitymesh_64px_configs_only_change_scout_resolution():
    smoke = Config.fromfile(SMOKE64)
    full = Config.fromfile(FULL64)

    _assert_cadf_config(smoke, 64)
    _assert_cadf_config(full, 64)
    assert smoke.workflow.max_train_iters == 2
    assert full.workflow.end_epoch == 60


@pytest.mark.parametrize(
    "config_path",
    [
        SMOKE32,
        FULL32,
        ALPHA0_32,
        ALPHA0_STABILITY_32,
        ALPHA0_ONLY_ACTIONNESS_32,
        ALPHA0_ONLY_ST_32,
        ALPHA0_ST_ACTIONNESS_32,
        ALPHA0_ONLY_AMP_32,
        STAGED32,
        CANDIDATE32,
        FORMAL32,
        FORMAL_FASTFIX32,
        DENSITY_LOSS_V2_DIAG32,
        SMOKE64,
        FULL64,
    ],
)
def test_cadf_densitymesh_configs_pass_shared_precheck_validator(config_path):
    validate_config(config_path)


def test_cadf_formal_fastfix_keeps_method_semantics_but_enables_speed_guards():
    cfg = Config.fromfile(FORMAL_FASTFIX32)

    _assert_cadf_config(cfg, 32)
    assert cfg.c3_claim_status == "formal_selector_candidate_locked"
    assert cfg.c3_speed_fix == "selector_cpu_once_repair_diag_off_amp_withcp_probe"
    assert cfg.model.frame_selector.fast_cpu_selection is True
    assert cfg.model.frame_selector.emit_selection_diagnostics is False
    assert cfg.model.frame_selector.selection_diagnostics_interval == 0
    assert cfg.model.frame_selector.max_gap_guard_count == 12
    assert cfg.model.frame_selector.st_local_radius == 2
    assert cfg.model.frame_selector.st_scale == 0.5
    assert cfg.model.frame_selector.density_alpha == pytest.approx(0.65)
    assert cfg.solver.train.batch_size == 2
    assert cfg.solver.train.num_workers >= 2
    assert cfg.solver.amp is True
    assert cfg.solver.fp16_compress is True
    validate_config(FORMAL_FASTFIX32)


def test_cadf_densitymesh_rejects_invalid_density_weight_configs():
    base = dict(
        type="PCOTMRASIndirectPreBackboneFrameSelector",
        target_len=4,
        dense_window_size=8,
        selection_unit=1,
        scout_spatial_size=4,
        strategy="cadf_density_mesh_st",
        scout=dict(
            type="PCOTMRASCADFDensityFrameScout",
            in_channels=48,
            hidden_channels=8,
            num_layers=1,
            with_boundary_head=False,
        ),
    )

    with pytest.raises(ValueError, match="density_weights"):
        build_selector(dict(base, density_weights=dict(action=-0.1, utility=1.0)))
    with pytest.raises(ValueError, match="density_weights"):
        build_selector(dict(base, density_weights=dict(action=0.0, utility=0.0, boundary=0.0)))


def test_cadf_densitymesh_boundary_density_weight_requires_boundary_scout_head():
    with pytest.raises(ValueError, match="boundary"):
        build_selector(
            dict(
                type="PCOTMRASIndirectPreBackboneFrameSelector",
                target_len=4,
                dense_window_size=8,
                selection_unit=1,
                scout_spatial_size=4,
                strategy="cadf_density_mesh_st",
                density_weights=dict(action=0.5, utility=0.4, boundary=0.1),
                scout=dict(
                    type="PCOTMRASCADFDensityFrameScout",
                    in_channels=48,
                    hidden_channels=8,
                    num_layers=1,
                    with_boundary_head=False,
                ),
            )
        )


def test_cadf_densitymesh_rejects_unknown_nonfinite_and_nonnumeric_density_weight_keys():
    base = dict(
        type="PCOTMRASIndirectPreBackboneFrameSelector",
        target_len=4,
        dense_window_size=8,
        selection_unit=1,
        scout_spatial_size=4,
        strategy="cadf_density_mesh_st",
        scout=dict(
            type="PCOTMRASCADFDensityFrameScout",
            in_channels=48,
            hidden_channels=8,
            num_layers=1,
            with_boundary_head=False,
        ),
    )

    invalid_weights = [
        dict(action=0.5, utility=0.4, saliency=0.1),
        dict(action=math.nan, utility=1.0),
        dict(action=math.inf, utility=1.0),
        dict(action="0.5", utility=0.5),
    ]
    for density_weights in invalid_weights:
        with pytest.raises(ValueError, match="density_weights"):
            build_selector(dict(base, density_weights=density_weights))


def test_cadf_densitymesh_allows_boundary_density_weight_with_boundary_scout_head():
    selector = build_selector(
        dict(
            type="PCOTMRASIndirectPreBackboneFrameSelector",
            target_len=4,
            dense_window_size=8,
            selection_unit=1,
            scout_spatial_size=4,
            strategy="cadf_density_mesh_st",
            density_weights=dict(action=0.5, utility=0.4, boundary=0.1),
            scout=dict(
                type="PCOTMRASCADFDensityFrameScout",
                in_channels=48,
                hidden_channels=8,
                num_layers=1,
                with_boundary_head=True,
            ),
        )
    )

    assert selector.scout.boundary_head is not None


def test_shared_precheck_validator_rejects_boundary_density_without_boundary_head(tmp_path):
    cfg = Config.fromfile(SMOKE32)
    cfg.model.frame_selector.density_weights = dict(action=0.5, utility=0.4, boundary=0.1)
    cfg.model.frame_selector.scout.with_boundary_head = False
    bad_config = tmp_path / "bad_cadf_boundary_weight.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="boundary.*with_boundary_head"):
        validate_config(bad_config)


def test_shared_precheck_validator_rejects_unlocked_full_train_claim(tmp_path):
    cfg = Config.fromfile(FULL32)
    cfg.c3_full_train_claim_unlocked = True
    cfg.c3_claim_status = "paper_claim"
    bad_config = tmp_path / "bad_cadf_unlocked_claim.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="full-train.*claim"):
        validate_config(bad_config)


def test_shared_precheck_validator_rejects_physical_time_postprocess_claim_without_diagnostic_lock(tmp_path):
    cfg = Config.fromfile(SMOKE32)
    cfg.c3_physical_time_postprocess_enabled = True
    cfg.c3_physical_time_postprocess_mode = "official_eval"
    bad_config = tmp_path / "bad_cadf_physical_time.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="physical-time.*diagnostic"):
        validate_config(bad_config)


@pytest.mark.parametrize("route_token", ["divergent", "pvr_qc", "pvr-qc", "pvrqc", "pqr", "pqr_ranking", "bvr"])
def test_shared_precheck_validator_rejects_cadf_route_mix_tokens(tmp_path, route_token):
    cfg = Config.fromfile(SMOKE32)
    cfg.c3_route_mixed_note = f"forbidden {route_token} route mix"
    bad_config = tmp_path / f"bad_cadf_{route_token.replace('-', '_')}.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="Forbidden old-route tokens"):
        validate_config(bad_config)


def test_shared_precheck_validator_requires_alpha_schedule_to_be_marked_nonresumable(tmp_path):
    cfg = Config.fromfile(STAGED32)
    cfg.c3_alpha_schedule_recoverable = True
    bad_config = tmp_path / "bad_cadf_resumable_schedule.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="alpha schedule.*not checkpoint-resumable"):
        validate_config(bad_config)


def test_shared_precheck_validator_rejects_formal_selector_candidate_without_launch_lock(tmp_path):
    cfg = Config.fromfile(FORMAL32)
    cfg.launch_locked_until_combo_pass = False
    cfg.c3_combo_old_window_pass_evidence = "logs/combo_gate_passed.txt"
    bad_config = tmp_path / "bad_cadf_formal_unlocked.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="formal selector.*locked"):
        validate_config(bad_config)


def test_shared_precheck_validator_rejects_formal_selector_candidate_with_amp_or_ema(tmp_path):
    cfg = Config.fromfile(FORMAL32)
    cfg.solver.amp = True
    cfg.solver.fp16_compress = True
    cfg.solver.ema = True
    bad_config = tmp_path / "bad_cadf_formal_amp_ema.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="formal selector.*AMP/fp16/EMA"):
        validate_config(bad_config)


def test_shared_precheck_validator_rejects_formal_selector_candidate_alpha0_backend(tmp_path):
    cfg = Config.fromfile(FORMAL32)
    cfg.model.frame_selector.density_alpha = 0.0
    bad_config = tmp_path / "bad_cadf_formal_alpha0.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="formal selector.*density_alpha"):
        validate_config(bad_config)
