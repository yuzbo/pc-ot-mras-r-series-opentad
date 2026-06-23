_base_ = ["ctf_bdi_pc_ot_mras_r18_aux_diag_candidate.py"]

# Local-only scaffold for the Pro-approved P2/NIIQ quality-ranking route.
# This config enables the optional P2 quality calibration branch so code review
# and local contract tests can inspect it. It is not a launch permission.

r18_pc_ot_mras_aux_diag_gate = None

p2_quality_rank_calibrator_v0_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="P2_NIIQ_QualityRank_Calibrator_v0_local_scaffold",
    reviewed_predecessor="GPT_5_5_Pro_quality_route_decision_20260623",
    default_off=True,
    explicit_config_opt_in=True,
    local_synthetic_gate_only=True,
    allow_detector_training=False,
    allow_remote_sync=False,
    allow_precheck_only=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_detector_map=False,
    requires_launch_gate=True,
    launch_gate_passed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "disabled_path_invariance",
        "quality_calibration_no_gt_eval_guard",
        "synthetic_quality_loss_smoke_when_torch_available",
    ),
    forbidden_checks=(
        "remote_sync",
        "remote_precheck",
        "slurm_or_gpu",
        "tools_train",
        "tools_test_or_map",
        "runtime_or_flops_claim",
        "deploy_claim",
        "metric_claim",
        "paper_claim",
        "validation_gt_score_formula_tuning",
        "raw_prediction_cache",
    ),
)

model = dict(
    rpn_head=dict(
        area_head=dict(
            quality_calibration=dict(
                enable=True,
                hidden_dim=256,
                quality_loss_weight=0.50,
                boundary_loss_weight=0.25,
                rank_loss_weight=0.05,
                boundary_tau=1.0,
                rank_positive_iou=0.70,
                rank_negative_iou=0.30,
                rank_margin=0.25,
                rank_sample_size=64,
                score_beta=1.0,
                boundary_gamma=1.0,
                base_delta=1.0,
                score_eps=1e-6,
            ),
        ),
    ),
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_local"
