_base_ = ["./pc_ot_mras_prebackbone_c3_hybrid_reader_candidate_n16r4.py"]


variant_id = "BH-SDC-BoundaryHazard-SparseDense-LocalPrecheck"
route_id = "bh_sdc_boundary_hazard_sparse_dense"
route_label = "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"
stage_id = "bh_sdc_boundary_hazard_sparse_dense_local_precheck"
reader_type = "BoundaryHazardTemporalScout"
reader_family = "BH-SDC"

dense_window_size = 768
min_budget = 256
target_budget = 384
max_budget = 448
budget_step = 32

experiment_scope = dict(
    _delete_=True,
    variant_id=variant_id,
    route=route_id,
    route_label=route_label,
    route_family="BH_SDC_DIVERGENT_INNOVATION_ROUTE",
    combo_status="NO_COMBO_ROUTE_APPROVED",
    stage=stage_id,
    detector_stack="original_adatad_actionformer_adapter",
    backend="OriginalAdaTAD",
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame",
    route_summary=(
        "BH-SDC uses a deploy-visible temporal CNN scout to estimate actionness, "
        "start/end boundary hazard, difficulty, uncertainty, and redundancy; a "
        "dynamic budget controller allocates per-video frame budgets; sparse raw "
        "frames are processed by the backbone; a sparse-to-dense completion bridge "
        "restores the detector's 768-step physical temporal axis before projection/head."
    ),
    budget_protocol="dynamic_min256_target384_max448_over_dense768_boundary_hazard_sparse_to_dense",
    changed_surface=(
        "input_sampling",
        "dynamic_budget_policy",
        "backbone_input_length",
        "sparse_to_dense_token_completion",
        "temporal_metadata",
    ),
    scout="BoundaryHazardTemporalScout",
    policy="BoundaryHazardAcquisitionPolicy",
    completion_bridge="PCOTMRASBoundaryHazardSparseToDenseBridge",
    attribution_boundary=(
        "This is BH-SDC divergent innovation, not C3/C3-Pro optimization. "
        "BH-SDC evidence must not be used to explain C3 failures, and C3 diagnostics "
        "must not be used as BH-SDC validation without an explicit COMBO_ROUTE_APPROVED gate."
    ),
    local_synthetic_gate_only=True,
    pro_decision="GO_WITH_CONSTRAINTS_IMPLEMENT_BH_SDC_LOCAL_PROTOTYPE_ONLY",
    pro_code_or_launch_approval=False,
    uses_p2=False,
    uses_offline_ledger=False,
    uses_teacher=False,
    uses_test_gt=False,
    uses_oracle=False,
    uses_raw_prediction_cache=False,
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=False,
    metric_claim_allowed=False,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

bh_sdc_gate = dict(
    _delete_=True,
    route=route_id,
    route_label=route_label,
    stage=stage_id,
    default_off=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_remote_sync=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_train_validation_map=False,
    allow_long_training=False,
    allow_precheck_only=True,
    allow_dataset_access=False,
    allow_pretrained_initialization=False,
    allow_checkpoint_write=False,
    allow_checkpoint_load=False,
    allow_resume=False,
    allow_raw_prediction_cache=False,
    allow_metric_claim=False,
    allow_paper_claim=False,
    allowed_entrypoints=(),
    forbidden_true_keys=(
        "allow_remote_sync",
        "allow_slurm",
        "allow_gpu",
        "allow_tools_train",
        "allow_tools_test",
        "allow_detector_map",
        "allow_long_training",
        "allow_checkpoint_load",
        "allow_resume",
        "allow_raw_prediction_cache",
        "metric_claim_allowed",
        "paper_claim_allowed",
        "runtime_flops_claim_allowed",
        "deploy_claim_allowed",
    ),
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    _delete_=True,
    disabled_by_bh_sdc=True,
    route=route_id,
    stage=stage_id,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_remote_sync=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_long_training=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        _delete_=True,
        type="PCOTMRASBoundaryHazardSparseDenseFrameSelector",
        input_channels=3,
        dense_window_size=dense_window_size,
        min_budget=min_budget,
        target_budget=target_budget,
        max_budget=max_budget,
        budget_step=budget_step,
        scout_hidden_dim=128,
        scout_num_layers=3,
        scout_kernel_size=5,
        scout_dropout=0.05,
        coverage_ratio=0.20,
        boundary_ratio=0.45,
        difficulty_ratio=0.20,
        uncertainty_ratio=0.15,
        max_dense_gap=64,
        aux_hazard_loss_weight=0.05,
        aux_budget_entropy_loss_weight=0.001,
    ),
    token_compressor=dict(
        type="PCOTMRASBoundaryHazardSparseToDenseBridge",
        dense_window_size=dense_window_size,
        target_len=dense_window_size,
        interpolation_temperature=6.0,
        refine_channels=384,
        refine_layers=2,
        smoothness_loss_weight=0.001,
    ),
    backbone=dict(
        backbone=dict(total_frames=max_budget),
        custom=dict(pretrain=None),
    ),
    projection=dict(max_seq_len=dense_window_size),
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=60,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/bh_sdc_boundary_hazard_sparse_dense_local_precheck"
