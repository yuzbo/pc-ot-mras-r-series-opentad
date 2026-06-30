_base_ = ["./input_abr_active_bracket_refinement_adapter_irregular_headv3.py"]

ABR_SHORTDIAG_ROUTE_LABEL = "DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3"

abr_route = dict(
    route_label=ABR_SHORTDIAG_ROUTE_LABEL,
    method="abr_active_bracket_refinement",
    stage="SHORT_DIAGNOSTIC_ONLY",
    diagnostic_only=True,
    full_train_unlocked=False,
    metric_claim=False,
    sparse_compute_claim=False,
    runtime_claim=False,
    deploy_claim=False,
    paper_claim=False,
    claim_status="short_diagnostic_only_no_metric_no_runtime_no_deploy_no_paper_claim",
)

shortdiag_gate = dict(
    route_label=ABR_SHORTDIAG_ROUTE_LABEL,
    method="abr_active_bracket_refinement",
    diagnostic_only=True,
    allowed_next_action="ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY",
    full_train_unlocked=False,
    metric_claim=False,
    sparse_compute_claim=False,
    runtime_claim=False,
    deploy_claim=False,
    paper_claim=False,
    tools_test_py_allowed=False,
    evaluation_allowed=False,
    checkpoint_allowed=False,
    result_detection_allowed=False,
    max_epochs=1,
    require_no_eval=True,
    require_no_checkpoint=True,
    require_finite_train_loss=True,
    n16r4_child_gpu_context_only=True,
)

workflow = dict(
    end_epoch=1,
    checkpoint_interval=999999,
    disable_checkpoint=True,
    val_start_epoch=999999,
    val_loss_interval=-1,
    val_eval_interval=-1,
    logging_interval=10,
    runtime_debug_interval=-1,
)

inference = dict(
    load_from_raw_predictions=False,
    save_raw_prediction=False,
)

post_processing = dict(
    save_dict=False,
)

work_dir = "exps/thumos/adatad/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag"
