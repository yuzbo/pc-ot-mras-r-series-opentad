_base_ = ["./input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py"]

route_label = "DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3"
short_diagnostic_only = True
full_train_unlocked = False
no_metric_claim = True
no_runtime_claim = True
no_deploy_claim = True
no_paper_claim = True

workflow = dict(
    logging_interval=20,
    checkpoint_interval=999,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=2,
    disable_checkpoint=True,
    runtime_debug_interval=20,
)

work_dir = "exps/thumos/adatad/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag_only"
