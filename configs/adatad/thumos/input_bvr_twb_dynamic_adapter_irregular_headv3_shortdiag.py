_base_ = ["./input_bvr_twb_dynamic_adapter_irregular_headv3.py"]

route_label = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
diagnostic_stage = "short_diagnostic_smoke_only"
diagnostic_only = True
shortdiag_expected_base_commit = "478325af8da10646f747f955a54378d53fffd3ef"
full_train_unlocked = False
metric_claim = False
sparse_compute_claim = False

required_pretrain_path = "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"

model = dict(
    backbone=dict(
        custom=dict(
            pretrain=required_pretrain_path,
        ),
    ),
    rpn_head=dict(
        max_reg_log_distance=6.0,
        regression_head_fp32=True,
        regression_loss_fp32=True,
        filter_invalid_regression_samples=False,
        min_regression_segment_length=1e-6,
    ),
)

solver = dict(
    amp=False,
    fp16_compress=False,
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=999,
    disable_checkpoint=True,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
    runtime_debug_interval=1,
)

work_dir = "exps/thumos/adatad/diagnostic_only/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag"
