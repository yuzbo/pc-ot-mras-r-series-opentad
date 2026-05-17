_base_ = ["./input_random_fixed_50pct_adapter_quality_rescore_detached.py"]

# Reduced-negative assigned-IoU ablation: preserve the original positive
# assignment semantics while reducing dense negative quality BCE pressure.
# First evaluation disables quality-score fusion; alpha sweeps should only be
# run after the checkpoint recovers the random-fixed Adapter baseline band.
model = dict(
    rpn_head=dict(
        quality_head_cfg=dict(
            enabled=True,
            target_mode="assigned_iou",
            positive_weight=1.0,
            negative_weight=0.25,
            loss_normalizer="weighted",
            loss_weight=0.10,
            score_alpha=0.0,
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0"
