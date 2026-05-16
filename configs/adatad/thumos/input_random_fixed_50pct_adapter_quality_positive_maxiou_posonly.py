_base_ = ["./input_random_fixed_50pct_adapter_quality_rescore_detached.py"]

# Next-round ablation candidate, not the active run:
# keep Adapter + ActionFormer and random-fixed 50% input unchanged, but train
# the quality head only on positive locations. This avoids the dense negative
# BCE pressure that can collapse quality scores toward zero for most proposals.
model = dict(
    rpn_head=dict(
        quality_head_cfg=dict(
            enabled=True,
            kernel_size=3,
            bias_init=4.59511985013459,
            weight_init=0.0,
            target_mode="positive_max_iou",
            positive_weight=1.0,
            negative_weight=0.0,
            loss_normalizer="positive",
            loss_weight=0.10,
            score_alpha=0.25,
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly"
