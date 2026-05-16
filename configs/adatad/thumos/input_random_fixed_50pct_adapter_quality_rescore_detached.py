_base_ = ["./input_random_fixed_50pct_adapter.py"]

# Baseline-preserving quality reranking: keep Adapter + ActionFormer and the
# random-fixed train/val/test input contract, and add only a detached ranking
# signal trained from train-set decoded-proposal IoU.
model = dict(
    rpn_head=dict(
        quality_head_cfg=dict(
            enabled=True,
            kernel_size=3,
            # sigmoid(4.595) ~= 0.99, so early quality reranking is nearly
            # neutral instead of multiplying all class scores by 0.5**alpha.
            bias_init=4.59511985013459,
            weight_init=0.0,
            loss_weight=0.10,
            score_alpha=0.25,
        ),
    ),
)

workflow = dict(
    checkpoint_interval=10,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_quality_rescore_detached"
