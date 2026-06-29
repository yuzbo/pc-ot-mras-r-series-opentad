_base_ = ["./input_random_fixed_50pct_adapter.py"]

# ActionFormer-only calibration: emphasize temporal boundary regression while
# preserving the Adapter backbone, random-fixed sampler, and baseline inference.
model = dict(
    rpn_head=dict(
        loss_weight=1.5,
    ),
)

workflow = dict(
    checkpoint_interval=10,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_regloss15"
