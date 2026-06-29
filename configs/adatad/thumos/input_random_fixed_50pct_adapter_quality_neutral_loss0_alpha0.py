_base_ = ["./input_random_fixed_50pct_adapter_quality_rescore_detached.py"]

# Structural diagnostic: keep the detached quality module in the graph but
# remove both quality supervision and inference fusion. This tests whether the
# branch integration, optimizer/DDP/EMA path, or static graph handling changes
# the Adapter + ActionFormer baseline even without a quality objective.
model = dict(
    rpn_head=dict(
        quality_head_cfg=dict(
            enabled=True,
            keep_loss_graph_when_weight_zero=True,
            loss_weight=0.0,
            score_alpha=0.0,
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0"
