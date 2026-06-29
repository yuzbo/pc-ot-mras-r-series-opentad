_base_ = ["./input_random_fixed_50pct_adapter.py"]

model = dict(
    rpn_head=dict(
        assignment_debug=dict(enabled=True),
        assigner=dict(
            type="AnchorFreeSimOTAAssigner",
            cls_weight=1.0,
            iou_weight=3.0,
            center_radius=2.5,
            keep_percent=0.65,
            confuse_weight=1.0,
            topk=9,
            min_k=4,
            filter_shortest_gt=False,
            dynamic_k=dict(
                type="dynamic_k_matching",
                mode="iou_sum",
                min_candidate_iou=0.05,
            ),
        ),
    ),
)

workflow = dict(
    checkpoint_interval=10,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
    disable_checkpoint=False,
    runtime_debug_interval=1,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_simota_mink4_w1"
