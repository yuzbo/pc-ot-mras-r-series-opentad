_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x.py"]

abr_route = dict(
    route_label="DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3",
    method="abr_active_bracket_refinement",
    claim_status="precheck_only_fail_closed_candidate",
)

window_size = 384
dense_window_size = 768
scale_factor = 1

abr_loader = dict(
    method="abr_active_bracket_refinement",
    abr_config=dict(
        k0=96,
        k1_cap=160,
        k2_cap=64,
        max_total_k=384,
        max_gap=24,
        target_frame_num=384,
        round2_enabled=True,
        deadline_ms=80.0,
        action_threshold=0.60,
        background_threshold=0.35,
        uncertainty_band=0.16,
        derivative_threshold=0.24,
        resolve_width=3,
        round2_min_width=8,
        outside_witness_offset=2,
        route_label="DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3",
        bracket_policy="deploy_visible_multiscale_graydiff_bracket_v2",
        first_round_max_temporal_coverage_fraction=0.70,
        first_round_max_bracket_width_fraction=0.30,
        event_train_peak_quantile=0.88,
        event_train_gradient_quantile=0.90,
        event_train_max_gap_fraction=0.16,
        event_train_context_fraction=0.04,
        event_train_min_anchors=2,
        allow_diagnostic_fallback_scout=True,
        fallback_stage="PRECHECK_ONLY",
    ),
)

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="abr_active_bracket_refinement",
                method_base="random_trunc",
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                source_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
                abr_allow_gt_after_selection=True,
                abr_config=abr_loader["abr_config"],
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="abr_active_bracket_refinement",
                method_base="sliding_window",
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                scale_factor=scale_factor,
                abr_allow_gt_after_selection=False,
                abr_config=abr_loader["abr_config"],
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="abr_active_bracket_refinement",
                method_base="sliding_window",
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                scale_factor=scale_factor,
                abr_allow_gt_after_selection=False,
                abr_config=abr_loader["abr_config"],
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

work_dir = "exps/thumos/adatad/input_abr_active_bracket_refinement_adapter_irregular_headv3"
