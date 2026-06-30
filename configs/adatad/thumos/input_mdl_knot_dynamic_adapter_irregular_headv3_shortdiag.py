import os

_base_ = ["./input_mdl_knot_dynamic_adapter_irregular_headv3.py"]

route_label = "DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"
route_status = "LOCAL_FINAL_CODE_CANDIDATE_SHORT_DIAGNOSTIC_ONLY_GATE_LOCKED_NO_EVAL_NO_CHECKPOINT_NO_METRIC_CLAIMS"

thumos_root = os.environ.get("THUMOS_ROOT", "/data/home/sczc063/run/yuzibo/thumos14")
annotation_path = os.path.join(thumos_root, "annotations", "thumos_14_anno.json")
class_map = os.path.join(thumos_root, "annotations", "category_idx.txt")
train_data_path = os.path.join(thumos_root, "train")
test_data_path = os.path.join(thumos_root, "test")

diagnostic_only = True
full_train_unlocked = False
metric_claim = False
sparse_compute_claim = False

shortdiag_gate = dict(
    route_label=route_label,
    diagnostic_only=True,
    full_train_unlocked=False,
    metric_claim=False,
    sparse_compute_claim=False,
    max_epochs=1,
    workflow=[("train", 1)],
    evaluation_locked=True,
    checkpoint_locked=True,
    tools_test_py_locked=True,
    map_claim_locked=True,
    claim_locked=True,
    no_eval=True,
    no_checkpoint=True,
    no_tools_test_py=True,
    no_map=True,
    no_metric_claim=True,
    no_runtime_claim=True,
    no_deploy_claim=True,
    no_paper_claim=True,
)

mdl_knot_acquisition = dict(
    route_label=route_label,
    method="mdl_knot_dynamic_subsample",
    bridge="fixed_pad",
    deploy_scout_source="raw_frame_motion_scout_with_metadata_fallback",
    scout_stride=8,
    scout_max_frames=96,
    dense_window_size=768,
    window_size=384,
    max_k=384,
    handoff_audit_mode="sampled_raw",
    real_scout_unavailable=False,
    synthetic_fallback_allowed=False,
    diagnostic_only=True,
    full_train_unlocked=False,
    metric_claim=False,
    sparse_compute_claim=False,
    no_val_test_gt_selector=True,
    no_teacher=True,
    no_prediction_cache=True,
    no_dense_raw_backbone_handoff=True,
    no_metric_claims=True,
    no_runtime_claims=True,
    no_deploy_claims=True,
    no_paper_claims=True,
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1000,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
    disable_checkpoint=True,
)

_shortdiag_window_size = 384
_shortdiag_dense_window_size = 768
_shortdiag_scale_factor = 1
_shortdiag_load_train = dict(
    type="LoadFrames",
    num_clips=1,
    method="mdl_knot_dynamic_subsample",
    method_base="random_trunc",
    keep_ratio=0.5,
    remap_gt_to_selected_axis=False,
    target_len=_shortdiag_window_size,
    source_len=_shortdiag_dense_window_size,
    trunc_thresh=0.75,
    crop_ratio=[0.9, 1.0],
    scale_factor=_shortdiag_scale_factor,
    mdl_knot_bridge=mdl_knot_acquisition["bridge"],
    mdl_knot_min_k=4,
    mdl_knot_max_k=mdl_knot_acquisition["max_k"],
    mdl_knot_target_weighted_error=0.02,
    mdl_knot_max_gap=32,
    mdl_knot_deploy_scout_source=mdl_knot_acquisition["deploy_scout_source"],
    mdl_knot_scout_stride=mdl_knot_acquisition["scout_stride"],
    mdl_knot_scout_max_frames=mdl_knot_acquisition["scout_max_frames"],
    mdl_knot_allow_synthetic_fallback=mdl_knot_acquisition["synthetic_fallback_allowed"],
    mdl_knot_no_gt_selector=True,
    mdl_knot_no_teacher=True,
    mdl_knot_no_prediction_cache=True,
    mdl_knot_no_dense_raw_backbone_handoff=True,
    mdl_knot_handoff_audit_mode=mdl_knot_acquisition["handoff_audit_mode"],
)
total_epochs = 1
max_epochs = 1
evaluation = dict(shortdiag_disabled=True)
checkpoint = dict(shortdiag_disabled=True, save_last=False, max_keep_ckpts=0)
solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
)
dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            _shortdiag_load_train,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
)
work_dir = "exps/thumos/adatad/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag"
