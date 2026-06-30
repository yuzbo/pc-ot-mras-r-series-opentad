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

workflow = [("train", 1)]
total_epochs = 1
max_epochs = 1
evaluation = dict(shortdiag_disabled=True)
checkpoint = dict(shortdiag_disabled=True, save_last=False, max_keep_ckpts=0)
dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
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
