import os

_base_ = ["./input_random_fixed_50pct_adapter_quality_rescore_detached.py"]

thumos_root = os.environ.get("THUMOS_ROOT", os.path.expanduser("~/run/yuzibo/thumos14"))
annotation_path = os.path.join(thumos_root, "annotations", "thumos_14_anno.json")
class_map = os.path.join(thumos_root, "annotations", "category_idx.txt")
train_data_path = os.path.join(thumos_root, "train")
test_data_path = os.path.join(thumos_root, "test")

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

evaluation = dict(
    ground_truth_filename=annotation_path,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_quality_rescore_detached_n16r4"
