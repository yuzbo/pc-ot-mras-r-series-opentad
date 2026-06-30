_base_ = ["./input_bvr_twb_dynamic_adapter_irregular_headv3.py"]

annotation_path = "data/thumos-14/annotations/thumos_14_anno.json"
class_map = "data/thumos-14/annotations/category_idx.txt"
train_data_path = "data/thumos-14/raw_data/train"
test_data_path = "data/thumos-14/raw_data/test"

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

work_dir = "exps/thumos/adatad/input_bvr_twb_dynamic_adapter_irregular_headv3_n16r4"
