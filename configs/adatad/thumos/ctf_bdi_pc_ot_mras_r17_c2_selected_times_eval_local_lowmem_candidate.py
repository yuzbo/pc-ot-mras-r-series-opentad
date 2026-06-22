_base_ = ["ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py"]

# C2 eval-time ablation only: keep R17 checkpoint/eval protocol and local
# low-memory loader, but write bridge/head temporal metadata from selected_times.
# The repair mode keeps C2 diagnostic eval alive when soft selected-time
# centroids cross slot order by sorting slot-aligned bridge tensors by time and
# adding a tiny monotonic jitter only to duplicated metadata positions.
model = dict(neck=dict(metadata_position_source="selected_times", metadata_position_repair="sort_jitter"))

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r17_c2_selected_times_eval_local_lowmem_candidate"
