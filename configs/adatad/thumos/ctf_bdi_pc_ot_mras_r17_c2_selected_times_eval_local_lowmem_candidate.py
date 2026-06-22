_base_ = ["ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py"]

# C2 eval-time ablation only: keep R17 checkpoint/eval protocol and local
# low-memory loader, but write bridge/head temporal metadata from selected_times.
model = dict(neck=dict(metadata_position_source="selected_times"))

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r17_c2_selected_times_eval_local_lowmem_candidate"
