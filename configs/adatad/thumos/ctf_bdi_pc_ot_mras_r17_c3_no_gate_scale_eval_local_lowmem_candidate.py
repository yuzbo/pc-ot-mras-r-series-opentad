_base_ = ["ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py"]

# C3 eval-time ablation only: keep R17 checkpoint/eval protocol and local
# low-memory loader, but recompute selected detector tokens from allocation
# without per-slot gate scaling.
model = dict(neck=dict(no_gate_scale=True))

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r17_c3_no_gate_scale_eval_local_lowmem_candidate"
