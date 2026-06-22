_base_ = ["ctf_bdi_pc_ot_mras_r18_post_train_eval_candidate.py"]

# Local diagnostic eval only: reduce video-reader memory pressure without
# changing the reviewed R18 post-train eval model, gate, checkpoint, or metrics.
solver = dict(test=dict(batch_size=1, num_workers=0))
