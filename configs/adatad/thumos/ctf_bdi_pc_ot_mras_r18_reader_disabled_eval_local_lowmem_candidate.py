_base_ = ["ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py"]

# Local diagnostic eval only: reduce video-reader memory pressure while keeping
# the R18 reader-disabled exact-uniform override and eval gate unchanged.
solver = dict(test=dict(batch_size=1, num_workers=0))
