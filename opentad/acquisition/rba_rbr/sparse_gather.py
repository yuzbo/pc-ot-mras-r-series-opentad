import numpy as np

from .types import sorted_unique_positions


def sparse_gather(dense_inputs, selected_positions, temporal_dim=0):
    dense = np.asarray(dense_inputs)
    temporal_dim = int(temporal_dim)
    dense_T = int(dense.shape[temporal_dim])
    positions = sorted_unique_positions(selected_positions, dense_T)
    indexer = [slice(None)] * dense.ndim
    indexer[temporal_dim] = np.asarray(positions, dtype=np.int64)
    selected = dense[tuple(indexer)]
    evidence = {
        "dense_temporal_len": dense_T,
        "selected_temporal_len": int(selected.shape[temporal_dim]),
        "selected_inputs_is_gathered": True,
        "dense_raw_backbone_handoff": False,
        "temporal_decode_uses_original_time": True,
        "status": "local_gather_smoke_only",
        "sparse_compute_claim": False,
    }
    return selected, evidence
