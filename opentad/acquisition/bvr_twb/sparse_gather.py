import numpy as np

from .types import sorted_unique_positions


def _slice_along_temporal(inputs, positions, temporal_dim):
    arr = np.asarray(inputs)
    indexer = [slice(None)] * arr.ndim
    indexer[int(temporal_dim)] = np.asarray(positions, dtype=np.int64)
    return arr[tuple(indexer)]


def _fingerprints(arr, positions, temporal_dim):
    dense = np.asarray(arr)
    values = []
    for pos in positions:
        indexer = [slice(None)] * dense.ndim
        indexer[int(temporal_dim)] = int(pos)
        values.append(float(np.asarray(dense[tuple(indexer)], dtype=np.float64).sum()))
    return values


def sparse_gather(
    dense_inputs,
    selected_positions,
    temporal_dim=0,
    detector_forward_temporal_len=None,
    detector_forward_exists=False,
):
    dense = np.asarray(dense_inputs)
    temporal_dim = int(temporal_dim)
    dense_T = int(dense.shape[temporal_dim])
    positions = sorted_unique_positions(selected_positions, dense_T)
    selected = _slice_along_temporal(dense, positions, temporal_dim)
    selected_T = int(selected.shape[temporal_dim])
    expected = _fingerprints(dense, positions, temporal_dim)
    actual = _fingerprints(selected, list(range(selected_T)), temporal_dim)
    fingerprint_checked = bool(len(expected) == len(actual) and np.allclose(expected, actual))
    if detector_forward_exists:
        forward_len = int(detector_forward_temporal_len)
        status = "detector_forward_sparse_audited"
    else:
        forward_len = int(selected_T)
        status = "local_gather_smoke_only"
    evidence = {
        "dense_temporal_len": dense_T,
        "selected_temporal_len": selected_T,
        "detector_forward_temporal_len": forward_len,
        "selected_inputs_is_gathered": True,
        "dense_raw_backbone_handoff": False,
        "fingerprint_checked": fingerprint_checked,
        "temporal_decode_uses_original_time": True,
        "status": status,
        "sparse_compute_claim": bool(detector_forward_exists),
        "fingerprint_expected": expected,
        "fingerprint_actual": actual,
    }
    return selected, evidence

