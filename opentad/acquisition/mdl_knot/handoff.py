from __future__ import annotations

import time
from typing import MutableMapping, Sequence

from .selector import greedy_mdl_knot_select
from .types import MDLKnotConfig, ScoutCurve
from .validators import validate_real_sparse_handoff


def apply_mdl_knot_to_dense_window(
    results: MutableMapping[str, object],
    dense_window: Sequence[int],
    scout_curve: ScoutCurve,
    config: MDLKnotConfig,
    adapter_target_len: int | None = None,
) -> MutableMapping[str, object]:
    if len(dense_window) != scout_curve.dense_t:
        raise ValueError(f"dense_window length {len(dense_window)} must match scout dense_t {scout_curve.dense_t}")
    profile = results.get("mdl_knot_profile")
    if not isinstance(profile, dict):
        profile = {}
        results["mdl_knot_profile"] = profile
    selector_start = time.perf_counter()
    ledger = greedy_mdl_knot_select(
        scout_curve,
        config,
        video_id=str(results.get("video_name", "unknown")),
        window_id=int(results.get("window_id", 0)),
    )
    profile["selector_s"] = float(time.perf_counter() - selector_start)
    structural_start = time.perf_counter()
    selected_positions = list(ledger.selected_positions)
    frame_inds = [int(dense_window[pos]) for pos in selected_positions]
    valid_k = int(ledger.valid_k)
    if adapter_target_len is not None:
        adapter_target_len = int(adapter_target_len)
        if adapter_target_len < valid_k:
            raise ValueError(f"adapter_target_len {adapter_target_len} is smaller than MDL valid_k {valid_k}")
        if adapter_target_len > valid_k:
            frame_inds = frame_inds + [frame_inds[-1]] * (adapter_target_len - valid_k)
            masks = [True] * valid_k + [False] * (adapter_target_len - valid_k)
        else:
            masks = [True] * valid_k
    else:
        masks = [True] * valid_k
    sparse_meta = ledger.to_sparse_meta()

    results["frame_inds"] = frame_inds
    results["num_clips"] = 1
    results["clip_len"] = len(frame_inds)
    results["masks"] = masks
    results["mdl_knot_selected_positions"] = selected_positions
    results["mdl_knot_selected_roles"] = list(ledger.selected_roles)
    results["mdl_knot_valid_k"] = valid_k
    results["mdl_knot_adapter_target_len"] = int(adapter_target_len or valid_k)
    results["mdl_knot_bridge"] = "fixed_pad" if adapter_target_len is not None else "variable_sparse"
    results["mdl_knot_padding_counts_as_valid"] = False
    results["mdl_knot_ledger"] = ledger.to_dict()
    results["mdl_knot_sparse_meta"] = sparse_meta.to_dict()
    results["irregular_selected_positions"] = [float(v) for v in selected_positions]
    results["irregular_selected_valid_len"] = float(ledger.dense_t)
    results["irregular_native_axis"] = False
    profile["structural_handoff_s"] = float(time.perf_counter() - structural_start)

    validation_start = time.perf_counter()
    validate_real_sparse_handoff(
        batch={
            "selected_inputs": frame_inds[:valid_k],
            "dense_inputs": list(dense_window),
            "meta": sparse_meta.to_dict(),
        },
        ledger=ledger,
    )
    profile["handoff_validation_s"] = float(time.perf_counter() - validation_start)
    profile["selector_and_structural_handoff_s"] = float(
        profile["selector_s"] + profile["structural_handoff_s"] + profile["handoff_validation_s"]
    )
    return results
