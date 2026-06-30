from __future__ import annotations

from typing import MutableMapping, Sequence

from .selector import greedy_mdl_knot_select
from .types import MDLKnotConfig, ScoutCurve
from .validators import validate_real_sparse_handoff


def _gather_selected_inputs(dense_inputs: Sequence[object], selected_positions: Sequence[int]) -> list[object]:
    return [dense_inputs[int(pos)] for pos in selected_positions]


def apply_mdl_knot_to_dense_window(
    results: MutableMapping[str, object],
    dense_window: Sequence[int],
    scout_curve: ScoutCurve,
    config: MDLKnotConfig,
    adapter_target_len: int | None = None,
    dense_inputs: Sequence[object] | None = None,
) -> MutableMapping[str, object]:
    if len(dense_window) != scout_curve.dense_t:
        raise ValueError(f"dense_window length {len(dense_window)} must match scout dense_t {scout_curve.dense_t}")
    if dense_inputs is not None and len(dense_inputs) != len(dense_window):
        raise ValueError(f"dense_inputs audit length {len(dense_inputs)} must match dense_window {len(dense_window)}")
    ledger = greedy_mdl_knot_select(
        scout_curve,
        config,
        video_id=str(results.get("video_name", "unknown")),
        window_id=int(results.get("window_id", 0)),
    )
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

    if dense_inputs is None:
        raise ValueError("MDL-Knot true sparse handoff requires dense raw inputs for gather validation")
    selected_inputs = _gather_selected_inputs(dense_inputs, selected_positions)
    validate_real_sparse_handoff(
        batch={
            "selected_inputs": selected_inputs,
            "dense_inputs": dense_inputs,
            "meta": sparse_meta.to_dict(),
        },
        ledger=ledger,
    )
    first_shape = getattr(selected_inputs[0], "shape", None) if selected_inputs else None
    results["mdl_knot_real_sparse_handoff_validated"] = True
    results["mdl_knot_handoff_audit"] = {
        "selected_inputs_is_gathered": True,
        "selected_len": valid_k,
        "dense_len": len(dense_inputs),
        "raw_sample_shape": None if first_shape is None else [int(v) for v in first_shape],
        "raw_inputs_retained": False,
    }
    return results
