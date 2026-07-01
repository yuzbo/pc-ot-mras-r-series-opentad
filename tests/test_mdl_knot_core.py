import json
from pathlib import Path

import numpy as np
import pytest

from opentad.acquisition.mdl_knot import (
    MDLKnotConfig,
    build_pipeline_diagnostic,
    build_deploy_scout_curve,
    build_raw_frame_motion_scout_curve,
    build_synthetic_scout_curve,
    generate_matched_controls,
    greedy_mdl_knot_select,
    mdl_objective,
    piecewise_linear_reconstruct,
    summarize_pipeline_diagnostics,
    validate_knot_ledger,
    validate_no_forbidden_sources,
    validate_real_sparse_handoff,
)


ROUTE_LABEL = "DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"


def test_route_identity_has_exact_label_and_rejects_c3_source():
    cfg = MDLKnotConfig(route_label=ROUTE_LABEL)
    assert cfg.route_label == ROUTE_LABEL

    with pytest.raises(ValueError, match="forbidden route/source token"):
        MDLKnotConfig(route_label=ROUTE_LABEL, scout_source="C3_CACHE")


def test_forbidden_gt_teacher_cache_provenance_rejected():
    validate_no_forbidden_sources(
        {
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "dense_raw_backbone_handoff": False,
            "selected_inputs_is_gathered": True,
        }
    )

    with pytest.raises(ValueError, match="uses_gt"):
        validate_no_forbidden_sources({"uses_gt": True})


def test_selector_rejects_unsafe_scout_provenance_without_washing_it_safe():
    curve = build_deploy_scout_curve(
        p_action=[0.1, 0.2, 0.9, 0.2, 0.1],
        provenance={
            "uses_gt": True,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "dense_raw_backbone_handoff": False,
            "selected_inputs_is_gathered": True,
            "position_unit": "original_dense_time_index",
        },
    )

    with pytest.raises(ValueError, match="uses_gt"):
        greedy_mdl_knot_select(curve, MDLKnotConfig(route_label=ROUTE_LABEL))


def test_selected_positions_are_sorted_unique_in_range_and_original_time():
    curve = build_synthetic_scout_curve("two_islands", dense_t=96)
    ledger = greedy_mdl_knot_select(curve, MDLKnotConfig(route_label=ROUTE_LABEL, max_k=40))

    validate_knot_ledger(ledger)
    assert ledger.position_unit == "original_dense_time_index"
    assert ledger.selected_positions == sorted(set(ledger.selected_positions))
    assert all(0 <= pos < ledger.dense_t for pos in ledger.selected_positions)
    assert ledger.valid_k == len(ledger.selected_positions)
    assert "endpoint_anchor" in ledger.selected_roles


def test_selector_keeps_short_windows_strictly_sparse_under_large_cap():
    curve = build_synthetic_scout_curve("short_islands", dense_t=4)
    cfg = MDLKnotConfig(route_label=ROUTE_LABEL, min_k=4, max_k=384, target_weighted_error=0.0, max_gap=1)

    ledger = greedy_mdl_knot_select(curve, cfg)
    dense = list(range(ledger.dense_t))
    selected = [dense[pos] for pos in ledger.selected_positions]

    assert ledger.valid_k < ledger.dense_t
    validate_real_sparse_handoff(
        batch={"selected_inputs": selected, "dense_inputs": dense, "meta": ledger.to_sparse_meta().to_dict()},
        ledger=ledger,
    )


def test_piecewise_reconstruction_error_improves_with_extra_knots():
    curve = build_synthetic_scout_curve("sharp_transition", dense_t=64)
    sparse = [0, 63]
    richer = [0, 20, 31, 32, 44, 63]

    sparse_recon = piecewise_linear_reconstruct(curve.as_matrix(), sparse)
    richer_recon = piecewise_linear_reconstruct(curve.as_matrix(), richer)
    sparse_cost = mdl_objective(curve, sparse, MDLKnotConfig(route_label=ROUTE_LABEL)).weighted_reconstruction_error
    richer_cost = mdl_objective(curve, richer, MDLKnotConfig(route_label=ROUTE_LABEL)).weighted_reconstruction_error

    assert sparse_recon.shape == richer_recon.shape == curve.as_matrix().shape
    assert richer_cost < sparse_cost


def test_dynamic_k_is_nonconstant_on_synthetic_complexity():
    cfg = MDLKnotConfig(route_label=ROUTE_LABEL, min_k=4, max_k=48, target_weighted_error=0.015)
    easy = greedy_mdl_knot_select(build_synthetic_scout_curve("stable_background", dense_t=128), cfg)
    complex_case = greedy_mdl_knot_select(build_synthetic_scout_curve("short_islands", dense_t=128), cfg)

    assert easy.valid_k < complex_case.valid_k
    assert easy.stop_reason in {"residual_and_gap_safe", "marginal_gain_low", "no_positive_gain", "cap_reached"}
    assert complex_case.valid_k <= cfg.max_k


def test_short_action_and_transition_risk_guards_are_selected():
    cfg = MDLKnotConfig(route_label=ROUTE_LABEL, min_k=4, max_k=56, target_weighted_error=0.01)
    ledger = greedy_mdl_knot_select(build_synthetic_scout_curve("short_islands", dense_t=128), cfg)

    assert "short_risk_guard" in ledger.selected_roles
    assert "transition_guard" in ledger.selected_roles
    assert ledger.transition_bands
    assert ledger.estimated_islands


def test_same_k_controls_share_budget_and_validator_path():
    curve = build_synthetic_scout_curve("two_islands", dense_t=96)
    ledger = greedy_mdl_knot_select(curve, MDLKnotConfig(route_label=ROUTE_LABEL, max_k=36))
    controls = generate_matched_controls(curve, ledger, seed=7)

    expected_k = ledger.valid_k
    assert set(controls) >= {
        "per_video_same_k_uniform",
        "mean_k_exact_uniform",
        "random_same_k",
        "scaffold_only",
        "mdl_only",
        "mdl_plus_transition_gap_duration",
    }
    for control in controls.values():
        validate_knot_ledger(control)
        if control.control_name != "scaffold_only":
            assert control.valid_k == expected_k


def test_real_sparse_handoff_requires_gathered_inputs_and_valid_mask():
    curve = build_synthetic_scout_curve("two_islands", dense_t=32)
    ledger = greedy_mdl_knot_select(curve, MDLKnotConfig(route_label=ROUTE_LABEL, max_k=16))
    dense = list(range(ledger.dense_t))
    selected = [dense[pos] for pos in ledger.selected_positions]
    meta = ledger.to_sparse_meta()

    validate_real_sparse_handoff(
        batch={"selected_inputs": selected, "dense_inputs": dense, "meta": meta.to_dict()},
        ledger=ledger,
    )

    with pytest.raises(ValueError, match="dense_raw_backbone_handoff"):
        bad = ledger.to_dict()
        bad["provenance"]["dense_raw_backbone_handoff"] = True
        validate_real_sparse_handoff(
            batch={"selected_inputs": dense, "dense_inputs": dense, "meta": meta.to_dict()},
            ledger=bad,
        )

    with pytest.raises(ValueError, match="length .* must equal valid_k"):
        validate_real_sparse_handoff(
            batch={"selected_inputs": dense, "dense_inputs": dense, "meta": meta.to_dict()},
            ledger=ledger,
        )

    bad_meta = meta.to_dict()
    bad_meta["visibility_mask"] = [True] * (ledger.valid_k - 1) + [False]
    with pytest.raises(ValueError, match="visible"):
        validate_real_sparse_handoff(
            batch={"selected_inputs": selected, "dense_inputs": dense, "meta": bad_meta},
            ledger=ledger,
        )

    bad_route = ledger.to_dict()
    bad_route["route_label"] = "DIVERGENT_INNOVATION_MDL_KNOT_DRIFT"
    with pytest.raises(ValueError, match="route_label"):
        validate_real_sparse_handoff(
            batch={"selected_inputs": selected, "dense_inputs": dense, "meta": meta.to_dict()},
            ledger=bad_route,
        )

    full_dense_ledger = {
        "route_label": ROUTE_LABEL,
        "dense_T": 4,
        "selected_positions": [0, 1, 2, 3],
        "selected_roles": ["endpoint_anchor", "mdl_knot", "mdl_knot", "endpoint_anchor"],
        "valid_k": 4,
        "position_unit": "original_dense_time_index",
        "provenance": {
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "dense_raw_backbone_handoff": False,
            "selected_inputs_is_gathered": True,
            "position_unit": "original_dense_time_index",
        },
    }
    full_dense_meta = {
        "selected_positions": [0, 1, 2, 3],
        "visibility_mask": [True, True, True, True],
        "valid_k": 4,
        "position_unit": "original_dense_time_index",
    }
    with pytest.raises(ValueError, match="valid_k 4 must be shorter than dense_T 4"):
        validate_real_sparse_handoff(
            batch={"selected_inputs": [0, 1, 2, 3], "dense_inputs": [0, 1, 2, 3], "meta": full_dense_meta},
            ledger=full_dense_ledger,
        )


def test_deploy_scout_builder_uses_only_deploy_visible_inputs():
    p_action = [0.1, 0.2, 0.8, 0.7, 0.2]
    curve = build_deploy_scout_curve(p_action=p_action, uncertainty=[0.0] * 5)

    assert curve.source == "deploy_scout"
    assert curve.dense_t == 5
    assert curve.provenance["uses_gt"] is False
    assert curve.temporal_change[2] > curve.temporal_change[0]


def test_synthetic_scout_is_labeled_diagnostic_only():
    curve = build_synthetic_scout_curve("two_islands", dense_t=32)

    assert curve.source == "synthetic_precheck_diagnostic"
    assert curve.provenance["synthetic_precheck_only"] is True


def test_pipeline_diagnostics_report_raw_scout_gaps_masks_and_short_risk():
    frames = []
    for idx in range(8):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:, :, 0] = idx * 18
        frame[3:10, 3:10, 1] = idx * 25
        frames.append(frame)
    curve = build_raw_frame_motion_scout_curve(
        frames,
        probe_positions=[0, 4, 8, 12, 16, 20, 24, 31],
        dense_t=32,
    )
    ledger = greedy_mdl_knot_select(curve, MDLKnotConfig(route_label=ROUTE_LABEL, max_k=24))
    diagnostic = build_pipeline_diagnostic(
        ledger=ledger,
        sparse_meta=ledger.to_sparse_meta().to_dict(),
        masks=[True] * ledger.valid_k + [False] * (24 - ledger.valid_k),
        scout_source=curve.source,
        scout_provenance=curve.provenance,
        bridge="fixed_pad",
        adapter_target_len=24,
    )

    assert diagnostic["raw_frame_scout_used"] is True
    assert diagnostic["metadata_fallback_used"] is False
    assert diagnostic["synthetic_fallback_rejected"] is True
    assert diagnostic["mask_metadata_alignment"]["all_aligned"] is True
    assert diagnostic["fixed_pad_bridge_compute_boundary"]["sparse_compute_claim"] is False
    assert "short_island_total" in diagnostic["short_boundary_risk"]


def test_pipeline_diagnostic_summary_keeps_synthetic_precheck_from_formal_evidence():
    cfg = MDLKnotConfig(route_label=ROUTE_LABEL, max_k=40)
    diagnostics = []
    for pattern in ("stable_background", "short_islands"):
        curve = build_synthetic_scout_curve(pattern, dense_t=64)
        ledger = greedy_mdl_knot_select(curve, cfg)
        diagnostics.append(
            build_pipeline_diagnostic(
                ledger=ledger,
                sparse_meta=ledger.to_sparse_meta().to_dict(),
                masks=[True] * ledger.valid_k + [False] * (40 - ledger.valid_k),
                scout_source=curve.source,
                scout_provenance=curve.provenance,
                bridge="fixed_pad",
                adapter_target_len=40,
            )
        )
    summary = summarize_pipeline_diagnostics(diagnostics)

    assert summary["window_count"] == 2
    assert summary["synthetic_fallback_windows"] == 2
    assert summary["synthetic_fallback_rejected"] is False
    assert summary["valid_k_distribution"]["nonconstant"] is True
    assert summary["fixed_pad_bridge_compute_boundary"]["sparse_compute_claim"] is False
