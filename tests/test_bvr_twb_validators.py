import numpy as np
import pytest

from opentad.acquisition.bvr_twb.budget_controller import DynamicBudgetController
from opentad.acquisition.bvr_twb.scaffold import build_scaffold_packets
from opentad.acquisition.bvr_twb.sparse_gather import sparse_gather
from opentad.acquisition.bvr_twb.state_scout import build_scout_from_actionness
from opentad.acquisition.bvr_twb.types import BudgetConfig, CandidatePacket, ROUTE_LABEL
from opentad.acquisition.bvr_twb.validators import (
    build_original_time_metadata,
    build_selection_gap_diagnostics,
    validate_candidate_packet_ledger,
    validate_deploy_ledger,
    validate_dynamicity_and_uniform_mimicry,
    validate_no_leakage,
    validate_original_time_metadata,
    validate_route_identity,
    validate_selected_positions,
    validate_selection_row_schema,
    validate_sparse_gather_evidence,
    validate_summary_claim_status,
)


def test_route_identity_rejects_c3_and_other_route_tokens():
    assert validate_route_identity({"route_label": ROUTE_LABEL, "note": "local bvr"})
    with pytest.raises(ValueError, match="forbidden route token"):
        validate_route_identity({"route_label": ROUTE_LABEL, "note": "borrow C3 selector"})
    with pytest.raises(ValueError, match="forbidden route token"):
        validate_route_identity({"route_label": ROUTE_LABEL, "route": "BH_SDC"})
    with pytest.raises(ValueError, match="route_label"):
        validate_route_identity({"route_label": "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"})


def test_no_leakage_rejects_gt_teacher_dense_prediction_and_cache():
    clean = {
        "provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
        }
    }
    assert validate_no_leakage(clean)
    for key in ("gt_segments", "teacher_logits", "dense_detector_predictions", "proposal_cache", "oracle_boundary"):
        with pytest.raises(ValueError):
            validate_no_leakage({key: [1]})
    with pytest.raises(ValueError, match="must be false"):
        validate_no_leakage({"provenance": {"selection_uses_gt": True}})
    with pytest.raises(ValueError, match="forbidden leakage"):
        validate_no_leakage({"outer": [{"nested": {"ground_truth": [0, 1]}}]})


def test_state_scout_rejects_forbidden_metadata_recursively():
    with pytest.raises(ValueError, match="forbidden"):
        build_scout_from_actionness(
            [0.1, 0.2, 0.1, 0.3],
            metadata={"outer": [{"safe": True}, {"nested": {"teacher_logits": [0.3]}}]},
        )


def test_selected_positions_are_sorted_unique_original_dense_indices():
    assert validate_selected_positions([0, 2, 9], dense_T=10, valid_k=3)
    with pytest.raises(ValueError, match="sorted unique"):
        validate_selected_positions([0, 2, 2], dense_T=10)
    with pytest.raises(ValueError, match="sorted unique"):
        validate_selected_positions([2, 0], dense_T=10)
    with pytest.raises(ValueError, match="out of range"):
        validate_selected_positions([0, 10], dense_T=10)
    with pytest.raises(ValueError, match="valid_k"):
        validate_selected_positions([0, 2], dense_T=10, valid_k=3)


def test_original_time_metadata_rejects_selected_index_as_time():
    meta = build_original_time_metadata(10, [0, 4, 9], fps=5.0)
    assert meta["selected_times_sec"] == [0.0, 0.8, 1.8]
    assert validate_original_time_metadata(meta)
    bad = dict(meta)
    bad["selected_index_is_time"] = True
    with pytest.raises(ValueError, match="selected_index_is_time"):
        validate_original_time_metadata(bad)
    bad_unit = dict(meta)
    bad_unit["selected_positions_unit"] = "selected_axis_index"
    with pytest.raises(ValueError, match="original_dense_index"):
        validate_original_time_metadata(bad_unit)


def test_sparse_gather_fingerprint_and_local_status_contracts():
    dense = np.arange(20, dtype=np.float64).reshape(10, 2)
    selected, evidence = sparse_gather(dense, [0, 3, 7], temporal_dim=0, detector_forward_exists=False)
    assert selected.shape == (3, 2)
    assert evidence["status"] == "local_gather_smoke_only"
    assert evidence["sparse_compute_claim"] is False
    assert validate_sparse_gather_evidence(evidence, dense_T=10, valid_k=3, require_detector_forward=False)
    tampered = dict(evidence)
    tampered["fingerprint_checked"] = False
    with pytest.raises(ValueError, match="fingerprint"):
        validate_sparse_gather_evidence(tampered, dense_T=10, valid_k=3)
    dense_handoff = dict(evidence)
    dense_handoff["selected_temporal_len"] = 10
    with pytest.raises(ValueError, match="smaller than dense_T"):
        validate_sparse_gather_evidence(dense_handoff, dense_T=10, valid_k=10)

    _, audited = sparse_gather(
        dense,
        [0, 3, 7],
        temporal_dim=0,
        detector_forward_exists=True,
        detector_forward_temporal_len=3,
    )
    assert audited["status"] == "detector_forward_sparse_audited"
    assert validate_sparse_gather_evidence(audited, dense_T=10, valid_k=3, require_detector_forward=True)
    bad_forward = dict(audited, detector_forward_temporal_len=10)
    with pytest.raises(ValueError, match="temporal length must equal valid_k"):
        validate_sparse_gather_evidence(bad_forward, dense_T=10, valid_k=3, require_detector_forward=True)
    bad_dense = dict(audited, dense_raw_backbone_handoff=True)
    with pytest.raises(ValueError, match="dense_raw_backbone_handoff"):
        validate_sparse_gather_evidence(bad_dense, dense_T=10, valid_k=3, require_detector_forward=True)


def test_candidate_packet_and_selection_rows_require_component_schema():
    packet = CandidatePacket(
        packet_id=7,
        video_id="v",
        window_id=0,
        split="synthetic",
        source="twb",
        role="transition_center",
        positions=[3],
        dense_T=12,
        reason="center witness",
        feature_summary={"gap_if_omitted_frames": 4.0},
    )
    row = packet.to_ledger_dict()
    row["value_components"] = {
        "belief_width_gain": 0.2,
        "role_gain": 0.1,
        "gap_gain": 0.0,
        "short_action_gain": 0.0,
        "redundancy_repulsion_penalty": 0.0,
        "low_actionness_component": 0.03,
    }
    assert validate_candidate_packet_ledger(row)
    bad_packet = dict(row)
    bad_packet.pop("packet_positions")
    with pytest.raises(ValueError, match="candidate packet ledger missing"):
        validate_candidate_packet_ledger(bad_packet)

    selection_row = {
        "selected_decision": "selected",
        "selected_decision_subreason": "marginal_gain_positive",
        "constraint_state": {"max_gap_ok": True, "budget_ok": True, "duplicate": False},
        "value_components": row["value_components"],
    }
    assert validate_selection_row_schema(selection_row)
    bad_selection = dict(selection_row)
    bad_selection.pop("selected_decision_subreason")
    with pytest.raises(ValueError, match="selection row missing"):
        validate_selection_row_schema(bad_selection)


def test_summary_claim_status_is_locked_for_local_gather_smoke():
    assert validate_summary_claim_status(
        {"claim_status": "local_gather_smoke_only_no_sparse_compute_or_metric_claim"},
        claim_mode="local_gather_smoke",
    )
    with pytest.raises(ValueError, match="claim_status"):
        validate_summary_claim_status({"claim_status": "metric_claim_unlocked"}, claim_mode="local_gather_smoke")


def test_deploy_ledger_validator_rejects_dense_handoff_and_bad_decode():
    dense = np.arange(16, dtype=np.float64).reshape(8, 2)
    _, evidence = sparse_gather(dense, [0, 3, 6], temporal_dim=0)
    meta = build_original_time_metadata(8, [0, 3, 6], fps=4.0)
    ledger = {
        "route_label": ROUTE_LABEL,
        "method": "BVR-TWB",
        "video_id": "v",
        "split": "synthetic",
        "window_id": 0,
        "dense_T": 8,
        "selected_positions": [0, 3, 6],
        "selected_times_sec": meta["selected_times_sec"],
        "claim_mode": "local_gather_smoke",
        "valid_k": 3,
        "scaffold_k": 1,
        "min_k": 2,
        "max_k": 4,
        "budget_stop_reason": "candidate_exhausted",
        "selection_gap_diagnostics": build_selection_gap_diagnostics([0, 3, 6], 8, 3),
        "original_time_metadata": meta,
        "real_sparse_evidence": evidence,
        "forbidden_fields_absent": {
            "regret_label_absent": True,
            "gt_fields_absent": True,
            "teacher_fields_absent": True,
            "prediction_cache_absent": True,
        },
        "provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        },
    }
    assert validate_deploy_ledger(ledger)
    bad = dict(ledger)
    bad["real_sparse_evidence"] = dict(evidence, dense_raw_backbone_handoff=True)
    with pytest.raises(ValueError, match="dense_raw_backbone_handoff"):
        validate_deploy_ledger(bad)
    bad_meta = dict(meta, selected_index_is_time=True)
    bad_decode = dict(ledger, original_time_metadata=bad_meta)
    with pytest.raises(ValueError, match="selected_index_is_time"):
        validate_deploy_ledger(bad_decode)
    bad_gap = dict(ledger)
    bad_gap["selection_gap_diagnostics"] = build_selection_gap_diagnostics([0, 3, 6], 8, 1)
    with pytest.raises(ValueError, match="hard max-gap"):
        validate_deploy_ledger(bad_gap)


def test_deploy_ledger_rejects_forged_belief_width_safe_trace():
    dense = np.arange(16, dtype=np.float64).reshape(8, 2)
    _, evidence = sparse_gather(dense, [0, 3, 6], temporal_dim=0)
    meta = build_original_time_metadata(8, [0, 3, 6], fps=4.0)
    ledger = {
        "route_label": ROUTE_LABEL,
        "method": "BVR-TWB",
        "video_id": "v",
        "split": "synthetic",
        "window_id": 0,
        "dense_T": 8,
        "selected_positions": [0, 3, 6],
        "selected_times_sec": meta["selected_times_sec"],
        "claim_mode": "local_gather_smoke",
        "valid_k": 3,
        "scaffold_k": 1,
        "min_k": 2,
        "max_k": 4,
        "budget_stop_reason": "belief_width_safe",
        "selection_gap_diagnostics": build_selection_gap_diagnostics([0, 3, 6], 8, 3),
        "bracket_summary": {
            "active_belief_update_trace": [
                {
                    "bracket_id": 0,
                    "updated_from_selected_witness": False,
                    "belief_width_safe": True,
                }
            ],
            "all_active_beliefs_updated_and_safe": False,
        },
        "original_time_metadata": meta,
        "real_sparse_evidence": evidence,
        "forbidden_fields_absent": {
            "regret_label_absent": True,
            "gt_fields_absent": True,
            "teacher_fields_absent": True,
            "prediction_cache_absent": True,
        },
        "provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        },
    }
    with pytest.raises(ValueError, match="belief_width_safe"):
        validate_deploy_ledger(ledger)


def test_controller_fails_closed_when_max_gap_infeasible_under_budget():
    scaffold = build_scaffold_packets(
        dense_T=20,
        scaffold_k=2,
        max_gap=20,
        video_id="infeasible_gap",
        split="synthetic",
    )
    controller = DynamicBudgetController(BudgetConfig(min_k=2, max_k=3, max_gap=3))
    with pytest.raises(ValueError, match="hard max-gap contract"):
        controller.select(
            scaffold_packets=scaffold,
            candidate_packets=[],
            brackets=[],
            dense_T=20,
            video_id="infeasible_gap",
            split="synthetic",
        )


def test_gap_repair_rows_are_incremental_and_equal_max_gap_is_safe():
    scaffold = build_scaffold_packets(
        dense_T=64,
        scaffold_k=2,
        max_gap=64,
        video_id="gap_incremental",
        split="synthetic",
    )
    controller = DynamicBudgetController(BudgetConfig(min_k=2, max_k=9, max_gap=15))
    result = controller.select(
        scaffold_packets=scaffold,
        candidate_packets=[],
        brackets=[],
        dense_T=64,
        video_id="gap_incremental",
        split="synthetic",
    )
    repair_rows = [row for row in result.ledger_rows if row["selected_decision"] == "selected_gap_guard"]
    assert len(repair_rows) >= 2
    after_lengths = [len(row["selected_after_positions"]) for row in repair_rows]
    assert after_lengths == sorted(after_lengths)
    assert all((b - a) == 1 for a, b in zip(after_lengths, after_lengths[1:]))
    assert result.deploy_ledger["selection_gap_diagnostics"]["max_gap"] == 15
    assert result.stop_reason != "gap_guard"


def test_dynamicity_gate_rejects_constant_k_budget_cap_and_uniform_mimicry():
    base = {
        "route_label": ROUTE_LABEL,
        "dense_T": 20,
        "valid_k": 5,
        "scaffold_k": 1,
        "budget_stop_reason": "budget_cap",
        "selected_positions": [0, 5, 10, 15, 19],
    }
    with pytest.raises(ValueError, match="constant-K"):
        validate_dynamicity_and_uniform_mimicry([dict(base), dict(base), dict(base)], dynamic_enabled=True)
    varied = [
        dict(base, valid_k=5, selected_positions=[0, 3, 9, 12, 19]),
        dict(base, valid_k=6, selected_positions=[0, 2, 7, 11, 16, 19]),
        dict(base, valid_k=7, selected_positions=[0, 4, 8, 10, 12, 17, 19]),
    ]
    with pytest.raises(ValueError, match="budget-cap-only"):
        validate_dynamicity_and_uniform_mimicry(varied, dynamic_enabled=True, max_uniform_overlap=0.99)
    uniform = [
        dict(base, valid_k=5, budget_stop_reason="candidate_exhausted", selected_positions=[0, 5, 10, 14, 19]),
        dict(base, valid_k=6, budget_stop_reason="candidate_exhausted", selected_positions=[0, 4, 8, 11, 15, 19]),
        dict(base, valid_k=7, budget_stop_reason="candidate_exhausted", selected_positions=[0, 3, 6, 10, 13, 16, 19]),
    ]
    with pytest.raises(ValueError, match="exact-uniform"):
        validate_dynamicity_and_uniform_mimicry(uniform, dynamic_enabled=True, max_uniform_overlap=0.80)
