import numpy as np
import pytest

from opentad.acquisition.bvr_twb.boundary_belief import update_boundary_belief_trace
from opentad.acquisition.bvr_twb.budget_controller import DynamicBudgetController
from opentad.acquisition.bvr_twb.regret_labels import build_packet_regret_labels, validate_regret_label_schema
from opentad.acquisition.bvr_twb.scaffold import build_scaffold_packets
from opentad.acquisition.bvr_twb.types import BracketState, BudgetConfig, CandidatePacket, ROUTE_LABEL
from opentad.acquisition.bvr_twb.validators import (
    validate_belief_update_trace_schema,
    validate_voi_bbc_stop_contract,
    validate_voi_component_balance,
)
from tools.bvr_twb.build_synthetic_ledgers import run_bvr_case, synthetic_cases


def _bracket():
    return BracketState(
        bracket_id=0,
        video_id="v",
        window_id=0,
        split="synthetic",
        kind="start_like",
        left_pos=10,
        center_pos=15,
        right_pos=20,
        dense_T=40,
        peak_transition=0.8,
        mean_uncertainty=0.7,
        action_left_mean=0.1,
        action_right_mean=0.8,
        persistence=0.2,
        short_action_risk=0.3,
        gap_risk=0.5,
        entropy=0.8,
        width_p50_frames=5.0,
        width_p80_frames=9.0,
        confidence=0.2,
        two_sided_state_contrast=0.9,
        state="active",
    )


def _packet(packet_id, role, pos):
    return CandidatePacket(
        packet_id=packet_id,
        video_id="v",
        window_id=0,
        split="synthetic",
        source="twb",
        role=role,
        positions=[pos],
        dense_T=40,
        bracket_id=0,
        reason=role,
        feature_summary={
            "mean_actionness": 0.1,
            "max_transition": 0.8,
            "mean_uncertainty": 0.7,
            "short_action_risk": 0.3,
            "two_sided_state_contrast": 0.9,
            "bracket_width_frames": 11.0,
            "gap_if_omitted_frames": 8.0,
            "gap_risk": 0.5,
            "belief_entropy": 0.8,
        },
    )


def test_voi_bbc_belief_safe_requires_selected_two_sided_posterior_update():
    bracket = _bracket()
    prior = update_boundary_belief_trace([bracket], [], safe_width=5.0)[0]
    assert prior["updated_by_selected_witness"] is False
    assert prior["belief_width_safe"] is False

    selected = [
        _packet(1, "transition_before", 12),
        _packet(2, "transition_center", 15),
        _packet(3, "transition_after", 18),
    ]
    posterior = update_boundary_belief_trace([bracket], selected, safe_width=5.0)[0]
    assert posterior["updated_by_selected_witness"] is True
    assert posterior["two_sided_witness_coverage"] is True
    assert posterior["posterior_entropy"] < posterior["initial_entropy"]
    assert posterior["posterior_credible_width_frames"] < posterior["initial_width_p80_frames"]
    assert posterior["posterior_risk_mass"] < posterior["initial_risk_mass"]
    assert posterior["belief_width_safe"] is True
    assert validate_belief_update_trace_schema([posterior])


def test_voi_bbc_forged_safe_without_selected_witness_fails():
    forged = {
        "bracket_id": 0,
        "initial_entropy": 0.8,
        "posterior_entropy": 0.2,
        "initial_width_p80_frames": 9.0,
        "posterior_width_p80_frames": 3.0,
        "posterior_credible_width_frames": 3.0,
        "initial_risk_mass": 0.8,
        "posterior_risk_mass": 0.2,
        "two_sided_witness_coverage": True,
        "updated_by_selected_witness": False,
        "belief_width_safe": True,
    }
    with pytest.raises(ValueError, match="forged"):
        validate_belief_update_trace_schema([forged])


def test_scaffold_is_candidate_not_mandatory_first_in_low_risk_case():
    scaffold = build_scaffold_packets(dense_T=32, scaffold_k=4, max_gap=40, video_id="low", split="synthetic")
    controller = DynamicBudgetController(BudgetConfig(min_k=2, max_k=5, max_gap=40, min_marginal_value=0.99))
    result = controller.select(scaffold, [], [], dense_T=32, video_id="low", split="synthetic")
    assert result.deploy_ledger["controller_trace_summary"]["mandatory_scaffold_first"] is False
    assert len(result.selected_positions) < len(scaffold)
    assert not any(row["selected_decision"] == "required_scaffold" for row in result.ledger_rows)


def test_voi_components_reject_actionness_only_dominance():
    components = {
        "expected_entropy_reduction": 0.0,
        "expected_width_reduction": 0.0,
        "expected_gap_risk_reduction": 0.0,
        "short_action_value": 0.0,
        "two_sided_witness_value": 0.0,
        "predicted_regret": 0.01,
        "value_per_cost": 1.0,
        "actionness_component": 10.0,
    }
    with pytest.raises(ValueError, match="actionness-only"):
        validate_voi_component_balance(components, max_actionness_fraction=0.45)


def test_stop_reason_must_match_posterior_risk_state():
    ledger = {
        "budget_stop_reason": "risk_constraints_satisfied",
        "selection_gap_diagnostics": {"coverage_violation": False},
        "bracket_summary": {
            "max_posterior_risk_mass": 0.9,
            "belief_update_trace": [
                {
                    "bracket_id": 0,
                    "initial_entropy": 0.8,
                    "posterior_entropy": 0.6,
                    "initial_width_p80_frames": 9.0,
                    "posterior_width_p80_frames": 8.0,
                    "posterior_credible_width_frames": 8.0,
                    "initial_risk_mass": 0.9,
                    "posterior_risk_mass": 0.9,
                    "two_sided_witness_coverage": False,
                    "updated_by_selected_witness": True,
                    "belief_width_safe": False,
                }
            ],
        },
    }
    with pytest.raises(ValueError, match="low posterior risk"):
        validate_voi_bbc_stop_contract(ledger)


def test_train_only_voi_labels_schema_and_val_forbidden():
    bracket = _bracket()
    packet = _packet(10, "transition_center", 15)
    labels = build_packet_regret_labels([packet], gt_segments=np.array([[12.0, 19.0]], dtype=np.float32), dense_T=40, split="train")
    assert len(labels) == 1
    assert labels[0]["target_entropy_reduction"] >= 0.0
    assert labels[0]["target_width_reduction"] >= 0.0
    assert labels[0]["target_boundary_risk_reduction"] >= 0.0
    assert labels[0]["target_omission_regret"] == labels[0]["target_regret"]
    assert validate_regret_label_schema(labels[0])
    with pytest.raises(ValueError, match="train-only"):
        build_packet_regret_labels([packet], gt_segments=np.array([[12.0, 19.0]], dtype=np.float32), dense_T=40, split="val")


def test_controller_rows_record_marginal_voi_and_belief_before_after():
    name, p_action, motion = synthetic_cases()[1]
    result, _, _ = run_bvr_case(name, p_action, motion)
    assert result.deploy_ledger["voi_bbc_spec"] == "Value-of-Information Boundary Belief Controller"
    assert result.stop_reason in {
        "belief_width_safe",
        "risk_constraints_satisfied",
        "regret_saturation",
        "budget_cap",
        "candidate_exhausted",
        "gap_guard",
    }
    assert result.ledger_rows
    assert any("belief_state_before" in row and "belief_state_after" in row for row in result.ledger_rows)
    assert any("marginal_voi_score" in row for row in result.ledger_rows)
    assert result.deploy_ledger["controller_trace_summary"]["mandatory_scaffold_first"] is False


def test_no_c3_route_token_leakage_in_route_label_guard():
    assert ROUTE_LABEL == "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
