from collections import Counter

import numpy as np

from opentad.acquisition.bvr_twb.scaffold import gap_statistics
from opentad.acquisition.bvr_twb.boundary_belief import estimate_boundary_beliefs
from opentad.acquisition.bvr_twb.state_scout import build_scout_from_actionness
from opentad.acquisition.bvr_twb.value_predictor import PacketValuePredictor
from opentad.acquisition.bvr_twb.validators import validate_deploy_ledger, validate_dynamicity_and_uniform_mimicry
from opentad.acquisition.bvr_twb.witness_packets import build_witness_packets
from tools.bvr_twb.build_synthetic_ledgers import build_ledgers, run_bvr_case, synthetic_cases


def test_synthetic_smoke_runs_dynamic_bvr_and_validates_ledgers():
    ledgers = []
    stops = []
    ks = []
    for name, p_action, motion in synthetic_cases():
        result, candidates, _ = run_bvr_case(name, p_action, motion)
        assert validate_deploy_ledger(result.deploy_ledger)
        stats = gap_statistics(result.selected_positions, result.deploy_ledger["dense_T"])
        assert stats["max_gap"] <= 22
        assert result.deploy_ledger["real_sparse_evidence"]["status"] == "local_gather_smoke_only"
        assert result.deploy_ledger["real_sparse_evidence"]["sparse_compute_claim"] is False
        assert result.deploy_ledger["claim_mode"] == "local_gather_smoke"
        trace = result.deploy_ledger["bracket_summary"]["belief_update_trace"]
        assert isinstance(trace, list)
        assert result.deploy_ledger["bracket_summary"]["posterior_mean_belief_width_p80"] <= result.deploy_ledger[
            "bracket_summary"
        ]["initial_mean_belief_width_p80"]
        if result.stop_reason == "belief_width_safe":
            active_trace = result.deploy_ledger["bracket_summary"]["active_belief_update_trace"]
            assert active_trace
            assert all(row["updated_from_selected_witness"] for row in active_trace)
            assert all(row["belief_width_safe"] for row in active_trace)
        assert any(role in result.deploy_ledger["selected_packet_roles"] for role in ("transition_center", "gap_bridge", "short_action_guard"))
        assert all(packet.predicted_value is not None for packet in candidates)
        assert all("value_components" in packet.to_ledger_dict() for packet in candidates)
        ledgers.append(result.deploy_ledger)
        stops.append(result.stop_reason)
        ks.append(result.deploy_ledger["valid_k"])
    diag = validate_dynamicity_and_uniform_mimicry(ledgers, dynamic_enabled=True)
    assert len(set(ks)) > 1
    assert Counter(stops)["budget_cap"] < len(stops)
    assert diag["mean_uniform_overlap"] < 0.86


def test_packet_value_predictor_keeps_actionness_low_weight():
    name, p_action, motion = synthetic_cases()[1]
    result, candidates, _ = run_bvr_case(name, p_action, motion)
    predictor = PacketValuePredictor()
    diag = predictor.anti_actionness_only_diagnostics(candidates)
    assert diag["passes"] is True
    action_fractions = [
        packet.predicted_value.diagnostics["actionness_component_fraction"]
        for packet in candidates
        if packet.predicted_value is not None
    ]
    component_keys = set(candidates[0].predicted_value.diagnostics["value_components"])
    assert {
        "belief_width_gain",
        "role_gain",
        "gap_gain",
        "short_action_gain",
        "redundancy_repulsion_penalty",
        "low_actionness_component",
    }.issubset(component_keys)
    assert np.mean(action_fractions) < 0.18
    assert any(packet.role.startswith("transition") for packet in result.selected_packets)


def test_dynamic_stop_reasons_include_non_budget_outcomes():
    stops = []
    ks = []
    for name, p_action, motion in synthetic_cases():
        result, _, _ = run_bvr_case(name, p_action, motion)
        stops.append(result.stop_reason)
        ks.append(len(result.selected_positions))
    assert len(set(ks)) >= 2
    assert set(stops).difference({"budget_cap"})


def test_witness_gap_features_use_scaffold_positions_and_explicit_max_gap():
    dense_T = 48
    p_action = np.zeros(dense_T, dtype=np.float64) + 0.08
    p_action[22:27] = 0.8
    scout = build_scout_from_actionness(p_action)
    brackets = estimate_boundary_beliefs(scout, max_brackets=1, radius=3)
    scaffold_positions = [0, 1, 24, 47]
    packets = build_witness_packets(scout, brackets, scaffold_positions=scaffold_positions, max_gap=22)
    assert packets
    assert min(packet.feature_summary["gap_if_omitted_frames"] for packet in packets) <= 4.0
    assert not [packet for packet in packets if packet.role == "gap_bridge"]


def test_synthetic_summary_keeps_claim_status_locked_and_reports_diagnostics(tmp_path):
    out = tmp_path / ".tmp_bvr_twb_summary"
    summary = build_ledgers(out, overwrite=True, root=tmp_path)
    assert summary["claim_status"] == "local_gather_smoke_only_no_sparse_compute_or_metric_claim"
    assert summary["claim_mode"] == "local_gather_smoke"
    assert "mean_posterior_belief_width_p80" in summary
    assert "twb_no_regret_uniform_fallback_ratio" in summary
