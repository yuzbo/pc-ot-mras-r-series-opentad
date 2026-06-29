from collections import Counter

import numpy as np

from opentad.acquisition.bvr_twb.scaffold import gap_statistics
from opentad.acquisition.bvr_twb.value_predictor import PacketValuePredictor
from opentad.acquisition.bvr_twb.validators import validate_deploy_ledger, validate_dynamicity_and_uniform_mimicry
from tools.bvr_twb.build_synthetic_ledgers import run_bvr_case, synthetic_cases


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
        assert any(role in result.deploy_ledger["selected_packet_roles"] for role in ("transition_center", "gap_bridge", "short_action_guard"))
        assert all(packet.predicted_value is not None for packet in candidates)
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

