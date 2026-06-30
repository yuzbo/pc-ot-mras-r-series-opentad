import numpy as np
import pytest

from opentad.acquisition.rba_rbr import (
    DynamicBudgetController,
    RbaRbrBudgetConfig,
    build_probe_candidates,
    build_rba_rbr_open_tad_selection,
    build_regret_labels,
    build_risk_map,
    build_soft_brackets,
)
from opentad.acquisition.rba_rbr.types import ROUTE_LABEL
from opentad.acquisition.rba_rbr.validators import (
    validate_no_leakage,
    validate_route_identity,
    validate_selected_positions,
)


def _missed_boundary_case():
    dense_T = 80
    actionness = np.zeros(dense_T, dtype=np.float64) + 0.05
    actionness[18:31] = 0.82
    uncertainty = np.zeros(dense_T, dtype=np.float64) + 0.08
    transition = np.zeros(dense_T, dtype=np.float64) + 0.04
    transition[17:20] = 0.55
    transition[30:33] = 0.45
    transition[55:58] = 0.95
    uncertainty[54:59] = 0.92
    return actionness, uncertainty, transition


def test_route_identity_rejects_c3_and_other_route_mixing_tokens():
    assert ROUTE_LABEL == "DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3"
    assert validate_route_identity({"route_label": ROUTE_LABEL, "note": "local rba rbr"})
    with pytest.raises(ValueError, match="forbidden route token"):
        validate_route_identity({"route_label": ROUTE_LABEL, "note": "borrow C3 controller"})
    with pytest.raises(ValueError, match="forbidden route token"):
        validate_route_identity({"route_label": ROUTE_LABEL, "route": "CADF"})
    with pytest.raises(ValueError, match="route_label"):
        validate_route_identity({"route_label": "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"})


def test_selected_positions_must_be_sorted_unique_original_dense_indices():
    assert validate_selected_positions([0, 4, 9], dense_T=10, valid_k=3)
    with pytest.raises(ValueError, match="sorted unique"):
        validate_selected_positions([0, 4, 4], dense_T=10)
    with pytest.raises(ValueError, match="sorted unique"):
        validate_selected_positions([4, 0], dense_T=10)
    with pytest.raises(ValueError, match="out of range"):
        validate_selected_positions([0, 10], dense_T=10)


def test_rescue_probe_covers_boundary_missed_by_hard_bracket():
    actionness, uncertainty, transition = _missed_boundary_case()
    risk_map = build_risk_map(
        actionness=actionness,
        uncertainty=uncertainty,
        transition=transition,
        observed_positions=[0, 20, 40, 79],
    )
    brackets = build_soft_brackets(risk_map, action_threshold=0.55, rescue_threshold=0.70)
    probes = build_probe_candidates(
        risk_map,
        brackets,
        scaffold_positions=[0, 20, 40, 79],
        video_id="missed_boundary",
        split="synthetic",
    )

    hard_spans = [(bracket.hard_left, bracket.hard_right) for bracket in brackets if bracket.hard_bracket]
    rescue_probes = [probe for probe in probes if probe.stage == "rescue"]
    assert rescue_probes, "RBA-RBR must produce out-of-bracket rescue probes"
    assert any(not any(left <= probe.center_pos <= right for left, right in hard_spans) for probe in rescue_probes)
    assert any(abs(probe.center_pos - 56) <= 1 for probe in rescue_probes)

    selected = DynamicBudgetController(RbaRbrBudgetConfig(min_k=5, max_k=10, scaffold_k=4)).select(
        risk_map,
        brackets,
        probes,
        scaffold_positions=[0, 20, 40, 79],
        video_id="missed_boundary",
        split="synthetic",
    )
    assert any(abs(pos - 56) <= 1 for pos in selected.selected_positions)
    assert selected.deploy_ledger["rescue_outside_hard_bracket_count"] >= 1
    assert selected.stop_reason in {"risk_satisfied", "budget_cap", "candidate_exhausted", "regret_saturation"}


def test_train_only_regret_labels_reject_val_test_and_deploy():
    actionness, uncertainty, transition = _missed_boundary_case()
    risk_map = build_risk_map(actionness=actionness, uncertainty=uncertainty, transition=transition)
    brackets = build_soft_brackets(risk_map)
    probes = build_probe_candidates(risk_map, brackets, scaffold_positions=[0, 20, 40, 79], split="train")
    labels = build_regret_labels(probes, gt_segments=np.array([[18.0, 31.0]], dtype=np.float32), dense_T=80, split="train")
    assert labels
    assert all(row["train_only"] is True for row in labels)
    for split in ("val", "test", "deploy"):
        with pytest.raises(ValueError, match="train-only"):
            build_regret_labels(probes, gt_segments=np.array([[18.0, 31.0]], dtype=np.float32), dense_T=80, split=split)


def test_val_test_selection_allows_downstream_gt_payload_without_regret_labels():
    dense_window = np.arange(80, dtype=np.int64)
    clean_results = {
        "video_name": "clean_val",
        "rba_rbr_preview_actionness": _missed_boundary_case()[0],
        "rba_rbr_preview_uncertainty": _missed_boundary_case()[1],
        "rba_rbr_preview_transition": _missed_boundary_case()[2],
        "gt_segments": np.array([[18.0, 31.0]], dtype=np.float32),
        "gt_labels": np.array([1], dtype=np.int64),
    }
    for split in ("val", "test"):
        selected = build_rba_rbr_open_tad_selection(
            dict(clean_results, video_name=f"clean_{split}"),
            dense_window=dense_window,
            target_frame_num=16,
            split=split,
            train_value_labels=False,
            allow_diagnostic_preview_fallback=False,
        )
        assert selected["ledger"]["value_labels_used_at_test"] is False
        assert selected["ledger"]["train_value_labels_present"] is False
        assert selected["regret_labels"] == []
        assert selected["ledger"]["selector_provenance"]["selection_uses_gt"] is False


def test_val_test_reject_selector_facing_leakage_but_not_plain_gt_payload():
    dense_window = np.arange(80, dtype=np.int64)
    base_results = {
        "video_name": "leakage_check",
        "rba_rbr_preview_actionness": _missed_boundary_case()[0],
        "rba_rbr_preview_uncertainty": _missed_boundary_case()[1],
        "rba_rbr_preview_transition": _missed_boundary_case()[2],
        "gt_segments": np.array([[18.0, 31.0]], dtype=np.float32),
        "gt_labels": np.array([1], dtype=np.int64),
    }
    leakage_cases = [
        {"teacher_logits": [0.1, 0.2]},
        {"proposal_cache": {"path": "forbidden"}},
        {"oracle_boundary": [18, 31]},
        {"raw_detector_predictions": [1.0]},
        {"rba_rbr_selector_provenance": {"selection_uses_gt": True}},
    ]
    for split in ("val", "test"):
        for extra in leakage_cases:
            with pytest.raises(ValueError):
                build_rba_rbr_open_tad_selection(
                    dict(base_results, **extra),
                    dense_window=dense_window,
                    target_frame_num=16,
                    split=split,
                    train_value_labels=False,
                    allow_diagnostic_preview_fallback=False,
                )


def test_open_tad_selection_keeps_train_labels_train_only():
    dense_window = np.arange(80, dtype=np.int64)
    clean_results = {
        "video_name": "clean_train",
        "rba_rbr_preview_actionness": _missed_boundary_case()[0],
        "rba_rbr_preview_uncertainty": _missed_boundary_case()[1],
        "rba_rbr_preview_transition": _missed_boundary_case()[2],
    }
    train_results = dict(clean_results, video_name="train_ok")
    train = build_rba_rbr_open_tad_selection(
        train_results,
        dense_window=dense_window,
        target_frame_num=16,
        split="train",
        gt_segments=np.array([[18.0, 31.0]], dtype=np.float32),
        train_value_labels=True,
        allow_diagnostic_preview_fallback=False,
    )
    assert train["ledger"]["train_value_labels_present"] is True
    assert train["regret_labels"]
    assert validate_no_leakage(train["ledger"]["selector_provenance"])

    with pytest.raises(ValueError, match="train_value_labels"):
        build_rba_rbr_open_tad_selection(
            dict(clean_results, gt_segments=np.array([[18.0, 31.0]], dtype=np.float32)),
            dense_window=dense_window,
            target_frame_num=16,
            split="val",
            gt_segments=np.array([[18.0, 31.0]], dtype=np.float32),
            train_value_labels=True,
            allow_diagnostic_preview_fallback=False,
        )


def test_dynamic_budget_records_variable_k_and_stop_reasons_across_cases():
    ledgers = []
    for offset, peak in [(22, 0.55), (42, 0.85), (58, 0.95)]:
        dense_T = 96
        actionness = np.zeros(dense_T, dtype=np.float64) + 0.05
        actionness[18:32] = peak
        uncertainty = np.zeros(dense_T, dtype=np.float64) + 0.08
        transition = np.zeros(dense_T, dtype=np.float64) + 0.04
        uncertainty[offset : offset + 4] = peak
        transition[offset : offset + 4] = peak
        result = build_rba_rbr_open_tad_selection(
            {
                "video_name": f"case_{offset}",
                "rba_rbr_preview_actionness": actionness,
                "rba_rbr_preview_uncertainty": uncertainty,
                "rba_rbr_preview_transition": transition,
            },
            dense_window=np.arange(dense_T, dtype=np.int64),
            target_frame_num=16,
            split="synthetic",
            train_value_labels=False,
            allow_diagnostic_preview_fallback=False,
        )
        ledgers.append(result["ledger"])

    assert len({row["valid_k"] for row in ledgers}) > 1
    assert all(row["budget_stop_reason"] for row in ledgers)
    assert all(row["selected_positions"] == sorted(set(row["selected_positions"])) for row in ledgers)
