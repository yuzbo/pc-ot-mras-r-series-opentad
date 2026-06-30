import json

import pytest

from opentad.acquisition.abr import (
    ABRConfig,
    ABR_ROUTE_LABEL,
    BracketState,
    select_active_bracket_refinement,
)
from opentad.acquisition.abr.policy import (
    refine_or_split_bracket,
    state_at_position,
    two_sided_witness_missing,
)
from opentad.acquisition.abr.validators import (
    ABRValidationError,
    assert_no_forbidden_route_tokens,
    assert_real_sparse_handoff,
)


def test_route_identity_is_isolated_from_forbidden_route_tokens():
    assert ABR_ROUTE_LABEL == "DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3"
    assert_no_forbidden_route_tokens({"route_label": ABR_ROUTE_LABEL, "method": "abr_active_bracket_refinement"})
    with pytest.raises(ABRValidationError):
        assert_no_forbidden_route_tokens({"method": "abr_active_bracket_refinement", "notes": "use GlobalRank helper"})
    with pytest.raises(ABRValidationError):
        old_label = "DIVERGENT_INNOVATION_" + "BOUNDARY" + "_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
        assert_no_forbidden_route_tokens({"route_label": old_label})


def test_selection_outputs_sorted_unique_original_positions_and_no_leakage():
    curve = [0.02] * 20 + [0.85] * 10 + [0.08] * 22 + [0.9] * 8 + [0.03] * 20
    result = select_active_bracket_refinement(
        dense_t=len(curve),
        fps=25.0,
        video_id="synthetic_two_action",
        scout_curve=curve,
        config=ABRConfig(k0=8, k1_cap=12, k2_cap=4, max_total_k=28, max_gap=14, deadline_ms=50.0),
    )

    assert result.route_label == ABR_ROUTE_LABEL
    assert result.selected_positions == sorted(set(result.selected_positions))
    assert min(result.selected_positions) >= 0
    assert max(result.selected_positions) < len(curve)
    assert len(result.selected_positions) == result.valid_k
    assert result.provenance == {
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "uses_detector_feedback": False,
        "dense_raw_backbone_handoff": False,
        "selected_inputs_is_gathered": True,
    }
    assert result.cost.detector_forward_count == 1
    assert result.scout_source == "unspecified"
    assert result.diagnostic_fallback_used is False
    assert result.cost.selected_k == result.valid_k
    assert len({ledger.round_id for ledger in result.round_ledgers}) >= 2
    assert all(ledger.cumulative_k <= result.config.max_total_k for ledger in result.round_ledgers)
    round0 = result.round_ledgers[0].diagnostics
    assert round0["bracket_policy"] == "deploy_visible_multiscale_graydiff_bracket_v2"
    assert round0["bracket_policy_inputs"] == {
        "deploy_visible_scout_curve": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "uses_detector_feedback": False,
    }
    assert "multiscale_peak_brackets" in round0["bracket_policy_mechanisms"]
    assert all(bracket.left <= bracket.right for bracket in result.brackets)
    assert [(bracket.left, bracket.right, bracket.bracket_id) for bracket in result.brackets] == sorted(
        (bracket.left, bracket.right, bracket.bracket_id) for bracket in result.brackets
    )


def test_dynamic_k_is_nonconstant_on_synthetic_easy_and_boundary_cases():
    easy = [0.05] * 96
    boundary_rich = (
        [0.02] * 10
        + [0.82] * 5
        + [0.14] * 8
        + [0.91] * 6
        + [0.08] * 11
        + [0.88] * 7
        + [0.03] * 49
    )
    cfg = ABRConfig(k0=8, k1_cap=14, k2_cap=6, max_total_k=32, max_gap=16, round2_enabled=True)
    easy_result = select_active_bracket_refinement(96, scout_curve=easy, config=cfg)
    rich_result = select_active_bracket_refinement(96, scout_curve=boundary_rich, config=cfg)

    assert easy_result.valid_k < rich_result.valid_k
    assert easy_result.cost.stop_reason in {"no_brackets", "saturated", "budget_cap", "deadline"}
    assert rich_result.cost.rounds_used >= easy_result.cost.rounds_used


def test_bracket_narrowing_split_stale_and_resolved_states():
    cfg = ABRConfig(resolve_width=2, keep_stale_confidence=0.20)

    single = BracketState(bracket_id=1, kind="start", left=10, right=20, confidence=0.9, uncertainty=0.7)
    narrowed = refine_or_split_bracket(
        single,
        observations={10: "background", 14: "background", 16: "action", 20: "action"},
        config=cfg,
        round_id=1,
    )
    assert len(narrowed) == 1
    assert narrowed[0].left == 14
    assert narrowed[0].right == 16
    assert narrowed[0].status == "resolved"

    multi = BracketState(bracket_id=2, kind="unknown", left=30, right=50, confidence=0.8, uncertainty=0.8)
    children = refine_or_split_bracket(
        multi,
        observations={30: "background", 34: "action", 40: "background", 46: "action", 50: "action"},
        config=cfg,
        round_id=1,
    )
    assert len(children) == 2
    assert {child.status for child in children} == {"active"}
    assert all(child.parent_id == 2 for child in children)

    stale = BracketState(bracket_id=3, kind="end", left=60, right=70, confidence=0.7, uncertainty=0.3)
    stale_result = refine_or_split_bracket(
        stale,
        observations={60: "background", 65: "background", 70: "background"},
        config=cfg,
        round_id=1,
    )
    assert stale_result[0].status == "stale"
    assert stale_result[0].confidence < 0.7


def test_two_sided_witness_logic_and_state_thresholds():
    start = BracketState(
        bracket_id=10,
        kind="start",
        left=8,
        right=16,
        has_pre_background_witness=False,
        has_action_core_witness=True,
        has_post_background_witness=True,
    )
    assert two_sided_witness_missing(start)
    start.has_pre_background_witness = True
    assert not two_sided_witness_missing(start)
    assert state_at_position([0.1, 0.45, 0.8], 0) == "background"
    assert state_at_position([0.1, 0.45, 0.8], 1) == "ambiguous"
    assert state_at_position([0.1, 0.45, 0.8], 2) == "action"


def test_round2_is_bounded_and_only_for_unresolved_high_value_brackets():
    curve = [0.05] * 18 + [0.42, 0.48, 0.52, 0.49, 0.55, 0.46, 0.58] * 6 + [0.9] * 10 + [0.04] * 20
    result = select_active_bracket_refinement(
        dense_t=len(curve),
        scout_curve=curve,
        config=ABRConfig(k0=7, k1_cap=8, k2_cap=2, max_total_k=19, max_gap=18, round2_enabled=True),
    )

    round2 = [ledger for ledger in result.round_ledgers if ledger.round_id == 2]
    assert len(round2) <= 1
    if round2:
        assert len(round2[0].selected_positions) <= 2
        assert round2[0].selected_source == ["round2_probe"] * len(round2[0].selected_positions)
    assert result.valid_k <= 19


def test_real_sparse_handoff_validator_rejects_padding_as_valid_and_dense_handoff():
    handoff = {
        "selected_positions_window_local": [0, 3, 7, 12],
        "selected_positions_original_dense": [10, 13, 17, 22],
        "selected_mask": [True, True, True, False],
        "dense_T": 16,
        "valid_k": 3,
        "dense_window": list(range(10, 26)),
        "frame_inds_raw": [10, 13, 17, 17],
        "provenance": {
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "uses_detector_feedback": False,
            "dense_raw_backbone_handoff": False,
            "selected_inputs_is_gathered": True,
        },
    }
    assert_real_sparse_handoff(handoff)

    bad = json.loads(json.dumps(handoff))
    bad["valid_k"] = 4
    with pytest.raises(ABRValidationError):
        assert_real_sparse_handoff(bad)

    bad = json.loads(json.dumps(handoff))
    bad["provenance"]["dense_raw_backbone_handoff"] = True
    with pytest.raises(ABRValidationError):
        assert_real_sparse_handoff(bad)

    bad = json.loads(json.dumps(handoff))
    bad["frame_inds_raw"][1] = 99
    with pytest.raises(ABRValidationError):
        assert_real_sparse_handoff(bad)

    bad = json.loads(json.dumps(handoff))
    bad.pop("selected_positions_original_dense")
    with pytest.raises(ABRValidationError):
        assert_real_sparse_handoff(bad)
