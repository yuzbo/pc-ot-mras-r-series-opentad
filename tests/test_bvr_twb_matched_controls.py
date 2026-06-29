from collections import Counter

import pytest

from opentad.acquisition.bvr_twb.matched_controls import build_matched_controls
from opentad.acquisition.bvr_twb.validators import validate_deploy_ledger
from tools.bvr_twb.build_synthetic_ledgers import run_bvr_case, safe_prepare_output_dir, synthetic_cases


def _build_smoke_inputs():
    bvr_ledgers = []
    candidates_by_video = {}
    dense_inputs_by_video = {}
    for name, p_action, motion in synthetic_cases()[:4]:
        result, candidates, dense_inputs = run_bvr_case(name, p_action, motion)
        bvr_ledgers.append(result.deploy_ledger)
        candidates_by_video[name] = candidates
        dense_inputs_by_video[name] = dense_inputs
    return bvr_ledgers, candidates_by_video, dense_inputs_by_video


def test_matched_controls_cover_required_families_and_validate_same_path():
    bvr_ledgers, candidates_by_video, dense_inputs_by_video = _build_smoke_inputs()
    controls = build_matched_controls(
        bvr_ledgers,
        candidate_packets_by_video=candidates_by_video,
        dense_inputs_by_video=dense_inputs_by_video,
        scaffold_k=4,
        max_gap=22,
    )
    counts = Counter(row["control_name"] for row in controls)
    assert counts == {
        "same_k_uniform": 4,
        "same_mean_k_exact_uniform": 4,
        "random_same_k": 4,
        "scaffold_only": 4,
        "twb_no_regret": 4,
    }
    for row in controls:
        assert validate_deploy_ledger(row)
        assert row["uses_same_gather_time_validator_path"] is True
        assert row["dense_handoff_used"] is False
        assert row["real_sparse_evidence"]["status"] == "local_gather_smoke_only"
        assert row["claim_mode"] == "local_gather_smoke"
        assert row["selection_gap_diagnostics"]["max_gap"] <= row["selection_gap_diagnostics"]["max_allowed_gap"]
        if row["control_name"] == "random_same_k":
            assert isinstance(row["control_random_seed"], int)
            assert row["control_jitter"] >= 0
        if row["control_name"] == "scaffold_only":
            assert row["control_policy"] == "scaffold_only_max_gap_repaired"
        if row["control_name"] == "twb_no_regret":
            assert 0.0 <= row["uniform_fallback_ratio"] <= 1.0


def test_same_k_controls_match_bvr_k_and_share_original_time_axis():
    bvr_ledgers, candidates_by_video, dense_inputs_by_video = _build_smoke_inputs()
    controls = build_matched_controls(
        bvr_ledgers,
        candidate_packets_by_video=candidates_by_video,
        dense_inputs_by_video=dense_inputs_by_video,
    )
    by_video = {(row["video_id"], row["control_name"]): row for row in controls}
    for bvr in bvr_ledgers:
        same_k = by_video[(bvr["video_id"], "same_k_uniform")]
        random_k = by_video[(bvr["video_id"], "random_same_k")]
        twb = by_video[(bvr["video_id"], "twb_no_regret")]
        assert same_k["valid_k"] == bvr["valid_k"]
        assert random_k["valid_k"] == bvr["valid_k"]
        assert twb["valid_k"] == bvr["valid_k"]
        for row in (same_k, random_k, twb):
            assert row["selected_positions_unit"] == "original_dense_index"
            assert row["original_time_metadata"]["time_axis_mode"] == "irregular_original_time"
            assert row["original_time_metadata"]["selected_index_is_time"] is False


def test_matched_controls_inherit_per_case_bvr_max_gap_by_default():
    bvr_ledgers, candidates_by_video, dense_inputs_by_video = _build_smoke_inputs()
    for idx, row in enumerate(bvr_ledgers):
        row["selection_gap_diagnostics"] = dict(row["selection_gap_diagnostics"], max_allowed_gap=18 + idx)
    controls = build_matched_controls(
        bvr_ledgers,
        candidate_packets_by_video=candidates_by_video,
        dense_inputs_by_video=dense_inputs_by_video,
        random_seed=123,
        max_gap=None,
    )
    by_video = {(row["video_id"], row["control_name"]): row for row in controls}
    for bvr in bvr_ledgers:
        expected = bvr["selection_gap_diagnostics"]["max_allowed_gap"]
        for name in ("same_k_uniform", "same_mean_k_exact_uniform", "random_same_k", "scaffold_only", "twb_no_regret"):
            assert by_video[(bvr["video_id"], name)]["selection_gap_diagnostics"]["max_allowed_gap"] == expected


def test_per_case_uniform_overlap_gate_reports_each_case():
    bvr_ledgers, candidates_by_video, dense_inputs_by_video = _build_smoke_inputs()
    controls = build_matched_controls(
        bvr_ledgers,
        candidate_packets_by_video=candidates_by_video,
        dense_inputs_by_video=dense_inputs_by_video,
        max_gap=22,
    )
    for row in controls:
        assert "uniform_overlap_ratio" in row
        assert 0.0 <= row["uniform_overlap_ratio"] <= 1.0


def test_synthetic_tool_overwrite_guard_rejects_unsafe_paths(tmp_path):
    with pytest.raises(ValueError, match="outside worktree"):
        safe_prepare_output_dir(tmp_path / "bvr_twb_out", overwrite=True)
    with pytest.raises(ValueError, match="without bvr_twb marker"):
        safe_prepare_output_dir(tmp_path / "plain_output", overwrite=True, root=tmp_path)
    allowed = safe_prepare_output_dir(tmp_path / ".tmp_bvr_twb_safe", overwrite=True, root=tmp_path)
    assert allowed.exists()
