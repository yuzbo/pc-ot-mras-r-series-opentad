from pathlib import Path

from tools.bata.diagnose_pc_ot_mras_r17_r18_mechanisms import (
    classify_summary_verdict,
    geometry_contract_audit,
    proposal_survival_audit,
    reader_selection_utility,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reader_selection_utility_uses_high_score_unique_positions():
    reader_out = {
        "valid_mask": [[1, 1, 1, 1, 1, 0]],
        "acquisition_matrix": [
            [
                [0.1, 0.90, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.85, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.00, 0.0, 0.8, 0.1, 0.0],
            ]
        ],
    }

    audit = reader_selection_utility(
        reader_out,
        batch_idx=0,
        batch_size=1,
        budget=3,
        gt_segments=[[1.0, 3.0]],
        boundary_radius=0.5,
    )

    assert audit["matrix_key"] == "acquisition_matrix"
    assert audit["selected_positions"] == [0, 1, 3]
    assert audit["duplicate_slot_count"] == 1
    assert audit["selected_count"] == 3
    assert audit["gt_coverage"]["gt_touched_fraction"] == 1.0
    assert audit["gt_coverage"]["boundary_touched_fraction"] == 1.0


def test_geometry_contract_audit_flags_bad_temporal_metadata_and_proposals():
    meta = {
        "irregular_native_axis": True,
        "irregular_selected_positions": [0.0, 4.0, 3.0],
        "irregular_selected_count": 2,
        "irregular_selected_valid_len": 4.0,
    }
    d1 = {"selected_positions": [0, 3, 4]}
    proposal_rows = [
        {"start_coord": 3.5, "end_coord": 4.5},
        {"start_coord": 2.0, "end_coord": 1.5},
    ]

    audit = geometry_contract_audit(meta, d1, proposal_rows)

    assert "unsorted_irregular_selected_positions" in audit["failure_flags"]
    assert "selected_count_metadata_mismatch" in audit["failure_flags"]
    assert "proposal_outside_selected_valid_len" in audit["failure_flags"]
    assert "proposal_non_positive_duration" in audit["failure_flags"]
    assert audit["proposal_outside_valid_len_count"] == 1
    assert audit["proposal_non_positive_duration_count"] == 1


def test_proposal_survival_audit_reports_score_iou_signal_and_topk_hits():
    rows = [
        {"final_score": 0.9, "diagnostic_max_gt_iou": 0.8, "level": 0},
        {"final_score": 0.5, "diagnostic_max_gt_iou": 0.4, "level": 1},
        {"final_score": 0.1, "diagnostic_max_gt_iou": 0.0, "level": 1},
    ]

    audit = proposal_survival_audit(rows, topk=(1, 3), iou_thresholds=(0.5, 0.7), score_bins=5)

    assert audit["candidate_count"] == 3
    assert audit["level_counts"] == {"0": 1, "1": 2}
    assert audit["topk_any_iou_at_threshold"]["1"]["0.7"] is True
    assert audit["topk_any_iou_at_threshold"]["3"]["0.5"] is True
    assert audit["score_iou_pearson"] is not None
    assert audit["score_iou_pearson"] > 0.9
    assert audit["post_nms_available"] is False


def test_summary_verdict_prioritizes_d2_then_d3_then_d1():
    d1_good = [{"gt_coverage": {"status": "computed_on_batch_gt_axis_assumed_compatible", "boundary_touched_fraction": 1.0}}]
    d3_good = [{"candidate_count": 10}]
    d2_bad = [{"failure_flags": ["missing_irregular_selected_positions"]}]

    assert classify_summary_verdict(all_d1=d1_good, all_d2=d2_bad, all_d3=d3_good)["primary_failure_mode"] == "D2_bad"

    d2_ok = [{"failure_flags": []}]
    d3_bad = [{"candidate_count": 0}, {"candidate_count": 0}]
    assert classify_summary_verdict(all_d1=d1_good, all_d2=d2_ok, all_d3=d3_bad)["primary_failure_mode"] == "D3_bad"

    d1_low_boundary = [
        {"gt_coverage": {"status": "computed_on_batch_gt_axis_assumed_compatible", "boundary_touched_fraction": 0.0}}
    ]
    verdict = classify_summary_verdict(all_d1=d1_low_boundary, all_d2=d2_ok, all_d3=d3_good)
    assert verdict["primary_failure_mode"] == "D1_suspect"
    assert verdict["verdict"] == "warning"


def test_n16r4_launcher_is_diagnostic_only_and_precheck_gated():
    launcher = ROOT / "scripts" / "run_pc_ot_mras_r17_r18_mechanism_diagnostics_n16r4.sbatch"
    text = launcher.read_text(encoding="utf-8")

    assert "tools/bata/diagnose_pc_ot_mras_r17_r18_mechanisms.py" in text
    assert "ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py" in text
    assert "ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py" in text
    assert "tools/train.py" not in text
    assert "tools/test.py" not in text
    assert "ALLOW_PCOTMRAS_R17_R18_MECH_DIAG" in text
    assert "PRECHECK_ONLY" in text
    assert text.index('if [ "$PRECHECK_ONLY" = "1" ]') < text.index("verify_file R17_MECH_CHECKPOINT")
    assert "[[ ! \"$RUN_TAG\" =~ ^[A-Za-z0-9._-]+$ ]]" in text
