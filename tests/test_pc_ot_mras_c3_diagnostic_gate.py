import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.bata.analyze_actionformer_post_nms_overload import run_post_nms_overload_audit
from tools.bata.analyze_p2_proposal_localization import run_localization_attribution
from tools.bata.analyze_pc_ot_mras_selector_posttrain_diagnostics import analyze_selector_payload
from tools.bata.validate_pc_ot_mras_c3_diagnostic_gate import (
    READY,
    NO_GO,
    validate_c3_diagnostic_gate_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "validate_pc_ot_mras_c3_diagnostic_gate.py"


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _selector_summary():
    return analyze_selector_payload(
        {
            "samples": [
                {
                    "sample_id": "video_a",
                    "selected_dense_indices": [0, 2, 4, 6],
                    "valid_len": 8,
                    "gt_segments": [[1, 5]],
                    "selector_scores": [0.1, 0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.1],
                    "pc_ot_mras_prebackbone_raw_slot_dense_indices": [0, 2, 2, 4],
                    "pc_ot_mras_prebackbone_reader_fill_count": 1,
                    "pc_ot_mras_prebackbone_st_active_row_count": 3,
                    "irregular_selected_positions": [0, 2, 4, 6],
                    "irregular_dense_valid_len": 8,
                    "irregular_selected_valid_len": 8,
                },
                {
                    "sample_id": "video_b",
                    "selected_dense_indices": [1, 3, 5, 7],
                    "valid_len": 8,
                    "gt_segments": [[2, 7]],
                    "selector_scores": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9],
                    "pc_ot_mras_prebackbone_raw_slot_dense_indices": [1, 3, 5, 7],
                    "pc_ot_mras_prebackbone_reader_fill_count": 0,
                    "pc_ot_mras_prebackbone_st_active_row_count": 4,
                    "irregular_selected_positions": [1, 3, 5, 7],
                    "irregular_dense_valid_len": 8,
                    "irregular_selected_valid_len": 8,
                },
            ]
        },
        boundary_radius=1.0,
    )


def _proposal_summary(tmp_path: Path):
    topk = tmp_path / "topk_iou_rank_summary.json"
    reliability = tmp_path / "score_iou_reliability.json"
    _write_json(
        topk,
        {
            "schema_version": "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_READY",
            "diagnostic": "top_k_iou_rank_dump",
            "aggregate": {"top100_iou0p70_gt_recall": {"mean": 0.25}},
            "diagnostic_only": True,
            "uses_validation_gt": True,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
    )
    _write_json(
        reliability,
        {
            "schema_version": "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_READY",
            "diagnostic": "score_vs_iou_reliability",
            "bins": [{"bin_index": 0, "count": 4}],
            "diagnostic_only": True,
            "uses_validation_gt": True,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
    )
    return {
        "schema_version": "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_READY",
        "decision": "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_READY",
        "proposal_rows": 8,
        "joined_rows": 8,
        "topk_iou_rank_summary_json": str(topk),
        "score_iou_reliability_json": str(reliability),
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def _overload_summary():
    return {
        "schema_version": "ACTIONFORMER_POST_NMS_OVERLOAD_AUDIT_READY",
        "decision": "ACTIONFORMER_POST_NMS_OVERLOAD_AUDIT_READY",
        "diagnostic": "actionformer_post_nms_overload_audit",
        "summary": {
            "log_eval_count": 1,
            "latest_predictions": 4000,
            "annotation_validation_video_count": 2,
            "post_processing_max_seg_num": 2000,
            "expected_dataset_prediction_cap": 4000,
            "all_logged_prediction_counts_equal_cap": True,
            "cap_saturation_ratio": 1.0,
            "latest_predictions_per_video": 2000.0,
            "result_detection_counts": {
                "video_count": 2,
                "total_predictions": 4000,
                "per_video_min": 2000,
                "per_video_max": 2000,
                "per_video_mean": 2000.0,
                "per_video_median": 2000.0,
                "per_video_count_histogram": {"2000": 2},
            },
            "result_detection_total_matches_latest_log": True,
        },
        "protocol_flags": {
            "diagnostic_only": True,
            "no_training": True,
            "no_tools_test": True,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_raw_prediction": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
    }


def test_c3_diagnostic_gate_passes_only_when_selector_ranking_and_cap_evidence_exist(tmp_path):
    selector = _selector_summary()
    proposal = _proposal_summary(tmp_path)
    overload = _overload_summary()

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector,
        proposal_summary=proposal,
        overload_summary=overload,
        min_selector_samples=2,
    )

    assert payload["decision"] == READY
    assert payload["gate"]["selector_dump"]["status"] == "PASS"
    assert payload["gate"]["proposal_ranking"]["status"] == "PASS"
    assert payload["gate"]["proposal_cap"]["status"] == "PASS"
    assert payload["gate"]["proposal_cap"]["observed"]["result_detection_cap_hit_video_count"] == 2
    assert payload["gate"]["proposal_cap"]["observed"]["result_detection_cap_hit_video_ratio"] == 1.0
    assert payload["protocol_flags"]["diagnostic_only"] is True
    assert payload["protocol_flags"]["tools_train_allowed"] is False
    assert payload["protocol_flags"]["metric_claim_allowed"] is False


def test_c3_diagnostic_gate_rejects_summaries_without_matching_run_provenance(tmp_path):
    selector = _selector_summary()
    proposal = _proposal_summary(tmp_path)
    overload = _overload_summary()
    run_root = "/run/current_c3"
    work_dir = "/run/current_c3/train_workdir/gpu1_id0"
    train_stdout = "/run/current_c3/current.train.stdout.log"
    result_detection = "/run/current_c3/train_workdir/gpu1_id0/result_detection.json"

    for summary in (selector, proposal, overload):
        summary["provenance"] = {
            "run_root": "/run/old_c3",
            "work_dir": work_dir,
            "train_stdout": train_stdout,
            "result_detection_json": result_detection,
        }
    overload["summary"]["result_detection_counts"]["result_detection_json"] = "/run/old_c3/result_detection.json"

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector,
        proposal_summary=proposal,
        overload_summary=overload,
        min_selector_samples=2,
        expected_run_root=run_root,
        expected_work_dir=work_dir,
        expected_train_stdout=train_stdout,
        expected_result_detection_json=result_detection,
    )

    assert payload["decision"] == NO_GO
    assert payload["gate"]["provenance"]["status"] == "NO_GO"
    assert any("run_root must match current C3 run" in item for item in payload["gate"]["provenance"]["missing"])
    assert any("result_detection_json must match current C3 result file" in item for item in payload["gate"]["provenance"]["missing"])


def test_c3_diagnostic_gate_accepts_actual_producer_outputs_with_current_run_provenance(tmp_path):
    run_root = tmp_path / "run"
    work_dir = run_root / "train_workdir" / "gpu1_id0"
    train_stdout = run_root / "c3.train.stdout.log"
    result_detection = work_dir / "result_detection.json"
    annotation = tmp_path / "anno.json"
    class_map = tmp_path / "category_idx.txt"
    proposal_jsonl = run_root / "proposals.jsonl"
    proposal_out = run_root / "proposal_diag"
    overload_out = run_root / "overload_diag"
    work_dir.mkdir(parents=True)
    run_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_root": str(run_root),
        "work_dir": str(work_dir),
        "train_stdout": str(train_stdout),
        "result_detection_json": str(result_detection),
    }

    selector_summary = analyze_selector_payload(
        {
            "samples": [
                {
                    "sample_id": "video_a",
                    "selected_dense_indices": [0, 2, 4, 6],
                    "valid_len": 8,
                    "gt_segments": [[1, 5]],
                    "selector_scores": [0.1, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1],
                    "pc_ot_mras_prebackbone_raw_slot_dense_indices": [0, 2, 4, 6],
                    "pc_ot_mras_prebackbone_reader_fill_count": 0,
                    "pc_ot_mras_prebackbone_st_active_row_count": 4,
                    "irregular_selected_positions": [0, 2, 4, 6],
                    "irregular_dense_valid_len": 8,
                    "irregular_selected_valid_len": 8,
                }
            ]
        },
        boundary_radius=1.0,
        provenance=provenance,
    )

    _write_json(
        annotation,
        {
            "database": {
                "video_a": {
                    "subset": "validation",
                    "duration": 10.0,
                    "annotations": [{"segment": [1.0, 3.0], "label": "A"}],
                },
                "video_b": {
                    "subset": "validation",
                    "duration": 10.0,
                    "annotations": [{"segment": [2.0, 4.0], "label": "A"}],
                },
            }
        },
    )
    class_map.write_text("0 A\n", encoding="utf-8")
    _write_jsonl(
        proposal_jsonl,
        [
            {
                "video_id": "video_a",
                "sample_id": "video_a|window_start_frame=0",
                "class_id": 0,
                "label": "A",
                "fps": 10.0,
                "snippet_stride": 2.0,
                "window_start_frame": 0.0,
                "offset_frames": 0.0,
                "window_size": 64.0,
                "window_start_seconds": 0.0,
                "window_end_seconds": 12.8,
                "duration_seconds": 10.0,
                "segment": [5.0, 15.0],
                "segment_seconds": [1.0, 3.0],
                "segment_frames": [10.0, 30.0],
                "final_score": 0.9,
                "duration": 10.0,
            }
        ],
    )
    proposal_summary = run_localization_attribution(
        proposal_jsonl=proposal_jsonl,
        output_dir=proposal_out,
        annotation=annotation,
        class_map=class_map,
        topk=(1,),
        iou_thresholds=(0.5,),
        score_bins=1,
        provenance=provenance,
    )

    train_stdout.write_text(
        "\n".join(
            [
                "2026-06-24 00:00:00 Train INFO: Number of predictions: 4",
                "2026-06-24 00:00:00 Train INFO: Average-mAP: 1.00 (%)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "cfg.py"
    config.write_text("post_processing = dict(nms=dict(max_seg_num=2))\n", encoding="utf-8")
    _write_json(
        result_detection,
        {
            "results": {
                "video_a": [
                    {"segment": [1.0, 2.0], "label": "A", "score": 0.9},
                    {"segment": [2.0, 3.0], "label": "A", "score": 0.8},
                ],
                "video_b": [
                    {"segment": [2.0, 3.0], "label": "A", "score": 0.7},
                    {"segment": [3.0, 4.0], "label": "A", "score": 0.6},
                ],
            }
        },
    )
    overload_summary = run_post_nms_overload_audit(
        train_log=train_stdout,
        annotation=annotation,
        output_dir=overload_out,
        config=config,
        result_detection_json=result_detection,
        provenance=provenance,
    )

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector_summary,
        proposal_summary=proposal_summary,
        overload_summary=overload_summary,
        min_selector_samples=1,
        expected_run_root=str(run_root),
        expected_work_dir=str(work_dir),
        expected_train_stdout=str(train_stdout),
        expected_result_detection_json=str(result_detection),
    )

    assert payload["decision"] == READY
    assert payload["gate"]["provenance"]["status"] == "PASS"


@pytest.mark.parametrize(
    "case_name,expected_missing",
    [
        ("logged_counts_below_cap", "every logged prediction count must equal proposal cap"),
        ("cap_saturation_below_one", "proposal cap saturation ratio must be 1.0"),
        ("partial_result_detection_cap_hit", "every result_detection video must hit proposal cap"),
    ],
)
def test_c3_diagnostic_gate_rejects_under_cap_or_partial_video_cap(tmp_path, case_name, expected_missing):
    selector = _selector_summary()
    proposal = _proposal_summary(tmp_path)
    overload = _overload_summary()
    summary = overload["summary"]

    if case_name == "logged_counts_below_cap":
        summary["all_logged_prediction_counts_equal_cap"] = False
    elif case_name == "cap_saturation_below_one":
        summary["cap_saturation_ratio"] = 0.75
    elif case_name == "partial_result_detection_cap_hit":
        summary["result_detection_counts"]["per_video_count_histogram"] = {"2000": 1, "1000": 1}
        summary["result_detection_counts"]["per_video_min"] = 1000
        summary["result_detection_counts"]["per_video_mean"] = 1500.0
    else:
        raise AssertionError(case_name)

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector,
        proposal_summary=proposal,
        overload_summary=overload,
        min_selector_samples=2,
    )

    assert payload["decision"] == NO_GO
    assert payload["gate"]["proposal_cap"]["status"] == "NO_GO"
    assert any(expected_missing in item for item in payload["gate"]["proposal_cap"]["missing"])


@pytest.mark.parametrize(
    "case_name,expected_missing",
    [
        ("ready_with_warnings", "selector diagnostic decision must be strict READY"),
        ("non_finite", "selector dump must not contain NaN/Inf values"),
        ("metadata_inconsistent", "selector metadata consistency must be clean"),
    ],
)
def test_c3_diagnostic_gate_rejects_selector_warnings_nonfinite_or_metadata_inconsistency(
    tmp_path,
    case_name,
    expected_missing,
):
    selector = _selector_summary()
    proposal = _proposal_summary(tmp_path)
    overload = _overload_summary()

    if case_name == "ready_with_warnings":
        selector["decision"] = "PC_OT_MRAS_SELECTOR_POSTTRAIN_DIAGNOSTIC_READY_WITH_WARNINGS"
    elif case_name == "non_finite":
        selector["non_finite"] = {"count": 1, "examples": [{"path": "$.samples[0].debug_nan", "value": "nan"}]}
    elif case_name == "metadata_inconsistent":
        selector.setdefault("aggregate", {}).setdefault("metadata_consistency", {})[
            "inconsistent_sample_count"
        ] = 1
        selector["aggregate"]["metadata_consistency"]["inconsistent_sample_ids"] = ["video_a"]
    else:
        raise AssertionError(case_name)

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector,
        proposal_summary=proposal,
        overload_summary=overload,
        min_selector_samples=2,
    )

    assert payload["decision"] == NO_GO
    assert payload["gate"]["selector_dump"]["status"] == "NO_GO"
    assert any(expected_missing in item for item in payload["gate"]["selector_dump"]["missing"])


def test_c3_diagnostic_gate_rejects_dataset_cap_without_per_video_counts(tmp_path):
    selector = _selector_summary()
    proposal = _proposal_summary(tmp_path)
    overload = _overload_summary()
    del overload["summary"]["result_detection_counts"]
    del overload["summary"]["result_detection_total_matches_latest_log"]

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector,
        proposal_summary=proposal,
        overload_summary=overload,
        min_selector_samples=2,
    )

    assert payload["decision"] == NO_GO
    assert payload["gate"]["proposal_cap"]["status"] == "NO_GO"
    assert any("per-video result_detection proposal counts" in item for item in payload["gate"]["proposal_cap"]["missing"])


def test_c3_diagnostic_gate_rejects_non_ready_source_decisions(tmp_path):
    selector = _selector_summary()
    proposal = _proposal_summary(tmp_path)
    overload = _overload_summary()
    proposal["decision"] = "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_WARN_ONLY"
    overload["decision"] = "ACTIONFORMER_POST_NMS_OVERLOAD_AUDIT_WARN_ONLY"

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector,
        proposal_summary=proposal,
        overload_summary=overload,
        min_selector_samples=2,
    )

    assert payload["decision"] == NO_GO
    assert payload["gate"]["proposal_ranking"]["status"] == "NO_GO"
    assert payload["gate"]["proposal_cap"]["status"] == "NO_GO"
    assert any("must be a recognized READY decision" in item for item in payload["gate"]["proposal_ranking"]["missing"])
    assert any("must be a recognized READY decision" in item for item in payload["gate"]["proposal_cap"]["missing"])


def test_c3_diagnostic_gate_fails_without_raw_slot_and_proposal_cap_evidence(tmp_path):
    selector = _selector_summary()
    del selector["aggregate"]["slot_transport"]
    proposal = _proposal_summary(tmp_path)

    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=selector,
        proposal_summary=proposal,
        overload_summary=None,
        min_selector_samples=2,
    )

    assert payload["decision"] == NO_GO
    assert payload["gate"]["selector_dump"]["status"] == "NO_GO"
    assert payload["gate"]["proposal_cap"]["status"] == "NO_GO"
    assert any("raw_slot_duplicate" in item for item in payload["gate"]["selector_dump"]["missing"])
    assert any("proposal cap" in item for item in payload["gate"]["proposal_cap"]["missing"])


def test_c3_diagnostic_gate_cli_writes_summary(tmp_path):
    selector_path = tmp_path / "selector_summary.json"
    proposal_path = tmp_path / "proposal_summary.json"
    overload_path = tmp_path / "overload_summary.json"
    output_path = tmp_path / "gate.json"
    _write_json(selector_path, _selector_summary())
    _write_json(proposal_path, _proposal_summary(tmp_path))
    _write_json(overload_path, _overload_summary())

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--selector-summary",
            str(selector_path),
            "--proposal-summary",
            str(proposal_path),
            "--overload-summary",
            str(overload_path),
            "--output",
            str(output_path),
            "--min-selector-samples",
            "2",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    stdout_payload = json.loads(proc.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["decision"] == READY
