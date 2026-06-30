import json
import subprocess
import sys
from pathlib import Path

import pytest

from opentad.acquisition.abr import ABRConfig, ABR_ROUTE_LABEL
from opentad.acquisition.abr.validators import ABRValidationError
from tools.abr.audit_abr_first_round_bracket_recall import (
    _build_window_cases,
    assert_selector_payload_is_deploy_visible,
    build_selector_payload,
    run_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "tools" / "abr" / "audit_abr_first_round_bracket_recall.py"


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_bom_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8-sig")
    return path


def _annotation_payload(annotations):
    return {
        "database": {
            "video_0001": {
                "subset": "validation",
                "duration": 12.0,
                "frame": 12,
                "annotations": annotations,
            }
        }
    }


def _scout_payload(curve=None, **extra):
    payload = {
        "videos": {
            "video_0001": {
                "deploy_visible_scout_curve": curve
                if curve is not None
                else [0.05, 0.05, 0.9, 0.9, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05, 0.05, 0.05],
                "scout_source": "unit_deploy_visible_scout",
                **extra,
            }
        }
    }
    return payload


def test_selector_payload_rejects_gt_teacher_cache_detector_and_dense_handoff():
    clean = build_selector_payload(
        {"deploy_visible_scout_curve": [0.1, 0.9, 0.1], "total_frames": 3, "fps": 25.0},
        "video_clean",
        "window0",
        3,
    )
    assert clean["video_name"] == "video_clean"
    assert "gt_segments" not in clean
    assert "gt_labels" not in clean
    assert_selector_payload_is_deploy_visible(clean)

    for key in (
        "gt_segments",
        "gt_labels",
        "teacher_logits",
        "prediction_cache",
        "raw_detector_outputs",
        "dense_backbone_handoff",
    ):
        with pytest.raises(ABRValidationError):
            build_selector_payload(
                {"deploy_visible_scout_curve": [0.1, 0.9, 0.1], key: [1]},
                "video_leaky",
                "window0",
                3,
            )


def test_output_schema_claim_locks_and_recall_math_on_deploy_visible_scout(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload(
            [
                {"segment": [2.0, 4.0], "label": "BaseballPitch"},
                {"segment": [6.0, 6.5], "label": "GolfSwing"},
                {"segment": [8.0, 9.0], "label": "Ambiguous"},
            ]
        ),
    )
    scout = _write_json(tmp_path / "scout.json", _scout_payload())

    payload = run_audit(
        ann,
        scout,
        abr_config=ABRConfig(k0=12, k1_cap=0, k2_cap=0, max_total_k=12, max_gap=1, round2_enabled=False),
    )

    assert payload["route_label"] == ABR_ROUTE_LABEL
    assert payload["diagnostic_only"] is True
    assert payload["no_detector"] is True
    assert payload["no_training"] is True
    assert payload["selector_gt_visible"] is False
    assert payload["selector_payload_gt_keys_stripped"] is True
    assert payload["formal_thresholds"] == {
        "min_first_round_bracket_recall": 0.95,
        "min_first_round_transition_coverage": 0.95,
    }
    assert payload["formal_gate_passed"] is True
    assert payload["video_count"] == 1
    assert payload["window_count"] == 1
    assert payload["transition_count"] == 4
    assert payload["bracketed_transition_count"] == 4
    assert payload["missed_transition_count"] == 0
    assert payload["first_round_bracket_recall"] == pytest.approx(1.0)
    assert payload["first_round_transition_coverage"] == pytest.approx(1.0)
    assert payload["temporal_coverage_fraction"] > 0.0
    assert payload["bracket_width_stats"]["count"] > 0
    assert payload["false_positive_bracket_density"] >= 0.0
    assert payload["ambiguous_transition_count"] == 2
    assert payload["short_action_stratified_recall"]["short_le_1s"]["transition_count"] == 2
    assert payload["short_action_stratified_recall"]["short_le_1s"]["recall"] == pytest.approx(1.0)
    assert payload["class_misses"] == {}
    assert payload["window_misses"] == []
    assert payload["allowed_next_action"] == "FORMAL_REVIEW_PACKET_ONLY_WITH_REAL_SCOUT_RECALL_EVIDENCE"
    for key in (
        "formal_full_train_unlocked",
        "tools_test_allowed",
        "official_map_claim",
        "runtime_flops_claim",
        "deploy_claim",
        "sparse_compute_claim",
        "paper_claim",
    ):
        assert payload[key] is False


def test_selector_payload_has_no_gt_derived_ambiguous_or_scoring_fields(tmp_path):
    videos = {
        "video_0001": {
            "duration": 12.0,
            "frame": 12,
            "annotations": [
                {"segment": [2.0, 4.0], "label": "BaseballPitch"},
                {"segment": [8.0, 9.0], "label": "Ambiguous"},
            ],
        }
    }
    scout_records = {
        "video_0001": {
            "deploy_visible_scout_curve": [0.05, 0.05, 0.9, 0.9, 0.05, 0.05],
            "scout_source": "unit_deploy_visible_scout",
        }
    }
    cases = _build_window_cases(
        videos,
        scout_records,
        ABRConfig(k0=6, k1_cap=0, k2_cap=0, max_total_k=6, max_gap=1, round2_enabled=False),
        window_size=0,
        window_overlap_ratio=0.5,
        max_windows=None,
        allow_diagnostic_fallback_scout=False,
        fallback_stage="DIAGNOSTIC_ONLY",
    )

    assert len(cases) == 1
    selector_payload = cases[0].selector_payload
    forbidden_gt_derived = {
        "gt_segments",
        "gt_labels",
        "gt_instances",
        "ambiguous_transition_count_after_selection_only",
        "ambiguous_transition_count",
        "missed_transition_count",
        "bracketed_transition_count",
        "first_round_bracket_recall",
        "first_round_transition_coverage",
    }
    assert forbidden_gt_derived.isdisjoint(selector_payload.keys())
    assert_selector_payload_is_deploy_visible(selector_payload)

    ann = _write_json(tmp_path / "ann.json", _annotation_payload(videos["video_0001"]["annotations"]))
    scout = _write_json(tmp_path / "scout.json", _scout_payload(scout_records["video_0001"]["deploy_visible_scout_curve"]))
    payload = run_audit(
        ann,
        scout,
        abr_config=ABRConfig(k0=6, k1_cap=0, k2_cap=0, max_total_k=6, max_gap=1, round2_enabled=False),
    )
    assert payload["ambiguous_transition_count"] == 2


def test_recall_math_records_missed_class_video_and_window_entries(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    scout = _write_json(tmp_path / "scout.json", _scout_payload())

    payload = run_audit(
        ann,
        scout,
        abr_config=ABRConfig(k0=1, k1_cap=0, k2_cap=0, max_total_k=1, max_gap=0, round2_enabled=False),
    )

    assert payload["transition_count"] == 2
    assert payload["bracketed_transition_count"] == 0
    assert payload["missed_transition_count"] == 2
    assert payload["first_round_bracket_recall"] == pytest.approx(0.0)
    assert payload["first_round_transition_coverage"] == pytest.approx(0.0)
    assert payload["class_misses"] == {"BaseballPitch": 2}
    assert payload["video_misses"] == {"video_0001": 2}
    assert payload["window_misses"][0]["missed_transition_count"] == 2


def test_zero_transition_pseudo_perfect_is_rejected(tmp_path):
    ann = _write_json(tmp_path / "ann.json", _annotation_payload([]))
    scout = _write_json(tmp_path / "scout.json", _scout_payload())

    payload = run_audit(
        ann,
        scout,
        abr_config=ABRConfig(k0=12, k1_cap=0, k2_cap=0, max_total_k=12, max_gap=1, round2_enabled=False),
    )

    assert payload["transition_count"] == 0
    assert payload["first_round_bracket_recall"] == 0.0
    assert payload["first_round_transition_coverage"] == 0.0
    assert payload["zero_transition_pseudo_perfect_rejected"] is True
    assert payload["real_deploy_visible_recall_evidence"] is False
    assert payload["formal_thresholds"] == {
        "min_first_round_bracket_recall": 0.95,
        "min_first_round_transition_coverage": 0.95,
    }
    assert payload["formal_gate_passed"] is False
    assert payload["allowed_next_action"] == "LOCKED_ZERO_TRANSITION_NO_REAL_RECALL_EVIDENCE"


def test_missing_scout_fails_closed_unless_fallback_is_explicit_and_non_evidence(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )

    with pytest.raises(ABRValidationError, match="no deploy-visible scout"):
        run_audit(ann)

    payload = run_audit(
        ann,
        allow_diagnostic_fallback_scout=True,
        fallback_stage="DIAGNOSTIC_ONLY",
        abr_config=ABRConfig(
            k0=4,
            k1_cap=0,
            k2_cap=0,
            max_total_k=4,
            max_gap=8,
            round2_enabled=False,
            allow_diagnostic_fallback_scout=True,
            fallback_stage="DIAGNOSTIC_ONLY",
        ),
    )
    assert payload["diagnostic_fallback_used"] is True
    assert payload["real_deploy_visible_recall_evidence"] is False
    assert payload["allowed_next_action"] == "LOCKED_DIAGNOSTIC_FALLBACK_NOT_REAL_RECALL_EVIDENCE"


def test_forbidden_route_tokens_in_scout_source_are_rejected(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    scout = _write_json(tmp_path / "scout.json", _scout_payload(scout_source="GlobalRank shortcut"))

    with pytest.raises(ABRValidationError):
        run_audit(ann, scout)


def test_scout_record_with_annotations_is_rejected_as_not_deploy_visible(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    scout = _write_json(
        tmp_path / "scout_with_annotations.json",
        _scout_payload(annotations=[{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )

    with pytest.raises(ABRValidationError, match="annotations"):
        run_audit(ann, scout)


def test_fail_closed_missing_or_ambiguous_artifacts(tmp_path):
    invalid_ann = _write_json(
        tmp_path / "invalid_ann.json",
        _annotation_payload([{"segment": [2.0], "label": "BaseballPitch"}]),
    )
    scout = _write_json(tmp_path / "scout.json", _scout_payload())
    with pytest.raises(ABRValidationError, match="missing segment"):
        run_audit(invalid_ann, scout)

    ambiguous_only = _write_json(
        tmp_path / "ambiguous_ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "Ambiguous"}]),
    )
    payload = run_audit(
        ambiguous_only,
        scout,
        abr_config=ABRConfig(k0=12, k1_cap=0, k2_cap=0, max_total_k=12, max_gap=1, round2_enabled=False),
    )
    assert payload["ambiguous_transition_count"] == 2
    assert payload["zero_transition_pseudo_perfect_rejected"] is True
    assert payload["allowed_next_action"] == "LOCKED_ZERO_TRANSITION_NO_REAL_RECALL_EVIDENCE"


def test_cli_outputs_json_schema_and_never_claims_detector_training_or_test_py(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    scout = _write_json(tmp_path / "scout.json", _scout_payload())
    out = tmp_path / "audit.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--annotation-json",
            str(ann),
            "--scout-json",
            str(scout),
            "--out-json",
            str(out),
            "--k0",
            "12",
            "--k1-cap",
            "0",
            "--k2-cap",
            "0",
            "--max-total-k",
            "12",
            "--max-gap",
            "1",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_DIAGNOSTIC_ONLY_REAL_SCOUT_RECALL_EVIDENCE"
    assert payload["method"] == "abr_first_round_bracket_recall_diagnostic"
    assert payload["no_detector"] is True
    assert payload["no_training"] is True
    assert payload["tools_test_allowed"] is False
    assert payload["official_map_claim"] is False
    assert "tools/test.py" not in proc.stdout


def test_bom_annotation_and_scout_json_are_accepted(tmp_path):
    ann = _write_bom_json(
        tmp_path / "ann_bom.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    scout = _write_bom_json(tmp_path / "scout_bom.json", _scout_payload())

    payload = run_audit(
        ann,
        scout,
        abr_config=ABRConfig(k0=12, k1_cap=0, k2_cap=0, max_total_k=12, max_gap=1, round2_enabled=False),
    )

    assert payload["real_deploy_visible_recall_evidence"] is True
    assert payload["transition_count"] == 2
    assert payload["formal_gate_passed"] is True


def test_bom_scout_jsonl_is_accepted(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    scout_record = {
        "video_id": "video_0001",
        "deploy_visible_scout_curve": [0.05, 0.05, 0.9, 0.9, 0.05, 0.05],
        "scout_source": "unit_deploy_visible_scout_jsonl",
    }
    scout_jsonl = tmp_path / "scout_bom.jsonl"
    scout_jsonl.write_text(json.dumps(scout_record) + "\n", encoding="utf-8-sig")

    payload = run_audit(
        ann,
        scout_jsonl,
        abr_config=ABRConfig(k0=6, k1_cap=0, k2_cap=0, max_total_k=6, max_gap=1, round2_enabled=False),
    )

    assert payload["real_deploy_visible_recall_evidence"] is True
    assert payload["scout_sources"] == ["unit_deploy_visible_scout_jsonl"]


def test_low_recall_real_scout_stays_locked_below_formal_gate(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload([{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    scout = _write_json(tmp_path / "scout.json", _scout_payload())

    payload = run_audit(
        ann,
        scout,
        abr_config=ABRConfig(k0=1, k1_cap=0, k2_cap=0, max_total_k=1, max_gap=0, round2_enabled=False),
    )

    assert payload["real_deploy_visible_recall_evidence"] is True
    assert payload["formal_thresholds"]["min_first_round_bracket_recall"] == pytest.approx(0.95)
    assert payload["formal_thresholds"]["min_first_round_transition_coverage"] == pytest.approx(0.95)
    assert payload["first_round_bracket_recall"] < 0.95
    assert payload["first_round_transition_coverage"] < 0.95
    assert payload["missed_transition_count"] > 0
    assert payload["formal_gate_passed"] is False
    assert (
        payload["allowed_next_action"]
        == "LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT"
    )
