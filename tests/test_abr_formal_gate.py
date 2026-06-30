import json
import subprocess
import sys
from pathlib import Path

import pytest

from opentad.acquisition.abr import ABRConfig, ABR_ROUTE_LABEL, select_active_bracket_refinement
from opentad.acquisition.abr.validators import ABRValidationError, validate_formal_readiness_payload
from tools.abr.validate_abr_formal_gate import validate_formal_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = REPO_ROOT / "configs" / "adatad" / "thumos" / (
    "input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py"
)


def _formal_payload_from_diagnostics(diagnostics):
    return {
        "route_label": ABR_ROUTE_LABEL,
        "method": "abr_active_bracket_refinement",
        "status": "PASS_FORMAL_READINESS_EVIDENCE",
        "summary": {
            "scout_source": "abr_scout_curve:deploy_visible",
            "diagnostic_fallback_used": False,
            "fallback_stage": "FORMAL_READINESS_LOCKED",
            "detector_forward_count": 1,
            "first_round_bracket_diagnostics": diagnostics,
            "formal_thresholds": {
                "min_first_round_bracket_recall": 0.95,
                "min_first_round_transition_coverage": 0.95,
            },
        },
    }


def test_first_round_ledger_records_bracket_recall_coverage_and_missed_transitions():
    curve = [0.02] * 16 + [0.88] * 12 + [0.04] * 18 + [0.9] * 10 + [0.03] * 24
    result = select_active_bracket_refinement(
        dense_t=len(curve),
        scout_curve=curve,
        scout_source="abr_scout_curve:deploy_visible",
        config=ABRConfig(k0=24, k1_cap=8, k2_cap=2, max_total_k=40, max_gap=4),
    )

    diagnostics = result.round_ledgers[0].diagnostics
    assert diagnostics["scope"] == "first_round_bracket_recall_from_deploy_visible_scout"
    assert diagnostics["transition_count"] > 0
    assert diagnostics["bracketed_transition_count"] == diagnostics["transition_count"]
    assert diagnostics["missed_transition_count"] == 0
    assert diagnostics["first_round_bracket_recall"] == pytest.approx(1.0)
    assert diagnostics["first_round_transition_coverage"] == pytest.approx(1.0)
    assert diagnostics["diagnostic_fallback_used"] is False


@pytest.mark.parametrize(
    ("curve", "expected_kind"),
    [
        ([0.05, 0.45, 0.90], "start"),
        ([0.90, 0.45, 0.05], "end"),
    ],
)
def test_ambiguous_mediated_dense_transition_blocks_formal_readiness(curve, expected_kind):
    result = select_active_bracket_refinement(
        dense_t=len(curve),
        scout_curve=curve,
        scout_source="abr_scout_curve:deploy_visible",
        config=ABRConfig(k0=1, k1_cap=0, k2_cap=0, max_total_k=1, max_gap=0, round2_enabled=False),
    )

    diagnostics = result.round_ledgers[0].diagnostics
    assert diagnostics["transition_count"] == 1
    assert diagnostics["bracketed_transition_count"] == 0
    assert diagnostics["missed_transition_count"] == 1
    assert diagnostics["first_round_bracket_recall"] == pytest.approx(0.0)
    assert diagnostics["first_round_transition_coverage"] == pytest.approx(0.0)
    assert diagnostics["missed_transitions"] == [{"left": 0, "right": 2, "kind": expected_kind}]

    with pytest.raises(ABRValidationError, match="first_round_bracket_recall"):
        validate_formal_readiness_payload(_formal_payload_from_diagnostics(diagnostics))


@pytest.mark.parametrize(
    ("curve", "expected_kind"),
    [
        ([0.05, 0.45, 0.90], "start"),
        ([0.90, 0.45, 0.05], "end"),
    ],
)
def test_ambiguous_mediated_transition_passes_when_bracketed_and_nonzero(curve, expected_kind):
    result = select_active_bracket_refinement(
        dense_t=len(curve),
        scout_curve=curve,
        scout_source="abr_scout_curve:deploy_visible",
        config=ABRConfig(k0=2, k1_cap=0, k2_cap=0, max_total_k=2, max_gap=2, round2_enabled=False),
    )

    diagnostics = result.round_ledgers[0].diagnostics
    assert diagnostics["transition_count"] == 1
    assert diagnostics["bracketed_transition_count"] == 1
    assert diagnostics["missed_transition_count"] == 0
    assert diagnostics["first_round_bracket_recall"] == pytest.approx(1.0)
    assert diagnostics["first_round_transition_coverage"] == pytest.approx(1.0)
    assert diagnostics["covered_transitions"] == [{"left": 0, "right": 2, "kind": expected_kind}]

    decision = validate_formal_readiness_payload(_formal_payload_from_diagnostics(diagnostics))
    assert decision["formal_readiness_evidence_ok"] is True
    assert decision["full_train_unlocked"] is False
    assert decision["metrics"]["transition_count"] == 1


def test_formal_config_disables_diagnostic_fallback_and_stays_locked():
    decision = validate_formal_config(FORMAL_CONFIG)
    assert decision["formal_config_ok"] is True
    assert decision["full_train_unlocked"] is False
    assert decision["allowed_next_action"] == "FORMAL_REVIEW_PACKET_ONLY"
    assert "FORMAL_FULL_TRAIN_PENDING_REAL_SCOUT_RECALL_EVIDENCE" in decision["still_locked"]


def test_formal_payload_rejects_precheck_fallback_and_low_first_round_recall():
    good_diag = {
        "scout_source": "abr_scout_curve:deploy_visible",
        "diagnostic_fallback_used": False,
        "transition_count": 2,
        "bracketed_transition_count": 2,
        "missed_transition_count": 0,
        "first_round_bracket_recall": 1.0,
        "first_round_transition_coverage": 1.0,
    }
    decision = validate_formal_readiness_payload(_formal_payload_from_diagnostics(good_diag))
    assert decision["formal_readiness_evidence_ok"] is True
    assert decision["full_train_unlocked"] is False

    fallback_payload = _formal_payload_from_diagnostics(dict(good_diag))
    fallback_payload["summary"]["scout_source"] = "diagnostic_fallback:PRECHECK_ONLY"
    fallback_payload["summary"]["diagnostic_fallback_used"] = True
    fallback_payload["summary"]["fallback_stage"] = "PRECHECK_ONLY"
    fallback_payload["summary"]["first_round_bracket_diagnostics"]["scout_source"] = "diagnostic_fallback:PRECHECK_ONLY"
    fallback_payload["summary"]["first_round_bracket_diagnostics"]["diagnostic_fallback_used"] = True
    with pytest.raises(ABRValidationError):
        validate_formal_readiness_payload(fallback_payload)

    low_payload = _formal_payload_from_diagnostics(dict(good_diag))
    low_payload["summary"]["first_round_bracket_diagnostics"]["first_round_bracket_recall"] = 0.5
    low_payload["summary"]["first_round_bracket_diagnostics"]["missed_transition_count"] = 1
    with pytest.raises(ABRValidationError):
        validate_formal_readiness_payload(low_payload)


def test_formal_payload_rejects_zero_transition_perfect_score_pseudo_evidence():
    zero_transition_diag = {
        "scout_source": "abr_scout_curve:deploy_visible",
        "diagnostic_fallback_used": False,
        "transition_count": 0,
        "bracketed_transition_count": 0,
        "missed_transition_count": 0,
        "first_round_bracket_recall": 1.0,
        "first_round_transition_coverage": 1.0,
    }

    with pytest.raises(ABRValidationError, match="transition_count > 0"):
        validate_formal_readiness_payload(_formal_payload_from_diagnostics(zero_transition_diag))


def test_formal_cli_accepts_config_only_but_rejects_precheck_json(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "abr" / "validate_abr_formal_gate.py"),
            "--config",
            str(FORMAL_CONFIG),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["full_train_unlocked"] is False

    precheck_payload = {
        "route_label": ABR_ROUTE_LABEL,
        "method": "abr_active_bracket_refinement",
        "status": "PASS_PRECHECK_ONLY",
        "summary": {
            "scout_source": "diagnostic_fallback:PRECHECK_ONLY",
            "diagnostic_fallback_used": True,
            "fallback_stage": "PRECHECK_ONLY",
            "detector_forward_count": 1,
            "first_round_bracket_diagnostics": {},
        },
    }
    path = tmp_path / "precheck.json"
    path.write_text(json.dumps(precheck_payload), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "abr" / "validate_abr_formal_gate.py"),
            "--config",
            str(FORMAL_CONFIG),
            "--formal-json",
            str(path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "LOCKED" in proc.stdout
