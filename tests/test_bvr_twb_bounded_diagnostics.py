import json
import shutil
from pathlib import Path

from tools.bvr_twb.audit_pretrain_load import run_pretrain_audit
from tools.bvr_twb.dump_bridge_roundtrip import run_dump


ROOT = Path(__file__).resolve().parents[1]


def test_bvr_pretrain_audit_blocks_missing_resolved_pretrain():
    summary = run_pretrain_audit(
        ROOT / "configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py"
    )
    assert summary["blocked"] is True
    assert summary["verdict"] == "BLOCKER_PRETRAIN_MISSING_IN_RESOLVED_CONFIG"
    assert summary["resolved_pretrain"] is None
    assert summary["pretrain_resolves_videomae_s"] is False
    assert summary["full_training_unlocked"] is False


def test_bvr_bridge_roundtrip_dump_writes_dynamic_and_forced_uniform_rows():
    out = ROOT / ".tmp_bvr_twb_bridge_roundtrip_pytest" / "bvr_twb_bridge_roundtrip"
    try:
        summary = run_dump(out, overwrite=True)
        assert summary["all_ledgers_validated"] is True
        assert summary["all_seconds_roundtrip_match"] is True
        assert summary["forced_uniform_raw_valid_k"] == 192
        assert summary["forced_uniform_detector_feature_valid_k"] == 96
        assert summary["no_metric_claim"] is True

        rows = json.loads((out / "bridge_roundtrip_rows.json").read_text(encoding="utf-8"))
        assert [row["case"] for row in rows] == [
            "dynamic_bvr",
            "forced_uniform_through_bvr_bridge",
        ]
        assert all(row["ledger_validated"] for row in rows)
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)
