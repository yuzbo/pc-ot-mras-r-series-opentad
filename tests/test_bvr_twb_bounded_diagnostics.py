import json
import shutil
from pathlib import Path

from tools.bvr_twb.audit_pretrain_load import run_pretrain_audit
from tools.bvr_twb.dump_bridge_roundtrip import run_dump


ROOT = Path(__file__).resolve().parents[1]


def _bvr_config_variants():
    configs = sorted((ROOT / "configs/adatad/thumos").glob("*bvr_twb*.py"))
    assert configs
    return configs


def test_bvr_pretrain_audit_requires_resolved_videomae_s_for_all_variants():
    for config in _bvr_config_variants():
        summary = run_pretrain_audit(config)
        assert summary["blocked"] is False, config
        assert summary["verdict"] == "PASS_PRETRAIN_RESOLVED_STATIC_NO_TRAINING"
        assert summary["resolved_pretrain"] == (
            "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
        )
        assert summary["pretrain_resolves_videomae_s"] is True
        assert summary["pretrain_file_check_required"] is False
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
