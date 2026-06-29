import json

import pytest

from opentad.acquisition.bvr_twb.raw_handoff import build_sparse_raw_handoff
from opentad.acquisition.bvr_twb.sparse_forward_audit import (
    SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS,
    build_fake_sparse_forward_ledger,
)
from opentad.acquisition.bvr_twb.validators import (
    SPARSE_FORWARD_PASS_REAL_MODULE,
    SPARSE_FORWARD_PASS_SHAPE_ONLY,
    validate_sparse_forward_ledger,
)
from tools.bvr_twb.audit_sparse_forward_precheck import run_precheck


def _good_ledger():
    return build_fake_sparse_forward_ledger(
        video_name="unit_fake",
        window_id="unit_fake_0000",
        dense_T=32,
        selected_positions=[0, 5, 12, 21, 31],
        detector_pad_len=64,
    )


def test_sparse_forward_ledger_passes_shape_only_without_sparse_claim():
    row = _good_ledger()
    result = validate_sparse_forward_ledger(row)
    assert result["verdict"] == SPARSE_FORWARD_PASS_SHAPE_ONLY
    assert row["claim_status"] == SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS
    assert row["sparse_compute_claim"] is False
    assert row["detector_pad_len"] > row["detector_mask_true_count"]


def test_module_fake_forward_records_hook_like_evidence_without_sparse_claim():
    row = build_fake_sparse_forward_ledger(
        video_name="unit_module_fake",
        window_id="unit_module_fake_0000",
        dense_T=32,
        selected_positions=[0, 7, 15, 23, 31],
        module_forward=True,
    )
    result = validate_sparse_forward_ledger(row)
    assert result["verdict"] == SPARSE_FORWARD_PASS_REAL_MODULE
    assert row["module_forward_evidence"] is True
    assert row["sparse_compute_claim"] is False


def test_sparse_forward_validator_rejects_dense_raw_handoff():
    row = _good_ledger()
    row["dense_raw_backbone_handoff"] = True
    with pytest.raises(ValueError, match="FAIL_DENSE_RAW_HANDOFF"):
        validate_sparse_forward_ledger(row)

    row = _good_ledger()
    row["raw_frame_inds_in"] = list(range(row["dense_T"]))
    row["decoded_frame_count"] = row["dense_T"]
    row["decoded_unique_count"] = row["dense_T"]
    with pytest.raises(ValueError, match="FAIL_DENSE_RAW_HANDOFF"):
        validate_sparse_forward_ledger(row)


def test_sparse_forward_validator_rejects_dense_backbone_chunk_count():
    row = _good_ledger()
    row["dense_backbone_chunk_count"] = 16
    row["backbone_forward_chunk_count"] = 16
    with pytest.raises(ValueError, match="FAIL_BACKBONE_DENSE_CHUNK_COUNT"):
        validate_sparse_forward_ledger(row)


def test_sparse_forward_validator_rejects_detector_pad_counted_as_valid():
    row = _good_ledger()
    row["detector_mask_true_count"] = row["detector_pad_len"]
    row["rpn_valid_temporal_len"] = row["detector_pad_len"]
    with pytest.raises(ValueError, match="FAIL_DETECTOR_PAD_CONFUSED_AS_VALID"):
        validate_sparse_forward_ledger(row)


def test_sparse_forward_validator_rejects_missing_original_time_decode():
    row = _good_ledger()
    row["temporal_decode_uses_original_time"] = False
    with pytest.raises(ValueError, match="FAIL_ORIGINAL_TIME_DECODE_MISSING"):
        validate_sparse_forward_ledger(row)

    row = _good_ledger()
    row["original_time_metadata"] = dict(row["original_time_metadata"], selected_index_is_time=True)
    with pytest.raises(ValueError, match="FAIL_ORIGINAL_TIME_DECODE_MISSING"):
        validate_sparse_forward_ledger(row)


def test_sparse_forward_validator_rejects_leakage_and_route_mixing():
    row = _good_ledger()
    row["teacher_logits"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="FAIL_LEAKAGE_FIELD_PRESENT"):
        validate_sparse_forward_ledger(row)

    row = _good_ledger()
    row["notes"] = "borrow GlobalRank interval routing"
    with pytest.raises(ValueError, match="FAIL_ROUTE_MIXING"):
        validate_sparse_forward_ledger(row)


def test_shape_only_cannot_unlock_sparse_compute_claim():
    row = _good_ledger()
    row["sparse_compute_claim"] = True
    with pytest.raises(ValueError, match="FAIL_SPARSE_COMPUTE_CLAIM_UNLOCKED"):
        validate_sparse_forward_ledger(row)


def test_padded_duplicate_raw_handoff_is_not_counted_as_valid():
    row = _good_ledger()
    handoff = build_sparse_raw_handoff(row["selected_positions"], row["dense_T"], pad_to=8)
    row.update(handoff)
    assert row["decoded_frame_count"] == 8
    assert row["decoded_unique_count"] == row["valid_k"]
    assert row["padded_duplicate_count"] == 3
    result = validate_sparse_forward_ledger(row)
    assert result["verdict"] == SPARSE_FORWARD_PASS_SHAPE_ONLY
    assert row["detector_mask_true_count"] == row["valid_k"]


def test_sparse_forward_precheck_tool_writes_jsonl_and_summary(tmp_path):
    out = tmp_path / ".tmp_bvr_twb_sparse_forward_audit"
    summary = run_precheck("fake_raw", out, overwrite=True, root=tmp_path)
    assert summary["sparse_compute_claim"] is False
    assert summary["no_metric_claim"] is True
    ledger_path = out / "bvr_twb_sparse_forward_ledgers.jsonl"
    assert ledger_path.exists()
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == summary["num_ledgers"]
    assert all(validate_sparse_forward_ledger(row)["verdict"] == SPARSE_FORWARD_PASS_SHAPE_ONLY for row in rows)
