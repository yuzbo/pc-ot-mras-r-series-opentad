import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opentad.acquisition.bvr_twb.regret_labels import (
    build_packet_regret_labels,
    validate_regret_label_schema,
)
from opentad.acquisition.bvr_twb.adapter_bridge import (
    ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
    build_adapter_fixed_length_padded_bridge,
    build_detector_feature_centers_from_raw,
)
from opentad.acquisition.bvr_twb.trainable_value import (
    BVRPacketValueMLP,
    build_value_training_batch,
    packet_value_loss,
)
from opentad.acquisition.bvr_twb.types import FORBIDDEN_ROUTE_TOKENS, ROUTE_LABEL
from opentad.acquisition.bvr_twb.validators import validate_bvr_twb_pipeline_ledger
from tools.bvr_twb.audit_sparse_forward_precheck import safe_prepare_output_dir
from tools.bvr_twb.validate_bvr_twb_launch_gate import validate_launch_gate

WORKTREE_ROOT = Path(__file__).resolve().parents[1]

_torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

if _torch_probe.returncode == 0:
    import torch
    from opentad.datasets.transforms.end_to_end import LoadFrames
    from opentad.datasets.transforms.formatting import Collect
    from tools.bvr_twb.audit_opentad_bvr_twb_pipeline import run_pipeline_audit
    TORCH_IMPORT_ERROR = None
else:
    torch = None
    LoadFrames = None
    Collect = None
    run_pipeline_audit = None
    TORCH_IMPORT_ERROR = RuntimeError((_torch_probe.stderr or _torch_probe.stdout).strip().splitlines()[-1])


def _results(split="train", with_gt=True):
    dense_len = 64
    x = np.arange(dense_len, dtype=np.float32)
    out = {
        "video_name": f"bvr_unit_{split}",
        "data_path": "mock",
        "total_frames": 128,
        "avg_fps": 30.0,
        "fps": 30.0,
        "duration": 128.0 / 30.0,
        "snippet_stride": 1,
        "window_size": dense_len,
        "feature_start_idx": 0,
        "feature_end_idx": dense_len - 1,
        "split": split,
        "bvr_twb_preview_actionness": np.clip(
            0.08 + 0.6 * np.exp(-((x - 28.0) ** 2) / (2.0 * 5.0**2)),
            0.0,
            1.0,
        ),
    }
    if with_gt:
        out["gt_segments"] = np.asarray([[24.0, 34.0]], dtype=np.float32)
        out["gt_labels"] = np.asarray([2], dtype=np.int32)
    return out


def _loader(split="train", train_labels=False):
    return LoadFrames(
        num_clips=1,
        method="bvr_twb_dynamic_subsample",
        method_base="sliding_window",
        keep_ratio=0.5,
        target_len=32,
        scale_factor=1,
        remap_gt_to_selected_axis=False,
        bvr_twb_split=split,
        bvr_twb_min_keep=12,
        bvr_twb_max_keep=32,
        bvr_twb_max_gap=16,
        bvr_twb_scaffold_k=4,
        bvr_twb_train_value_labels=train_labels,
        bvr_twb_feature_stride=2,
        bvr_twb_adapter_bridge_mode=ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
    )


def test_bvr_loadframes_dynamic_subsample_outputs_sparse_sorted_metadata_and_labels():
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"torch/OpenTAD pipeline unavailable locally: {TORCH_IMPORT_ERROR}")
    transformed = _loader(split="train", train_labels=True)(_results("train", with_gt=True))
    ledger = transformed["bvr_twb_ledger"]
    assert validate_bvr_twb_pipeline_ledger(ledger)
    assert transformed["frame_inds"].shape[0] == ledger["adapter_input_frame_count"] == 32
    assert ledger["adapter_bridge_mode"] == ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
    assert ledger["adapter_padding_duplicate_count"] == 32 - ledger["valid_k"]
    assert ledger["adapter_padding_counts_as_valid"] is False
    assert len(ledger["selected_frame_inds"]) == ledger["valid_k"]
    assert ledger["valid_k"] < ledger["dense_T"]
    assert ledger["selected_positions"] == sorted(set(ledger["selected_positions"]))
    assert np.array_equal(transformed["frame_inds"], np.asarray(ledger["adapter_padded_frame_inds"]))
    assert int(transformed["masks"].sum().item()) == ledger["detector_feature_valid_k"]
    assert len(transformed["irregular_selected_positions"]) == ledger["detector_feature_valid_k"]
    assert np.allclose(transformed["irregular_selected_positions"], np.asarray(ledger["detector_feature_positions"]))
    assert len(transformed["bvr_twb_raw_selected_positions"]) == ledger["valid_k"]
    assert transformed["irregular_selected_valid_len"] == float(ledger["dense_T"])
    assert ledger["original_time_metadata"]["selected_index_is_time"] is False
    assert ledger["temporal_decode_uses_original_time"] is True
    assert transformed["bvr_twb_train_value_labels"]
    assert all(validate_regret_label_schema(row) for row in transformed["bvr_twb_train_value_labels"])


def test_bvr_loadframes_val_test_do_not_build_gt_value_labels_and_reject_if_requested():
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"torch/OpenTAD pipeline unavailable locally: {TORCH_IMPORT_ERROR}")
    val = _loader(split="val", train_labels=False)(_results("val", with_gt=True))
    assert validate_bvr_twb_pipeline_ledger(val["bvr_twb_ledger"])
    assert val["bvr_twb_train_value_labels"] == []
    assert val["bvr_twb_ledger"]["selector_provenance"]["selection_uses_gt"] is False

    with pytest.raises(ValueError, match="train_value_labels"):
        _loader(split="val", train_labels=True)(_results("val", with_gt=True))


def test_regret_label_schema_and_trainable_value_loss():
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"torch/OpenTAD pipeline unavailable locally: {TORCH_IMPORT_ERROR}")
    transformed = _loader(split="train", train_labels=True)(_results("train", with_gt=True))
    labels = transformed["bvr_twb_train_value_labels"]
    assert labels
    label_by_id = {row["packet_id"]: row for row in labels}
    assert all("boundary_coverage_loss" in row["regret_components"] for row in labels)

    packets = transformed["bvr_twb_ledger"]
    assert packets["num_candidate_packets"] >= len(labels) - 4

    # Rebuild a tiny train batch from labels using bridge candidate packets.
    from opentad.acquisition.bvr_twb.open_tad_bridge import build_bvr_twb_open_tad_selection

    dense_window = np.arange(64, dtype=np.int64)
    bridge = build_bvr_twb_open_tad_selection(
        _results("train", with_gt=True),
        dense_window=dense_window,
        target_frame_num=32,
        split="train",
        gt_segments=np.asarray([[24.0, 34.0]], dtype=np.float32),
        gt_labels=np.asarray([2], dtype=np.int32),
        min_keep=12,
        max_keep=32,
        max_gap=16,
        train_value_labels=True,
    )
    x, y = build_value_training_batch(bridge["candidate_packets"], bridge["regret_labels"], dense_T=64)
    model = BVRPacketValueMLP()
    pred = model(x)
    loss = packet_value_loss(pred, y)
    assert torch.isfinite(loss)
    assert set(label_by_id).issubset({row["packet_id"] for row in bridge["regret_labels"]})


def test_regret_labels_are_train_only():
    from opentad.acquisition.bvr_twb.types import CandidatePacket

    packet = CandidatePacket(
        packet_id=1,
        video_id="v",
        window_id=0,
        split="train",
        source="twb",
        role="transition_center",
        positions=[5],
        dense_T=16,
    )
    labels = build_packet_regret_labels([packet], np.asarray([[4.0, 8.0]], dtype=np.float32), 16, "train")
    assert validate_regret_label_schema(labels[0])
    with pytest.raises(ValueError, match="train-only"):
        build_packet_regret_labels([packet], np.asarray([[4.0, 8.0]], dtype=np.float32), 16, "val")


def test_collect_passes_bvr_metadata_to_metas():
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"torch/OpenTAD pipeline unavailable locally: {TORCH_IMPORT_ERROR}")
    transformed = _loader(split="train", train_labels=True)(_results("train", with_gt=True))
    transformed["imgs"] = np.zeros((transformed["frame_inds"].shape[0], 4, 4, 3), dtype=np.uint8)
    collected = Collect(inputs="imgs", keys=["masks"])(transformed)
    metas = collected["metas"]
    assert "bvr_twb_ledger" in metas
    assert "bvr_twb_train_value_labels" in metas
    assert metas["bvr_twb_ledger"]["route_label"] == ROUTE_LABEL


def test_bvr_config_uses_dynamic_method_and_excludes_unapproved_route_tokens():
    path = Path("configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py")
    text = path.read_text(encoding="utf-8")
    assert "bvr_twb_dynamic_subsample" in text
    normalized = text.replace(ROUTE_LABEL, "").replace("checkpoint_interval", "checkpoint_period")
    for token in FORBIDDEN_ROUTE_TOKENS:
        assert token.lower() not in normalized.lower()


def test_opentad_pipeline_audit_cli_function_writes_valid_summary(tmp_path):
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"torch/OpenTAD pipeline unavailable locally: {TORCH_IMPORT_ERROR}")
    out = WORKTREE_ROOT / ".tmp_bvr_twb_opentad_pipeline_pytest" / f"bvr_twb_{tmp_path.name}"
    try:
        summary = run_pipeline_audit(out, overwrite=True)
        assert summary["all_validated"] is True
        assert summary["sparse_compute_claim"] is False
        rows = [
            json.loads(line)
            for line in (out / "bvr_twb_opentad_pipeline_ledgers.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 3
        assert all(validate_bvr_twb_pipeline_ledger(row) for row in rows)
    finally:
        shutil.rmtree(out, ignore_errors=True)
        try:
            out.parent.rmdir()
        except OSError:
            pass


def test_opentad_pipeline_audit_rejects_outside_worktree_output(tmp_path):
    with pytest.raises(ValueError, match="outside worktree"):
        safe_prepare_output_dir(tmp_path / ".tmp_bvr_twb_opentad_pipeline", overwrite=True)


def test_bvr_launch_gate_requires_unblocked_precheck_summary(tmp_path):
    config = Path("configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py")
    blocked = tmp_path / "blocked_summary.json"
    blocked.write_text(
        json.dumps(
            {
                "route_label": ROUTE_LABEL,
                "blocked": True,
                "blocked_stage": "import_torch_or_loadframes",
                "blocked_reason": "local torch unavailable",
                "all_validated": False,
                "sparse_compute_claim": False,
                "no_training": True,
                "no_metric_claim": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blocked"):
        validate_launch_gate(config, blocked)

    passed = tmp_path / "passed_summary.json"
    passed.write_text(
        json.dumps(
            {
                "route_label": ROUTE_LABEL,
                "blocked": False,
                "all_validated": True,
                "sparse_compute_claim": False,
                "no_training": True,
                "no_metric_claim": True,
                "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
                "adapter_bridge_modes": [ADAPTER_FIXED_LENGTH_PADDED_BRIDGE],
                "adapter_padding_counts_as_valid": False,
            }
        ),
        encoding="utf-8",
    )
    result = validate_launch_gate(config, passed)
    assert result["allowed_next_action"] == "FINAL_READ_ONLY_REVIEW_THEN_LINUX_PRECHECK_ONLY"
    assert result["full_train_unlocked"] is False


def test_adapter_fixed_length_padded_bridge_keeps_valid_k_sparse_and_padding_invalid():
    bridge = build_adapter_fixed_length_padded_bridge(
        selected_positions=np.asarray([1, 5, 9, 12, 18], dtype=np.int64),
        selected_frame_inds=np.asarray([11, 15, 19, 22, 28], dtype=np.int64),
        target_frame_num=12,
        dense_T=32,
        feature_stride=2,
    )
    assert bridge["adapter_bridge_mode"] == ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
    assert bridge["adapter_input_frame_count"] == 12
    assert bridge["adapter_padding_duplicate_count"] == 7
    assert bridge["adapter_padding_counts_as_valid"] is False
    assert bridge["adapter_valid_raw_mask"].sum() == 5
    assert bridge["detector_mask_len"] == 6
    assert bridge["detector_valid_mask"].sum() == 3
    assert np.allclose(bridge["detector_feature_positions"], np.asarray([3.0, 10.5, 18.0], dtype=np.float32))
    assert bridge["adapter_padded_frame_inds"][-1] == 28


def test_detector_feature_centers_use_partial_raw_group_not_padding_duplicate():
    centers = build_detector_feature_centers_from_raw([1, 5, 9, 12, 18], feature_stride=2)
    assert np.allclose(centers, np.asarray([3.0, 10.5, 18.0], dtype=np.float32))


def test_pipeline_ledger_accepts_adapter_bridge_padding_but_not_valid_padding():
    from opentad.acquisition.bvr_twb.validators import build_original_time_metadata, build_selection_gap_diagnostics

    bridge = build_adapter_fixed_length_padded_bridge([0, 3, 7], [10, 13, 17], target_frame_num=8, dense_T=16, feature_stride=2)
    ledger = {
        "route_label": ROUTE_LABEL,
        "method": "bvr_twb_dynamic_subsample",
        "split": "train",
        "dense_T": 16,
        "selected_positions": [0, 3, 7],
        "selected_frame_inds": [10, 13, 17],
        "raw_selected_positions": [0, 3, 7],
        "valid_k": 3,
        "budget_stop_reason": "candidate_exhausted",
        "selection_gap_diagnostics": build_selection_gap_diagnostics([0, 3, 7], 16, 8),
        "original_time_metadata": build_original_time_metadata(16, [0, 3, 7], fps=4.0),
        "temporal_decode_uses_original_time": True,
        "selected_index_is_time": False,
        "dense_raw_backbone_handoff": False,
        "selected_inputs_is_gathered": True,
        "padding_duplicate_count": bridge["adapter_padding_duplicate_count"],
        "sparse_compute_claim": False,
        "claim_status": "bvr_twb_first_trainable_pipeline_no_metric_claim",
        "selector_provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        },
        "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        "adapter_target_frame_num": bridge["adapter_target_frame_num"],
        "adapter_input_frame_count": bridge["adapter_input_frame_count"],
        "adapter_padded_frame_inds": bridge["adapter_padded_frame_inds"].tolist(),
        "adapter_padded_positions": bridge["adapter_padded_positions"].tolist(),
        "adapter_valid_raw_mask": bridge["adapter_valid_raw_mask"].tolist(),
        "adapter_padding_duplicate_count": bridge["adapter_padding_duplicate_count"],
        "adapter_padding_counts_as_valid": bridge["adapter_padding_counts_as_valid"],
        "adapter_fixed_length_padded_bridge": True,
        "detector_feature_valid_k": bridge["detector_feature_valid_k"],
        "detector_feature_positions": bridge["detector_feature_positions"].tolist(),
        "detector_mask_len": bridge["detector_mask_len"],
        "detector_mask_true_count": bridge["detector_feature_valid_k"],
        "bvr_twb_feature_stride": 2,
    }
    assert validate_bvr_twb_pipeline_ledger(ledger)
    bad = dict(ledger, adapter_padding_counts_as_valid=True)
    with pytest.raises(ValueError, match="padding duplicates"):
        validate_bvr_twb_pipeline_ledger(bad)
    bad_positions = dict(ledger, detector_feature_positions=[0.0, 7.0])
    with pytest.raises(ValueError, match="feature.*center"):
        validate_bvr_twb_pipeline_ledger(bad_positions)


def test_launch_gate_rejects_c3_and_combo_tokens_in_config(tmp_path):
    base_config = Path("configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py").read_text(encoding="utf-8")
    summary = tmp_path / "passed_summary.json"
    summary.write_text(
        json.dumps(
            {
                "route_label": ROUTE_LABEL,
                "blocked": False,
                "all_validated": True,
                "sparse_compute_claim": False,
                "no_training": True,
                "no_metric_claim": True,
                "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
                "adapter_bridge_modes": [ADAPTER_FIXED_LENGTH_PADDED_BRIDGE],
                "adapter_padding_counts_as_valid": False,
            }
        ),
        encoding="utf-8",
    )
    for token in ("C3", "COMBO"):
        bad_config = tmp_path / f"bad_{token}.py"
        bad_config.write_text(base_config + f"\n# forbidden route token {token}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="forbidden route token"):
            validate_launch_gate(bad_config, summary)
