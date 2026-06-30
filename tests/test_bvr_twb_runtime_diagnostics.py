import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.bvr_twb.audit_postprocess_proposals import run_postprocess_audit
from tools.bvr_twb.audit_runtime_provenance import run_runtime_provenance
from tools.bvr_twb.trace_one_sample_pipeline import run_one_sample_trace


ROOT = Path(__file__).resolve().parents[1]

_torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
TORCH_IMPORT_ERROR = None if _torch_probe.returncode == 0 else RuntimeError(
    (_torch_probe.stderr or _torch_probe.stdout).strip().splitlines()[-1]
)


def _require_torch():
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"torch/OpenTAD runtime unavailable locally: {TORCH_IMPORT_ERROR}")


def test_runtime_provenance_source_contract_reports_required_tokens_without_gpu():
    summary = run_runtime_provenance()

    assert summary["route_label"] == "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
    assert summary["no_training"] is True
    assert summary["no_video_decode"] is True
    assert "method_literal" in summary["required_tokens_present"]
    assert "selection_call" in summary["required_tokens_present"]
    assert "fail_closed_unknown_method" in summary["required_tokens_present"]
    assert summary["runtime_contract"] in {"passed", "skipped"}
    if summary["runtime_contract"] == "passed":
        assert summary["dispatch_hit"] is True
        assert summary["loadframes_call_contains_selection_call"] is True


def test_unsupported_loadframes_method_is_fail_closed_in_source_and_runtime_if_available():
    text = (ROOT / "opentad/datasets/transforms/end_to_end.py").read_text(encoding="utf-8")
    assert "Unsupported LoadFrames method" in text

    _require_torch()
    from opentad.datasets.transforms.end_to_end import LoadFrames

    with pytest.raises(ValueError, match="Unsupported LoadFrames method"):
        LoadFrames(method="not_a_bvr_method")({"total_frames": 8, "avg_fps": 30.0})


def test_one_sample_pipeline_trace_outputs_required_shape_and_keys_without_gpu():
    summary = run_one_sample_trace()

    assert summary["audit"] == "bvr_twb_one_sample_pipeline_trace"
    assert summary["no_training"] is True
    assert summary["no_real_video"] is True
    assert summary["runtime_trace"] in {"passed", "skipped"}
    if summary["runtime_trace"] == "passed":
        assert summary["dispatch_hit"] is True
        assert summary["frame_idxs_count"] == 32
        assert summary["fake_decord_decode_input_count"] == 32
        assert summary["fake_backbone_input_ncthw_shape"] == [1, 3, 32, 4, 4]
        assert summary["selected_frame_inds_subset_of_decoded_unique"] is True
        assert summary["collect_meta_has_bvr_ledger"] is True
        assert summary["collect_meta_has_detector_positions"] is True
        assert summary["gt_native_axis_preserved"] is True
        assert summary["irregular_native_axis"] is True


def test_grid_audit_hook_is_env_gated_and_records_native_positions_if_runtime_available():
    detector_text = (ROOT / "opentad/models/detectors/irregular_actionformer.py").read_text(encoding="utf-8")
    assert "BVR_TWB_GRID_AUDIT" in detector_text
    assert "PASS_NATIVE_AXIS_POSITIONS_ENTERED_MODEL" in detector_text

    _require_torch()
    import torch

    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

    out_dir = ROOT / ".tmp_bvr_twb_runtime_diag_pytest"
    audit_path = out_dir / "grid_audit.jsonl"
    old_enabled = os.environ.get("BVR_TWB_GRID_AUDIT")
    old_path = os.environ.get("BVR_TWB_GRID_AUDIT_PATH")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BVR_TWB_GRID_AUDIT"] = "1"
        os.environ["BVR_TWB_GRID_AUDIT_PATH"] = str(audit_path)
        detector = object.__new__(IrregularActionFormer)
        masks = torch.tensor([[True, True, False, False]])
        meta = {
            "video_name": "bvr_grid_audit_unit",
            "irregular_native_axis": True,
            "bvr_twb_ledger": {"method": "bvr_twb_dynamic_subsample"},
            "bvr_twb_detector_feature_positions": np.asarray([4.0, 20.0], dtype=np.float32),
            "bvr_twb_detector_feature_valid_len": 64.0,
        }
        grid = detector._temporal_grid_from_metas([meta], masks)
        rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

        assert torch.allclose(grid["center"][0, :2], torch.tensor([4.0, 20.0]))
        assert rows[0]["dispatch_hit"] is True
        assert rows[0]["mask_true_count"] == 2
        assert rows[0]["grid_center_prefix"] == [4.0, 20.0]
        assert rows[0]["status"] == "PASS_NATIVE_AXIS_POSITIONS_ENTERED_MODEL"
    finally:
        if old_enabled is None:
            os.environ.pop("BVR_TWB_GRID_AUDIT", None)
        else:
            os.environ["BVR_TWB_GRID_AUDIT"] = old_enabled
        if old_path is None:
            os.environ.pop("BVR_TWB_GRID_AUDIT_PATH", None)
        else:
            os.environ["BVR_TWB_GRID_AUDIT_PATH"] = old_path
        shutil.rmtree(out_dir, ignore_errors=True)


def test_postprocess_proposal_audit_explains_365940_without_metric_claim():
    detector_text = (ROOT / "opentad/models/detectors/irregular_actionformer.py").read_text(encoding="utf-8")
    assert "BVR_TWB_POSTPROCESS_AUDIT" in detector_text
    assert "flattened_candidate_count" in detector_text
    assert "bvr_twb_postprocess_guard" in detector_text
    assert "bvr_twb_guarded_raw_cap_per_class" in detector_text

    summary = run_postprocess_audit()

    assert summary["no_training"] is True
    assert summary["no_checkpoint"] is True
    assert summary["no_metric_claim"] is True
    assert summary["full_training_unlocked"] is False
    assert summary["runtime_contract"] in {"passed", "skipped"}
    if summary["runtime_contract"] == "passed":
        assert summary["raw_proposal_count"] == 19260
        assert summary["num_classes"] == 19
        assert summary["flattened_candidate_count"] == 365940
        assert summary["expected_explosion_formula"] == "raw_proposal_count * num_classes"
        assert summary["explains_365940_predictions"] is True
        assert summary["pre_nms_selected_count"] == 2000
        assert summary["guard_runtime_contract"] == "passed"
        assert summary["guard_active"] is True
        assert summary["guard_candidate_generation_mode"] == "bvr_twb_guarded_raw_cap_per_class"
        assert summary["guard_raw_input_count"] == 19260
        assert summary["guard_raw_selected_count"] == 1024
        assert summary["guard_class_candidate_count_before_global_topk"] <= 32 * 19
        assert summary["guard_pre_nms_selected_count"] <= 512
        assert summary["guard_final_result_count"] <= 512
        assert summary["guard_no_metric_claim"] is True
        assert summary["guard_full_training_unlocked"] is False


def test_bvr_postprocess_guard_fails_closed_without_bvr_meta():
    _require_torch()
    import torch

    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

    detector = object.__new__(IrregularActionFormer)
    proposals = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    scores = torch.full((2, 3), 0.1, dtype=torch.float32)
    metas = [
        {
            "video_name": "non_bvr_guard_should_fail",
            "fps": 30.0,
            "duration": 2.0,
            "snippet_stride": 1,
            "offset_frames": 0,
            "window_start_frame": 0,
        }
    ]
    post_cfg = SimpleNamespace(
        pre_nms_thresh=0.001,
        pre_nms_topk=512,
        sliding_window=True,
        nms=None,
        bvr_twb_postprocess_guard=dict(
            enabled=True,
            require_bvr_meta=True,
            raw_proposal_cap=1024,
            per_class_topk=32,
            total_candidate_cap=512,
            min_score=0.001,
        ),
    )

    with pytest.raises(ValueError, match="no BVR-TWB metadata"):
        detector.post_processing(([proposals], [scores]), metas, post_cfg, ["a", "b", "c"])
