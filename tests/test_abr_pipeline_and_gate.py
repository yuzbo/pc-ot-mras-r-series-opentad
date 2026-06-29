import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from opentad.acquisition.abr import ABRConfig, ABR_ROUTE_LABEL
from opentad.acquisition.abr.integration import apply_abr_to_results
from opentad.acquisition.abr.validators import ABRValidationError, validate_launch_gate_payload
from tools.abr.audit_abr_pipeline_precheck import build_precheck_summary


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pseudo_boundary_helper_exists_for_clean_clone_import_dependency():
    helper_path = REPO_ROOT / "opentad" / "datasets" / "transforms" / "pseudo_boundary.py"
    assert helper_path.exists()

    spec = importlib.util.spec_from_file_location("abr_pseudo_boundary_helper", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in (
        "load_pseudo_boundary_cache",
        "select_pseudo_boundary_hybrid_positions",
        "select_pseudo_boundary_snap_positions",
    ):
        assert callable(getattr(module, name, None)), name


def test_apply_abr_to_results_sets_frame_inds_before_decode_and_keeps_padding_invalid():
    results = {
        "video_name": "mock_video",
        "total_frames": 80,
        "avg_fps": 25.0,
        "fps": 25.0,
        "duration": 3.2,
        "snippet_stride": 1,
        "window_size": 80,
        "feature_start_idx": 0,
        "feature_end_idx": 79,
        "abr_scout_curve": [0.02] * 20 + [0.82] * 8 + [0.10] * 20 + [0.88] * 8 + [0.04] * 24,
    }
    out = apply_abr_to_results(
        results,
        config=ABRConfig(k0=8, k1_cap=8, k2_cap=2, max_total_k=20, max_gap=14, target_frame_num=24),
    )

    assert "frame_inds" in out
    assert out["frame_inds"].shape[0] == 24
    assert out["abr_selected_valid_k"] < 24
    assert out["masks"].sum().item() == out["abr_selected_valid_k"]
    assert out["abr_selected_positions_window_local"].tolist() == out["abr_selected_positions"].tolist()
    assert out["frame_inds"][: out["abr_selected_valid_k"]].tolist() == out[
        "abr_selected_positions_original_dense"
    ].tolist()
    assert out["irregular_selected_positions"].shape[0] == out["abr_selected_valid_k"]
    assert out["irregular_selected_valid_len"] == 80.0
    assert out["abr_selection_ledger"]["route_label"] == ABR_ROUTE_LABEL
    assert out["abr_selection_ledger"]["cost"]["detector_forward_count"] == 1
    assert out["abr_selection_ledger"]["handoff"]["valid_k"] == out["abr_selected_valid_k"]
    assert out["abr_scout_source"] == "abr_scout_curve:deploy_visible"
    assert out["abr_diagnostic_fallback_used"] is False


def test_nonzero_dense_window_preserves_local_global_and_raw_frame_relationship():
    dense_window = list(range(120, 200))
    out = apply_abr_to_results(
        {
            "video_name": "nonzero_window",
            "total_frames": 240,
            "avg_fps": 25.0,
            "fps": 25.0,
            "duration": 9.6,
            "snippet_stride": 1,
            "abr_frame_signal": [0.02] * 140 + [0.9] * 12 + [0.1] * 32 + [0.8] * 10 + [0.03] * 46,
        },
        dense_window=dense_window,
        config=ABRConfig(k0=8, k1_cap=8, k2_cap=2, max_total_k=20, max_gap=14, target_frame_num=24),
    )

    valid_k = out["abr_selected_valid_k"]
    local = out["abr_selected_positions_window_local"].tolist()
    global_pos = out["abr_selected_positions_original_dense"].tolist()
    assert local == sorted(set(local))
    assert global_pos == [dense_window[pos] for pos in local]
    assert out["frame_inds"][:valid_k].tolist() == global_pos
    handoff = out["abr_selection_ledger"]["handoff"]
    assert handoff["selected_positions_window_local"][:valid_k] == local
    assert handoff["selected_positions_original_dense"][:valid_k] == global_pos
    assert handoff["frame_inds_raw"][:valid_k] == global_pos
    assert handoff["dense_window_start"] == 120
    assert out["abr_scout_source"] == "abr_frame_signal:deploy_visible_metadata"


def test_real_loadframes_method_abr_active_bracket_refinement_is_integrated_in_real_file():
    source = (REPO_ROOT / "opentad" / "datasets" / "transforms" / "end_to_end.py").read_text(encoding="utf-8")
    assert "abr_active_bracket_refinement" in source
    assert "apply_abr_to_results" in source
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from opentad.datasets.transforms.end_to_end import LoadFrames\n"
                "loader=LoadFrames(num_clips=1,method='abr_active_bracket_refinement',method_base='sliding_window',"
                "target_len=24,scale_factor=1,abr_config=dict(k0=8,k1_cap=8,k2_cap=2,max_total_k=20,max_gap=14,target_frame_num=24))\n"
                "out=loader(dict(video_name='real_loader_mock',total_frames=80,avg_fps=25.0,fps=25.0,duration=3.2,"
                "snippet_stride=1,window_size=80,feature_start_idx=0,feature_end_idx=79,"
                "abr_scout_curve=[0.02]*20+[0.82]*8+[0.10]*20+[0.88]*8+[0.04]*24))\n"
                "assert out['frame_inds'].shape[0] == 24\n"
                "assert out['clip_len'] == 24\n"
                "assert out['abr_route_label'] == 'DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3'\n"
                "assert out['abr_scout_source'] == 'abr_scout_curve:deploy_visible'\n"
                "assert out['irregular_native_axis'] == 'abr_original_dense_time'\n"
                "assert int(out['masks'].sum().item()) == out['abr_selected_valid_k']\n"
            ),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0 and ("torch" in proc.stderr.lower() or "c10.dll" in proc.stderr.lower()):
        pytest.skip(f"real LoadFrames smoke locked by local torch import failure: {proc.stderr.strip()}")
    assert proc.returncode == 0, proc.stderr


def test_apply_abr_rejects_eval_gt_teacher_cache_and_detector_feedback():
    base = {"video_name": "leaky", "total_frames": 32, "avg_fps": 25.0, "gt_segments": [[1.0, 2.0]]}
    with pytest.raises(ABRValidationError):
        apply_abr_to_results(base, config=ABRConfig())

    for key in ("teacher_logits", "prediction_cache_path", "detector_feedback"):
        leaky = {"video_name": "leaky", "total_frames": 32, "avg_fps": 25.0, key: object()}
        with pytest.raises(ABRValidationError):
            apply_abr_to_results(leaky, config=ABRConfig())
    for key in ("gt_scout_curve", "teacher_scout_curve", "prediction_scout_curve"):
        leaky = {"video_name": "leaky", "total_frames": 32, "avg_fps": 25.0, key: [0.1] * 32}
        with pytest.raises(ABRValidationError):
            apply_abr_to_results(leaky, config=ABRConfig())


def test_missing_scout_is_locked_unless_precheck_fallback_is_explicit():
    base = {
        "video_name": "missing_scout",
        "total_frames": 80,
        "avg_fps": 25.0,
        "fps": 25.0,
        "duration": 3.2,
        "snippet_stride": 1,
    }
    with pytest.raises(ABRValidationError):
        apply_abr_to_results(base, config=ABRConfig(target_frame_num=24))

    out = apply_abr_to_results(
        dict(base),
        config=ABRConfig(
            k0=8,
            k1_cap=8,
            k2_cap=2,
            max_total_k=20,
            max_gap=14,
            target_frame_num=24,
            allow_diagnostic_fallback_scout=True,
            fallback_stage="PRECHECK_ONLY",
        ),
    )
    assert out["abr_diagnostic_fallback_used"] is True
    assert out["abr_scout_source"] == "diagnostic_fallback:PRECHECK_ONLY"


def test_candidate_config_keeps_val_and_test_gt_selection_gate_closed():
    source = (REPO_ROOT / "configs" / "adatad" / "thumos" / "input_abr_active_bracket_refinement_adapter_irregular_headv3.py").read_text(
        encoding="utf-8"
    )
    assert "DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3" in source
    old_label = "DIVERGENT_INNOVATION_" + "BOUNDARY" + "_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
    assert old_label not in source
    assert "abr_allow_gt_after_selection=True" in source
    assert source.count("abr_allow_gt_after_selection=False") >= 2
    assert "allow_diagnostic_fallback_scout=True" in source
    assert 'fallback_stage="PRECHECK_ONLY"' in source


def test_mock_precheck_records_nonzero_window_and_gt_rejection():
    payload = build_precheck_summary(require_torch=False)
    summary = payload["summary"]
    assert payload["status"] == "PASS_PRECHECK_ONLY"
    assert summary["nonzero_window_ok"] is True
    assert summary["val_test_gt_rejection_ok"] is True
    assert summary["deploy_visible_scout_or_explicit_fallback_ok"] is True
    assert summary["nonzero_window_start"] == 120
    assert summary["nonzero_window_first_global"] >= 120


def test_launch_gate_requires_route_label_no_forbidden_tokens_and_validated_precheck_summary(tmp_path):
    good = {
        "route_label": ABR_ROUTE_LABEL,
        "method": "abr_active_bracket_refinement",
        "status": "PASS_PRECHECK_ONLY",
        "precheck_validated": True,
            "summary": {
                "real_sparse_handoff_ok": True,
                "forbidden_inputs_ok": True,
                "nonzero_window_ok": True,
                "val_test_gt_rejection_ok": True,
                "dynamic_k_nonconstant": True,
                "deploy_visible_scout_or_explicit_fallback_ok": True,
                "detector_forward_count": 1,
            },
    }
    assert validate_launch_gate_payload(good)["allowed_next_action"] == "REMOTE_PRECHECK_ONLY_REQUEST"

    bad = dict(good)
    bad["status"] = "INCOMPLETE"
    with pytest.raises(ABRValidationError):
        validate_launch_gate_payload(bad)

    bad = json.loads(json.dumps(good))
    bad["notes"] = "borrow C3-Pro signal"
    with pytest.raises(ABRValidationError):
        validate_launch_gate_payload(bad)

    payload = tmp_path / "precheck.json"
    payload.write_text(json.dumps(bad), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "abr" / "validate_abr_launch_gate.py"),
            "--precheck-json",
            str(payload),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "LOCKED" in proc.stdout

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "abr" / "validate_abr_launch_gate.py"),
            "--config",
            str(REPO_ROOT / "configs" / "adatad" / "thumos" / "input_abr_active_bracket_refinement_adapter_irregular_headv3.py"),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "LOCAL_PRECHECK_ONLY_VALIDATION" in proc.stdout
