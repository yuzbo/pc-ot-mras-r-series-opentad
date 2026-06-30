from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "input_mdl_knot_dynamic_adapter_irregular_headv3.py"
COLLECTOR_PATH = ROOT / "tools" / "mdl_knot" / "collect_mdl_knot_real_video_diagnostics.py"
VALIDATOR_PATH = ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"
ROUTE_LABEL = "DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"


def _write_shortdiag_log(tmp_path: Path) -> Path:
    log_path = tmp_path / "mdl_knot_shortdiag_one_epoch.log"
    log_path.write_text(
        "\n".join(
            [
                "Epoch [1][1/2] Loss 2.0 loss_cls: 0.5",
                "Epoch [1][2/2] loss_bbox=0.1 finite diagnostic train step complete",
                "Training short diagnostic stopped after one epoch without evaluation",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def _summary_from_collector(tmp_path: Path, *extra_args: str) -> tuple[dict, Path]:
    out_path = tmp_path / "mdl_knot_realdiag_formal_summary.json"
    cmd = [
        sys.executable,
        str(COLLECTOR_PATH),
        "--config",
        str(CONFIG_PATH),
        "--out",
        str(out_path),
        "--dry-run-fixture",
        "--window-count",
        "4",
        "--shortdiag-log",
        str(_write_shortdiag_log(tmp_path)),
        *extra_args,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(out_path.read_text(encoding="utf-8"))
    return summary, out_path


def _run_formal_validator(summary_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--config",
            str(CONFIG_PATH),
            "--formal-readiness-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def _mark_summary_as_real_video_for_validator(summary: dict) -> dict:
    raw_count = summary["real_video_pipeline_diagnostics"]["raw_frame_scout_count"]
    summary["source_mode"] = "real_video_pipeline"
    summary["dry_run_fixture"] = False
    summary["annotation_path"] = str(ROOT / "figures" / "cache" / "thumos_14_anno.json")
    summary["video_root"] = [str(ROOT / "test_videos")]
    summary["fixture_window_count"] = 0
    summary["real_video_reader_window_count"] = raw_count
    summary["real_video_raw_scout_count"] = raw_count
    summary["reader_backend"] = {
        "mode": "annotation_video_root",
        "decord_available": True,
        "real_video_reader_required_for_formal_readiness": True,
        "loader": "opentad.datasets.transforms.end_to_end.LoadFrames",
        "equivalent_loader_fallback": False,
    }
    return summary


def test_realdiag_collector_emits_fixture_schema_but_formal_gate_rejects_it(tmp_path):
    summary, summary_path = _summary_from_collector(tmp_path)

    assert summary["route_label"] == ROUTE_LABEL
    assert summary["source_mode"] == "fixture_schema_only"
    assert summary["dry_run_fixture"] is True
    assert summary["annotation_path"] is None
    assert summary["video_root"] == []
    assert summary["fixture_window_count"] == 4
    assert summary["real_video_reader_window_count"] == 0
    assert summary["reader_backend"]["mode"] == "fixture"
    assert isinstance(summary["reader_backend"]["decord_available"], bool)
    assert summary["synthetic"] is False
    assert summary["formal_train_unlocked"] is False
    assert summary["no_mAP"] is True
    assert summary["no_tools_test"] is True
    assert summary["no_train"] is True
    diag = summary["real_video_pipeline_diagnostics"]
    assert diag["synthetic"] is False
    assert diag["raw_frame_scout_count"] >= 1
    assert diag["raw_frame_scout_ratio"] > 0.0
    assert diag["metadata_fallback_count"] >= 1
    assert diag["synthetic_fallback_count"] == 0
    assert diag["valid_k_distribution"]["nonconstant"] is True
    assert diag["selected_gap_stats"]["count"] > 0
    assert diag["mask_meta_alignment_status"] == "aligned"
    assert diag["short_transition_guard_coverage"]["placeholder_or_measured"] in {"measured", "placeholder"}
    assert diag["fixed_pad_sparse_compute_claim_locked"] is True
    assert diag["route_drift_claim_lock"]["locked"] is True

    proc = _run_formal_validator(summary_path)
    assert proc.returncode != 0
    assert "fixture" in proc.stdout.lower() or "dry-run" in proc.stdout.lower()
    assert "Still locked" in proc.stdout


def test_realdiag_formal_validator_rejects_synthetic_summary(tmp_path):
    summary, summary_path = _summary_from_collector(tmp_path)
    summary["synthetic"] = True
    summary["real_video_pipeline_diagnostics"]["synthetic"] = True
    summary["real_video_pipeline_diagnostics"]["synthetic_fallback_count"] = 1
    summary["real_video_pipeline_diagnostics"]["synthetic_fallback_windows"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    proc = _run_formal_validator(summary_path)

    assert proc.returncode != 0
    assert "synthetic" in proc.stdout.lower()


def test_realdiag_formal_validator_rejects_constant_valid_k(tmp_path):
    summary, summary_path = _summary_from_collector(tmp_path)
    _mark_summary_as_real_video_for_validator(summary)
    summary["real_video_pipeline_diagnostics"]["valid_k_distribution"].update(
        {"min": 12, "max": 12, "mean": 12.0, "std": 0.0, "unique_count": 1, "nonconstant": False}
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    proc = _run_formal_validator(summary_path)

    assert proc.returncode != 0
    assert "valid_k" in proc.stdout or "dynamic" in proc.stdout


@pytest.mark.parametrize("token", ["GlobalRank", "C3"])
def test_realdiag_formal_validator_rejects_route_drift_tokens(tmp_path, token):
    summary, summary_path = _summary_from_collector(tmp_path)
    _mark_summary_as_real_video_for_validator(summary)
    summary["real_video_pipeline_diagnostics"]["route_drift_claim_lock"]["tokens"] = [token]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    proc = _run_formal_validator(summary_path)

    assert proc.returncode != 0
    assert "route drift" in proc.stdout


def test_realdiag_formal_validator_rejects_claim_marker(tmp_path):
    summary, summary_path = _summary_from_collector(tmp_path)
    _mark_summary_as_real_video_for_validator(summary)
    summary["real_video_pipeline_diagnostics"]["route_drift_claim_lock"]["claim_markers"] = [
        "sparse_compute_claim=true"
    ]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    proc = _run_formal_validator(summary_path)

    assert proc.returncode != 0
    assert "claim" in proc.stdout.lower()
