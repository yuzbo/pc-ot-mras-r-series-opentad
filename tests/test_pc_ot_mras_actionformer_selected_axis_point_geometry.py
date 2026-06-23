import json

import pytest

from tools.bata.analyze_actionformer_selected_axis_point_geometry import (
    READY,
    iter_snapshot_samples,
    run_selected_axis_point_geometry_audit,
)


def _snapshot_row(**overrides):
    row = {
        "schema_version": "pc_ot_mras_reader_snapshot_dump_v0",
        "snapshot_id": "synthetic",
        "sample_ids": ["video_0001"],
        "reader_out": {
            "selected_times": [[0.0, 2.0 / 7.0, 4.0 / 7.0, 1.0]],
            "selected_mask": [[1, 1, 1, 1]],
            "valid_lengths": [8],
            "valid_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
        },
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
    }
    row.update(overrides)
    return row


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_selected_axis_point_geometry_audit_reconstructs_normalized_selected_times(tmp_path):
    snapshot = tmp_path / "snapshot.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(snapshot, [_snapshot_row()])

    summary = run_selected_axis_point_geometry_audit(
        snapshot_jsonl=[snapshot],
        output_dir=output_dir,
        strides=(1, 2),
    )

    assert summary["decision"] == READY
    assert summary["sample_count"] == 1
    assert summary["position_source_counts"] == {"selected_times": 1}
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["no_training"] is True
    assert summary["no_model_forward"] is True

    rows = (output_dir / "per_sample_level_geometry.csv").read_text(encoding="utf-8").splitlines()
    assert "nominal_stride" in rows[0]
    stride1 = next(row for row in rows[1:] if ",1," in row)
    assert "0.42857142857142855" in stride1
    aggregate = summary["aggregate_by_nominal_stride"]["1"]
    assert aggregate["fake_span_to_physical_span_ratio_mean"] == pytest.approx(3.0 / 7.0)
    assert aggregate["abs_delta_mean"] == pytest.approx((0.0 + 1.0 + 2.0 + 4.0) / 4.0)


def test_selected_axis_point_geometry_audit_falls_back_to_acquisition_matrix(tmp_path):
    snapshot = tmp_path / "snapshot.jsonl"
    output_dir = tmp_path / "out"
    row = _snapshot_row(
        reader_out={
            "acquisition_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.25, 0.75],
                ]
            ],
            "selected_mask": [[1, 1, 1]],
            "valid_lengths": [6],
            "valid_mask": [[1, 1, 1, 1, 1, 1]],
        }
    )
    _write_jsonl(snapshot, [row])

    samples = iter_snapshot_samples(snapshot)
    assert samples[0]["position_source"] == "acquisition_matrix"
    assert samples[0]["positions"] == pytest.approx([0.0, 2.0, 4.75])

    summary = run_selected_axis_point_geometry_audit(
        snapshot_jsonl=[snapshot],
        output_dir=output_dir,
        strides=(1,),
    )
    assert summary["position_source_counts"] == {"acquisition_matrix": 1}
    assert summary["aggregate_by_nominal_stride"]["1"]["abs_delta_mean"] == pytest.approx(
        (0.0 + 1.0 + 2.75) / 3.0
    )


def test_selected_axis_point_geometry_audit_rejects_metric_claim_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot.jsonl"
    _write_jsonl(snapshot, [_snapshot_row(metric_claim_allowed=True)])

    with pytest.raises(ValueError, match="forbidden provenance flag"):
        run_selected_axis_point_geometry_audit(
            snapshot_jsonl=[snapshot],
            output_dir=tmp_path / "out",
            strides=(1,),
        )
