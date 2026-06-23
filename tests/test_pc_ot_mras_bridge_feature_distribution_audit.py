import json

from tools.bata.audit_pc_ot_mras_bridge_feature_distribution import (
    EVIDENCE_INSUFFICIENT,
    READY,
    build_audit,
    main,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base_reader_out():
    return {
        "acquisition_matrix": [
            [
                [0.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6],
            ]
        ],
        "allocation": [
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        "valid_mask": [[1, 1, 1, 1]],
        "selected_mask": [[1, 1]],
        "selected_times": [[0.25, 0.75]],
        "centers": [[0.20, 0.70]],
        "gates": [[0.8, 0.4]],
    }


def test_build_audit_reports_ready_when_selected_and_bridge_features_are_visible(tmp_path):
    input_jsonl = tmp_path / "feature_rows.jsonl"
    reader_out = _base_reader_out()
    reader_out["selected_tokens"] = [
        [
            [3.0, 4.0],
            [0.0, 2.0],
        ]
    ]
    _write_jsonl(
        input_jsonl,
        [
            {
                "sample_ids": ["video_test_000001"],
                "snapshot_id": "unit_epoch",
                "reader_out": reader_out,
                "bridge_output": [[1.0, 2.0, 2.0]],
            }
        ],
    )

    summary = build_audit([("unit", input_jsonl)])

    assert summary["decision"] == READY
    assert summary["feature_distribution_ready_count"] == 1
    run = summary["runs"][0]
    assert run["feature_distribution_evidence_ready"] is True
    assert run["missing_feature_evidence"] == []
    assert run["selected_token_norm"]["count"] == 2
    assert run["bridge_output_norm"]["count"] == 1
    assert run["bridge_to_selected_token_norm_mean_ratio"] is not None


def test_build_audit_marks_snapshot_only_rows_as_feature_evidence_insufficient(tmp_path):
    input_jsonl = tmp_path / "snapshot_rows.jsonl"
    _write_jsonl(
        input_jsonl,
        [
            {
                "sample_ids": ["video_test_000001"],
                "snapshot_id": "snapshot_only",
                "reader_out": _base_reader_out(),
            }
        ],
    )

    summary = build_audit([("snapshot", input_jsonl)])

    assert summary["decision"] == EVIDENCE_INSUFFICIENT
    assert summary["feature_distribution_ready_count"] == 0
    assert summary["missing_feature_evidence_by_label"] == {
        "snapshot": [
            "selected_token_values_not_visible_in_jsonl",
            "bridge_output_values_not_visible_in_jsonl",
        ]
    }
    assert summary["runs"][0]["reader_temporal_context"]["gate"]["count"] == 2
    assert summary["runs"][0]["tools_test_allowed"] is False


def test_cli_writes_summary_json_for_insufficient_feature_evidence(tmp_path, capsys):
    input_jsonl = tmp_path / "snapshot_rows.jsonl"
    output_json = tmp_path / "summary.json"
    _write_jsonl(
        input_jsonl,
        [
            {
                "sample_ids": ["video_test_000001"],
                "snapshot_id": "snapshot_only",
                "reader_out": _base_reader_out(),
            }
        ],
    )

    exit_code = main(["--input", f"snapshot={input_jsonl}", "--output-json", str(output_json)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["decision"] == EVIDENCE_INSUFFICIENT
    assert output_json.is_file()
