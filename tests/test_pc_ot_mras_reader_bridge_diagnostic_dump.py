import json

from tools.bata.dump_pc_ot_mras_reader_bridge_diagnostics import (
    NO_GO,
    READY,
    diagnose_reader_out_sample,
    main,
    run_jsonl_diagnostic,
    run_synthetic_diagnostic,
)


def _reader_out():
    return {
        "acquisition_matrix": [
            [
                [0.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6],
                [0.0, 0.7, 0.0, 0.0],
            ]
        ],
        "allocation": [
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ],
        "valid_mask": [[1, 1, 1, 1]],
        "selected_mask": [[1, 1, 1]],
        "selected_times": [[0.25, 0.75, 0.50]],
        "centers": [[0.25, 0.70, 0.40]],
        "gates": [[0.2, 0.8, 0.5]],
        "selected_tokens": [
            [
                [3.0, 4.0],
                [0.0, 2.0],
                [1.0, 2.0],
            ]
        ],
    }


def test_diagnose_reader_out_sample_computes_core_reader_bridge_stats():
    sample = diagnose_reader_out_sample(
        _reader_out(),
        sample_id="video_test_000001",
        snapshot_id="unit",
        row_bridge_norm=("bridge_output", [6.0]),
    )

    assert sample["schema_version"] == "pc_ot_mras_reader_bridge_diagnostic_dump_v0"
    assert sample["matrix_key"] == "acquisition_matrix"
    assert sample["centers_selected_times"]["abs_offset"]["count"] == 3
    assert abs(sample["centers_selected_times"]["abs_offset"]["max"] - 0.1) < 1.0e-9
    assert sample["selected_times_monotonicity"]["nondecreasing"] is False
    assert sample["selected_times_monotonicity"]["violation_count"] == 1
    assert sample["gates"]["histogram_0_0p25_0p5_0p75_1"]["[0,0.25)"] == 1
    assert sample["gates"]["histogram_0_0p25_0p5_0p75_1"]["[0.75,1]"] == 1
    assert sample["acquisition"]["top1_positions_preview"] == [1, 3, 1]
    assert sample["selected_token_norm"]["available"] is True
    assert sample["selected_token_norm"]["stats"]["max"] == 5.0
    assert sample["bridge_output_norm"]["stats"]["mean"] == 6.0
    assert sample["uses_gt"] is False
    assert sample["metric_claim_allowed"] is False


def test_jsonl_diagnostic_reads_snapshot_rows_and_writes_summary(tmp_path):
    input_jsonl = tmp_path / "snapshot.jsonl"
    output_json = tmp_path / "summary.json"
    row = {
        "sample_ids": ["sample_a"],
        "snapshot_id": "epoch_001",
        "reader_out": _reader_out(),
        "bridge_output": [[1.0, 2.0, 2.0]],
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_jsonl_diagnostic(input_jsonl, output_json=output_json)

    assert summary["decision"] == READY
    assert summary["source"] == "jsonl"
    assert summary["sample_count"] == 1
    assert summary["aggregate"]["selected_times_nondecreasing_failure_count"] == 1
    assert summary["aggregate"]["gate"]["min"] == 0.2
    assert summary["aggregate"]["gate"]["max"] == 0.8
    assert summary["aggregate"]["bridge_output_norm"]["mean"] == 3.0
    assert "_aggregate_values" not in json.dumps(summary)
    assert output_json.is_file()


def test_synthetic_diagnostic_is_torch_free_and_contains_expected_sections(tmp_path):
    output_json = tmp_path / "synthetic_summary.json"

    summary = run_synthetic_diagnostic(
        output_json=output_json,
        batch_size=2,
        slots=3,
        time=6,
        channels=2,
    )

    assert summary["decision"] == READY
    assert summary["source"] == "synthetic"
    assert summary["sample_count"] == 2
    assert summary["aggregate"]["acquisition_entropy"]["count"] == 6
    assert summary["aggregate"]["selected_token_norm"]["count"] == 6
    assert summary["aggregate"]["gate"]["min"] == 0.35
    assert summary["aggregate"]["gate"]["max"] == 0.65
    assert output_json.is_file()


def test_cli_empty_jsonl_returns_structured_no_go_without_traceback(tmp_path, capsys):
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")

    exit_code = main(["--mode", "jsonl", "--input-jsonl", str(empty_jsonl)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["decision"] == NO_GO
    assert payload["schema_version"] == "pc_ot_mras_reader_bridge_diagnostic_summary_v0"
    assert payload["error_type"] == "ValueError"
    assert "traceback" not in captured.out.lower()
