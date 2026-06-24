import json
from pathlib import Path

import pytest

from tools.bata.export_pc_ot_mras_hard_positions import (
    READY,
    resolve_pc_ot_mras_hard_positions,
    run_jsonl_export,
)


def test_hard_export_resolves_sorted_unique_exact_budget_and_repairs_duplicates():
    out = {
        "allocation": [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0],
                [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            ]
        ],
        "soft_selection": [[0.1, 0.2, 0.8, 0.6, 0.7, 0.4, 0.9, 0.3]],
        "valid_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
        "role_ids": [[10, 11, 12, 13, 14]],
        "round_ids": [[0, 1, 2, 3, 4]],
    }

    rows = resolve_pc_ot_mras_hard_positions(out, budget=5, sample_ids=["synthetic|0"])
    row = rows[0]

    assert row["sample_id"] == "synthetic|0"
    assert row["selected_positions"] == sorted(row["selected_positions"])
    assert len(row["selected_positions"]) == 5
    assert len(set(row["selected_positions"])) == 5
    assert row["selected_mask"] == [1 if idx in row["selected_positions"] else 0 for idx in range(8)]
    assert row["duplicate_repair_count"] == 1
    assert row["repair_fill_count"] == 1
    assert isinstance(row["soft_hard_time_error"], float)
    assert row["soft_hard_time_error"] >= 0
    assert row["resolver_generation"]["training_backprop_allowed"] is False
    assert row["resolver_generation"]["detached_reader_tensors"] is True
    assert isinstance(row["selected_positions"], list)


def test_hard_export_preserves_role_and_round_metadata_for_unique_slot_positions():
    rows = resolve_pc_ot_mras_hard_positions(
        {
            "allocation": [
                [
                    [0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.7, 0.0],
                ]
            ],
            "valid_mask": [[1, 1, 1, 1, 1, 1]],
            "role_ids": [[4, 2, 9]],
            "round_ids": [[0, 3, 1]],
        },
        budget=3,
        sample_ids=["meta|0"],
    )
    row = rows[0]

    assert row["selected_positions"] == [1, 3, 4]
    assert row["role_ids"] == [4, 2, 9]
    assert row["round_ids"] == [0, 3, 1]
    assert row["role_round_metadata"] == [
        {"position": 1, "role_id": 4, "round_id": 0},
        {"position": 3, "role_id": 2, "round_id": 3},
        {"position": 4, "role_id": 9, "round_id": 1},
    ]


def test_hard_export_supports_acquisition_matrix_only():
    rows = resolve_pc_ot_mras_hard_positions(
        {
            "acquisition_matrix": [
                [
                    [0.0, 0.10, 0.0, 0.0, 0.0],
                    [0.0, 0.85, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.20, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.95],
                ]
            ],
            "valid_mask": [[1, 1, 1, 1, 1]],
        },
        budget=2,
        sample_ids=["acq-only|0"],
    )

    assert rows[0]["selected_positions"] == [1, 4]
    assert rows[0]["repair_fill_count"] == 0


def test_hard_export_prefers_acquisition_matrix_when_allocation_conflicts():
    rows = resolve_pc_ot_mras_hard_positions(
        {
            "acquisition_matrix": [
                [
                    [0.0, 0.0, 0.0, 0.91, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.82],
                ]
            ],
            "allocation": [
                [
                    [0.99, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.98, 0.0, 0.0, 0.0],
                ]
            ],
            "transport_prob": [
                [
                    [0.0, 0.0, 0.97, 0.0, 0.0],
                    [0.96, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            "soft_selection": [[0.99, 0.98, 0.97, 0.01, 0.01]],
            "valid_mask": [[1, 1, 1, 1, 1]],
        },
        budget=2,
        sample_ids=["acq-conflict|0"],
    )

    assert rows[0]["selected_positions"] == [3, 4]


def test_hard_export_ranks_global_matrix_candidates_when_slots_exceed_budget():
    rows = resolve_pc_ot_mras_hard_positions(
        {
            "acquisition_matrix": [
                [
                    [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50],
                    [0.0, 0.0, 0.0, 0.0, 0.90, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.0],
                ]
            ],
            "allocation": [
                [
                    [0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.0],
                ]
            ],
            "valid_mask": [[1, 1, 1, 1, 1, 1, 1]],
        },
        budget=2,
        sample_ids=["acq-budget|0"],
    )

    assert rows[0]["selected_positions"] == [4, 5]


def test_hard_export_prefers_explicit_selected_positions_over_conflicting_acquisition_matrix():
    rows = resolve_pc_ot_mras_hard_positions(
        {
            "acquisition_matrix": [
                [
                    [0.0, 0.0, 0.0, 0.91, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.82],
                ]
            ],
            "selected_positions": [[0, 1]],
            "valid_mask": [[1, 1, 1, 1, 1]],
        },
        budget=2,
        sample_ids=["acq-vs-selected|0"],
    )

    assert rows[0]["selected_positions"] == [0, 1]
    assert rows[0]["resolver_generation"]["selected_position_source"] == "selected_positions"


def test_hard_export_prefers_explicit_hard_positions_over_conflicting_acquisition_matrix():
    rows = resolve_pc_ot_mras_hard_positions(
        {
            "acquisition_matrix": [
                [
                    [0.0, 0.0, 0.0, 0.91, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.82],
                ]
            ],
            "hard_selected_positions": [[0, 1]],
            "valid_mask": [[1, 1, 1, 1, 1]],
        },
        budget=2,
        sample_ids=["acq-vs-hard|0"],
    )

    assert rows[0]["selected_positions"] == [0, 1]
    assert rows[0]["resolver_generation"]["selected_position_source"] == "hard_selected_positions"


def test_hard_export_jsonl_cli_path_writes_rows_and_summary(tmp_path):
    input_jsonl = tmp_path / "reader_rows.jsonl"
    output_jsonl = tmp_path / "hard_rows.jsonl"
    summary_json = tmp_path / "summary.json"
    row = {
        "sample_id": "jsonl|0",
        "budget": 3,
        "dense_len": 5,
        "reader_out": {
            "allocation": [[[0.0, 0.8, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.0]]],
            "valid_mask": [[1, 1, 1, 1, 1]],
            "soft_selection": [[0.7, 0.8, 0.1, 0.9, 0.0]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_jsonl_export(input_jsonl, output_jsonl, budget=3, summary_json=summary_json)

    assert summary["decision"] == READY
    assert summary["row_count"] == 1
    written = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert written[0]["selected_positions"] == [0, 1, 3]
    assert Path(summary_json).is_file()


@pytest.mark.parametrize(
    ("score_key", "scores"),
    [
        ("selection_logits", [[0.1, 0.9, 0.2, 0.3]]),
        ("selection_prob", [[0.1, 0.9, 0.2, 0.3]]),
        ("soft_selection", [[0.1, 0.9, 0.2, 0.3]]),
    ],
)
def test_score_fallback_paths_remain_usable(score_key, scores):
    rows = resolve_pc_ot_mras_hard_positions(
        {
            score_key: scores,
            "valid_mask": [[1, 1, 1, 1]],
        },
        budget=2,
        sample_ids=[f"{score_key}|0"],
    )

    assert rows[0]["selected_positions"] == [1, 3]
    assert len(rows[0]["selected_positions"]) == 2
    assert len(set(rows[0]["selected_positions"])) == 2
    assert rows[0]["selected_mask"] == [0, 1, 0, 1]


def test_default_score_fallback_remains_usable_without_score_fields():
    rows = resolve_pc_ot_mras_hard_positions(
        {
            "valid_mask": [[1, 1, 1, 1, 1]],
        },
        budget=2,
        sample_ids=["default-score|0"],
    )

    assert rows[0]["selected_positions"] == [1, 2]
    assert len(rows[0]["selected_positions"]) == 2
    assert len(set(rows[0]["selected_positions"])) == 2


def test_hard_export_jsonl_reader_out_prefers_explicit_hard_fields_over_matrix(tmp_path):
    input_jsonl = tmp_path / "reader_rows.jsonl"
    output_jsonl = tmp_path / "hard_rows.jsonl"
    row = {
        "sample_id": "jsonl-mixed|0",
        "budget": 2,
        "dense_len": 5,
        "reader_out": {
            "acquisition_matrix": [
                [
                    [0.0, 0.0, 0.0, 0.91, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.82],
                ]
            ],
            "hard_selected_positions": [[0, 1]],
            "selected_positions": [[1, 2]],
            "valid_mask": [[1, 1, 1, 1, 1]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_jsonl_export(input_jsonl, output_jsonl, budget=2)

    assert summary["decision"] == READY
    written = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert written[0]["selected_positions"] == [0, 1]
    assert written[0]["resolver_generation"]["selected_position_source"] == "hard_selected_positions"


def test_hard_export_jsonl_rejects_row_budget_conflict(tmp_path):
    input_jsonl = tmp_path / "reader_rows.jsonl"
    output_jsonl = tmp_path / "hard_rows.jsonl"
    row = {
        "sample_id": "budget-conflict|0",
        "budget": 1,
        "dense_len": 4,
        "reader_out": {
            "acquisition_matrix": [[[0.0, 0.9, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0]]],
            "valid_mask": [[1, 1, 1, 1]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row budget conflicts with CLI budget"):
        run_jsonl_export(input_jsonl, output_jsonl, budget=2)


def test_hard_export_rejects_non_prefix_valid_mask():
    with pytest.raises(ValueError, match="valid_mask must be prefix-contiguous"):
        resolve_pc_ot_mras_hard_positions(
            {
                "allocation": [
                    [
                        [0.0, 0.9, 0.0, 0.0],
                        [0.0, 0.0, 0.8, 0.0],
                    ]
                ],
                "valid_mask": [[1, 0, 1, 1]],
            },
            budget=2,
            sample_ids=["non-prefix-mask|0"],
        )


def test_hard_export_jsonl_rejects_forbidden_deploy_invisible_keys(tmp_path):
    input_jsonl = tmp_path / "reader_rows.jsonl"
    output_jsonl = tmp_path / "hard_rows.jsonl"
    row = {
        "sample_id": "forbidden|0",
        "budget": 2,
        "dense_len": 4,
        "metadata": {"gt_segments": [[0.1, 0.2]]},
        "reader_out": {
            "acquisition_matrix": [[[0.0, 0.9, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0]]],
            "valid_mask": [[1, 1, 1, 1]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden deploy-invisible key"):
        run_jsonl_export(input_jsonl, output_jsonl, budget=2)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("feature_cache", "cached-features.json"),
        ("prediction", [0.1, 0.2]),
        ("predictions", {"path": "predictions.json"}),
        ("detection_result", "result_detection.json"),
        ("checkpoint_path", "epoch_12.pth"),
    ],
)
def test_hard_export_jsonl_rejects_forbidden_key_variants(tmp_path, key, value):
    input_jsonl = tmp_path / "reader_rows.jsonl"
    output_jsonl = tmp_path / "hard_rows.jsonl"
    row = {
        "sample_id": "forbidden-variant|0",
        "budget": 2,
        "dense_len": 4,
        "metadata": {key: value},
        "reader_out": {
            "acquisition_matrix": [[[0.0, 0.9, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0]]],
            "valid_mask": [[1, 1, 1, 1]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden deploy-invisible key"):
        run_jsonl_export(input_jsonl, output_jsonl, budget=2)


def test_malformed_acquisition_matrix_does_not_fallback_to_legacy_hard_fields():
    with pytest.raises(ValueError, match="acquisition_matrix sample must be \\[K,T\\]"):
        resolve_pc_ot_mras_hard_positions(
            {
                "acquisition_matrix": ["not-a-matrix"],
                "hard_selected_positions": [[0, 1]],
                "selected_positions": [[1, 2]],
                "valid_mask": [[1, 1, 1, 1]],
            },
            budget=2,
            sample_ids=["bad-matrix|0"],
        )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([], "acquisition_matrix sample must be \\[K,T\\]"),
        ([[]], "acquisition_matrix sample must be \\[K,T\\]"),
        ([[0.9, 0.0, 0.0, 0.0], [0.0, 0.8]], "acquisition_matrix sample must be rectangular \\[K,T\\]"),
        ([[0.9, 0.0], [0.0, 0.8]], "acquisition_matrix width must equal dense axis T=4"),
    ],
)
def test_malformed_acquisition_matrix_shape_fails_closed_without_legacy_fallback(matrix, message):
    with pytest.raises(ValueError, match=message):
        resolve_pc_ot_mras_hard_positions(
            {
                "acquisition_matrix": [matrix],
                "hard_selected_positions": [[0, 1]],
                "selected_positions": [[1, 2]],
                "valid_mask": [[1, 1, 1, 1]],
            },
            budget=2,
            sample_ids=["bad-matrix-shape|0"],
        )


@pytest.mark.parametrize("matrix_key", ["allocation", "transport_prob"])
def test_malformed_matrix_priority_fields_fail_closed_without_legacy_fallback(matrix_key):
    with pytest.raises(ValueError, match=f"{matrix_key} sample must be rectangular \\[K,T\\]"):
        resolve_pc_ot_mras_hard_positions(
            {
                matrix_key: [[[0.9, 0.0, 0.0, 0.0], [0.0, 0.8]]],
                "hard_selected_positions": [[0, 1]],
                "selected_positions": [[1, 2]],
                "selection_prob": [[0.1, 0.2, 0.9, 0.8]],
                "valid_mask": [[1, 1, 1, 1]],
            },
            budget=2,
            sample_ids=[f"bad-{matrix_key}|0"],
        )


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
        ("0.9", "must be numeric"),
        (True, "must be numeric"),
    ],
)
def test_matrix_priority_fields_reject_nonfinite_or_nonnumeric_values(bad_value, message):
    with pytest.raises(ValueError, match=message):
        resolve_pc_ot_mras_hard_positions(
            {
                "acquisition_matrix": [[[0.0, bad_value, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0]]],
                "hard_selected_positions": [[0, 1]],
                "valid_mask": [[1, 1, 1, 1]],
            },
            budget=2,
            sample_ids=["bad-value|0"],
        )
