import torch
import pytest

from tools.bata.dump_pc_ot_mras_reader_snapshots import (
    READER_OUTPUT_KEYS,
    ReaderOutputHook,
    make_snapshot_row,
    sample_ids_from_metas,
    serialize_reader_outputs,
)


def _reader_outputs():
    return {
        "acquisition_matrix": torch.tensor(
            [
                [
                    [0.0, 0.9, 0.0],
                    [0.8, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        "allocation": torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        "valid_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "gates": torch.tensor([[0.75, 0.25]], dtype=torch.float32),
        "centers": torch.tensor([[0.2, 0.7]], dtype=torch.float32),
        "role_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "round_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "regularizers": {"total_regularizer": torch.tensor(0.1)},
        "slot_state": torch.ones(1, 2, 4),
    }


def test_serialize_reader_outputs_keeps_whitelist_and_jsonable_values():
    serialized = serialize_reader_outputs(_reader_outputs(), float_digits=4)

    assert "acquisition_matrix" in serialized
    assert "allocation" in serialized
    assert "valid_mask" in serialized
    assert "regularizers" not in serialized
    assert "slot_state" not in serialized
    assert set(serialized).issubset(set(READER_OUTPUT_KEYS))
    assert serialized["acquisition_matrix"][0][0] == [0.0, 0.9, 0.0]
    assert serialized["valid_mask"] == [[True, True, True]]


def test_serialize_reader_outputs_requires_valid_mask_and_matrix_signal():
    with pytest.raises(ValueError, match="valid_mask"):
        serialize_reader_outputs({"allocation": torch.ones(1, 2, 3)})

    with pytest.raises(ValueError, match="acquisition_matrix, allocation, or transport_prob"):
        serialize_reader_outputs({"valid_mask": torch.ones(1, 3, dtype=torch.bool)})


def test_make_snapshot_row_is_diagnostic_only_and_visualizer_compatible():
    row = make_snapshot_row(
        sample_ids=["video_test_000001"],
        reader_outputs=_reader_outputs(),
        snapshot_id="r17_epoch_012",
        epoch=12,
    )

    assert row["schema_version"] == "pc_ot_mras_reader_snapshot_dump_v0"
    assert row["snapshot_id"] == "r17_epoch_012"
    assert row["epoch"] == 12
    assert row["sample_ids"] == ["video_test_000001"]
    assert row["diagnostic_only"] is True
    assert row["uses_gt"] is False
    assert row["uses_teacher"] is False
    assert row["uses_raw_prediction"] is False
    assert row["metric_claim_allowed"] is False
    assert "checkpoint" not in row
    assert "reader_out" in row


def test_sample_ids_from_metas_uses_safe_identifiers_only():
    metas = [
        {"video_name": "video_test_000001", "gt_segments": torch.tensor([[0.0, 1.0]])},
        {"filename": "feature_000002.npy", "labels": [1, 2]},
        {"duration": 12.0},
    ]

    assert sample_ids_from_metas(metas, seen_count=5) == [
        "video_test_000001",
        "feature_000002.npy",
        "sample_7",
    ]


def test_reader_output_hook_captures_and_clears_mapping():
    hook = ReaderOutputHook()
    payload = _reader_outputs()

    hook(None, None, payload)
    assert hook.pop() is payload
    with pytest.raises(RuntimeError, match="did not capture"):
        hook.pop()


def test_reader_output_hook_rejects_non_mapping():
    hook = ReaderOutputHook()

    with pytest.raises(ValueError, match="expected mapping output"):
        hook(None, None, torch.ones(1))
