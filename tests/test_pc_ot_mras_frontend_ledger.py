import json
import importlib.util
import sys
from pathlib import Path

from mmengine.config import Config

from tools.bata.convert_pc_ot_mras_hard_positions_to_value_transport_ledger import READY, run_conversion
from tools.bata.dump_pc_ot_mras_reader_snapshots import sample_ids_from_metas


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = ROOT / "opentad" / "datasets" / "transforms" / "boundary_acquisition.py"
BOUNDARY_SPEC = importlib.util.spec_from_file_location("pcot_frontend_test_boundary_acquisition", BOUNDARY_PATH)
BOUNDARY_MODULE = importlib.util.module_from_spec(BOUNDARY_SPEC)
sys.modules[BOUNDARY_SPEC.name] = BOUNDARY_MODULE
BOUNDARY_SPEC.loader.exec_module(BOUNDARY_MODULE)
validate_value_transport_selection_row = BOUNDARY_MODULE.validate_value_transport_selection_row


def test_reader_snapshot_sample_ids_match_value_transport_window_key():
    metas = [
        {"video_name": "video_test_0001", "window_start_frame": 0},
        {"video_id": "video_test_0001", "window_start_frame": 768.0},
    ]

    assert sample_ids_from_metas(metas) == ["video_test_0001|0", "video_test_0001|768"]


def test_pc_ot_mras_hard_positions_convert_to_diagnostic_value_transport_ledger(tmp_path):
    input_jsonl = tmp_path / "hard_positions.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    summary_json = tmp_path / "summary.json"
    hard_row = {
        "schema_version": "pc_ot_mras_hard_positions_v0",
        "sample_id": "video_test_0001|0",
        "batch_index": 0,
        "budget": 3,
        "dense_len": 6,
        "valid_len": 6,
        "selected_positions": [0, 2, 5],
        "duplicate_repair_count": 1,
        "invalid_repair_count": 0,
        "repair_fill_count": 1,
        "soft_hard_time_error": 0.25,
        "role_round_metadata": [{"position": 0, "role_id": 1, "round_id": 0}],
    }
    input_jsonl.write_text(json.dumps(hard_row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_conversion(
        input_jsonl,
        output_jsonl,
        target_len=3,
        summary_json=summary_json,
        require_selected_count=3,
    )

    assert summary["decision"] == READY
    assert summary["row_count"] == 1
    written = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    row = written[0]
    assert row["schema_version"] == "pc_ot_mras_frontend_value_transport_ledger_v0"
    assert row["sample_id"] == "video_test_0001|0"
    assert row["selected_positions_unit"] == "local_dense_index"
    assert row["selected_positions"] == [0, 2, 5]
    assert row["target_len"] == 3
    assert row["selected_count"] == 3
    assert row["diagnostic_only"] is True
    assert row["deploy_selection_ledger"] is False
    assert row["uses_gt"] is False
    assert row["uses_teacher"] is False
    assert row["uses_raw_prediction"] is False
    validate_value_transport_selection_row(row, line_no=1, require_deployable=False)
    assert Path(summary_json).is_file()


def test_pc_ot_mras_frontend_eval_config_keeps_original_detector_stack_and_selected_axis_loader():
    cfg = Config.fromfile(str(ROOT / "configs" / "adatad" / "thumos" / "pc_ot_mras_frontend_hard_ledger_fixed50_adapter_n16r4.py"))

    assert int(cfg.window_size) == 384
    assert int(cfg.dense_window_size) == 768
    assert int(cfg.model.backbone.backbone.total_frames) == 384
    assert int(cfg.model.projection.max_seq_len) == 384
    assert "pc_ot_mras_reader" not in cfg.model
    assert "frame_selector" not in cfg.model
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.changes_input_sampling is True
    assert cfg.experiment_scope.changes_detector_head is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    for split in ("val", "test"):
        loadframes = cfg.dataset[split].pipeline[2]
        assert loadframes.method == "bata_value_transport_ledger_subsample"
        assert loadframes.method_base == "sliding_window"
        assert int(loadframes.target_len) == 384
        assert loadframes.remap_gt_to_selected_axis is True
        assert loadframes.bata_value_transport_allow_missing_fallback is False
        assert loadframes.bata_value_transport_require_deployable is False
        assert loadframes.bata_value_transport_source == "pc_ot_mras_frontend_hard_positions"
