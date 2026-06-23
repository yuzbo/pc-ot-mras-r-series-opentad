import json
import importlib.util
import sys
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.convert_pc_ot_mras_hard_positions_to_value_transport_ledger import READY as LEDGER_READY, run_conversion
from tools.bata.dump_pc_ot_mras_reader_snapshots import sample_ids_from_metas
from tools.bata.export_pc_ot_mras_hard_positions import READY as HARD_READY, run_jsonl_export


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_pc_ot_mras_frontend_ledger_eval_n16r4.sbatch"
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


def test_reader_snapshot_row_exports_to_ledger_with_original_window_sample_id(tmp_path):
    snapshot_jsonl = tmp_path / "reader_snapshots.jsonl"
    hard_jsonl = tmp_path / "hard_positions.jsonl"
    hard_summary_json = tmp_path / "hard_summary.json"
    ledger_jsonl = tmp_path / "value_transport_ledger.jsonl"
    ledger_summary_json = tmp_path / "ledger_summary.json"
    snapshot_row = {
        "sample_ids": ["video_test_0001|0"],
        "budget": 3,
        "dense_len": 6,
        "valid_len": 6,
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "reader_out": {
            "allocation": [
                [
                    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
                ]
            ],
            "soft_selection": [[0.1, 0.8, 0.2, 0.9, 0.3, 0.7]],
            "valid_mask": [[1, 1, 1, 1, 1, 1]],
            "valid_lengths": [6],
        },
    }
    snapshot_jsonl.write_text(json.dumps(snapshot_row, sort_keys=True) + "\n", encoding="utf-8")

    hard_summary = run_jsonl_export(snapshot_jsonl, hard_jsonl, budget=3, summary_json=hard_summary_json)
    ledger_summary = run_conversion(
        hard_jsonl,
        ledger_jsonl,
        target_len=3,
        summary_json=ledger_summary_json,
        require_selected_count=3,
    )

    assert hard_summary["decision"] == HARD_READY
    assert ledger_summary["decision"] == LEDGER_READY
    hard_row = json.loads(hard_jsonl.read_text(encoding="utf-8").splitlines()[0])
    ledger_row = json.loads(ledger_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert hard_row["sample_id"] == "video_test_0001|0"
    assert ledger_row["sample_id"] == "video_test_0001|0"
    assert ledger_row["selected_positions"] == [1, 3, 5]
    assert ledger_row["uses_gt"] is False
    assert ledger_row["uses_teacher"] is False
    assert ledger_row["uses_oracle"] is False
    assert ledger_row["uses_raw_prediction"] is False
    assert ledger_row["uses_checkpoint"] is False
    validate_value_transport_selection_row(ledger_row, line_no=1, require_deployable=False)


def test_reader_snapshot_hard_export_rejects_true_or_payload_forbidden_sources(tmp_path):
    input_jsonl = tmp_path / "reader_snapshots.jsonl"
    output_jsonl = tmp_path / "hard_positions.jsonl"
    base_row = {
        "sample_ids": ["video_test_0001|0"],
        "budget": 1,
        "dense_len": 2,
        "reader_out": {
            "allocation": [[[0.0, 0.9]]],
            "valid_mask": [[1, 1]],
        },
    }

    true_flag = dict(base_row, uses_gt=True)
    input_jsonl.write_text(json.dumps(true_flag, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uses_gt must be JSON false"):
        run_jsonl_export(input_jsonl, output_jsonl, budget=1)

    payload_row = dict(base_row, gt_segments=[[0.0, 1.0]])
    input_jsonl.write_text(json.dumps(payload_row, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden deploy-invisible key"):
        run_jsonl_export(input_jsonl, output_jsonl, budget=1)


@pytest.mark.parametrize(
    ("forbidden_key", "forbidden_value"),
    [
        ("ground_truth", [0.0, 1.0]),
        ("teacher_logits", [0.1, 0.9]),
        ("oracle_scores", [1.0, 0.0]),
        ("prediction_cache_path", "cache/preds.json"),
        ("checkpoint_path", "checkpoint/best.pth"),
    ],
)
def test_reader_snapshot_hard_export_rejects_forbidden_payload_keys(tmp_path, forbidden_key, forbidden_value):
    input_jsonl = tmp_path / "reader_snapshots.jsonl"
    output_jsonl = tmp_path / "hard_positions.jsonl"
    row = {
        "sample_ids": ["video_test_0001|0"],
        "budget": 1,
        "dense_len": 2,
        "reader_out": {
            "allocation": [[[0.0, 0.9]]],
            "valid_mask": [[1, 1]],
            "valid_lengths": [2],
            forbidden_key: forbidden_value,
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden deploy-invisible key"):
        run_jsonl_export(input_jsonl, output_jsonl, budget=1)


@pytest.mark.parametrize("bad_false", [0, 0.0, None, "", "0", "false", "no"])
def test_reader_snapshot_hard_export_requires_json_boolean_false_for_guard_flags(tmp_path, bad_false):
    input_jsonl = tmp_path / "reader_snapshots.jsonl"
    output_jsonl = tmp_path / "hard_positions.jsonl"
    row = {
        "sample_ids": ["video_test_0001|0"],
        "budget": 1,
        "dense_len": 2,
        "uses_gt": bad_false,
        "reader_out": {
            "allocation": [[[0.0, 0.9]]],
            "valid_mask": [[1, 1]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uses_gt must be JSON false"):
        run_jsonl_export(input_jsonl, output_jsonl, budget=1)


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

    assert summary["decision"] == LEDGER_READY
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
    assert row["uses_oracle"] is False
    assert row["uses_raw_prediction"] is False
    assert row["uses_checkpoint"] is False
    validate_value_transport_selection_row(row, line_no=1, require_deployable=False)
    assert Path(summary_json).is_file()


@pytest.mark.parametrize("flag", ["uses_oracle", "uses_checkpoint"])
def test_value_transport_ledger_validator_rejects_oracle_and_checkpoint_flags(flag):
    row = {
        "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
        "sample_id": "video_test_0001|0",
        "selected_positions_unit": "local_dense_index",
        "selected_positions": [0, 2, 5],
        "target_len": 3,
        "selected_count": 3,
        "valid_len": 6,
        "dense_len": 6,
        "deploy_selection_ledger": False,
        "diagnostic_only": True,
        flag: True,
    }

    with pytest.raises(ValueError, match=rf"forbidden deploy-invisible flag {flag}=true"):
        validate_value_transport_selection_row(row, line_no=1, require_deployable=False)


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
    assert cfg.experiment_scope.changes_post_processing is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    for split in ("val", "test"):
        loadframes = cfg.dataset[split].pipeline[2]
        assert loadframes.method == "bata_value_transport_ledger_subsample"
        assert loadframes.method_base == "sliding_window"
        assert int(loadframes.target_len) == 384
        assert "selection_unit" not in loadframes
        assert loadframes.remap_gt_to_selected_axis is True
        assert loadframes.bata_value_transport_allow_missing_fallback is False
        assert loadframes.bata_value_transport_require_deployable is False
        assert loadframes.bata_value_transport_source == "pc_ot_mras_frontend_hard_positions"


def test_frontend_launcher_defaults_to_review_safe_precheck_and_uses_supported_dump_cli():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_FRONTEND_ADATAD_EVAL="${ALLOW_FRONTEND_ADATAD_EVAL:-0}"' in text
    assert 'fail "ALLOW_FRONTEND_ADATAD_EVAL=1 is required before dump, ledger generation, and detector mAP"' in text
    assert 'require_file_sha "$PC_OT_MRAS_CHECKPOINT" "$PC_OT_MRAS_CHECKPOINT_SHA256" "pc_ot_mras_checkpoint"' in text
    assert 'require_file_sha "$ADATAD_CHECKPOINT" "$ADATAD_CHECKPOINT_SHA256" "adatad_checkpoint"' in text
    assert "DUMP_ARGS+=(--use-amp)" in text
    assert "DUMP_ARGS+=(--amp)" not in text
    assert "tests/test_bata_post_processing_selected_axis.py" in text
