import json
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from mmengine.config import Config

from tools.bata.convert_pc_ot_mras_hard_positions_to_value_transport_ledger import READY as LEDGER_READY, run_conversion
from tools.bata.convert_lowres_probe_samples_to_value_transport_ledger import (
    READY as LOWRES_LEDGER_READY,
    run_conversion as run_lowres_probe_conversion,
)
from tools.bata.dump_pc_ot_mras_reader_snapshots import sample_ids_from_metas
from tools.bata.export_pc_ot_mras_hard_positions import READY as HARD_READY, run_jsonl_export


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_pc_ot_mras_frontend_ledger_eval_n16r4.sbatch"
LOWRES_LEDGER_EXPORT_LAUNCHER = ROOT / "scripts" / "run_c3_lowres_probe_ledger_export_gpu1_20260702.sh"
BOUNDARY_PATH = ROOT / "opentad" / "datasets" / "transforms" / "boundary_acquisition.py"
END_TO_END_PATH = ROOT / "opentad" / "datasets" / "transforms" / "end_to_end.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
BOUNDARY_SPEC = importlib.util.spec_from_file_location("pcot_frontend_test_boundary_acquisition", BOUNDARY_PATH)
BOUNDARY_MODULE = importlib.util.module_from_spec(BOUNDARY_SPEC)
sys.modules[BOUNDARY_SPEC.name] = BOUNDARY_MODULE
BOUNDARY_SPEC.loader.exec_module(BOUNDARY_MODULE)
validate_value_transport_selection_row = BOUNDARY_MODULE.validate_value_transport_selection_row


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


class _TorchArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def bool(self):
        return self.value.astype(bool)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_loadframes_class():
    stubs = {}
    previous = {}
    for name in ("opentad", "opentad.datasets", "opentad.datasets.transforms"):
        module = types.ModuleType(name)
        module.__path__ = []
        stubs[name] = module
    builder = types.ModuleType("opentad.datasets.builder")
    builder.PIPELINES = _Registry()
    stubs["opentad.datasets.builder"] = builder
    stubs["opentad.datasets.transforms.boundary_acquisition"] = BOUNDARY_MODULE
    torch_stub = types.ModuleType("torch")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_functional_stub = types.ModuleType("torch.nn.functional")
    torch_nn_stub.functional = torch_functional_stub
    torch_stub.nn = torch_nn_stub
    torch_stub.ones = lambda *shape: _TorchArray(np.ones(shape if len(shape) != 1 else shape[0]))
    torch_stub.zeros = lambda *shape: _TorchArray(np.zeros(shape if len(shape) != 1 else shape[0]))
    torch_stub.cat = lambda tensors: _TorchArray(
        np.concatenate([item.value if isinstance(item, _TorchArray) else np.asarray(item) for item in tensors])
    )
    stubs["torch"] = torch_stub
    stubs["torch.nn"] = torch_nn_stub
    stubs["torch.nn.functional"] = torch_functional_stub
    module_name = "opentad.datasets.transforms.end_to_end_frontend_test"
    try:
        for name, module in stubs.items():
            previous[name] = sys.modules.get(name)
            sys.modules[name] = module
        spec = importlib.util.spec_from_file_location(module_name, END_TO_END_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.LoadFrames
    finally:
        sys.modules.pop(module_name, None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


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


def test_lowres_probe_samples_convert_strategy_to_value_transport_ledger(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    summary_json = tmp_path / "summary.json"
    sample_row = {
        "sample_id": "video_test_0001|768",
        "probe_model": "temporal-tcn",
        "tcn_variant": "gated",
        "spatial_size": 64,
        "dense_len": 8,
        "valid_len": 8,
        "budget": 3,
        "selected_positions": [0, 2, 4],
        "strategy_selected_positions": {
            "delta_p_action": [1, 3, 6],
            "topk_action_logit": [0, 2, 4],
        },
        "boundary_support_r1": 0.75,
    }
    input_jsonl.write_text(json.dumps(sample_row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_lowres_probe_conversion(
        input_jsonl,
        output_jsonl,
        strategy="delta_p_action",
        target_len=3,
        summary_json=summary_json,
        require_selected_count=3,
    )

    assert summary["decision"] == LOWRES_LEDGER_READY
    assert summary["row_count"] == 1
    row = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert row["schema_version"] == "pc_ot_mras_frontend_value_transport_ledger_v0"
    assert row["sample_id"] == "video_test_0001|768"
    assert row["selected_positions"] == [1, 3, 6]
    assert row["diagnostics"]["source_strategy"] == "delta_p_action"
    assert row["diagnostics"]["source_tcn_variant"] == "gated"
    assert row["diagnostics"]["diagnostic_boundary_support_r1_ignored_by_selection"] == 0.75
    assert row["uses_gt"] is False
    assert row["uses_teacher"] is False
    assert row["uses_oracle"] is False
    validate_value_transport_selection_row(row, line_no=1, require_deployable=False)


def test_lowres_probe_deploy_ledger_strips_gt_derived_diagnostics(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    sample_row = {
        "sample_id": "video_test_0001|768",
        "probe_model": "temporal-tcn",
        "tcn_variant": "gated",
        "spatial_size": 64,
        "dense_len": 8,
        "valid_len": 8,
        "strategy_selected_positions": {"delta_p_action": [1, 3, 6]},
        "boundary_support_r1": 0.75,
    }
    input_jsonl.write_text(json.dumps(sample_row, sort_keys=True) + "\n", encoding="utf-8")

    run_lowres_probe_conversion(
        input_jsonl,
        output_jsonl,
        strategy="delta_p_action",
        target_len=3,
        require_selected_count=3,
        deploy_selection_ledger=True,
    )

    row = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert row["deploy_selection_ledger"] is True
    assert row["diagnostic_only"] is False
    assert "diagnostic_boundary_support_r1_ignored_by_selection" not in row["diagnostics"]
    assert row["uses_gt"] is False
    validate_value_transport_selection_row(row, line_no=1, require_deployable=True)


def test_lowres_probe_conversion_rejects_video_only_or_duplicate_sample_ids(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    base = {
        "sample_id": "video_only",
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"delta_p_action": [0, 2]},
    }
    input_jsonl.write_text(json.dumps(base, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample_id must match video_name"):
        run_lowres_probe_conversion(input_jsonl, output_jsonl, strategy="delta_p_action", target_len=2)

    duplicate_a = dict(base, sample_id="video_test_0001|0")
    duplicate_b = dict(base, sample_id="video_test_0001|0")
    input_jsonl.write_text(
        json.dumps(duplicate_a, sort_keys=True) + "\n" + json.dumps(duplicate_b, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        run_lowres_probe_conversion(input_jsonl, output_jsonl, strategy="delta_p_action", target_len=2)


def test_lowres_probe_conversion_can_deduplicate_identical_window_rows(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    summary_json = tmp_path / "summary.json"
    sample_row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"delta_p_action": [0, 2]},
    }
    input_jsonl.write_text(
        json.dumps(sample_row, sort_keys=True) + "\n" + json.dumps(sample_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = run_lowres_probe_conversion(
        input_jsonl,
        output_jsonl,
        strategy="delta_p_action",
        target_len=2,
        require_selected_count=2,
        summary_json=summary_json,
        deduplicate_sample_id=True,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert summary["decision"] == LOWRES_LEDGER_READY
    assert summary["row_count"] == 1
    assert summary["deduplicate_sample_id"] is True
    assert summary["duplicate_sample_id_count"] == 1
    assert rows[0]["sample_id"] == "video_test_0001|0"
    assert rows[0]["selected_positions"] == [0, 2]


def test_lowres_probe_conversion_rejects_conflicting_duplicate_window_rows(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    duplicate_a = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"delta_p_action": [0, 2]},
    }
    duplicate_b = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"delta_p_action": [0, 3]},
    }
    input_jsonl.write_text(
        json.dumps(duplicate_a, sort_keys=True) + "\n" + json.dumps(duplicate_b, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting selected_positions"):
        run_lowres_probe_conversion(
            input_jsonl,
            output_jsonl,
            strategy="delta_p_action",
            target_len=2,
            require_selected_count=2,
            deduplicate_sample_id=True,
        )


def test_lowres_probe_conversion_can_uniform_fill_to_required_count(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    sample_row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 8,
        "valid_len": 8,
        "strategy_selected_positions": {"delta_p_action": [2, 5]},
    }
    input_jsonl.write_text(json.dumps(sample_row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_lowres_probe_conversion(
        input_jsonl,
        output_jsonl,
        strategy="delta_p_action",
        target_len=4,
        require_selected_count=4,
        fill_to_target_count=True,
    )

    row = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert summary["decision"] == LOWRES_LEDGER_READY
    assert len(row["selected_positions"]) == 4
    assert set([2, 5]).issubset(set(row["selected_positions"]))
    assert row["diagnostics"]["uniform_visible_fill_count"] == 2
    assert row["uses_gt"] is False
    assert row["uses_teacher"] is False
    assert row["uses_oracle"] is False
    assert row["uses_raw_prediction"] is False
    assert row["uses_checkpoint"] is False
    validate_value_transport_selection_row(row, line_no=1, require_deployable=False)


def test_lowres_probe_conversion_can_require_short_valid_ratio_count(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    sample_row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 8,
        "valid_len": 5,
        "strategy_selected_positions": {"delta_p_action": [1, 3]},
    }
    input_jsonl.write_text(json.dumps(sample_row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_lowres_probe_conversion(
        input_jsonl,
        output_jsonl,
        strategy="delta_p_action",
        target_len=4,
        require_selected_count=4,
        fill_to_target_count=True,
        allow_short_valid_ratio_count=True,
    )

    row = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert summary["decision"] == LOWRES_LEDGER_READY
    assert summary["allow_short_valid_ratio_count"] is True
    assert len(row["selected_positions"]) == 3
    assert row["diagnostics"]["required_selected_count"] == 3
    assert row["diagnostics"]["uniform_visible_fill_count"] == 1
    assert set([1, 3]).issubset(set(row["selected_positions"]))
    validate_value_transport_selection_row(row, line_no=1, require_deployable=False)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ([0, 1.9, 3], "must be an integer"),
        ([0, 3, 2], "must be sorted"),
        ([0, 2, 2], "must be unique"),
    ],
)
def test_lowres_probe_conversion_rejects_non_strict_positions(tmp_path, positions, message):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    sample_row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 8,
        "valid_len": 8,
        "strategy_selected_positions": {"delta_p_action": positions},
    }
    input_jsonl.write_text(json.dumps(sample_row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_lowres_probe_conversion(input_jsonl, output_jsonl, strategy="delta_p_action", target_len=3)


@pytest.mark.parametrize("sample_id", ["video_test_0001|", "|0", "video_test_0001|abc", "video_test_0001|-1"])
def test_lowres_probe_conversion_rejects_malformed_window_sample_ids(tmp_path, sample_id):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    sample_row = {
        "sample_id": sample_id,
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"delta_p_action": [0, 2]},
    }
    input_jsonl.write_text(json.dumps(sample_row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample_id must match video_name"):
        run_lowres_probe_conversion(input_jsonl, output_jsonl, strategy="delta_p_action", target_len=2)


def test_lowres_probe_video_only_escape_is_diagnostic_only(tmp_path):
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "value_transport_ledger.jsonl"
    sample_row = {
        "sample_id": "video_test_0001",
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"delta_p_action": [0, 2]},
    }
    input_jsonl.write_text(json.dumps(sample_row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_lowres_probe_conversion(
        input_jsonl,
        output_jsonl,
        strategy="delta_p_action",
        target_len=2,
        require_window_sample_id=False,
    )
    row = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert summary["decision"] == LOWRES_LEDGER_READY
    assert row["diagnostic_only"] is True
    assert row["deploy_selection_ledger"] is False

    with pytest.raises(ValueError, match="deploy_selection_ledger requires strict"):
        run_lowres_probe_conversion(
            input_jsonl,
            output_jsonl,
            strategy="delta_p_action",
            target_len=2,
            require_window_sample_id=False,
            deploy_selection_ledger=True,
        )


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
    guard = _load_module("training_guard_for_frontend_eval_config_test", GUARD_PATH)

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
    assert cfg.pc_ot_mras_frontend_hard_ledger_eval_gate.requires_launch_gate is True
    assert cfg.pc_ot_mras_frontend_hard_ledger_eval_gate.launch_gate_passed is False
    assert cfg.pc_ot_mras_frontend_hard_ledger_eval_gate.allow_tools_train is False
    assert cfg.pc_ot_mras_frontend_hard_ledger_eval_gate.allow_tools_test is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    with pytest.raises(RuntimeError, match="requires_launch_gate=True"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    for split in ("val", "test"):
        loadframes = cfg.dataset[split].pipeline[2]
        assert loadframes.method == "bata_value_transport_ledger_subsample"
        assert loadframes.method_base == "sliding_window"
        assert int(loadframes.target_len) == 384
        assert "selection_unit" not in loadframes
        assert loadframes.remap_gt_to_selected_axis is True
        assert loadframes.bata_value_transport_allow_missing_fallback is False
        assert loadframes.bata_value_transport_require_deployable is False
        assert int(loadframes.bata_value_transport_require_selected_count) == 384
        assert loadframes.bata_value_transport_source == "pc_ot_mras_frontend_hard_positions"


def test_value_transport_loader_rejects_short_ledger_when_exact_count_required(tmp_path):
    LoadFrames = _load_loadframes_class()
    ledger_path = tmp_path / "value_transport_ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [0, 2],
                "selected_count": 2,
                "target_len": 3,
                "valid_len": 4,
                "dense_len": 4,
                "deploy_selection_ledger": False,
                "diagnostic_only": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    loader = LoadFrames(
        num_clips=1,
        scale_factor=1,
        method="bata_value_transport_ledger_subsample",
        method_base="sliding_window",
        target_len=3,
        bata_value_transport_ledger_path=str(ledger_path),
        bata_value_transport_require_deployable=False,
        bata_value_transport_require_selected_count=3,
    )

    with pytest.raises(ValueError, match="bata_value_transport_require_selected_count=3"):
        loader(
            {
                "video_name": "video_test_0001",
                "window_start_frame": 0,
                "window_size": 4,
                "feature_start_idx": 0,
                "feature_end_idx": 3,
                "total_frames": 16,
                "avg_fps": 30,
                "snippet_stride": 1,
            }
        )


def test_value_transport_loader_accepts_short_tail_ratio_count_when_enabled(tmp_path):
    LoadFrames = _load_loadframes_class()
    ledger_path = tmp_path / "value_transport_ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [0, 2],
                "selected_count": 2,
                "target_len": 3,
                "valid_len": 4,
                "dense_len": 6,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    loader = LoadFrames(
        num_clips=1,
        scale_factor=1,
        method="bata_value_transport_ledger_subsample",
        method_base="sliding_window",
        target_len=3,
        bata_value_transport_ledger_path=str(ledger_path),
        bata_value_transport_require_deployable=True,
        bata_value_transport_require_selected_count=3,
        bata_value_transport_allow_short_valid_ratio_count=True,
    )

    out = loader(
        {
            "video_name": "video_test_0001",
            "window_start_frame": 0,
            "window_size": 6,
            "feature_start_idx": 0,
            "feature_end_idx": 3,
            "total_frames": 16,
            "avg_fps": 30,
            "snippet_stride": 1,
        }
    )

    assert out["frame_inds"].tolist() == [0, 2, 2]
    assert out["masks"].tolist() == [True, True, False]
    assert out["bata_selected_dense_indices"].tolist() == [0, 2]
    assert out["selected_valid_len"] == 2
    assert out["irregular_selected_valid_len"] == 4.0


def test_frontend_launcher_defaults_to_review_safe_precheck_and_uses_supported_dump_cli():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '[[ -z "$RUN_TAG"' in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_FRONTEND_ADATAD_EVAL="${ALLOW_FRONTEND_ADATAD_EVAL:-0}"' in text
    assert 'fail "ALLOW_FRONTEND_ADATAD_EVAL=1 is required before dump, ledger generation, and detector mAP"' in text
    assert 'require_file_sha "$PC_OT_MRAS_CHECKPOINT" "$PC_OT_MRAS_CHECKPOINT_SHA256" "pc_ot_mras_checkpoint"' in text
    assert 'require_file_sha "$ADATAD_CHECKPOINT" "$ADATAD_CHECKPOINT_SHA256" "adatad_checkpoint"' in text
    assert "DUMP_ARGS+=(--use-amp)" in text
    assert "DUMP_ARGS+=(--amp)" not in text
    assert "tests/test_bata_post_processing_selected_axis.py" in text


def test_lowres_probe_ledger_export_launcher_is_gpu1_coverage_only_and_strict():
    text = LOWRES_LEDGER_EXPORT_LAUNCHER.read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES}" != "1"' in text
    assert "PROBE_CHECKPOINT is required" in text
    assert 'VAL_SUBSET_NAME="${VAL_SUBSET_NAME:-training}"' in text
    assert 'VAL_SUBSET_NAME="${VAL_SUBSET_NAME:-validation}"' in text
    assert "--coverage-only" in text
    assert '--probe-window-size "${DENSE_WINDOW_SIZE}"' in text
    assert '--eval-window-overlap-ratio "${EVAL_WINDOW_OVERLAP_RATIO}"' in text
    assert "--eval-include-all-windows" in text
    assert 'EVAL_WINDOW_OVERLAP_RATIO="${EVAL_WINDOW_OVERLAP_RATIO:-0.5}"' in text
    assert 'DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE:-768}"' in text
    assert 'TARGET_LEN="${TARGET_LEN:-384}"' in text
    assert "--deploy-selection-ledger" in text
    assert '--require-selected-count "${TARGET_LEN}"' in text
    assert "--allow-short-valid-ratio-count" in text
    assert "--fill-to-target-count" in text
    assert "--deduplicate-sample-id" in text
    assert "--allow-video-only-sample-id" not in text
