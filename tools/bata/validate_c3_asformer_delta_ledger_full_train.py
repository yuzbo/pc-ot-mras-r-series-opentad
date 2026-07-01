import argparse
import json
import math
import os
from pathlib import Path

from mmengine.config import Config


CONFIG_DEFAULT = "configs/adatad/thumos/c3_official_asformer_delta_ledger_original_adatad_full_train.py"
READY = "C3_ASFORMER_DELTA_LEDGER_FULL_TRAIN_GATE_PASS"


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _as_bool(value):
    return bool(value)


def _find_loadframes(pipeline):
    matches = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    _require(len(matches) == 1, f"expected exactly one LoadFrames step, got {len(matches)}")
    return matches[0]


def _find_collect(pipeline):
    matches = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    _require(len(matches) == 1, f"expected exactly one Collect step, got {len(matches)}")
    return matches[0]


def _validate_loader(name, dataset_cfg, expected_ledger_path):
    loader = _find_loadframes(dataset_cfg.pipeline)
    _require(loader.get("method") == "bata_value_transport_ledger_subsample", f"{name}: wrong LoadFrames method")
    _require(loader.get("method_base") == "sliding_window", f"{name}: method_base must be sliding_window")
    _require(int(loader.get("target_len")) == 384, f"{name}: target_len must be 384")
    _require(int(loader.get("scale_factor", 1)) == 1, f"{name}: scale_factor must be 1")
    _require(int(loader.get("bata_value_transport_require_selected_count")) == 384, f"{name}: require count must be 384")
    _require(_as_bool(loader.get("bata_value_transport_allow_short_valid_ratio_count")), f"{name}: short-tail ratio gate off")
    _require(_as_bool(loader.get("bata_value_transport_require_deployable")), f"{name}: deployable ledger not required")
    _require(not _as_bool(loader.get("bata_value_transport_allow_missing_fallback")), f"{name}: missing fallback must be off")
    _require(_as_bool(loader.get("remap_gt_to_selected_axis")), f"{name}: selected-axis GT remap must be on")
    _require(
        loader.get("bata_value_transport_ledger_path") == expected_ledger_path,
        f"{name}: unexpected ledger path {loader.get('bata_value_transport_ledger_path')}",
    )
    _require(loader.get("bata_value_transport_source") == "c3_official_asformer_delta_p_action", f"{name}: wrong source")


def _validate_dataset(cfg):
    _require(cfg.dataset.train.type == "ThumosSlidingDataset", "train must use ThumosSlidingDataset")
    _require(cfg.dataset.val.type == "ThumosSlidingDataset", "val must use ThumosSlidingDataset")
    _require(cfg.dataset.test.type == "ThumosSlidingDataset", "test must use ThumosSlidingDataset")
    _require(cfg.dataset.train.subset_name == "training", "train subset must be training")
    _require(cfg.dataset.val.subset_name == "validation", "val subset must be validation")
    _require(cfg.dataset.test.subset_name == "validation", "test subset must be validation")
    _require(int(cfg.dataset.train.window_size) == 768, "train dense window must be 768")
    _require(int(cfg.dataset.val.window_size) == 768, "val dense window must be 768")
    _require(int(cfg.dataset.test.window_size) == 768, "test dense window must be 768")
    _validate_loader("train", cfg.dataset.train, cfg.train_ledger_path)
    _validate_loader("val", cfg.dataset.val, cfg.val_ledger_path)
    _validate_loader("test", cfg.dataset.test, cfg.test_ledger_path)

    train_collect = _find_collect(cfg.dataset.train.pipeline)
    val_collect = _find_collect(cfg.dataset.val.pipeline)
    test_collect = _find_collect(cfg.dataset.test.pipeline)
    _require("gt_segments" in train_collect.get("keys", []), "train must collect gt_segments")
    _require("gt_labels" in train_collect.get("keys", []), "train must collect gt_labels")
    _require("gt_segments" in val_collect.get("keys", []), "val must collect gt_segments")
    _require("gt_labels" in val_collect.get("keys", []), "val must collect gt_labels")
    _require("gt_segments" not in test_collect.get("keys", []), "test must not collect gt_segments")
    _require("gt_labels" not in test_collect.get("keys", []), "test must not collect gt_labels")
    for split_name, collect in (("train", train_collect), ("val", val_collect), ("test", test_collect)):
        meta_keys = set(collect.get("meta_keys", []))
        for key in ("irregular_selected_positions", "irregular_selected_valid_len", "bata_selected_dense_indices"):
            _require(key in meta_keys, f"{split_name}: missing meta key {key}")


def _validate_model_and_train(cfg):
    _require(int(cfg.window_size) == 384, "selected window_size must be 384")
    _require(int(cfg.dense_window_size) == 768, "dense_window_size must be 768")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "backbone total_frames must be 384")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection max_seq_len must be 384")
    _require(not _as_bool(cfg.inference.load_from_raw_predictions), "raw prediction loading must be off")
    _require(not _as_bool(cfg.inference.save_raw_prediction), "raw prediction saving must be off")
    _require(int(cfg.workflow.end_epoch) == 60, "formal full train must run 60 epochs")
    _require(cfg.workflow.get("max_train_iters", None) is None, "full train must not cap train iterations")
    _require(int(cfg.workflow.val_eval_interval) == 5, "validation interval must be 5")
    explicit_eval_epochs = list(cfg.workflow.get("val_eval_epochs", []))
    _require(2 in explicit_eval_epochs, "first scheduled validation must include epoch 2")
    _require(60 in explicit_eval_epochs, "final scheduled validation must include epoch 60")
    _require(int(cfg.workflow.get("val_eval_interval_anchor_epoch", 0)) == 2, "validation anchor must be epoch 2")
    _require(not _as_bool(cfg.solver.get("ema", True)), "EMA should be off for this diagnostic comparison")


def _validate_gate(cfg, *, allow_launch_unlocked=False):
    gate = cfg.c3_asformer_delta_ledger_full_train_gate
    _require(gate.route == "C3_MAINLINE_OPTIMIZATION", "wrong route")
    _require(gate.route_variant == "C3_ORIGINAL_OPTIMIZATION_ROUTE", "wrong route variant")
    _require(_as_bool(gate.full_train_candidate), "not marked as full train candidate")
    _require(_as_bool(gate.requires_launch_gate), "launch gate must be required")
    if allow_launch_unlocked:
        _require(_as_bool(gate.launch_gate_passed), "execution config must pass launch gate")
        _require(_as_bool(gate.get("reviewed_execution_config", False)), "execution config must be reviewed")
    else:
        _require(not _as_bool(gate.launch_gate_passed), "config must be locked by default")
    _require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), "only tools/train.py may be allowed")
    _require("tools/test.py" in tuple(gate.forbidden_entrypoints), "tools/test.py must be forbidden by this launcher")
    text = repr(
        dict(
            experiment_scope=cfg.experiment_scope,
            model=cfg.model,
            dataset=cfg.dataset,
            inference=cfg.inference,
        )
    ).lower()
    for forbidden in ("bh_sdc", "event_surprise", "boundary_microscope", "frame_token_hybrid"):
        _require(forbidden not in text, f"forbidden route token present: {forbidden}")
    for forbidden in ("load_from_raw_predictions': true", "save_raw_prediction': true"):
        _require(forbidden not in text, f"forbidden raw prediction flag present: {forbidden}")


def _validate_ledger_file(path, *, require_exists):
    path = Path(path)
    _require(str(path) and "REPLACE_WITH" not in str(path), f"ledger path is unresolved: {path}")
    if not require_exists:
        return
    _require(path.is_file(), f"ledger file missing: {path}")
    seen = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            sample_id = row.get("sample_id")
            _require(sample_id and "|" in sample_id, f"{path}:{line_no}: invalid sample_id")
            _require(sample_id not in seen, f"{path}:{line_no}: duplicate sample_id {sample_id}")
            seen.add(sample_id)
            _require(row.get("deploy_selection_ledger") is True, f"{path}:{line_no}: deploy flag is not true")
            _require(row.get("diagnostic_only") is not True, f"{path}:{line_no}: diagnostic ledger row")
            for key in ("uses_gt", "uses_teacher", "uses_oracle", "uses_cache", "uses_raw_prediction", "uses_checkpoint"):
                _require(row.get(key) is not True, f"{path}:{line_no}: forbidden flag {key}=True")
            valid_len = int(row.get("valid_len"))
            dense_len = int(row.get("dense_len"))
            target_len = int(row.get("target_len"))
            positions = [int(item) for item in row.get("selected_positions", [])]
            _require(dense_len == 768, f"{path}:{line_no}: dense_len must be 768")
            _require(target_len == 384, f"{path}:{line_no}: target_len must be 384")
            _require(positions == sorted(set(positions)), f"{path}:{line_no}: positions must be sorted unique")
            _require(len(positions) == int(row.get("selected_count")), f"{path}:{line_no}: selected_count mismatch")
            _require(all(0 <= item < valid_len for item in positions), f"{path}:{line_no}: position outside valid_len")
            expected = 384
            if valid_len < dense_len:
                expected = max(1, min(384, valid_len, int(math.ceil(valid_len * 384.0 / dense_len))))
            _require(len(positions) == expected, f"{path}:{line_no}: expected {expected} selected positions")
    _require(rows > 0, f"ledger file has no rows: {path}")


def validate_config(config_path=CONFIG_DEFAULT, *, require_ledger_files=False, allow_launch_unlocked=False):
    cfg = Config.fromfile(str(config_path))
    _validate_gate(cfg, allow_launch_unlocked=allow_launch_unlocked)
    _validate_dataset(cfg)
    _validate_model_and_train(cfg)
    for ledger_path in (cfg.train_ledger_path, cfg.val_ledger_path, cfg.test_ledger_path):
        _validate_ledger_file(ledger_path, require_exists=require_ledger_files)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--require-ledger-files", action="store_true")
    parser.add_argument("--allow-launch-unlocked", action="store_true")
    args = parser.parse_args(argv)
    validate_config(
        args.config,
        require_ledger_files=args.require_ledger_files,
        allow_launch_unlocked=args.allow_launch_unlocked,
    )
    print(READY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
