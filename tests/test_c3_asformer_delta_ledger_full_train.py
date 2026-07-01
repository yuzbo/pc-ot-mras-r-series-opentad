import importlib.util
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_official_asformer_delta_ledger_original_adatad_full_train.py"
EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_official_asformer_delta_ledger_original_adatad_full_train_exec.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_c3_asformer_delta_ledger_full_train.py"
LAUNCHER = ROOT / "scripts" / "run_c3_asformer_delta_ledger_adatad_full_train_gpu1.sh"
TRAIN = ROOT / "tools" / "train.py"
SCHEDULE = ROOT / "opentad" / "utils" / "train_schedule.py"
GUARD = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_c3_asformer_delta_ledger_full_train_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loadframes(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    assert len(matches) == 1
    return matches[0]


def _collect(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    assert len(matches) == 1
    return matches[0]


def test_asformer_delta_ledger_full_train_config_is_original_adatad_selected_axis():
    cfg = Config.fromfile(str(CONFIG))

    assert cfg.experiment_scope.route == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.experiment_scope.route_variant == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.changes_detector_head is False
    assert cfg.experiment_scope.changes_post_processing is False
    assert int(cfg.window_size) == 384
    assert int(cfg.dense_window_size) == 768
    assert int(cfg.model.backbone.backbone.total_frames) == 384
    assert int(cfg.model.projection.max_seq_len) == 384
    assert "frame_selector" not in repr(cfg.model)
    assert "pc_ot_mras_reader" not in repr(cfg.model)
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    assert cfg.dataset.train.type == "ThumosSlidingDataset"
    assert cfg.dataset.train.subset_name == "training"
    assert cfg.dataset.val.subset_name == "validation"
    assert cfg.dataset.test.subset_name == "validation"
    for split in ("train", "val", "test"):
        loader = _loadframes(cfg.dataset[split])
        assert loader.method == "bata_value_transport_ledger_subsample"
        assert loader.method_base == "sliding_window"
        assert int(loader.target_len) == 384
        assert int(loader.bata_value_transport_require_selected_count) == 384
        assert loader.bata_value_transport_require_deployable is True
        assert loader.bata_value_transport_allow_missing_fallback is False
        assert loader.bata_value_transport_allow_short_valid_ratio_count is True
        assert loader.remap_gt_to_selected_axis is True

    assert "gt_segments" not in _collect(cfg.dataset.test).get("keys", [])
    assert "gt_labels" not in _collect(cfg.dataset.test).get("keys", [])
    assert cfg.workflow.val_eval_epochs == [2, 60]
    assert int(cfg.workflow.val_eval_interval) == 5
    assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 2
    assert cfg.workflow.max_train_iters is None


def test_asformer_delta_ledger_full_train_validator_passes_without_local_files(monkeypatch):
    monkeypatch.setenv("C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH", "/tmp/c3_asformer_delta_train.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_VAL_LEDGER_PATH", "/tmp/c3_asformer_delta_val.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_TEST_LEDGER_PATH", "/tmp/c3_asformer_delta_test.jsonl")
    validator = _load_validator()
    cfg = validator.validate_config(str(CONFIG), require_ledger_files=False)
    assert cfg.c3_asformer_delta_ledger_full_train_gate.requires_launch_gate is True
    assert cfg.c3_asformer_delta_ledger_full_train_gate.launch_gate_passed is False


def test_asformer_delta_ledger_exec_config_passes_training_guard(monkeypatch):
    monkeypatch.setenv("C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH", "/tmp/c3_asformer_delta_train.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_VAL_LEDGER_PATH", "/tmp/c3_asformer_delta_val.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_TEST_LEDGER_PATH", "/tmp/c3_asformer_delta_test.jsonl")
    validator = _load_validator()
    cfg = validator.validate_config(str(EXEC_CONFIG), require_ledger_files=False, allow_launch_unlocked=True)
    guard_spec = importlib.util.spec_from_file_location("training_guard_c3_asformer_delta_test", GUARD)
    guard = importlib.util.module_from_spec(guard_spec)
    guard_spec.loader.exec_module(guard)
    guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    assert cfg.c3_asformer_delta_ledger_full_train_gate.launch_gate_passed is True
    assert cfg.c3_asformer_delta_ledger_full_train_gate.reviewed_execution_config is True


def test_training_eval_schedule_keeps_legacy_and_runs_epoch60():
    schedule_spec = importlib.util.spec_from_file_location("train_schedule_c3_asformer_delta_test", SCHEDULE)
    schedule_module = importlib.util.module_from_spec(schedule_spec)
    schedule_spec.loader.exec_module(schedule_module)

    cfg = Config.fromfile(str(CONFIG))
    eval_epochs = [
        epoch + 1
        for epoch in range(60)
        if epoch >= cfg.workflow.val_start_epoch and schedule_module.should_eval_epoch(epoch, cfg.workflow)
    ]
    assert 2 in eval_epochs
    assert 60 in eval_epochs
    assert eval_epochs[:3] == [2, 7, 12]

    class LegacyWorkflow:
        val_eval_interval = 5

        def __contains__(self, key):
            return False

        def get(self, key, default=None):
            return default

    assert schedule_module.should_eval_epoch(4, LegacyWorkflow()) is True
    assert schedule_module.should_eval_epoch(3, LegacyWorkflow()) is False


def test_asformer_delta_ledger_launcher_is_gpu1_precheck_first_and_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN="${ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN:-0}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
    assert "EXEC_CONFIG" in text
    assert "--allow-launch-unlocked" in text
    assert "--require-ledger-files" in text
    assert "validate_c3_asformer_delta_ledger_full_train.py" in text
    assert "tests/test_c3_asformer_delta_ledger_full_train.py" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text
