import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_frontend_r18_frozen_reader_adapter_retrain_candidate_n16r4.py"
)
READER_DUMP_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_frontend_r18_sliding_reader_dump_n16r4.py"
)
LAUNCHER = ROOT / "scripts" / "run_pc_ot_mras_frontend_r18_frozen_reader_adapter_retrain_n16r4.sbatch"
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_pc_ot_mras_frontend_retrain_gate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
HARD_EXPORT_PATH = ROOT / "tools" / "bata" / "export_pc_ot_mras_hard_positions.py"
SNAPSHOT_DUMP_PATH = ROOT / "tools" / "bata" / "dump_pc_ot_mras_reader_snapshots.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_text(path, text):
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _good_gate_payload(
    manifest="manifest-sha",
    resolved="resolved-sha",
    pcot_ckpt_sha="pcot-ckpt-sha",
    pretrained_sha="pretrained-sha",
    train_ledger_sha="train-ledger-sha",
    val_ledger_sha="val-ledger-sha",
    test_ledger_sha="test-ledger-sha",
):
    return {
        "decision": "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN",
        "route": "pc_ot_mras_frontend_original_adatad",
        "active_sha256_manifest_sha256": manifest,
        "resolved_config_sha256": resolved,
        "pc_ot_mras_checkpoint_sha256": pcot_ckpt_sha,
        "pretrained_sha256": pretrained_sha,
        "train_ledger_sha256": train_ledger_sha,
        "val_ledger_sha256": val_ledger_sha,
        "test_ledger_sha256": test_ledger_sha,
        "budget": 384,
        "dense_window_size": 768,
        "max_epochs": 60,
        "val_start_epoch": 40,
        "val_eval_interval": 2,
        "allow_slurm": True,
        "allow_gpu": True,
        "single_gpu": True,
        "allow_reader_dump": True,
        "allow_hard_position_export": True,
        "allow_value_transport_ledger": True,
        "allow_tools_train": True,
        "allow_dataset_access": True,
        "allow_pretrained_initialization": True,
        "allow_checkpoint_write": True,
        "allow_train_validation_map": True,
        "allow_long_training": True,
        "tools_test": False,
        "allow_tools_test": False,
        "direct_tools_test": False,
        "detector_map": False,
        "allow_detector_map": False,
        "checkpoint_load": False,
        "allow_checkpoint_load": False,
        "resume": False,
        "allow_resume": False,
        "raw_prediction_cache": False,
        "allow_raw_prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction": False,
        "metric_claim": False,
        "metric_claim_allowed": False,
        "paper_claim": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim": False,
        "deploy_claim_allowed": False,
    }


def test_frontend_retrain_config_is_gated_and_uses_three_reader_ledgers(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_module(GUARD_PATH, "training_guard_for_frontend_retrain_test")
    pcot_ckpt = tmp_path / "r18_epoch_59.pth"
    pretrained = tmp_path / "videomae_pretrained.pth"
    train_ledger = tmp_path / "train_ledger.jsonl"
    val_ledger = tmp_path / "val_ledger.jsonl"
    test_ledger = tmp_path / "test_ledger.jsonl"
    pcot_ckpt_sha = _sha_text(pcot_ckpt, "pcot checkpoint bytes")
    pretrained_sha = _sha_text(pretrained, "pretrained bytes")
    train_ledger_sha = _sha_text(train_ledger, '{"split":"train"}\n')
    val_ledger_sha = _sha_text(val_ledger, '{"split":"val"}\n')
    test_ledger_sha = _sha_text(test_ledger, '{"split":"test"}\n')
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH", str(train_ledger))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH", str(val_ledger))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH", str(test_ledger))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_RETRAIN_PCOT_CKPT_PATH", str(pcot_ckpt))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_RETRAIN_PRETRAINED_PATH", str(pretrained))

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.window_size == 384
    assert cfg.dense_window_size == 768
    assert cfg.experiment_scope.stage == "r18_frozen_reader_adatad_retrain_fixed50_sliding"
    assert cfg.experiment_scope.changes_input_sampling is True
    assert cfg.experiment_scope.changes_detector_head is False
    assert cfg.experiment_scope.changes_loss_assignment is False
    assert "pc_ot_mras_reader" not in cfg.model
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384

    gate = cfg.pc_ot_mras_frontend_retrain_gate
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_train_validation_map is True
    assert gate.allow_long_training is True
    assert gate.entrypoint_gate_context.required is True
    assert gate.entrypoint_gate_context.allowed_decisions == (
        "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN",
    )
    bindings = {item["gate_key"]: item["path_env"] for item in gate.entrypoint_gate_context.sha256_file_bindings}
    assert bindings == {
        "pc_ot_mras_checkpoint_sha256": "PC_OT_MRAS_FRONTEND_RETRAIN_PCOT_CKPT_PATH",
        "pretrained_sha256": "PC_OT_MRAS_FRONTEND_RETRAIN_PRETRAINED_PATH",
        "train_ledger_sha256": "PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH",
        "val_ledger_sha256": "PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH",
        "test_ledger_sha256": "PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH",
    }
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    expected_ledgers = {
        "train": str(train_ledger),
        "val": str(val_ledger),
        "test": str(test_ledger),
    }
    for split, ledger_path in expected_ledgers.items():
        dataset = cfg.dataset[split]
        assert dataset.type == "ThumosSlidingDataset"
        assert dataset.window_size == 768
        loadframes = dataset.pipeline[2]
        assert loadframes.method == "bata_value_transport_ledger_subsample"
        assert loadframes.method_base == "sliding_window"
        assert loadframes.target_len == 384
        assert loadframes.bata_value_transport_ledger_path == ledger_path
        assert loadframes.bata_value_transport_allow_missing_fallback is False
        assert loadframes.bata_value_transport_require_selected_count == 384

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "frontend_retrain_gate.json"
    gate_json.write_text(
        json.dumps(
            _good_gate_payload(
                pcot_ckpt_sha=pcot_ckpt_sha,
                pretrained_sha=pretrained_sha,
                train_ledger_sha=train_ledger_sha,
                val_ledger_sha=val_ledger_sha,
                test_ledger_sha=test_ledger_sha,
            )
        ),
        encoding="utf-8",
    )
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_frontend_retrain_tools_train_guard_rejects_ledger_sha_mismatch(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_module(GUARD_PATH, "training_guard_for_frontend_retrain_sha_test")
    pcot_ckpt = tmp_path / "r18_epoch_59.pth"
    pretrained = tmp_path / "videomae_pretrained.pth"
    train_ledger = tmp_path / "train_ledger.jsonl"
    val_ledger = tmp_path / "val_ledger.jsonl"
    test_ledger = tmp_path / "test_ledger.jsonl"
    pcot_ckpt_sha = _sha_text(pcot_ckpt, "pcot checkpoint bytes")
    pretrained_sha = _sha_text(pretrained, "pretrained bytes")
    _sha_text(train_ledger, '{"split":"train","actual":true}\n')
    val_ledger_sha = _sha_text(val_ledger, '{"split":"val"}\n')
    test_ledger_sha = _sha_text(test_ledger, '{"split":"test"}\n')

    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH", str(train_ledger))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH", str(val_ledger))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH", str(test_ledger))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_RETRAIN_PCOT_CKPT_PATH", str(pcot_ckpt))
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_RETRAIN_PRETRAINED_PATH", str(pretrained))

    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    gate_json = tmp_path / "frontend_retrain_bad_gate.json"
    gate_json.write_text(
        json.dumps(
            _good_gate_payload(
                pcot_ckpt_sha=pcot_ckpt_sha,
                pretrained_sha=pretrained_sha,
                train_ledger_sha="wrong-train-ledger-sha",
                val_ledger_sha=val_ledger_sha,
                test_ledger_sha=test_ledger_sha,
            )
        ),
        encoding="utf-8",
    )
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_RESOLVED_CONFIG_SHA256", "resolved-sha")

    with pytest.raises(RuntimeError, match="train ledger sha256 mismatch"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_frontend_r18_reader_dump_config_uses_sliding_train_windows():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(READER_DUMP_CONFIG))

    assert cfg.window_size == 384
    assert cfg.dense_window_size == 768
    assert cfg.model.pc_ot_mras_reader.num_slots == 384
    assert cfg.model.projection.max_seq_len == 768
    for split in ("train", "val", "test"):
        dataset = cfg.dataset[split]
        assert dataset.type == "ThumosSlidingDataset"
        assert dataset.window_size == 768
        assert dataset.pipeline[2].method == "sliding_window"
        assert dataset.pipeline[2].scale_factor == 1
        assert "random_trunc" not in repr(dataset.pipeline).lower()


def test_frontend_retrain_gate_validator_accepts_bound_payload(tmp_path):
    validator = _load_module(VALIDATOR_PATH, "validate_frontend_retrain_gate_test")
    gate_json = tmp_path / "gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    payload = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        pc_ot_mras_checkpoint_sha256="pcot-ckpt-sha",
        pretrained_sha256="pretrained-sha",
        train_ledger_sha256="train-ledger-sha",
        val_ledger_sha256="val-ledger-sha",
        test_ledger_sha256="test-ledger-sha",
        budget=384,
        dense_window_size=768,
    )
    assert payload["decision"] == "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("decision", "ALLOW_P2QR_FORMAL_TRAIN", "decision is not allowed"),
        ("allow_tools_train", False, "allow_tools_train=true"),
        ("tools_test", True, "tools_test=false/absent"),
        ("detector_map", True, "detector_map=false/absent"),
        ("raw_prediction_cache", True, "raw_prediction_cache=false/absent"),
        ("checkpoint_load", True, "checkpoint_load=false/absent"),
        ("uses_gt", True, "uses_gt=false/absent"),
        ("uses_teacher", True, "uses_teacher=false/absent"),
        ("metric_claim", True, "metric_claim=false/absent"),
        ("max_epochs", 2, "max_epochs=60"),
        ("budget", 192, "budget=384"),
        ("dense_window_size", 384, "dense_window_size=768"),
    ],
)
def test_frontend_retrain_gate_validator_rejects_unsafe_payloads(key, value, match):
    validator = _load_module(VALIDATOR_PATH, "validate_frontend_retrain_gate_negative_test")
    payload = _good_gate_payload()
    payload[key] = value
    with pytest.raises(ValueError, match=match):
        validator.validate_gate_payload(
            payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pc_ot_mras_checkpoint_sha256="pcot-ckpt-sha",
            pretrained_sha256="pretrained-sha",
            train_ledger_sha256="train-ledger-sha",
            val_ledger_sha256="val-ledger-sha",
            test_ledger_sha256="test-ledger-sha",
            budget=384,
            dense_window_size=768,
        )


def test_frontend_retrain_gate_validator_rejects_unknown_keys():
    validator = _load_module(VALIDATOR_PATH, "validate_frontend_retrain_gate_unknown_test")
    payload = _good_gate_payload()
    payload["allow_secret_eval_mode"] = False
    with pytest.raises(ValueError, match="unknown or unallowlisted key"):
        validator.validate_gate_payload(
            payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pc_ot_mras_checkpoint_sha256="pcot-ckpt-sha",
            pretrained_sha256="pretrained-sha",
            train_ledger_sha256="train-ledger-sha",
            val_ledger_sha256="val-ledger-sha",
            test_ledger_sha256="test-ledger-sha",
            budget=384,
            dense_window_size=768,
        )


def test_frontend_retrain_gate_validator_requires_exact_route():
    validator = _load_module(VALIDATOR_PATH, "validate_frontend_retrain_gate_route_test")
    payload = _good_gate_payload()
    del payload["route"]
    with pytest.raises(ValueError, match="route mismatch"):
        validator.validate_gate_payload(
            payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pc_ot_mras_checkpoint_sha256="pcot-ckpt-sha",
            pretrained_sha256="pretrained-sha",
            train_ledger_sha256="train-ledger-sha",
            val_ledger_sha256="val-ledger-sha",
            test_ledger_sha256="test-ledger-sha",
            budget=384,
            dense_window_size=768,
        )


def test_frontend_retrain_launcher_generates_three_ledgers_then_trains():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J pcot_front_train" in text
    assert "pc_ot_mras_frontend_r18_frozen_reader_adapter_retrain_candidate_n16r4.py" in text
    assert "pc_ot_mras_frontend_r18_sliding_reader_dump_n16r4.py" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_FRONTEND_RETRAIN=1" in text
    assert "generate_split_ledger train" in text
    assert "generate_split_ledger val" in text
    assert "generate_split_ledger test" in text
    assert "PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH" in text
    assert "PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH" in text
    assert "PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH" in text
    assert "validate_pc_ot_mras_frontend_retrain_gate.py" in text
    assert "frontend_retrain_predump_gate.json" in text
    assert "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_LEDGER_GENERATION" in text
    assert "--reader-dump-gate-mode frontend_retrain_predump" in text
    assert "--reader-dump-gate-json \"$FRONTEND_RETRAIN_PREDUMP_GATE_JSON\"" in text
    assert "--pretrained-sha256 \"$PRETRAINED_SHA256\"" in text
    assert "DUMP_TRAIN_LIMIT_BATCHES/DUMP_VAL_LIMIT_BATCHES/DUMP_TEST_LIMIT_BATCHES must all be 0" in text
    assert "PC_OT_MRAS_FRONTEND_RETRAIN_PCOT_CKPT_PATH" in text
    assert "PC_OT_MRAS_FRONTEND_RETRAIN_PRETRAINED_PATH" in text
    assert "OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_JSON" in text
    assert "OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_RESOLVED_CONFIG_SHA256" in text
    assert "tools/train.py \"$CONFIG\"" in text
    assert "tools/test.py \"$CONFIG\"" not in text
    assert "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN" in text
    assert "FRONTEND_RETRAIN_PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "FRONTEND_RETRAIN_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS" in text


def test_reader_snapshot_dump_supports_frontend_retrain_predump_gate():
    text = SNAPSHOT_DUMP_PATH.read_text(encoding="utf-8")

    assert "--reader-dump-gate-mode" in text
    assert "frontend_retrain_predump" in text
    assert "--pretrained-sha256" in text
    assert "validate_predump_gate_file" in text


def test_hard_export_prefers_reader_selected_positions_over_matrix_argmax():
    hard_export = _load_module(HARD_EXPORT_PATH, "hard_export_selected_positions_priority_test")
    row = hard_export._resolve_sample(
        {
            "selected_positions": [1, 3],
            "acquisition_matrix": [
                [0.0, 0.0, 0.0, 0.0, 0.99, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.98],
            ],
            "valid_mask": [1, 1, 1, 1, 1, 1],
        },
        batch_idx=0,
        batch_size=1,
        budget=2,
        sample_id="sample_a",
        dense_len=6,
        valid_len=6,
    )

    assert row["selected_positions"] == [1, 3]
    assert row["resolver_generation"]["selected_position_source"] == "selected_positions"


def test_hard_export_rounds_fractional_reader_selected_positions_before_repair():
    hard_export = _load_module(HARD_EXPORT_PATH, "hard_export_fractional_selected_positions_test")
    row = hard_export._resolve_sample(
        {
            "selected_positions": [1.2, 3.6],
            "acquisition_matrix": [
                [0.0, 0.0, 0.0, 0.0, 0.99, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.98],
            ],
            "valid_mask": [1, 1, 1, 1, 1, 1],
        },
        batch_idx=0,
        batch_size=1,
        budget=2,
        sample_id="sample_b",
        dense_len=6,
        valid_len=6,
    )

    assert row["selected_positions"] == [1, 4]
    assert row["resolver_generation"]["selected_position_source"] == "selected_positions"
