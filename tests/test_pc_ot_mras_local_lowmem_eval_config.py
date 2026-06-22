import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"

CASES = (
    dict(
        label="R17_C0",
        config="ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r17_post_train_eval_candidate.py",
        gate_name="r17_pc_ot_mras_post_train_eval_gate",
        old_gate_name="r17_pc_ot_mras_formal_train_gate",
        decision="ALLOW_R17_POST_TRAIN_EVAL",
        stage="R17_post_train_eval_candidate",
        reader_disabled=False,
        r18_aux=False,
        r17_reader=True,
    ),
    dict(
        label="R18_C0",
        config="ctf_bdi_pc_ot_mras_r18_post_train_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r18_post_train_eval_candidate.py",
        gate_name="r18_pc_ot_mras_post_train_eval_gate",
        old_gate_name="r18_pc_ot_mras_aux_diag_gate",
        decision="ALLOW_R18_POST_TRAIN_EVAL",
        stage="R18_post_train_eval_candidate",
        reader_disabled=False,
        r18_aux=True,
        r17_reader=False,
    ),
    dict(
        label="R17_C1",
        config="ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py",
        gate_name="r17_pc_ot_mras_reader_disabled_eval_gate",
        old_gate_name="r17_pc_ot_mras_post_train_eval_gate",
        decision="ALLOW_R17_READER_DISABLED_EVAL",
        stage="R17_reader_disabled_eval_candidate",
        reader_disabled=True,
        r18_aux=False,
        r17_reader=True,
    ),
    dict(
        label="R18_C1",
        config="ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py",
        gate_name="r18_pc_ot_mras_reader_disabled_eval_gate",
        old_gate_name="r18_pc_ot_mras_post_train_eval_gate",
        decision="ALLOW_R18_READER_DISABLED_EVAL",
        stage="R18_reader_disabled_eval_candidate",
        reader_disabled=True,
        r18_aux=True,
        r17_reader=False,
    ),
)


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_local_lowmem_eval_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gate(tmp_path, *, decision):
    gate_json = tmp_path / f"{decision}.json"
    gate_json.write_text(
        json.dumps(
            {
                "decision": decision,
                "active_sha256_manifest_sha256": "manifest-sha",
                "resolved_config_sha256": "resolved-sha",
                "checkpoint_sha256": "checkpoint-sha",
                "completed_training_evidence": True,
                "tools_train": False,
                "direct_tools_train": False,
                "raw_prediction_cache": False,
                "paper_claim": False,
                "runtime_flops_claim": False,
                "deploy_claim": False,
                "dynamic_budget_claim": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return gate_json, hashlib.sha256(gate_json.read_bytes()).hexdigest()


def _as_plain(node):
    if hasattr(node, "to_dict"):
        return node.to_dict()
    if isinstance(node, dict):
        return {key: _as_plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_as_plain(value) for value in node]
    return node


@pytest.mark.parametrize("case", CASES, ids=[case["label"] for case in CASES])
def test_local_lowmem_eval_configs_inherit_parent_and_only_reduce_test_loader(case):
    mmengine_config = pytest.importorskip("mmengine.config")
    config_path = CONFIG_DIR / case["config"]
    parent_path = CONFIG_DIR / case["parent"]

    text = config_path.read_text(encoding="utf-8")
    assert f'_base_ = ["{case["parent"]}"]' in text

    cfg = mmengine_config.Config.fromfile(str(config_path))
    parent = mmengine_config.Config.fromfile(str(parent_path))

    assert cfg.solver.test.batch_size == 1
    assert cfg.solver.test.num_workers == 0
    assert parent.solver.test.batch_size == 2
    assert parent.solver.test.num_workers == 2
    assert _as_plain(cfg.solver.train) == _as_plain(parent.solver.train)
    assert _as_plain(cfg.solver.val) == _as_plain(parent.solver.val)
    assert _as_plain(cfg.model) == _as_plain(parent.model)
    assert _as_plain(cfg.inference) == _as_plain(parent.inference)
    assert _as_plain(cfg.evaluation) == _as_plain(parent.evaluation)
    assert cfg.work_dir == parent.work_dir


@pytest.mark.parametrize("case", CASES, ids=[case["label"] for case in CASES])
def test_local_lowmem_eval_configs_keep_test_only_entrypoint_gate(tmp_path, monkeypatch, case):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()
    cfg = mmengine_config.Config.fromfile(str(CONFIG_DIR / case["config"]))

    assert cfg.get(case["old_gate_name"]) is None
    gate = cfg.get(case["gate_name"])
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == case["stage"]
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is True
    assert gate.allow_detector_map is True
    assert gate.allow_post_train_checkpoint_eval is True
    assert gate.detector_map_reporting_allowed is True
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert gate.runtime_flops_claim_allowed is False
    assert gate.deploy_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ("tools/test.py",)
    assert tuple(gate.entrypoint_gate_context.allowed_decisions) == (case["decision"],)
    assert gate.entrypoint_gate_context.require_resolved_config_sha256 is True
    assert "raw_prediction_cache" in gate.entrypoint_gate_context.forbidden_true_keys
    assert "paper_claim" in gate.entrypoint_gate_context.forbidden_true_keys

    with pytest.raises(RuntimeError, match="allow_tools_train=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")

    gate_json, gate_sha = _write_gate(tmp_path, decision=case["decision"])
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "resolved-sha")
    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py") is None


@pytest.mark.parametrize("case", [case for case in CASES if case["reader_disabled"]], ids=["R17_C1", "R18_C1"])
def test_local_lowmem_reader_disabled_configs_keep_exact_uniform_override(case):
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG_DIR / case["config"]))

    gate = cfg.get(case["gate_name"])
    assert gate.reader_disabled_eval_candidate is True
    assert "reader_disabled_exact_uniform_eval_override" in gate.allowed_checks
    override = cfg.model.pc_ot_mras_reader_eval_override
    assert override.enabled is True
    assert override.mode == "exact_uniform"
    assert override.num_slots == 384
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"


@pytest.mark.parametrize("case", [case for case in CASES if case["r18_aux"]], ids=["R18_C0", "R18_C1"])
def test_local_lowmem_r18_configs_keep_auxiliary_model_surface(case):
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG_DIR / case["config"]))

    assert cfg.model.pc_ot_mras_reader_aux_loss.enabled is True
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2"


@pytest.mark.parametrize("case", [case for case in CASES if case["r17_reader"]], ids=["R17_C0", "R17_C1"])
def test_local_lowmem_r17_configs_keep_reader_pair_distribution_disabled_when_parent_has_patch(case):
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG_DIR / case["config"]))
    parent = mmengine_config.Config.fromfile(str(CONFIG_DIR / case["parent"]))

    if "emit_pair_distribution" in parent.model.pc_ot_mras_reader:
        assert parent.model.pc_ot_mras_reader.emit_pair_distribution is False
        assert cfg.model.pc_ot_mras_reader.emit_pair_distribution is False
