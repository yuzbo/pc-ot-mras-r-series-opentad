import importlib.util
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_guard_module():
    path = ROOT / "opentad" / "models" / "utils" / "pc_ot_mras_raw_prediction_guard.py"
    spec = importlib.util.spec_from_file_location("pc_ot_mras_raw_prediction_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(model, *, load=False, save=False):
    return types.SimpleNamespace(
        model=model,
        inference=types.SimpleNamespace(
            load_from_raw_predictions=load,
            save_raw_prediction=save,
        ),
    )


def test_pc_ot_mras_test_tool_gate_rejects_raw_prediction_load_and_save():
    module = _load_guard_module()
    model = {
        "type": "ActionFormer",
        "projection": {"type": "SyntheticProjection"},
        "pc_ot_mras_reader": {"type": "PCOTMRASReader"},
        "neck": {"type": "PCOTMRASDetectorBridge"},
    }

    with pytest.raises(ValueError, match="raw prediction caches bypass"):
        module.assert_no_raw_prediction_shortcut_for_pc_ot_mras(_cfg(model, load=True))

    with pytest.raises(ValueError, match="raw prediction caches bypass"):
        module.assert_no_raw_prediction_shortcut_for_pc_ot_mras(_cfg(model, save=True))


def test_pc_ot_mras_test_tool_gate_detects_nested_bridge_without_reader_key():
    module = _load_guard_module()
    model = {
        "type": "ActionFormer",
        "neck": [
            {"type": "SomeLegacyNeck"},
            {"type": "PCOTMRASDetectorBridge"},
        ],
    }

    with pytest.raises(ValueError, match="raw prediction caches bypass"):
        module.assert_no_raw_prediction_shortcut_for_pc_ot_mras(_cfg(model, load=True))


def test_pc_ot_mras_test_tool_gate_allows_legacy_raw_prediction_tuning():
    module = _load_guard_module()
    legacy_model = {
        "type": "ActionFormer",
        "projection": {"type": "ActionFormerProjection"},
        "neck": {"type": "FPN"},
        "rpn_head": {"type": "ActionFormerHead"},
    }

    module.assert_no_raw_prediction_shortcut_for_pc_ot_mras(_cfg(legacy_model, load=True))
    module.assert_no_raw_prediction_shortcut_for_pc_ot_mras(_cfg(legacy_model, save=True))
