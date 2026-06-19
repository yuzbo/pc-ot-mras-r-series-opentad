from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


def _ensure_package(name, path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pc_ot_mras_classes():
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.necks", ROOT / "opentad" / "models" / "necks")

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    builder.NECKS = _Registry()
    sys.modules["opentad.models.builder"] = builder

    _load_module(
        "opentad.ctf_bdi_role_constants",
        ROOT / "opentad" / "ctf_bdi_role_constants.py",
    )
    _load_module(
        "opentad.models.selectors.lowcost_acquisition_browser",
        ROOT / "opentad" / "models" / "selectors" / "lowcost_acquisition_browser.py",
    )
    reader_module = _load_module(
        "opentad.models.selectors.pc_ot_mras_reader",
        ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py",
    )
    bridge_module = _load_module(
        "opentad.models.necks.pc_ot_mras_detector_bridge",
        ROOT / "opentad" / "models" / "necks" / "pc_ot_mras_detector_bridge.py",
    )
    return reader_module.PCOTMRASReader, bridge_module.PCOTMRASDetectorBridge
