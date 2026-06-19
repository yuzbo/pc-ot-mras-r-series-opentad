from collections.abc import Mapping


def contains_pc_ot_mras_live_path(node):
    if isinstance(node, Mapping):
        node_type = node.get("type", "")
        if node_type in {"PCOTMRASReader", "PCOTMRASDetectorBridge"}:
            return True
        if "pc_ot_mras_reader" in node and node.get("pc_ot_mras_reader") is not None:
            return True
        return any(contains_pc_ot_mras_live_path(value) for value in node.values())
    if isinstance(node, (list, tuple)):
        return any(contains_pc_ot_mras_live_path(value) for value in node)
    return False


def assert_no_raw_prediction_shortcut_for_pc_ot_mras(cfg):
    if not contains_pc_ot_mras_live_path(cfg.model):
        return
    if getattr(cfg.inference, "load_from_raw_predictions", False) or getattr(cfg.inference, "save_raw_prediction", False):
        raise ValueError(
            "PC-OT-MRAS configs forbid inference.load_from_raw_predictions/save_raw_prediction because "
            "raw prediction caches bypass the live reader/bridge path"
        )
