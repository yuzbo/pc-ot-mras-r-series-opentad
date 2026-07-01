import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.rba_rbr.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE  # noqa: E402
from opentad.acquisition.rba_rbr.adapter_bridge import build_adapter_fixed_length_padded_bridge  # noqa: E402
from opentad.acquisition.rba_rbr.open_tad_bridge import build_rba_rbr_open_tad_selection  # noqa: E402
from opentad.acquisition.rba_rbr.types import FORBIDDEN_ROUTE_TOKENS, ROUTE_LABEL  # noqa: E402
from opentad.acquisition.rba_rbr.validators import validate_rba_rbr_control_bridge_metadata  # noqa: E402


def _read_config_family_text(config_path, seen=None):
    config_path = Path(config_path).resolve()
    seen = set() if seen is None else seen
    if config_path in seen:
        return ""
    seen.add(config_path)
    text = config_path.read_text(encoding="utf-8")
    parts = [text]
    for match in re.finditer(r"_base_\s*=\s*\[(.*?)\]", text, flags=re.DOTALL):
        for base_name in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
            base_path = (config_path.parent / base_name).resolve()
            if base_path.exists():
                parts.append(_read_config_family_text(base_path, seen=seen))
    return "\n".join(parts)


def _resolved_config_is_clean(config_path):
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    annotation_path = str(cfg.annotation_path)
    ground_truth_filename = str(cfg.evaluation.ground_truth_filename)
    if ground_truth_filename != annotation_path:
        raise ValueError(
            "RBA-RBR launch gate requires evaluation.ground_truth_filename "
            "to match the route-owned annotation_path"
        )
    if "/root/autodl-tmp" in ground_truth_filename.replace("\\", "/"):
        raise ValueError("RBA-RBR launch gate rejects legacy /root/autodl-tmp evaluation paths")
    return True


def _config_text_is_clean(config_path):
    text = _read_config_family_text(config_path)
    if "rba_rbr_recoverable_bracketing" not in text:
        raise ValueError("RBA-RBR launch gate requires method='rba_rbr_recoverable_bracketing'")
    if ROUTE_LABEL not in text:
        raise ValueError("RBA-RBR launch gate requires explicit route label in config")
    normalized = text.replace(ROUTE_LABEL, "").replace("checkpoint_interval", "checkpoint_period")
    for token in FORBIDDEN_ROUTE_TOKENS:
        if token.lower() in normalized.lower():
            raise ValueError(f"RBA-RBR launch gate rejects forbidden route token in config: {token}")
    required = {
        "full_train_unlocked = False": "full-train lock",
        "no_metric_claim = True": "metric claim lock",
        "no_runtime_claim = True": "runtime claim lock",
        "no_deploy_claim = True": "deploy claim lock",
        "no_paper_claim = True": "paper claim lock",
        "ground_truth_filename=annotation_path": "N16R4 evaluation annotation path override",
        "rba_rbr_train_value_labels=True": "train-only labels enabled only in train pipeline",
        "rba_rbr_train_value_labels=False": "train-only labels disabled in val/test pipeline",
        "rba_rbr_allow_diagnostic_preview_fallback=False": "formal preview fallback lock",
        "rba_rbr_scout_sample_count=32": "deploy-visible raw scout sample count",
        "rba_rbr_min_detector_feature_keep=32": "recoverable detector feature floor",
        "rba_rbr_max_raw_gap=16": "recoverable raw gap guard",
        "rba_rbr_max_detector_gap=24": "recoverable detector gap guard",
        f'rba_rbr_adapter_bridge_mode="{ADAPTER_FIXED_LENGTH_PADDED_BRIDGE}"': "adapter bridge mode",
        "rba_rbr_postprocess_guard=dict(": "RBA-RBR postprocess guard",
        "require_rba_meta=True": "RBA-RBR postprocess metadata fail-closed guard",
        "raw_proposal_cap=1024": "RBA-RBR raw proposal cap",
        "per_class_topk=32": "RBA-RBR per-class postprocess cap",
        "total_candidate_cap=512": "RBA-RBR total postprocess cap",
    }
    for token, label in required.items():
        if token not in text:
            raise ValueError(f"RBA-RBR launch gate requires {label}: {token}")
    if "rba_rbr_allow_diagnostic_preview_fallback=True" in text:
        raise ValueError("RBA-RBR formal config must not enable diagnostic preview fallback")
    return True


def _field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _rba_step(cfg, split):
    pipeline = cfg.dataset[split].pipeline
    for step in pipeline:
        if _field(step, "method") == "rba_rbr_recoverable_bracketing":
            return step
    raise ValueError(f"RBA-RBR launch gate could not find LoadFrames RBA step in {split} pipeline")


def _audit_control_config(cfg):
    if not bool(_field(cfg, "rba_rbr_control_diagnostic_only", False)):
        return None
    if not bool(_field(cfg, "short_diagnostic_only", False)):
        raise ValueError("RBA-RBR control configs must be short_diagnostic_only=true")
    if bool(_field(cfg, "full_train_unlocked", True)):
        raise ValueError("RBA-RBR control configs must keep full_train_unlocked=false")
    for claim_key in ("no_metric_claim", "no_runtime_claim", "no_deploy_claim", "no_paper_claim", "no_sparse_compute_claim"):
        if not bool(_field(cfg, claim_key, False)):
            raise ValueError(f"RBA-RBR control configs must keep {claim_key}=true")

    for split in ("train", "val", "test"):
        step = _rba_step(cfg, split)
        if _field(step, "rba_rbr_control_mode") != "uniform_raw":
            raise ValueError(f"RBA-RBR control {split} pipeline requires rba_rbr_control_mode=uniform_raw")
        if bool(_field(step, "rba_rbr_train_value_labels", True)):
            raise ValueError(f"RBA-RBR control {split} pipeline must keep train value labels disabled")
        if bool(_field(step, "rba_rbr_allow_diagnostic_preview_fallback", True)):
            raise ValueError(f"RBA-RBR control {split} pipeline must keep diagnostic preview fallback disabled")
        if int(_field(step, "rba_rbr_feature_stride", 1)) != 2:
            raise ValueError(f"RBA-RBR control {split} pipeline requires feature_stride=2")
        if _field(step, "rba_rbr_adapter_bridge_mode") != ADAPTER_FIXED_LENGTH_PADDED_BRIDGE:
            raise ValueError(f"RBA-RBR control {split} pipeline requires fixed-length adapter bridge")

    step = _rba_step(cfg, "test")
    dense_T = int(_field(cfg, "dense_window_size", 384))
    target_frame_num = int(_field(step, "target_len")) * int(max(_field(step, "scale_factor", 1), 1))
    feature_stride = int(_field(step, "rba_rbr_feature_stride", 2))
    result = build_rba_rbr_open_tad_selection(
        {"video_name": "rba_rbr_control_gate_audit"},
        dense_window=list(range(dense_T)),
        target_frame_num=target_frame_num,
        split="test",
        train_value_labels=False,
        min_keep=_field(step, "rba_rbr_min_keep"),
        max_keep=_field(step, "rba_rbr_max_keep"),
        scaffold_k=_field(step, "rba_rbr_scaffold_k", 4),
        allow_diagnostic_preview_fallback=False,
        scout_sample_count=_field(step, "rba_rbr_scout_sample_count", 32),
        min_detector_feature_keep=_field(step, "rba_rbr_min_detector_feature_keep"),
        feature_stride=feature_stride,
        max_raw_gap=_field(step, "rba_rbr_max_raw_gap"),
        max_detector_gap=_field(step, "rba_rbr_max_detector_gap"),
        control_mode=_field(step, "rba_rbr_control_mode"),
        control_keep=_field(step, "rba_rbr_control_keep"),
    )
    bridge = build_adapter_fixed_length_padded_bridge(
        selected_positions=result["keep_positions"],
        selected_frame_inds=result["selected_frame_inds"],
        target_frame_num=target_frame_num,
        dense_T=dense_T,
        feature_stride=feature_stride,
    )
    bridge_meta = {
        "irregular_native_axis": True,
        "rba_rbr_raw_selected_positions": result["keep_positions"],
        "rba_rbr_raw_selected_valid_len": float(dense_T),
        "rba_rbr_detector_feature_positions": bridge["detector_feature_positions"],
        "rba_rbr_detector_feature_valid_len": float(dense_T),
        "detector_valid_mask": bridge["detector_valid_mask"],
        "adapter_valid_raw_mask": bridge["adapter_valid_raw_mask"],
    }
    validate_rba_rbr_control_bridge_metadata(result["ledger"], bridge_meta)
    return {
        "control_mode": "uniform_raw",
        "raw_valid_k": int(result["ledger"]["valid_k"]),
        "target_frame_num": int(target_frame_num),
        "detector_feature_valid_k": int(result["ledger"]["detector_feature_valid_k"]),
        "detector_mask_len": int(bridge["detector_mask_len"]),
        "raw_density": float(result["ledger"]["control_raw_density"]),
        "detector_density": float(result["ledger"]["control_detector_density"]),
    }


def validate_launch_gate(config_path, precheck_summary_path):
    _config_text_is_clean(config_path)
    _resolved_config_is_clean(config_path)
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    control_audit = _audit_control_config(cfg)
    summary = json.loads(Path(precheck_summary_path).read_text(encoding="utf-8"))
    if summary.get("route_label") != ROUTE_LABEL:
        raise ValueError("RBA-RBR launch gate summary has wrong route_label")
    if summary.get("all_validated") is not True:
        raise ValueError("RBA-RBR launch gate requires all_validated=true")
    if summary.get("recovery_diagnostic", {}).get("recovered_boundary_by_rescue") is not True:
        raise ValueError("RBA-RBR launch gate requires rescue recovery diagnostic")
    if summary.get("recovery_diagnostic", {}).get("hard_bracket_would_miss_boundary") is not True:
        raise ValueError("RBA-RBR launch gate requires a hard-bracket miss contrast")
    if bool(summary.get("full_train_unlocked", True)):
        raise ValueError("RBA-RBR launch gate keeps full_train_unlocked=false")
    if summary.get("no_training") is not True or summary.get("no_metric_claim") is not True:
        raise ValueError("RBA-RBR local precheck must be no-training and no-metric")
    if summary.get("no_runtime_claim") is not True:
        raise ValueError("RBA-RBR local precheck must not claim runtime/FLOPs")
    if summary.get("no_deploy_claim") is not True:
        raise ValueError("RBA-RBR local precheck must not claim deployment readiness")
    if summary.get("no_paper_claim") is not True:
        raise ValueError("RBA-RBR local precheck must not claim paper readiness")
    result = {
        "route_label": ROUTE_LABEL,
        "gate_pass": True,
        "allowed_next_action": "FINAL_READ_ONLY_REVIEW_THEN_LOCAL_PRECHECK_ONLY",
        "full_train_unlocked": False,
        "remote_sync_unlocked_by_local_gate": False,
        "sparse_compute_claim": False,
        "deploy_claim_unlocked": False,
        "paper_claim_unlocked": False,
    }
    if control_audit is not None:
        result["control_audit"] = control_audit
        result["allowed_next_action"] = "FINAL_READ_ONLY_REVIEW_THEN_SHORT_DIAGNOSTIC_ONLY"
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate RBA-RBR local launch gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--precheck-summary")
    parser.add_argument(
        "--audit-out-dir",
        default=str(Path(tempfile.gettempdir()) / "rba_rbr_launch_gate_precheck"),
    )
    args = parser.parse_args()
    precheck_summary = args.precheck_summary
    if precheck_summary is None:
        from tools.rba_rbr.build_synthetic_ledgers import build_ledgers

        build_ledgers(args.audit_out_dir, overwrite=True)
        precheck_summary = str(Path(args.audit_out_dir) / "summary.json")
    try:
        result = validate_launch_gate(args.config, precheck_summary)
    except Exception as exc:
        print(json.dumps({"gate_pass": False, "reason": str(exc)}, ensure_ascii=False, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
