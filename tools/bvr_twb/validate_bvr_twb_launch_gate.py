import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
from opentad.acquisition.bvr_twb.types import FORBIDDEN_ROUTE_TOKENS, ROUTE_LABEL


FORMAL_PREVIEW_SOURCES = {"deploy_visible_metadata_actionness", "raw_rgb_lowres_scout"}
FORMAL_SCOUT_SOURCES = {
    "deploy_visible_raw_or_metadata_scout",
    "deploy_visible_metadata_scout",
    "raw_rgb_lowres_scout",
}
FORMAL_VALUE_MODES = {"deploy_heuristic_voi"}
REQUIRED_PRETRAIN_PATH = "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"


def _require_formal_pretrain(config_path, text):
    escaped = re.escape(REQUIRED_PRETRAIN_PATH)
    explicit_line = re.compile(rf"^\s*pretrain\s*=\s*['\"]{escaped}['\"]\s*,?\s*$", re.MULTILINE)
    if not explicit_line.search(text):
        raise ValueError(
            "BVR-TWB formal config must explicitly declare VideoMAE-S pretrain path "
            f"inside model.backbone.custom: {REQUIRED_PRETRAIN_PATH}"
        )
    try:
        from mmengine.config import Config
    except Exception as exc:
        raise ValueError(f"BVR-TWB launch gate cannot verify resolved pretrain without mmengine: {exc}") from exc

    cfg = Config.fromfile(str(config_path))
    custom_cfg = cfg.model.backbone.custom
    pretrain = custom_cfg.get("pretrain") if hasattr(custom_cfg, "get") else getattr(custom_cfg, "pretrain", None)
    if pretrain != REQUIRED_PRETRAIN_PATH:
        raise ValueError(
            "BVR-TWB resolved config must retain VideoMAE-S pretrain path; "
            f"got {pretrain!r}, expected {REQUIRED_PRETRAIN_PATH!r}"
        )
    return True


def _require_resolved_bvr_dynamic_method(config_path):
    try:
        from mmengine.config import Config
    except Exception as exc:
        raise ValueError(f"BVR-TWB launch gate cannot verify resolved LoadFrames method without mmengine: {exc}") from exc

    cfg = Config.fromfile(str(config_path))
    bad_splits = []
    for split in ("train", "val", "test"):
        dataset = getattr(cfg.dataset, split)
        load_steps = [step for step in dataset.pipeline if step.get("type") == "LoadFrames"]
        if not load_steps:
            bad_splits.append(f"{split}:missing LoadFrames")
            continue
        methods = [step.get("method") for step in load_steps]
        if "bvr_twb_dynamic_subsample" not in methods:
            bad_splits.append(f"{split}:{methods}")
    if bad_splits:
        raise ValueError(
            "BVR-TWB launch gate requires resolved train/val/test LoadFrames "
            f"method='bvr_twb_dynamic_subsample'; bad_splits={bad_splits}"
        )
    return True


def _required_string_set(summary, key):
    if key not in summary:
        raise ValueError(f"BVR-TWB launch gate requires {key} evidence in precheck summary")
    values = summary[key]
    if isinstance(values, str) or not isinstance(values, (list, tuple, set)):
        raise ValueError(f"BVR-TWB launch gate requires {key} to be a non-empty list/set of strings")
    values = set(values)
    if not values:
        raise ValueError(f"BVR-TWB launch gate requires non-empty {key} evidence")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"BVR-TWB launch gate requires {key} entries to be non-empty strings")
    return values


def _config_text_is_clean(config_path):
    text = Path(config_path).read_text(encoding="utf-8")
    if "bvr_twb_dynamic_subsample" not in text:
        raise ValueError("BVR-TWB launch gate requires method='bvr_twb_dynamic_subsample'")
    if ROUTE_LABEL not in text:
        raise ValueError("BVR-TWB launch gate requires explicit route label in config")
    normalized = text.replace(ROUTE_LABEL, "").replace("checkpoint_interval", "checkpoint_period")
    for token in FORBIDDEN_ROUTE_TOKENS:
        if token.lower() in normalized.lower():
            raise ValueError(f"BVR-TWB launch gate rejects forbidden route token in config: {token}")
    _require_formal_pretrain(config_path, text)
    _require_resolved_bvr_dynamic_method(config_path)
    if f'bvr_twb_adapter_bridge_mode="{ADAPTER_FIXED_LENGTH_PADDED_BRIDGE}"' not in text and (
        f"bvr_twb_adapter_bridge_mode='{ADAPTER_FIXED_LENGTH_PADDED_BRIDGE}'" not in text
    ):
        raise ValueError("BVR-TWB launch gate requires adapter_fixed_length_padded_bridge metadata in config")
    if "bvr_twb_train_value_labels=True" not in text:
        raise ValueError("BVR-TWB train config must enable train-only value labels for train split")
    if "bvr_twb_train_value_labels=False" not in text:
        raise ValueError("BVR-TWB val/test config must explicitly disable train value labels")
    required_gate_tokens = {
        "full_train_unlocked = False": "formal train lock",
        "metric_claim = False": "metric claim lock",
        "sparse_compute_claim = False": "sparse compute claim lock",
        "formal_readiness_requires_linux_torch_precheck = True": "Linux torch precheck requirement",
        "formal_readiness_requires_finite_gradient_evidence = True": "finite-gradient evidence requirement",
        "formal_readiness_requires_no_skipped_reg_head = True": "no skipped regression-head evidence requirement",
    }
    for token, description in required_gate_tokens.items():
        if token not in text:
            raise ValueError(f"BVR-TWB launch gate requires {description}: {token}")
    required_formal_tokens = {
        'bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout"': "formal scout source",
        "bvr_twb_require_deploy_visible_scout=True": "deploy-visible scout requirement",
        "bvr_twb_allow_diagnostic_preview_fallback=False": "diagnostic deterministic fallback lock",
        'bvr_twb_value_mode="deploy_heuristic_voi"': "deploy-visible VOI value mode",
    }
    for token, description in required_formal_tokens.items():
        if token not in text:
            raise ValueError(f"BVR-TWB launch gate requires {description}: {token}")
    if "bvr_twb_allow_diagnostic_preview_fallback=True" in text:
        raise ValueError("BVR-TWB formal config must not enable diagnostic preview fallback")
    if "diagnostic_deterministic_preview" in text:
        raise ValueError("BVR-TWB formal config must not request diagnostic deterministic preview")
    if "learned_packet_value" in text:
        raise ValueError("BVR-TWB formal local/precheck config must not silently require an unloaded learned value model")
    return True


def validate_launch_gate(config_path, precheck_summary_path):
    _config_text_is_clean(config_path)
    summary = json.loads(Path(precheck_summary_path).read_text(encoding="utf-8"))
    if summary.get("route_label") != ROUTE_LABEL:
        raise ValueError("BVR-TWB launch gate summary has wrong route_label")
    if bool(summary.get("blocked", False)):
        raise ValueError(f"BVR-TWB launch gate blocked at {summary.get('blocked_stage')}: {summary.get('blocked_reason')}")
    if summary.get("all_validated") is not True:
        raise ValueError("BVR-TWB launch gate requires all_validated=true")
    bridge_modes = set(summary.get("adapter_bridge_modes", []))
    bridge_mode = summary.get("adapter_bridge_mode")
    if bridge_mode is not None:
        bridge_modes.add(bridge_mode)
    if ADAPTER_FIXED_LENGTH_PADDED_BRIDGE not in bridge_modes:
        raise ValueError("BVR-TWB launch gate requires adapter_fixed_length_padded_bridge precheck evidence")
    if summary.get("adapter_padding_counts_as_valid") is not False:
        raise ValueError("BVR-TWB launch gate requires adapter padding duplicates to be invalid")
    if summary.get("raw_frame_handoff_stages") != ["pre_decode_selected_raw_frames"]:
        raise ValueError("BVR-TWB launch gate requires selected raw frames before decode/backbone evidence")
    if summary.get("selected_raw_frames_before_decode") is not True:
        raise ValueError("BVR-TWB launch gate requires selected raw frame handoff before decode")
    if summary.get("fixed_padded_bridge_sparse_compute_claim") is not False:
        raise ValueError("BVR-TWB launch gate rejects fixed padded bridge sparse-compute claims")
    if summary.get("adapter_padding_invalid_for_detector") is not True:
        raise ValueError("BVR-TWB launch gate requires adapter padding to be invalid for detector/head")
    preview_sources = _required_string_set(summary, "preview_sources")
    if not preview_sources.issubset(FORMAL_PREVIEW_SOURCES):
        raise ValueError(f"BVR-TWB launch gate rejects non-formal preview sources: {sorted(preview_sources)}")
    scout_sources = _required_string_set(summary, "scout_sources")
    if not scout_sources.issubset(FORMAL_SCOUT_SOURCES):
        raise ValueError(f"BVR-TWB launch gate rejects non-formal scout sources: {sorted(scout_sources)}")
    if bool(summary.get("deterministic_preview_fallback_used", False)):
        raise ValueError("BVR-TWB launch gate rejects deterministic preview fallback")
    value_modes = _required_string_set(summary, "value_modes")
    if value_modes != FORMAL_VALUE_MODES:
        raise ValueError(f"BVR-TWB launch gate rejects unexpected value_modes: {sorted(value_modes)}")
    if bool(summary.get("value_labels_used_at_test", False)):
        raise ValueError("BVR-TWB launch gate rejects value labels used at test/deploy")
    if summary.get("loadframes_source_dispatch_contract") != "passed":
        raise ValueError("BVR-TWB launch gate requires LoadFrames source dispatch proof")
    if summary.get("loadframes_dispatch_method") != "bvr_twb_dynamic_subsample":
        raise ValueError("BVR-TWB launch gate requires BVR-TWB dynamic LoadFrames dispatch method proof")
    if summary.get("loadframes_dispatch_calls_bvr_bridge") is not True:
        raise ValueError("BVR-TWB launch gate requires LoadFrames dispatch to call BVR bridge")
    if summary.get("loadframes_dispatch_assigns_bridge") is not True:
        raise ValueError("BVR-TWB launch gate requires LoadFrames dispatch to assign bridge from BVR bridge")
    if bool(summary.get("sparse_compute_claim", False)):
        raise ValueError("BVR-TWB precheck summary must not claim sparse compute")
    if summary.get("no_training") is not True or summary.get("no_metric_claim") is not True:
        raise ValueError("BVR-TWB precheck summary must be no-training and no-metric")
    return {
        "route_label": ROUTE_LABEL,
        "allowed_next_action": "FINAL_READ_ONLY_REVIEW_THEN_LINUX_PRECHECK_ONLY",
        "full_train_unlocked": False,
        "remote_sync_unlocked_by_local_gate": False,
        "sparse_compute_claim": False,
        "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate BVR-TWB local launch gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--precheck-summary")
    parser.add_argument(
        "--audit-out-dir",
        default=".tmp_bvr_twb_launch_gate_precheck",
        help="Used only when --precheck-summary is omitted.",
    )
    args = parser.parse_args()
    precheck_summary = args.precheck_summary
    if precheck_summary is None:
        from tools.bvr_twb.audit_opentad_bvr_twb_pipeline import run_pipeline_audit

        summary = run_pipeline_audit(args.audit_out_dir, overwrite=True)
        precheck_summary = str(Path(args.audit_out_dir) / "summary.json")
        if summary.get("blocked", False):
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    try:
        result = validate_launch_gate(args.config, precheck_summary)
    except Exception as exc:
        print(json.dumps({"gate_pass": False, "reason": str(exc)}, ensure_ascii=False, sort_keys=True))
        raise SystemExit(1)
    result["gate_pass"] = True
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
