import argparse
import json
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
    if f'bvr_twb_adapter_bridge_mode="{ADAPTER_FIXED_LENGTH_PADDED_BRIDGE}"' not in text and (
        f"bvr_twb_adapter_bridge_mode='{ADAPTER_FIXED_LENGTH_PADDED_BRIDGE}'" not in text
    ):
        raise ValueError("BVR-TWB launch gate requires adapter_fixed_length_padded_bridge metadata in config")
    if "bvr_twb_train_value_labels=True" not in text:
        raise ValueError("BVR-TWB train config must enable train-only value labels for train split")
    if "bvr_twb_train_value_labels=False" not in text:
        raise ValueError("BVR-TWB val/test config must explicitly disable train value labels")
    required_formal_tokens = {
        'bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout"': "formal scout source",
        "bvr_twb_require_deploy_visible_scout=True": "deploy-visible scout requirement",
        "bvr_twb_allow_diagnostic_preview_fallback=False": "diagnostic deterministic fallback lock",
        'bvr_twb_value_mode="deploy_heuristic_voi"': "deploy-visible VOI value mode",
        "bvr_twb_min_detector_keep=64": "effective detector-token floor",
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
    if bool(summary.get("sparse_compute_claim", False)):
        raise ValueError("BVR-TWB precheck summary must not claim sparse compute")
    if summary.get("no_training") is not True or summary.get("no_metric_claim") is not True:
        raise ValueError("BVR-TWB precheck summary must be no-training and no-metric")
    min_raw_valid_k = int(summary.get("min_raw_valid_k", 0))
    configured_min_raw_keep = int(summary.get("configured_min_raw_keep", 0))
    if min_raw_valid_k < configured_min_raw_keep or configured_min_raw_keep <= 0:
        raise ValueError(
            "BVR-TWB launch gate rejects under-budget raw selection: "
            f"min_raw_valid_k={min_raw_valid_k} configured_min_raw_keep={configured_min_raw_keep}"
        )
    min_detector_feature_valid_k = int(summary.get("min_detector_feature_valid_k", 0))
    configured_min_detector_feature_keep = int(summary.get("configured_min_detector_feature_keep", 0))
    if min_detector_feature_valid_k < configured_min_detector_feature_keep or configured_min_detector_feature_keep <= 0:
        raise ValueError(
            "BVR-TWB launch gate rejects effective detector-token floor violation: "
            f"min_detector_feature_valid_k={min_detector_feature_valid_k} "
            f"configured_min_detector_feature_keep={configured_min_detector_feature_keep}"
        )
    max_padding_ratio = float(summary.get("max_adapter_padding_duplicate_ratio", 1.0))
    max_allowed_padding_ratio = float(summary.get("max_allowed_adapter_padding_duplicate_ratio", 0.5))
    if max_padding_ratio > max_allowed_padding_ratio + 1e-12:
        raise ValueError(
            "BVR-TWB launch gate rejects duplicate padding dominance: "
            f"max_adapter_padding_duplicate_ratio={max_padding_ratio:.4f} "
            f"max_allowed_adapter_padding_duplicate_ratio={max_allowed_padding_ratio:.4f}"
        )
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
