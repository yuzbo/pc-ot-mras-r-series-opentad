import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.rba_rbr.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE  # noqa: E402
from opentad.acquisition.rba_rbr.types import FORBIDDEN_ROUTE_TOKENS, ROUTE_LABEL  # noqa: E402


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
    text = Path(config_path).read_text(encoding="utf-8")
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
    }
    for token, label in required.items():
        if token not in text:
            raise ValueError(f"RBA-RBR launch gate requires {label}: {token}")
    if "rba_rbr_allow_diagnostic_preview_fallback=True" in text:
        raise ValueError("RBA-RBR formal config must not enable diagnostic preview fallback")
    return True


def validate_launch_gate(config_path, precheck_summary_path):
    _config_text_is_clean(config_path)
    _resolved_config_is_clean(config_path)
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
    return {
        "route_label": ROUTE_LABEL,
        "gate_pass": True,
        "allowed_next_action": "FINAL_READ_ONLY_REVIEW_THEN_LOCAL_PRECHECK_ONLY",
        "full_train_unlocked": False,
        "remote_sync_unlocked_by_local_gate": False,
        "sparse_compute_claim": False,
        "deploy_claim_unlocked": False,
        "paper_claim_unlocked": False,
    }


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
