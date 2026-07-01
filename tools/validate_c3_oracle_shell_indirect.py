import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

_PSEUDO_BOUNDARY_PATH = REPO_ROOT / "opentad/datasets/transforms/pseudo_boundary.py"
_PSEUDO_SPEC = importlib.util.spec_from_file_location("_c3_oracle_shell_pseudo_boundary", _PSEUDO_BOUNDARY_PATH)
_PSEUDO_MODULE = importlib.util.module_from_spec(_PSEUDO_SPEC)
_PSEUDO_SPEC.loader.exec_module(_PSEUDO_MODULE)
validate_coarse_score_cache_manifest = _PSEUDO_MODULE.validate_coarse_score_cache_manifest


ROUTE_LABELS = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
FORBIDDEN_TOKENS = [
    "teacher",
    "raw_prediction_cache",
    "p2",
    "pqr",
    "pqr_ranking",
    "bh_sdc",
    "bhsdc",
    "divergent",
    "event-surprise",
    "event_surprise",
    "boundary_microscope",
    "frame_token_hybrid",
]


def _assert_no_forbidden_tokens(cfg_text):
    found = [token for token in FORBIDDEN_TOKENS if token in cfg_text]
    if found:
        raise AssertionError(f"Forbidden route/cache/teacher tokens in C3 oracle-shell indirect config: {found}")


def _load_frame_steps(cfg):
    steps = []
    for split in ("train", "val", "test"):
        split_cfg = cfg.dataset[split]
        load = next(step for step in split_cfg.pipeline if step["type"] == "LoadFrames")
        steps.append((split, load))
    return steps


def _collect_steps(cfg):
    steps = []
    for split in ("train", "val", "test"):
        split_cfg = cfg.dataset[split]
        collect = next(step for step in split_cfg.pipeline if step["type"] == "Collect")
        steps.append((split, collect))
    return steps


def _validate_pipeline(cfg):
    for split, load in _load_frame_steps(cfg):
        if load["method"] not in {"coarse_score_oracle_shell_subsample", "coarse_actionness_oracle_shell_subsample"}:
            raise AssertionError(f"{split} pipeline must use coarse oracle-shell indirect LoadFrames method")
        if load.get("method_base") not in {"random_trunc", "sliding_window"}:
            raise AssertionError(f"{split} pipeline must keep oracle shell method_base semantics")
        if int(load.get("target_len", -1)) != 384:
            raise AssertionError(f"{split} target_len must be 384")
        if split == "train" and int(load.get("source_len", -1)) != 768:
            raise AssertionError("train source_len must be 768")
        if split in {"val", "test"} and int(cfg.dataset[split].get("window_size", -1)) != 768:
            raise AssertionError(f"{split} window_size must be 768")
        if bool(load.get("coarse_score_allow_missing", False)):
            raise AssertionError(f"{split} must fail closed when coarse score is missing")
        if bool(load.get("coarse_score_debug_fallback", False)):
            raise AssertionError(f"{split} debug fallback must be disabled in route configs")
        cache_dir = str(load.get("coarse_score_cache_dir", ""))
        if not cache_dir:
            raise AssertionError(f"{split} requires coarse_score_cache_dir")
        if cache_dir != "REPLACE_WITH_C3_COARSE_SCORE_CACHE_DIR" and Path(cache_dir).exists():
            validate_coarse_score_cache_manifest(Path(cache_dir) / "manifest.json", expected_axis="global_snippet_index")
    required_meta = {
        "irregular_selected_positions",
        "irregular_selected_valid_len",
        "irregular_native_axis",
        "coarse_oracle_shell_score_source",
        "coarse_oracle_shell_uses_gt_for_selection",
        "coarse_oracle_shell_score_axis",
        "coarse_oracle_shell_selected_positions",
    }
    for split, collect in _collect_steps(cfg):
        meta_keys = set(collect.get("meta_keys", []))
        missing = sorted(required_meta - meta_keys)
        if missing:
            raise AssertionError(f"{split} Collect must preserve oracle-shell metadata: {missing}")
        keys = set(collect.get("keys", []))
        if split == "test" and ({"gt_segments", "gt_labels"} & keys):
            raise AssertionError("test Collect must not pass gt_segments/gt_labels into the model")
    if not bool(cfg.dataset.test.get("test_mode", False)):
        raise AssertionError("test split must use test_mode=True to avoid validation/test GT protocol pollution")


def _validate_schedule(cfg):
    claim_status = cfg.get("c3_claim_status", "")
    if claim_status == "precheck_only":
        if int(cfg.workflow.get("max_train_iters", 0)) <= 0:
            raise AssertionError("precheck must bound max_train_iters")
        if int(cfg.workflow.get("val_eval_interval", 0)) != -1:
            raise AssertionError("precheck must disable validation")
        return
    if claim_status != "formal_training_candidate_locked":
        raise AssertionError(f"Unsupported C3 oracle-shell indirect claim_status: {claim_status}")
    if cfg.get("c3_full_train_launch_locked_until_user_or_main_process", None) is not True:
        raise AssertionError("full-train config must remain launch-locked for the main process")
    if int(cfg.workflow.get("end_epoch", 0)) != 60:
        raise AssertionError("full train must use end_epoch=60")
    if cfg.workflow.get("max_train_iters", "not-none") is not None:
        raise AssertionError("full train must not bound max_train_iters")
    if int(cfg.workflow.get("val_start_epoch", -1)) != 2:
        raise AssertionError("full train must first validate at epoch2")
    if [int(x) for x in cfg.workflow.get("val_eval_epochs", [])] != [2]:
        raise AssertionError("full train must include explicit epoch2 validation")
    if int(cfg.workflow.get("val_eval_interval", -1)) != 5:
        raise AssertionError("full train must validate every 5 epochs after epoch2")
    if int(cfg.workflow.get("val_eval_interval_anchor_epoch", -1)) != 2:
        raise AssertionError("full train validation interval must anchor at epoch2")


def validate_config(config_path):
    cfg = Config.fromfile(config_path)
    cfg_text = cfg.pretty_text.lower()
    _assert_no_forbidden_tokens(cfg_text)
    if cfg.get("c3_route_label", None) != "C3_MAINLINE_OPTIMIZATION":
        raise AssertionError("config must use C3_MAINLINE_OPTIMIZATION route label")
    if cfg.get("c3_route_labels", []) != ROUTE_LABELS:
        raise AssertionError("config must use exact C3 route label list")
    if cfg.get("c3_method", None) != "C3-OracleShell-Indirect-CoarseScore-OriginalAdaTAD":
        raise AssertionError("config must use the C3 oracle-shell indirect method marker")
    if cfg.get("c3_no_test_gt_selection", None) is not True:
        raise AssertionError("config must explicitly mark no test GT selection")
    if cfg.get("c3_deploy_selection_uses_gt", None) is not False:
        raise AssertionError("deploy selection must not use GT")
    if cfg.get("c3_oracle_gt_diagnostic_only", None) is not True:
        raise AssertionError("oracle GT comparison must be diagnostic-only")
    if cfg.model.type != "ActionFormer" or cfg.model.rpn_head.type != "ActionFormerHead":
        raise AssertionError("config must keep Original AdaTAD ActionFormer backend")
    if cfg.model.get("frame_selector", None) is not None:
        raise AssertionError("oracle-shell indirect route must not use model.frame_selector/P2/PQR/teacher shortcuts")
    if cfg.model.backbone.backbone.total_frames != 384:
        raise AssertionError("backbone total_frames must be 384")
    if cfg.model.projection.max_seq_len != 384:
        raise AssertionError("projection max_seq_len must be 384")
    if cfg.inference.load_from_raw_predictions is not False or cfg.inference.save_raw_prediction is not False:
        raise AssertionError("raw prediction cache load/save must be disabled")
    _validate_pipeline(cfg)
    _validate_schedule(cfg)
    print(f"PASS_C3_ORACLE_SHELL_INDIRECT_CONFIG {Path(config_path).as_posix()}")


def main():
    parser = argparse.ArgumentParser(description="Fail-closed validator for C3 oracle-shell indirect configs.")
    parser.add_argument("config")
    args = parser.parse_args()
    validate_config(args.config)


if __name__ == "__main__":
    main()
