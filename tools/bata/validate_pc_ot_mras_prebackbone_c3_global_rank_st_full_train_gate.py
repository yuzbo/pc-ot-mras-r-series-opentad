from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ALLOWED_VARIANT = "C3-GlobalRankST-BoundaryDifficulty-OriginalAdaTAD"
ALLOWED_ROUTE = "pc_ot_mras_prebackbone_c3_global_rank_st"
ALLOWED_STAGE = "c3_global_rank_st_boundary_full_train_n16r4"
ALLOWED_READER = "PCOTMRASBoundaryDifficultyTemporalFrameScout"
ALLOWED_STRATEGY = "frame_score_global_rank_st"
ALLOWED_SURROGATE = "global_rank_topk"
PASS_MESSAGE = "C3_GLOBAL_RANK_ST_FULL_TRAIN_GATE_VALIDATION_PASS"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _get(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(key, default)
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                pass
    try:
        return node[key]
    except Exception:
        return getattr(node, key, default)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _repr_lower(value: Any) -> str:
    return repr(value).lower()


def validate_config(cfg_path: str | Path) -> bool:
    from mmengine.config import Config

    cfg = Config.fromfile(str(cfg_path))
    scope = cfg.experiment_scope
    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    selector = cfg.model.frame_selector

    _require(cfg.variant_id == ALLOWED_VARIANT, f"variant_id must be {ALLOWED_VARIANT}")
    _require(cfg.route_id == ALLOWED_ROUTE, f"route_id must be {ALLOWED_ROUTE}")
    _require(scope.route_family == "C3_ORIGINAL_OPTIMIZATION_ROUTE", "route_family must be C3 original")
    _require(scope.route == ALLOWED_ROUTE, f"experiment_scope.route must be {ALLOWED_ROUTE}")
    _require(scope.variant_id == ALLOWED_VARIANT, f"experiment_scope.variant_id must be {ALLOWED_VARIANT}")
    _require(scope.selection_strategy == ALLOWED_STRATEGY, "selection_strategy must be frame_score_global_rank_st")
    _require(scope.rank_transport_surrogate == ALLOWED_SURROGATE, "rank_transport_surrogate must be global_rank_topk")
    _require(scope.selector_reader == ALLOWED_READER, f"selector_reader must be {ALLOWED_READER}")
    _require(scope.backend == "OriginalAdaTAD", "backend must stay OriginalAdaTAD")
    _require(scope.detector_stack == "original_adatad_actionformer_adapter", "detector stack must stay original")

    for key in ("uses_p2", "uses_offline_ledger", "uses_teacher", "uses_test_gt", "uses_raw_prediction_cache"):
        _require(getattr(scope, key) is False, f"experiment_scope.{key} must be false")
    for key in ("changes_detector_head", "changes_neck", "changes_loss_assignment", "changes_post_processing"):
        _require(getattr(scope, key) is False, f"experiment_scope.{key} must be false")

    _require(gate.route == ALLOWED_ROUTE, f"gate.route must be {ALLOWED_ROUTE}")
    _require(gate.formal_train_candidate is True, "formal_train_candidate must be true")
    _require(gate.launch_gate_passed is True, "launch_gate_passed must be true")
    _require(gate.allow_tools_train is True, "allow_tools_train must be true")
    _require(gate.allow_long_training is True, "allow_long_training must be true")
    _require(gate.allow_train_validation_map is True, "allow_train_validation_map must be true")
    _require(gate.allow_pretrained_initialization is True, "allow_pretrained_initialization must be true")
    _require(gate.allow_checkpoint_write is True, "allow_checkpoint_write must be true")
    _require(gate.allow_slurm is True and gate.allow_gpu is True, "Slurm GPU training must be explicitly allowed")
    _require(gate.allow_tools_test is False, "allow_tools_test must be false")
    _require(gate.allow_detector_map is False, "allow_detector_map must be false")
    _require(gate.allow_checkpoint_load is False, "allow_checkpoint_load must be false")
    _require(gate.allow_resume is False, "allow_resume must be false")
    _require(gate.allow_raw_prediction_cache is False, "allow_raw_prediction_cache must be false")
    _require(gate.offline_ledger is False, "offline_ledger must be false")
    _require(gate.post_projection_bridge is False, "post_projection_bridge must be false")
    _require(_as_tuple(gate.allowed_entrypoints) == ("tools/train.py",), "allowed_entrypoints must be only tools/train.py")
    _require(cfg.stage_id == ALLOWED_STAGE, f"stage_id must be {ALLOWED_STAGE}")
    _require(scope.stage == ALLOWED_STAGE, f"experiment_scope.stage must be {ALLOWED_STAGE}")
    _require(gate.stage == ALLOWED_STAGE, f"gate.stage must be {ALLOWED_STAGE}")

    _require(selector.type == "PCOTMRASPreBackboneFrameSelector", "selector type must be prebackbone frame selector")
    _require(selector.reader.type == ALLOWED_READER, f"reader type must be {ALLOWED_READER}")
    _require(selector.selection_strategy == ALLOWED_STRATEGY, "selector strategy must be frame_score_global_rank_st")
    _require(selector.frame_score_st_surrogate == ALLOWED_SURROGATE, "frame score ST surrogate must be global_rank_topk")
    _require(int(selector.target_len) == 384, "target_len must be 384")
    _require(int(selector.dense_window_size) == 768, "dense_window_size must be 768")
    _require(int(selector.global_rank_st_topk) == 384, "global_rank_st_topk must be 384")
    _require(int(selector.max_dense_gap) == 0, "max_dense_gap must be 0")
    _require(int(selector.max_gap_guard_count) == 0, "max_gap_guard_count must be 0")
    _require(float(selector.aux_frame_score_boundary_loss_weight) > 0.0, "boundary loss weight must be > 0")
    _require(float(selector.aux_gt_acquisition_loss_weight) >= 0.0, "action frame-score loss weight must be finite/nonnegative")

    _require(cfg.model.rpn_head.type == "ActionFormerHead", "rpn_head must stay ActionFormerHead")
    _require("PCOTMRASDetectorBridge" not in repr(cfg.model), "detector bridge must not be introduced")
    _require(cfg.post_processing.save_dict is True, "post_processing.save_dict must be true")
    _require(cfg.inference.load_from_raw_predictions is False, "raw prediction loading must be false")
    _require(cfg.inference.save_raw_prediction is False, "raw prediction saving must be false")

    forbidden_tokens = (
        "bata_value_transport_ledger_subsample",
        "hard_positions",
        "teacher",
        "oracle",
        "raw_prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "divergent_innovation",
        "bh_sdc",
    )
    for split in ("train", "val", "test"):
        pipeline_text = _repr_lower(cfg.dataset[split].pipeline)
        for token in forbidden_tokens:
            _require(token not in pipeline_text, f"{split} pipeline contains forbidden token {token}")

    full_text = _repr_lower(cfg)
    for token in ("divergent_innovation", "bh_sdc", "event-surprise", "boundary microscope", "frame/token hybrid"):
        _require(token not in full_text, f"config contains forbidden route token {token}")

    _require(_get(cfg, "load_from") in (None, "", "none"), "checkpoint load_from must stay disabled")
    _require(_get(cfg, "resume") in (None, False, "", "none"), "checkpoint resume must stay disabled")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the C3 GlobalRank-ST Boundary full-train route gate."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        validate_config(args.config)
    except Exception as exc:
        print(f"C3_GLOBAL_RANK_ST_FULL_TRAIN_GATE_VALIDATION_FAIL: {exc}", file=sys.stderr)
        return 1
    print(PASS_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
