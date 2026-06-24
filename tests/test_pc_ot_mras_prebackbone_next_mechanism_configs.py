from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"

INTERVAL_CONFIG = CONFIG_DIR / "pc_ot_mras_prebackbone_c3_interval_packet_full_train_candidate_n16r4.py"
GLOBAL_RANK_CONFIG = CONFIG_DIR / "pc_ot_mras_prebackbone_c3_global_rank_st_full_train_candidate_n16r4.py"
DYNAMIC_CONFIG = (
    CONFIG_DIR / "pc_ot_mras_prebackbone_c3_pro_dynamic_marginal_budget_guard_candidate_n16r4.py"
)
PHYSICAL_GRID_CONFIG = (
    CONFIG_DIR / "pc_ot_mras_prebackbone_c3_physical_grid_actionformer_fixed384_candidate_n16r4.py"
)


def _load(path: Path):
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(path))


def _assert_no_deploy_leakage(scope):
    assert scope.uses_p2 is False
    assert scope.uses_teacher is False
    assert scope.uses_test_gt is False
    assert scope.uses_raw_prediction_cache is False
    assert scope.metric_claim_allowed is False
    assert scope.paper_claim_allowed is False
    assert scope.deploy_claim_allowed is False


def test_interval_packet_candidate_config_actually_enables_interval_selector_and_global_rank_st():
    cfg = _load(INTERVAL_CONFIG)

    selector = cfg.model.frame_selector
    assert selector.target_len == 384
    assert selector.reader.num_slots == 384
    assert selector.selection_strategy == "interval_boundary_packet"
    assert selector.frame_score_st_surrogate == "global_softmax"
    assert selector.interval_boundary_budget_ratio == 0.50
    assert selector.interval_candidate_topk == 24
    assert selector.max_dense_gap == 0
    assert selector.max_gap_guard_count == 0
    assert cfg.experiment_scope.budget_protocol == (
        "fixed384_over_dense768_interval_boundary_packet_global_rank_st"
    )
    assert cfg.experiment_scope.changes_input_sampling is True
    assert cfg.experiment_scope.changes_detector_head is False
    _assert_no_deploy_leakage(cfg.experiment_scope)


def test_global_rank_st_candidate_config_keeps_frame_score_hard_path_but_changes_surrogate():
    cfg = _load(GLOBAL_RANK_CONFIG)

    selector = cfg.model.frame_selector
    assert selector.target_len == 384
    assert selector.reader.num_slots == 384
    assert selector.selection_strategy == "frame_score_topk"
    assert selector.frame_score_st_surrogate == "global_softmax"
    assert selector.max_dense_gap == 0
    assert selector.max_gap_guard_count == 0
    assert cfg.experiment_scope.budget_protocol == (
        "fixed384_over_dense768_frame_score_first_global_rank_st"
    )
    _assert_no_deploy_leakage(cfg.experiment_scope)


def test_dynamic_marginal_budget_candidate_remains_isolated_from_interval_and_physical_grid():
    cfg = _load(DYNAMIC_CONFIG)

    selector = cfg.model.frame_selector
    assert selector.target_len == 448
    assert selector.reader.num_slots == 448
    assert selector.selection_strategy == "frame_score_topk"
    assert selector.dynamic_budget.enabled is True
    assert selector.dynamic_budget.min_budget == 320
    assert selector.dynamic_budget.target_budget == 384
    assert selector.dynamic_budget.average_budget == 384
    assert selector.dynamic_budget.max_budget == 448
    assert selector.max_dense_gap == 64
    assert selector.max_gap_guard_count == 12
    assert not hasattr(cfg.model.get("rpn_head", {}), "temporal_grid")
    assert cfg.experiment_scope.dynamic_budget_claim_allowed is False
    _assert_no_deploy_leakage(cfg.experiment_scope)


def test_physical_grid_candidate_is_detector_geometry_isolation_not_dynamic_budget_combo():
    cfg = _load(PHYSICAL_GRID_CONFIG)

    selector = cfg.model.frame_selector
    assert selector.target_len == 384
    assert selector.selection_strategy == "frame_score_topk"
    assert selector.remap_gt_to_selected_axis is False
    assert not hasattr(selector, "dynamic_budget") or selector.dynamic_budget is None
    temporal_grid = cfg.model.rpn_head.temporal_grid
    assert temporal_grid.enabled is True
    assert temporal_grid.temporal_grid_mode == "physical"
    assert temporal_grid.decode_axis == "dense"
    assert temporal_grid.required is True
    assert temporal_grid.strict is True
    assert cfg.experiment_scope.changes_detector_head is True
    assert cfg.experiment_scope.changes_loss_assignment is True
    assert cfg.experiment_scope.temporal_grid_mode == "physical"
    _assert_no_deploy_leakage(cfg.experiment_scope)
