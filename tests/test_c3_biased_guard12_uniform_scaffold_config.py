from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_biased_guard12_uniform_scaffold_c3_physical_grid_actionformer_n16r4.py"
)
UNIFORM_BIASED_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer_n16r4.py"
)


def test_biased_guard12_config_is_smaller_bias_ablation_than_uniform_biased():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    baseline = mmengine_config.Config.fromfile(str(UNIFORM_BIASED_CONFIG))

    selector = cfg.model.frame_selector
    baseline_selector = baseline.model.frame_selector

    assert cfg.experiment_scope.selection_strategy == "uniform_scaffold_tiny_actionness_uncertainty_bias_guard12_maxgap3"
    assert cfg.experiment_scope.budget_protocol == (
        "fixed384_over_dense768_uniform320_action48_uncertainty16_guard12_maxgap3_no_change"
    )
    assert selector.coarse_uniform_count == 320
    assert selector.coarse_action_count == 48
    assert selector.coarse_uncertainty_count == 16
    assert selector.max_gap_guard_count == 12
    assert selector.max_dense_gap == 3
    assert selector.coarse_change_count == 0
    assert selector.coarse_background_count == 0
    assert selector.coarse_change_weight == 0.0

    assert selector.coarse_uniform_count > baseline_selector.coarse_uniform_count
    assert selector.coarse_action_count < baseline_selector.coarse_action_count
    assert selector.coarse_uncertainty_count < baseline_selector.coarse_uncertainty_count
    assert selector.max_gap_guard_count == baseline_selector.max_gap_guard_count
    assert selector.max_dense_gap == baseline_selector.max_dense_gap
