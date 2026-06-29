from pathlib import Path
import math

from mmengine.config import Config
import pytest

from opentad.models.builder import build_selector


ROOT = Path(__file__).resolve().parents[1]
SMOKE32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_short_smoke.py"
FULL32 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_full_train.py"
SMOKE64 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_64px_short_smoke.py"
FULL64 = ROOT / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_64px_full_train.py"


def _assert_cadf_config(cfg, scout_size):
    cfg_text = cfg.pretty_text.lower()
    selector_text = repr(cfg.model.frame_selector).lower()

    assert cfg.c3_route_label == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.c3_route_labels == ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
    assert cfg.c3_method == "C3-CADF-DensityMesh-ST-OriginalAdaTAD"
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.frame_selector.type == "PCOTMRASIndirectPreBackboneFrameSelector"
    assert cfg.model.frame_selector.strategy == "cadf_density_mesh_st"
    assert "quotas" not in cfg.model.frame_selector
    assert "quota" not in selector_text
    assert "category" not in selector_text
    assert cfg.model.frame_selector.density_repulsion_loss_weight == 0.0
    assert cfg.model.frame_selector.scout.type == "PCOTMRASCADFDensityFrameScout"
    assert cfg.model.frame_selector.scout_spatial_size == scout_size
    assert cfg.model.frame_selector.scout.in_channels == 3 * scout_size * scout_size
    assert cfg.model.frame_selector.max_gap_guard_count == 0
    assert cfg.model.frame_selector.boundary_loss_weight == 0.0

    assert cfg.c3_dense_window_size == 768
    assert cfg.window_size == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    for forbidden in [
        "physicalgrid",
        "physical_grid",
        "rf50",
        "bh_sdc",
        "p2head",
        "teacher",
        "raw_prediction_cache",
        "start_head",
        "end_head",
        "coarse_actionness_uncertainty",
    ]:
        assert forbidden not in cfg_text


def test_cadf_densitymesh_32px_smoke_config_is_original_adatad_and_fail_closed():
    cfg = Config.fromfile(SMOKE32)

    _assert_cadf_config(cfg, 32)
    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 2
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.solver.train.batch_size == 1


def test_cadf_densitymesh_32px_full_config_unlocks_only_training_schedule():
    cfg = Config.fromfile(FULL32)

    _assert_cadf_config(cfg, 32)
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.disable_checkpoint is False
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.solver.train.batch_size == 2
    assert cfg.solver.ema is True


def test_cadf_densitymesh_64px_configs_only_change_scout_resolution():
    smoke = Config.fromfile(SMOKE64)
    full = Config.fromfile(FULL64)

    _assert_cadf_config(smoke, 64)
    _assert_cadf_config(full, 64)
    assert smoke.workflow.max_train_iters == 2
    assert full.workflow.end_epoch == 60


def test_cadf_densitymesh_rejects_invalid_density_weight_configs():
    base = dict(
        type="PCOTMRASIndirectPreBackboneFrameSelector",
        target_len=4,
        dense_window_size=8,
        selection_unit=1,
        scout_spatial_size=4,
        strategy="cadf_density_mesh_st",
        scout=dict(
            type="PCOTMRASCADFDensityFrameScout",
            in_channels=48,
            hidden_channels=8,
            num_layers=1,
            with_boundary_head=False,
        ),
    )

    with pytest.raises(ValueError, match="density_weights"):
        build_selector(dict(base, density_weights=dict(action=-0.1, utility=1.0)))
    with pytest.raises(ValueError, match="density_weights"):
        build_selector(dict(base, density_weights=dict(action=0.0, utility=0.0, boundary=0.0)))


def test_cadf_densitymesh_boundary_density_weight_requires_boundary_scout_head():
    with pytest.raises(ValueError, match="boundary"):
        build_selector(
            dict(
                type="PCOTMRASIndirectPreBackboneFrameSelector",
                target_len=4,
                dense_window_size=8,
                selection_unit=1,
                scout_spatial_size=4,
                strategy="cadf_density_mesh_st",
                density_weights=dict(action=0.5, utility=0.4, boundary=0.1),
                scout=dict(
                    type="PCOTMRASCADFDensityFrameScout",
                    in_channels=48,
                    hidden_channels=8,
                    num_layers=1,
                    with_boundary_head=False,
                ),
            )
        )


def test_cadf_densitymesh_rejects_unknown_nonfinite_and_nonnumeric_density_weight_keys():
    base = dict(
        type="PCOTMRASIndirectPreBackboneFrameSelector",
        target_len=4,
        dense_window_size=8,
        selection_unit=1,
        scout_spatial_size=4,
        strategy="cadf_density_mesh_st",
        scout=dict(
            type="PCOTMRASCADFDensityFrameScout",
            in_channels=48,
            hidden_channels=8,
            num_layers=1,
            with_boundary_head=False,
        ),
    )

    invalid_weights = [
        dict(action=0.5, utility=0.4, saliency=0.1),
        dict(action=math.nan, utility=1.0),
        dict(action=math.inf, utility=1.0),
        dict(action="0.5", utility=0.5),
    ]
    for density_weights in invalid_weights:
        with pytest.raises(ValueError, match="density_weights"):
            build_selector(dict(base, density_weights=density_weights))


def test_cadf_densitymesh_allows_boundary_density_weight_with_boundary_scout_head():
    selector = build_selector(
        dict(
            type="PCOTMRASIndirectPreBackboneFrameSelector",
            target_len=4,
            dense_window_size=8,
            selection_unit=1,
            scout_spatial_size=4,
            strategy="cadf_density_mesh_st",
            density_weights=dict(action=0.5, utility=0.4, boundary=0.1),
            scout=dict(
                type="PCOTMRASCADFDensityFrameScout",
                in_channels=48,
                hidden_channels=8,
                num_layers=1,
                with_boundary_head=True,
            ),
        )
    )

    assert selector.scout.boundary_head is not None
