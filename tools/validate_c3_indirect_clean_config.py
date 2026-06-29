import argparse
from pathlib import Path

from mmengine.config import Config


COMMON_FORBIDDEN_CONFIG_TOKENS = [
    "physicalgrid",
    "physical_grid",
    "temporal_grid",
    "rf50",
    "bh_sdc",
    "p2head",
    "teacher",
    "oracle",
    "raw_prediction_cache",
    "start_head",
    "end_head",
]

COARSE_ONLY_FORBIDDEN_CONFIG_TOKENS = ["boundary_head"]


def _assert_no_forbidden_tokens(cfg_text, forbidden_tokens):
    found = [token for token in forbidden_tokens if token in cfg_text]
    if found:
        raise AssertionError(f"Forbidden old-route tokens in config: {found}")


def _validate_common_contract(cfg):
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.frame_selector.type == "PCOTMRASIndirectPreBackboneFrameSelector"
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.scout_spatial_size in (32, 64)
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    train_load = next(step for step in cfg.dataset.train.pipeline if step["type"] == "LoadFrames")
    assert train_load["trunc_len"] == 768
    assert cfg.dataset.val.window_size == 768
    assert cfg.dataset.test.window_size == 768


def _validate_coarse_actionness_route(cfg, cfg_text):
    assert cfg.model.frame_selector.scout.type == "PCOTMRASCoarseActionnessFrameScout"
    assert cfg.model.frame_selector.strategy == "coarse_actionness_uncertainty"
    _assert_no_forbidden_tokens(cfg_text, COMMON_FORBIDDEN_CONFIG_TOKENS + COARSE_ONLY_FORBIDDEN_CONFIG_TOKENS)


def _validate_cadf_densitymesh_route(cfg, cfg_text):
    selector = cfg.model.frame_selector
    assert selector.scout.type == "PCOTMRASCADFDensityFrameScout"
    assert selector.strategy == "cadf_density_mesh_st"
    _assert_no_forbidden_tokens(cfg_text, COMMON_FORBIDDEN_CONFIG_TOKENS)

    density_weights = selector.get("density_weights", {})
    boundary_weight = float(density_weights.get("boundary", 0.0))
    with_boundary_head = bool(selector.scout.get("with_boundary_head", False))
    if boundary_weight > 0.0 and not with_boundary_head:
        raise AssertionError("CADF boundary density weight requires scout.with_boundary_head=True")


def validate_config(config_path):
    cfg = Config.fromfile(config_path)
    cfg_text = cfg.pretty_text.lower()
    _validate_common_contract(cfg)

    strategy = cfg.model.frame_selector.strategy
    scout_type = cfg.model.frame_selector.scout.type
    if strategy == "coarse_actionness_uncertainty" and scout_type == "PCOTMRASCoarseActionnessFrameScout":
        _validate_coarse_actionness_route(cfg, cfg_text)
        route = "coarse_actionness"
    elif strategy == "cadf_density_mesh_st" and scout_type == "PCOTMRASCADFDensityFrameScout":
        _validate_cadf_densitymesh_route(cfg, cfg_text)
        route = "cadf_density_mesh"
    else:
        raise AssertionError(f"Unsupported C3 selector route: strategy={strategy}, scout={scout_type}")

    print(f"PASS_C3_INDIRECT_CLEAN_CONFIG route={route} {Path(config_path).as_posix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="C3 indirect clean config path")
    args = parser.parse_args()
    validate_config(args.config)


if __name__ == "__main__":
    main()
