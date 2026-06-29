import argparse
from pathlib import Path

from mmengine.config import Config


FORBIDDEN_CONFIG_TOKENS = [
    "physical_grid",
    "temporal_grid",
    "rf50",
    "bh_sdc",
    "p2head",
    "teacher",
    "oracle",
    "raw_prediction_cache",
    "boundary_head",
    "start_head",
    "end_head",
]


def validate_config(config_path):
    cfg = Config.fromfile(config_path)
    cfg_text = cfg.pretty_text.lower()
    found = [token for token in FORBIDDEN_CONFIG_TOKENS if token in cfg_text]
    if found:
        raise AssertionError(f"Forbidden old-route tokens in config: {found}")

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.frame_selector.type == "PCOTMRASIndirectPreBackboneFrameSelector"
    assert cfg.model.frame_selector.scout.type == "PCOTMRASCoarseActionnessFrameScout"
    assert cfg.model.frame_selector.strategy == "coarse_actionness_uncertainty"
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

    print(f"PASS_C3_INDIRECT_CLEAN_CONFIG {Path(config_path).as_posix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="C3 indirect clean config path")
    args = parser.parse_args()
    validate_config(args.config)


if __name__ == "__main__":
    main()
