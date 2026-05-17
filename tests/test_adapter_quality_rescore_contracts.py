from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_detached_quality_rescore_head_preserves_baseline_when_alpha_zero():
    source = read("opentad/models/dense_heads/anchor_free_head.py")

    assert "quality_head_cfg" in source
    assert "quality_score_alpha" in source
    assert "quality_head_enabled" in source
    assert "quality_score_alpha <= 0" in source
    assert "return new_proposals, new_scores" in source
    assert "score * quality_score.pow(self.quality_score_alpha)" in source


def test_detached_quality_rescore_uses_detached_reg_features_and_all_valid_targets():
    source = read("opentad/models/dense_heads/anchor_free_head.py")

    assert "self.quality_head(reg_feat.detach())" in source
    assert "quality_loss_weight" in source
    assert "quality_target = torch.zeros_like(valid_mask" in source
    assert "quality_target[pos_mask] = self._segment_iou_1d(" in source
    assert "quality_pred = quality_pred.float()" in source
    assert "F.binary_cross_entropy_with_logits" in source
    assert "quality_logits = quality_pred[valid_mask]" in source
    assert "losses[\"quality_loss\"] = quality_loss * self.quality_loss_weight" in source


def test_detached_quality_rescore_excludes_quality_head_from_main_grad_clipping():
    detector = read("opentad/models/detectors/actionformer.py")
    train_engine = read("opentad/cores/train_engine.py")

    assert "def grad_clip_parameters" in detector
    assert 'name.startswith("rpn_head.quality_head.")' in detector
    assert "target.grad_clip_parameters()" in train_engine
    assert "clip_grad_norm_(grad_clip_parameters, clip_grad_l2norm)" in train_engine


def test_adapter_quality_rescore_config_and_launcher_keep_random_fixed_contract():
    adapter_base = read("configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py")
    config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py")
    launch_script = read("scripts/run_adapter_quality_rescore.sh")

    assert "train=dict(batch_size=2, num_workers=2)" in adapter_base
    assert "val=dict(batch_size=2, num_workers=2)" in adapter_base
    assert "test=dict(batch_size=2, num_workers=2)" in adapter_base

    assert '_base_ = ["./input_random_fixed_50pct_adapter.py"]' in config
    assert "quality_head_cfg=dict(" in config
    assert "enabled=True" in config
    assert "score_alpha=0.25" in config
    assert "loss_weight=0.10" in config
    assert "checkpoint_interval=10" in config
    assert "disable_checkpoint=False" in config
    assert "input_pdrop=0.2" not in config
    assert "oracle" not in config
    assert "pseudo_boundary" not in config
    assert "weighted_random" not in config

    assert "input_random_fixed_50pct_adapter_quality_rescore_detached.py" in launch_script
    assert 'cfg.model.type == "ActionFormer"' in launch_script
    assert 'cfg.model.backbone.backbone.type == "VisionTransformerAdapter"' in launch_script
    assert 'cfg.model.rpn_head.type == "ActionFormerHead"' in launch_script
    assert "cfg.model.rpn_head.quality_head_cfg.enabled" in launch_script
    assert 'train_load.method == "random_fixed_subsample"' in launch_script
    assert 'val_load.method == "random_fixed_subsample"' in launch_script
    assert 'test_load.method == "random_fixed_subsample"' in launch_script
    assert "EXPECT_BATCH_SIZE" in launch_script
    assert "solver.train.batch_size" in launch_script
    assert "input_pdrop" in launch_script


def test_quality_loss_zero_can_keep_static_ddp_graph_when_requested():
    source = read("opentad/models/dense_heads/anchor_free_head.py")

    assert "quality_keep_loss_graph_when_weight_zero" in source
    assert 'self.quality_head_cfg.get("keep_loss_graph_when_weight_zero", False)' in source
    assert "quality_loss_weight <= 0" in source
    assert "quality_zero_loss = sum(pred.float().sum() for pred in quality_pred) * 0" in source
    assert 'losses["quality_loss"] = quality_zero_loss' in source


def test_adapter_quality_next_diagnostic_configs_and_launcher_contracts():
    neutral = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0.py")
    neg025 = read(
        "configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0.py"
    )
    launch_script = read("scripts/run_adapter_quality_rescore.sh")

    assert '_base_ = ["./input_random_fixed_50pct_adapter_quality_rescore_detached.py"]' in neutral
    assert "keep_loss_graph_when_weight_zero=True" in neutral
    assert "loss_weight=0.0" in neutral
    assert "score_alpha=0.0" in neutral
    assert "input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0" in neutral
    assert "positive_max_iou" not in neutral
    assert "weighted_random" not in neutral
    assert "oracle" not in neutral

    assert '_base_ = ["./input_random_fixed_50pct_adapter_quality_rescore_detached.py"]' in neg025
    assert 'target_mode="assigned_iou"' in neg025
    assert "positive_weight=1.0" in neg025
    assert "negative_weight=0.25" in neg025
    assert 'loss_normalizer="weighted"' in neg025
    assert "loss_weight=0.10" in neg025
    assert "score_alpha=0.0" in neg025
    assert "input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0" in neg025
    assert "positive_max_iou" not in neg025
    assert "weighted_random" not in neg025
    assert "oracle" not in neg025

    assert "EXPECT_QUALITY_LOSS_WEIGHT" in launch_script
    assert "EXPECT_QUALITY_SCORE_ALPHA" in launch_script
    assert "EXPECT_QUALITY_TARGET_MODE" in launch_script
    assert "EXPECT_QUALITY_NEGATIVE_WEIGHT" in launch_script
    assert "EXPECT_QUALITY_LOSS_NORMALIZER" in launch_script
    assert "EXPECT_QUALITY_KEEP_ZERO_GRAPH" in launch_script
    assert 'quality.get("target_mode", "assigned_iou")' in launch_script
    assert 'quality.get("loss_normalizer", "valid")' in launch_script
    assert 'quality.get("keep_loss_graph_when_weight_zero", False)' in launch_script
