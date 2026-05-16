from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def load_module(rel_path, name):
    module_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tara_has_zero_initialized_whole_branch_residual():
    source = read("opentad/models/backbones/time_aligned_rasterizer.py")

    assert "residual_init" in source
    assert "branch_scale" in source
    assert "mix_with_source" in source
    assert "source + self.branch_scale" in source


def test_adapter_uses_source_conv_when_tara_branch_is_zero():
    source = read("opentad/models/backbones/vit_adapter.py")

    assert "use_branch_residual" in source
    assert "source_attn = self._temporal_conv" in source
    assert "mix_with_source(source_attn, tara_attn)" in source


def test_fixed_adapter_configs_exist_and_are_zero_or_identity_gated():
    late_linear = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_late_linear_zero.py")
    tara = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_tara_content_residual.py")
    remap = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_virtual_baseline_native_axis.py")

    assert "time_embed_scale=0.0" in late_linear
    assert 'adapter_tara_cfg=dict(mode="content_adaptive", residual=True, residual_init=0.0)' in tara
    assert "remap_gt_to_selected_axis=False" in remap


def test_vit_checkpoint_uses_non_reentrant_ddp_safe_path():
    source = read("opentad/models/backbones/vit_adapter.py")

    assert "cp.checkpoint(_inner_forward, x, use_reentrant=False)" in source


def test_time_embed_gate_is_bounded_and_epoch_warmup_controlled():
    source = read("opentad/models/backbones/vit_adapter.py")

    assert "time_embed_scale_max" in source
    assert "time_embed_warmup_epochs" in source
    assert "def set_train_epoch" in source
    assert "def _time_embed_scale_value" in source
    assert "torch.tanh(scale)" in source
    assert "scale = scale * 0.0" in source
    assert "scale.detach().new_zeros" not in source


def test_actionformer_delegates_train_epoch_to_head_and_backbone():
    source = read("opentad/models/detectors/actionformer.py")

    assert "def set_train_epoch" in source
    assert "self.rpn_head.set_train_epoch(curr_epoch)" in source
    assert "self.backbone.set_train_epoch(curr_epoch)" in source


def test_actionformer_head_cls_residual_is_zero_gated():
    source = read("opentad/models/dense_heads/anchor_free_head.py")

    assert "cls_residual_cfg" in source
    assert "cls_residual_scale" in source
    assert 'cls_residual_cfg.get("init_scale", 0.0)' in source
    assert "cls_logits = cls_logits + self.cls_residual_scale" in source


def test_adapter_multiscale_branch_is_zero_gated():
    source = read("opentad/models/backbones/vit_adapter.py")
    config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_multiscale_safe.py")

    assert "adapter_multiscale_cfg" in source
    assert "multiscale_scale" in source
    assert "rng_state = torch.get_rng_state()" in source
    assert "torch.set_rng_state(rng_state)" in source
    assert "source_attn + scale * branch_attn" in source
    assert "init_scale=0.0" in config
    assert "scale_max=0.20" in config


def test_actionformer_head_reg_residual_is_zero_gated():
    source = read("opentad/models/dense_heads/anchor_free_head.py")
    detector = read("opentad/models/detectors/actionformer.py")
    config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_head_regres_safe.py")

    assert "reg_residual_cfg" in source
    assert "reg_residual_scale" in source
    assert 'reg_residual_cfg.get("init_scale", 0.0)' in source
    assert "reg_raw = reg_raw + self.reg_residual_scale" in source
    assert '"reg_residual_scale"' in detector
    assert "reg_residual_cfg=dict(" in config
    assert "init_scale=0.0" in config


def test_new_safe_performance_configs_exist():
    adapter = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_timegate_safe.py")
    head = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_head_clsres_safe.py")
    multiscale = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_multiscale_safe.py")
    regres = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_head_regres_safe.py")

    assert "time_embed_scale=0.0" in adapter
    assert "time_embed_scale_max=0.03" in adapter
    assert "time_embed_warmup_epochs=5" in adapter
    assert 'dict(name="time_embed", lr=2e-5, weight_decay=0.05)' in adapter

    assert "cls_residual_cfg=dict(" in head
    assert "init_scale=0.0" in head
    assert "input_pdrop=0.2" not in head

    assert "adapter_multiscale_cfg=dict(" in multiscale
    assert "init_scale=0.0" in multiscale
    assert "input_pdrop=0.2" not in multiscale

    assert "reg_residual_cfg=dict(" in regres
    assert "init_scale=0.0" in regres
    assert "input_pdrop=0.2" not in regres


def test_boundary_weighted_adapter_configs_train_only_no_eval_leakage():
    boundary = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_train_boundary_weighted.py")
    action_boundary = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_train_action_boundary_weighted.py")
    launch_script = read("scripts/run_adapter_boundary_train_pair.sh")

    for config in (boundary, action_boundary):
        assert '_base_ = ["./input_random_fixed_50pct_adapter.py"]' in config
        assert "input_pdrop=0.2" not in config
        assert "checkpoint_interval=10" in config
        assert "disable_checkpoint=False" in config
        assert 'dict(type="mmaction.RandomResizedCrop")' in config
        assert 'dict(type="mmaction.ImgAug", transforms="default")' in config
        assert "sampling_boundary_weight=4.0" in config
        assert "oracle_boundary_radius=2" in config

    assert 'method="weighted_random_boundary_subsample"' in boundary
    assert 'method="weighted_random_action_boundary_subsample"' in action_boundary
    assert "val=dict" not in boundary
    assert "test=dict" not in boundary
    assert "val=dict" not in action_boundary
    assert "test=dict" not in action_boundary
    assert "weighted_random_boundary_subsample" in launch_script
    assert "weighted_random_action_boundary_subsample" in launch_script
    assert 'val_load.method == "random_fixed_subsample"' in launch_script
    assert 'test_load.method == "random_fixed_subsample"' in launch_script


def test_uniform_stride2_adapter_configs_keep_adapter_actionformer_contract():
    uniform = read("configs/adatad/thumos/input_stride2_uniform_50pct_adapter.py")
    center25 = read("configs/adatad/thumos/input_stride2_uniform_50pct_adapter_center25.py")
    launch_script = read("scripts/run_adapter_uniform_pair.sh")

    assert '_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]' in uniform
    assert 'type="VisionTransformerAdapter"' not in uniform
    assert "sample_stride=2" in uniform
    assert 'method="random_trunc"' in uniform
    assert 'method="sliding_window"' in uniform
    assert "window_size = 384" in uniform
    assert "max_seq_len=window_size" in uniform
    assert "input_pdrop=0.2" not in uniform
    assert "disable_checkpoint=True" in uniform

    assert '_base_ = ["./input_stride2_uniform_50pct_adapter.py"]' in center25
    assert "center_sample_radius=2.5" in center25
    assert "assignment_debug=dict(enabled=True)" in center25
    assert "input_pdrop=0.2" not in center25

    assert "input_stride2_uniform_50pct_adapter.py" in launch_script
    assert "input_stride2_uniform_50pct_adapter_center25.py" in launch_script
    assert "VisionTransformerAdapter" in launch_script
    assert 'cfg.model.projection.type == "Conv1DTransformerProj"' in launch_script
    assert 'cfg.model.rpn_head.type == "ActionFormerHead"' in launch_script
    assert 'cfg.model.neck.type == "FPNIdentity"' in launch_script
    assert 'assert int(pre.t1) == 24' in launch_script
    assert "mmaction.RandomResizedCrop" in launch_script
    assert "mmaction.ColorJitter" in launch_script
    assert 'train_load.method == "random_trunc"' in launch_script
    assert 'val_load.method == "sliding_window"' in launch_script
    assert "train.sample_stride) == 2" in launch_script


def test_adapter_postmortem_next_configs_are_low_risk_controls():
    nms = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_virtual_baseline_nms_sigma05_minscore001.py")
    frozen_recipe = read("configs/adatad/thumos/input_stride2_uniform_50pct_adapter_frozen_recipe.py")
    launch_script = read("scripts/run_adapter_postmortem_next.sh")

    assert '_base_ = ["./input_random_fixed_50pct_adapter_virtual_baseline.py"]' in nms
    assert "load_from_raw_predictions=False" in nms
    assert "save_raw_prediction=True" in nms
    assert "pre_nms_thresh=0.001" in nms
    assert "pre_nms_topk=2000" in nms
    assert "sigma=0.5" in nms
    assert "min_score=0.001" in nms

    assert '_base_ = ["./input_stride2_uniform_50pct_adapter.py"]' in frozen_recipe
    assert 'dict(type="mmaction.CenterCrop", crop_size=160)' in frozen_recipe
    assert "mmaction.RandomResizedCrop" not in frozen_recipe
    assert "mmaction.ImgAug" not in frozen_recipe
    assert "input_pdrop=0.2" not in frozen_recipe
    assert "sigma=0.5" in frozen_recipe
    assert "min_score=0.001" in frozen_recipe

    assert "tools/test.py" in launch_script
    assert "tools/train.py" in launch_script
    assert "BASELINE_CKPT" in launch_script
    assert "input_random_fixed_50pct_adapter_virtual_baseline_nms_sigma05_minscore001.py" in launch_script
    assert "input_stride2_uniform_50pct_adapter_frozen_recipe.py" in launch_script
    assert 'task == "nms_eval"' in launch_script
    assert 'task == "frozen_recipe_train"' in launch_script


def test_adapter_actionformer_regloss_config_is_narrow_random_fixed_control():
    config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_regloss15.py")
    launch_script = read("scripts/run_adapter_actionformer_regloss.sh")

    assert '_base_ = ["./input_random_fixed_50pct_adapter.py"]' in config
    assert "loss_weight=1.5" in config
    assert "checkpoint_interval=10" in config
    assert "disable_checkpoint=False" in config
    assert "input_pdrop=0.2" not in config

    assert "input_random_fixed_50pct_adapter_regloss15.py" in launch_script
    assert 'cfg.model.type == "ActionFormer"' in launch_script
    assert 'cfg.model.backbone.backbone.type == "VisionTransformerAdapter"' in launch_script
    assert 'cfg.model.rpn_head.type == "ActionFormerHead"' in launch_script
    assert "cfg.model.rpn_head.loss_weight" in launch_script
    assert 'train_load.method == "random_fixed_subsample"' in launch_script
    assert 'val_load.method == "random_fixed_subsample"' in launch_script
    assert 'test_load.method == "random_fixed_subsample"' in launch_script


def test_quality_rescore_supports_explicit_max_iou_targets_without_default_drift():
    source = read("opentad/models/dense_heads/anchor_free_head.py")
    config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py")
    posmax_config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py")

    assert 'self.quality_target_mode = self.quality_head_cfg.get("target_mode", "assigned_iou")' in source
    assert 'self.quality_positive_weight = float(self.quality_head_cfg.get("positive_weight", 1.0))' in source
    assert 'self.quality_negative_weight = float(self.quality_head_cfg.get("negative_weight", 1.0))' in source
    assert 'self.quality_loss_normalizer = self.quality_head_cfg.get("loss_normalizer", "valid")' in source
    assert '"positive_max_iou"' in source
    assert '"quality loss_normalizer=\'positive\' requires negative_weight=0.0"' in source
    assert 'elif self.quality_target_mode in ("max_iou", "positive_max_iou")' in source
    assert "dtype=quality_pred.dtype" in source
    assert 'if self.quality_target_mode == "positive_max_iou"' in source
    assert "quality_target[pos_mask] = max_iou_target[pos_mask]" in source
    assert "if pos_mask.any()" in source

    assert "target_mode" not in config
    assert "positive_weight" not in config
    assert "negative_weight" not in config
    assert "loss_normalizer" not in config

    assert '_base_ = ["./input_random_fixed_50pct_adapter_quality_rescore_detached.py"]' in posmax_config
    assert 'target_mode="positive_max_iou"' in posmax_config
    assert "negative_weight=0.0" in posmax_config
    assert 'loss_normalizer="positive"' in posmax_config
    assert "input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly" in posmax_config


def test_stratified_random_fixed_sampler_is_no_gt_and_train_eval_aligned():
    source = read("opentad/datasets/transforms/end_to_end.py")
    frame_config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_stratified.py")
    tubelet_config = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_stratified_tubelet2.py")
    launch_script = read("scripts/run_adapter_stratified_pair.sh")

    assert "stratified_random_fixed_subsample" in source
    assert "def _select_stratified_random_fixed_positions" in source
    assert "target_count + 1" in source
    assert "rng.choice(bucket_units" in source
    assert 'sample_profile = "random_fixed"' in source
    assert "sample_profile = f\"{self.method}|{self.selection_unit}|{self._selection_group_size()}\"" in source
    assert "_select_stratified_random_fixed_positions(valid_len, frame_num, sample_key)" in source

    for config in (frame_config, tubelet_config):
        assert '_base_ = ["./input_random_fixed_50pct_adapter.py"]' in config
        assert "weighted_random_boundary_subsample" not in config
        assert "oracle_boundary_subsample" not in config
        assert "input_pdrop=0.2" not in config
        assert "checkpoint_interval=10" in config
        assert "disable_checkpoint=False" in config
        assert config.count('method="stratified_random_fixed_subsample"') == 3
        assert 'method_base="random_trunc"' in config
        assert 'method_base="sliding_window"' in config

    assert 'selection_unit="tubelet"' not in frame_config
    assert 'selection_unit="tubelet"' in tubelet_config
    assert "selection_tubelet_size=2" in tubelet_config

    assert "input_random_fixed_50pct_adapter_stratified.py" in launch_script
    assert "input_random_fixed_50pct_adapter_stratified_tubelet2.py" in launch_script
    assert 'train_load.method == "stratified_random_fixed_subsample"' in launch_script
    assert 'val_load.method == "stratified_random_fixed_subsample"' in launch_script
    assert 'test_load.method == "stratified_random_fixed_subsample"' in launch_script
    assert "for split_name, split in" in launch_script


def test_pseudo_boundary_hybrid_selects_window_local_teacher_boundaries():
    pseudo = load_module("opentad/datasets/transforms/pseudo_boundary.py", "pseudo_boundary")

    full_scores = [0.0] * 400
    full_scores[100] = 0.99
    full_scores[300] = 0.95
    window_scores = pseudo.slice_global_scores_for_window(full_scores, range(64, 364))
    keep = pseudo.select_pseudo_boundary_hybrid_positions(
        valid_len=300,
        target_frame_num=128,
        sample_key="video_test|64|363",
        boundary_scores=window_scores,
        pseudo_quota=8,
        pseudo_radius=0,
        fallback="random_fixed",
    )

    assert keep.shape[0] == 128
    assert keep.tolist() == sorted(set(keep.tolist()))
    assert 36 in keep.tolist()
    assert 236 in keep.tolist()


def test_pseudo_boundary_hybrid_falls_back_to_random_fixed_without_cache():
    pseudo = load_module("opentad/datasets/transforms/pseudo_boundary.py", "pseudo_boundary")

    keep = pseudo.select_pseudo_boundary_hybrid_positions(
        valid_len=768,
        target_frame_num=384,
        sample_key="video_missing_cache|0|767",
        boundary_scores=None,
        pseudo_quota=64,
        fallback="random_fixed",
    )
    fallback = pseudo.select_random_fixed_positions(768, 384, "video_missing_cache|0|767")

    assert keep.tolist() == fallback.tolist()


def test_pseudo_boundary_snap_keeps_random_fixed_distribution_local():
    pseudo = load_module("opentad/datasets/transforms/pseudo_boundary.py", "pseudo_boundary")

    sample_key = "video_snap|0|767"
    base = pseudo.select_random_fixed_positions(768, 384, sample_key)
    base_set = set(base.tolist())
    candidate = None
    source = None
    for pos in base.tolist():
        for delta in (1, -1, 2, -2):
            snapped = pos + delta
            if 0 <= snapped < 768 and snapped not in base_set:
                source = pos
                candidate = snapped
                break
        if candidate is not None:
            break

    assert candidate is not None
    boundary_scores = [0.0] * 768
    boundary_scores[candidate] = 0.99

    keep = pseudo.select_pseudo_boundary_snap_positions(
        valid_len=768,
        target_frame_num=384,
        sample_key=sample_key,
        boundary_scores=boundary_scores,
        pseudo_quota=1,
        pseudo_snap_distance=2,
        pseudo_min_score=0.001,
        fallback="random_fixed",
    )

    assert keep.shape[0] == 384
    assert keep.tolist() == sorted(set(keep.tolist()))
    assert candidate in keep.tolist()
    assert source not in keep.tolist()
    assert len(set(keep.tolist()) ^ base_set) == 2
    assert max(abs(pos - min(base, key=lambda base_pos: abs(base_pos - pos))) for pos in keep) <= 2


def test_pseudo_boundary_snap_falls_back_to_random_fixed_without_cache():
    pseudo = load_module("opentad/datasets/transforms/pseudo_boundary.py", "pseudo_boundary")

    keep = pseudo.select_pseudo_boundary_snap_positions(
        valid_len=768,
        target_frame_num=384,
        sample_key="video_snap_missing_cache|0|767",
        boundary_scores=None,
        pseudo_quota=64,
        fallback="random_fixed",
    )
    fallback = pseudo.select_random_fixed_positions(768, 384, "video_snap_missing_cache|0|767")

    assert keep.tolist() == fallback.tolist()


def test_pseudo_boundary_configs_are_no_gt_teacher_guided_inputs():
    source = read("opentad/datasets/transforms/end_to_end.py")
    cache_source = read("opentad/datasets/transforms/pseudo_boundary.py")
    q64 = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_q64.py")
    q96 = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_q96.py")
    teacher_train = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_teacher_cache_train.py")
    teacher_val = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_teacher_cache_val.py")
    cache_builder = read("scripts/build_pseudo_boundary_cache.py")
    launch_script = read("scripts/run_adapter_pseudo_boundary_pair.sh")

    assert "pseudo_boundary_hybrid_subsample" in source
    assert "load_boundary_scores(" in source
    assert "select_pseudo_boundary_hybrid_positions(" in source
    assert '"oracle_boundary_subsample"' not in q64
    assert '"weighted_random_boundary_subsample"' not in q64
    assert "uses_gt" in cache_source
    assert "raise ValueError(f\"pseudo-boundary cache must not use GT" in cache_source

    for config in (q64, q96):
        assert "input_pdrop=0.2" not in config
        assert config.count('method="pseudo_boundary_hybrid_subsample"') == 3
        assert 'method_base="random_trunc"' in config
        assert 'method_base="sliding_window"' in config
        assert 'pseudo_boundary_fallback="random_fixed"' in config
        assert "pseudo_boundary_cache_dir=pseudo_boundary_cache_train" in config
        assert "pseudo_boundary_cache_dir=pseudo_boundary_cache_val" in config
        assert 'dict(type="ConvertToTensor", keys=["imgs"])' in config
        assert 'dict(type="Collect", inputs="imgs", keys=["masks"])' in config

    assert "pseudo_boundary_quota = 64" in q64
    assert "pseudo_boundary_quota = 96" in q96
    assert "save_raw_prediction=False" in teacher_train
    assert "save_raw_prediction=False" in teacher_val
    assert "save_dict=True" in teacher_train
    assert "save_dict=True" in teacher_val
    assert "test_mode=True" in teacher_train
    assert "test_mode=True" in teacher_val

    assert "uses_gt=False" in cache_builder
    assert "boundary_score" in cache_builder
    assert "--subset" in cache_builder
    assert "input_random_fixed_50pct_adapter_pseudo_boundary_q64.py" in launch_script
    assert "input_random_fixed_50pct_adapter_pseudo_boundary_q96.py" in launch_script
    assert "build_pseudo_boundary_cache.py" in launch_script
    assert "tools/test.py" in launch_script
    assert "--not_eval" in launch_script


def test_pseudo_boundary_snap_configs_are_local_random_fixed_preserving_inputs():
    source = read("opentad/datasets/transforms/end_to_end.py")
    q32 = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_snap_q32.py")
    q64 = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_snap_q64.py")
    launch_script = read("scripts/run_adapter_pseudo_boundary_snap_pair.sh")

    assert "pseudo_boundary_snap_subsample" in source
    assert "select_pseudo_boundary_snap_positions(" in source
    assert "pseudo_boundary_snap_distance" in source

    for config in (q32, q64):
        assert "input_pdrop=0.2" not in config
        assert config.count('method="pseudo_boundary_snap_subsample"') == 3
        assert 'method_base="random_trunc"' in config
        assert 'method_base="sliding_window"' in config
        assert 'pseudo_boundary_fallback="random_fixed"' in config
        assert "pseudo_boundary_cache_dir=pseudo_boundary_cache_train" in config
        assert "pseudo_boundary_cache_dir=pseudo_boundary_cache_val" in config
        assert "pseudo_boundary_snap_distance=2" in config
        assert 'dict(type="ConvertToTensor", keys=["imgs"])' in config
        assert 'dict(type="Collect", inputs="imgs", keys=["masks"])' in config

    assert "pseudo_boundary_quota = 32" in q32
    assert "pseudo_boundary_quota = 64" in q64
    assert "input_random_fixed_50pct_adapter_pseudo_boundary_snap_q32.py" in launch_script
    assert "input_random_fixed_50pct_adapter_pseudo_boundary_snap_q64.py" in launch_script
    assert 'load.method == "pseudo_boundary_snap_subsample"' in launch_script
