import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD_PATH = REPO_ROOT / "opentad/models/dense_heads/native_irregular_area_head_p2.py"
CONFIG_PATH = (
    REPO_ROOT
    / "configs/adatad/thumos/ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_local.py"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method_source(source: str, cls: ast.ClassDef, name: str) -> str:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"method {name} not found")


def test_quality_calibration_is_default_off_and_keeps_disabled_score_path():
    source = _source(HEAD_PATH)
    tree = ast.parse(source)
    cls = _class_node(tree, "NativeIrregularAreaHeadP2")

    init_src = _method_source(source, cls, "__init__")
    assert 'quality_cfg.get("enable", False)' in init_src
    assert "self.enable_quality_calibration = bool" in init_src
    assert "self.quality_calibrator = nn.Sequential" in init_src

    score_src = _method_source(source, cls, "_score_pair_candidates")
    guard_idx = score_src.index("if not self.enable_quality_calibration:")
    return_idx = score_src.index("return base_score", guard_idx)
    calibrator_idx = score_src.index("self._quality_calibration_logits")
    assert guard_idx < return_idx < calibrator_idx

    forward_test_src = _method_source(source, cls, "forward_test")
    assert "if self.enable_quality_calibration:" in forward_test_src
    assert "self._reject_quality_eval_gt_kwargs(kwargs)" in forward_test_src


def test_quality_targets_are_train_loss_only_and_class_aware():
    source = _source(HEAD_PATH)
    tree = ast.parse(source)
    cls = _class_node(tree, "NativeIrregularAreaHeadP2")

    losses_src = _method_source(source, cls, "losses")
    assert "if self.enable_quality_calibration:" in losses_src
    assert "self._quality_calibration_losses(preds, area_grids, gt_segments, gt_labels, normalizer)" in losses_src

    quality_loss_src = _method_source(source, cls, "_quality_calibration_losses")
    assert "with torch.no_grad():" in quality_loss_src
    assert "self._pair_iou_quality_target(candidates, gt_segment, gt_label)" in quality_loss_src
    assert "self._pair_boundary_quality_target(" in quality_loss_src
    assert "F.binary_cross_entropy_with_logits" in quality_loss_src
    assert "self._quality_rank_loss(quality_logit, quality_target)" in quality_loss_src

    sampler_src = _method_source(source, cls, "_sample_quality_training_rows")
    assert 'hand_score = candidates["hand_score"].detach().to(dtype=quality_target.dtype)' in sampler_src

    boundary_src = _method_source(source, cls, "_pair_boundary_quality_target")
    assert "same_cls = cls_idx == label" in boundary_src
    assert 'candidates["pair_start"][same_cls]' in boundary_src
    assert 'candidates["pair_end"][same_cls]' in boundary_src
    assert "quality = quality.to(dtype=target.dtype)" in boundary_src
    assert "target[same_cls] = torch.maximum(target[same_cls], quality)" in boundary_src

    iou_src = _method_source(source, cls, "_pair_iou_quality_target")
    assert "iou = iou.to(dtype=target.dtype)" in iou_src

    pair_scorer_src = _method_source(source, cls, "_pair_scorer_loss")
    assert 'hand_score = candidates["hand_score"].detach().to(dtype=target.dtype)' in pair_scorer_src

    logits_src = _method_source(source, cls, "_quality_calibration_logits")
    assert 'candidates["pair_features"]' in logits_src
    assert "return logits[:, 0], logits[:, 1]" in logits_src


def test_quality_eval_gt_guard_rejects_oracle_like_kwargs():
    source = _source(HEAD_PATH)
    tree = ast.parse(source)
    cls = _class_node(tree, "NativeIrregularAreaHeadP2")

    guard_src = _method_source(source, cls, "_reject_quality_eval_gt_kwargs")
    for key in (
        "gt_segments",
        "gt_labels",
        "gt_bboxes",
        "gt_masks",
        "targets",
        "quality_targets",
        "oracle_targets",
    ):
        assert f'"{key}"' in guard_src
    assert "kwargs.get(key) is not None" in guard_src
    assert "raise ValueError" in guard_src


def test_local_quality_rank_config_is_explicitly_non_launchable():
    config = _source(CONFIG_PATH)
    ast.parse(config)

    assert '_base_ = ["ctf_bdi_pc_ot_mras_r18_aux_diag_candidate.py"]' in config
    assert "r18_pc_ot_mras_aux_diag_gate = None" in config
    assert "p2_quality_rank_calibrator_v0_gate = dict" in config
    assert "default_off=True" in config
    assert "explicit_config_opt_in=True" in config
    assert "local_synthetic_gate_only=True" in config
    assert "pc_ot_mras_reader_aux_loss=None" in config
    for field in (
        "allow_detector_training",
        "allow_remote_sync",
        "allow_precheck_only",
        "allow_slurm",
        "allow_gpu",
        "allow_tools_train",
        "allow_tools_test",
        "allow_detector_map",
        "launch_gate_passed",
        "metric_claim_allowed",
        "paper_claim_allowed",
        "runtime_flops_claim_allowed",
        "deploy_claim_allowed",
    ):
        assert f"{field}=False" in config
    assert "quality_calibration=dict(" in config
    assert "enable=True" in config
