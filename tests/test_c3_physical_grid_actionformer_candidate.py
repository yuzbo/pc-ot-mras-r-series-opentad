import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGE = "c3_physical_grid_actionformer_runtime"


torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
TORCH_AVAILABLE = torch_probe.returncode == 0

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


def read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def load_mmengine_config_or_skip(rel_path):
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(ROOT / rel_path))


def test_torch_runtime_is_available_for_c3_precheck():
    if not TORCH_AVAILABLE:
        detail = torch_probe.stderr.strip() or torch_probe.stdout.strip() or f"exit {torch_probe.returncode}"
        pytest.fail(f"torch unavailable: C3 physical-grid precheck is an environment blocker, not a pass ({detail})")


class _Registry:
    def __init__(self):
        self._items = {}

    def register_module(self):
        def _decorator(cls):
            self._items[cls.__name__] = cls
            return cls

        return _decorator

    def build(self, cfg):
        cfg = dict(cfg)
        type_name = cfg.pop("type")
        return self._items[type_name](**cfg)


if TORCH_AVAILABLE:

    class _DummyLoss(nn.Module):
        def forward(self, inputs, targets, reduction="none", **kwargs):
            loss = inputs.sum() * 0.0
            if reduction in {"mean", "sum"}:
                return loss
            return inputs.new_zeros(inputs.shape[:-1])

    class _Scale(nn.Module):
        def __init__(self, init_value=1.0):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(float(init_value)))

        def forward(self, x):
            return x * self.scale

    class _ConvModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("physical-grid head tests use num_convs=0")


def _ensure_package(name, path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_head_runtime_or_skip():
    if not TORCH_AVAILABLE:
        pytest.skip(
            "torch unavailable in this process: "
            + (
                torch_probe.stderr.strip().splitlines()[-1]
                if torch_probe.stderr.strip()
                else f"exit {torch_probe.returncode}"
            )
        )

    _ensure_package(RUNTIME_PACKAGE, ROOT / "opentad")
    _ensure_package(f"{RUNTIME_PACKAGE}.models", ROOT / "opentad" / "models")
    _ensure_package(f"{RUNTIME_PACKAGE}.models.dense_heads", ROOT / "opentad" / "models" / "dense_heads")
    _ensure_package(
        f"{RUNTIME_PACKAGE}.models.dense_heads.prior_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator",
    )

    builder = types.ModuleType(f"{RUNTIME_PACKAGE}.models.builder")
    builder.HEADS = _Registry()
    builder.PRIOR_GENERATORS = _Registry()
    builder.LOSSES = _Registry()
    builder.build_prior_generator = lambda cfg: builder.PRIOR_GENERATORS.build(cfg)
    builder.build_loss = lambda cfg: _DummyLoss()
    sys.modules[f"{RUNTIME_PACKAGE}.models.builder"] = builder

    bricks = types.ModuleType(f"{RUNTIME_PACKAGE}.models.bricks")
    bricks.ConvModule = _ConvModule
    bricks.Scale = _Scale
    sys.modules[f"{RUNTIME_PACKAGE}.models.bricks"] = bricks

    _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.prior_generator.point_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator" / "point_generator.py",
    )
    _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.anchor_free_head",
        ROOT / "opentad" / "models" / "dense_heads" / "anchor_free_head.py",
    )
    actionformer_head = _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.actionformer_head",
        ROOT / "opentad" / "models" / "dense_heads" / "actionformer_head.py",
    )
    return actionformer_head.ActionFormerHead


def _make_head(**kwargs):
    ActionFormerHead = _install_head_runtime_or_skip()
    head = ActionFormerHead(
        num_classes=2,
        in_channels=2,
        feat_channels=2,
        num_convs=0,
        prior_generator=dict(type="PointGenerator", strides=[1], regression_range=[(0, 10000)]),
        loss=types.SimpleNamespace(cls_loss=dict(type="DummyLoss"), reg_loss=dict(type="DummyLoss")),
        **kwargs,
    )
    with torch.no_grad():
        head.cls_head.weight.zero_()
        head.cls_head.bias.zero_()
        head.reg_head.weight.zero_()
        head.reg_head.bias.zero_()
        head.scale[0].scale.fill_(1.0)
    return head


def _features_and_mask():
    return [torch.zeros(1, 2, 4)], [torch.tensor([[True, True, True, False]])]


def _irregular_meta(**extra):
    meta = {
        "video_name": "synthetic",
        "irregular_selected_positions": [0.0, 2.0, 5.0],
        "selected_dense_indices": [0, 2, 5],
        "selected_valid_len": 3,
        "irregular_selected_valid_len": 10.0,
        "irregular_native_axis": False,
        "remap_gt_to_selected_axis": False,
    }
    meta.update(extra)
    return meta


def _dense_axis_meta(**extra):
    return _irregular_meta(irregular_native_axis=True, remap_gt_to_selected_axis=False, **extra)


def test_physical_grid_opt_in_decodes_on_dense_positions_and_masks_padded_tail():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta()]

    proposals, scores = head.forward_test(feat_list, mask_list, metas=metas)

    expected_centers = torch.tensor([0.0, 2.0, 5.0])
    assert torch.allclose(proposals[0][:, 0], expected_centers)
    assert torch.allclose(proposals[0][:, 1], expected_centers)
    assert proposals[0].shape == (3, 2)
    assert scores[0].shape == (3, 2)
    assert metas[0]["irregular_native_axis"] is True
    assert metas[0]["physical_grid_actionformer"] is True


def test_physical_grid_default_keeps_selected_axis_with_same_metadata():
    head = _make_head()
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta()]

    proposals, _scores = head.forward_test(feat_list, mask_list, metas=metas)

    assert torch.allclose(proposals[0][:, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(proposals[0][:, 1], torch.tensor([0.0, 1.0, 2.0]))
    assert metas[0]["irregular_native_axis"] is False
    assert "physical_grid_actionformer" not in metas[0]


def test_physical_grid_missing_metadata_fails_closed():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()

    with pytest.raises(ValueError, match="physical-grid ActionFormer requires"):
        head.forward_test(feat_list, mask_list, metas=[{"video_name": "missing"}])


def test_physical_grid_rejects_selected_axis_gt_remap_metadata():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta(remap_gt_to_selected_axis=True)]

    with pytest.raises(ValueError, match="dense-axis GT"):
        head.forward_train(
            feat_list,
            mask_list,
            gt_segments=[torch.tensor([[1.5, 2.5]], dtype=torch.float32)],
            gt_labels=[torch.tensor([1], dtype=torch.long)],
            metas=metas,
        )


def test_physical_grid_training_rejects_selected_axis_gt_native_axis_false():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta(irregular_native_axis=False, remap_gt_to_selected_axis=False)]

    with pytest.raises(ValueError, match="dense-axis GT"):
        head.forward_train(
            feat_list,
            mask_list,
            gt_segments=[torch.tensor([[1.5, 2.5]], dtype=torch.float32)],
            gt_labels=[torch.tensor([1], dtype=torch.long)],
            metas=metas,
        )


def test_prepare_targets_batched_physical_points_use_per_sample_centers_without_assigner():
    head = _make_head(physical_grid_actionformer=dict(enabled=False))
    points = [
        torch.tensor(
            [
                [[0.0, 0.0, 100.0, 1.0], [2.0, 0.0, 100.0, 1.0], [5.0, 0.0, 100.0, 1.0]],
                [[10.0, 0.0, 100.0, 1.0], [12.0, 0.0, 100.0, 1.0], [15.0, 0.0, 100.0, 1.0]],
            ],
            dtype=torch.float32,
        )
    ]
    gt_cls, gt_reg = head.prepare_targets(
        points,
        gt_segments=[
            torch.tensor([[1.5, 2.5]], dtype=torch.float32),
            torch.tensor([[11.5, 12.5]], dtype=torch.float32),
        ],
        gt_labels=[torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)],
    )

    assert torch.equal(gt_cls[0].argmax(dim=1), torch.tensor([0, 0, 0]))
    assert torch.equal(gt_cls[0].sum(dim=1) > 0, torch.tensor([False, True, False]))
    assert torch.equal(gt_cls[1].argmax(dim=1), torch.tensor([0, 1, 0]))
    assert torch.equal(gt_cls[1].sum(dim=1) > 0, torch.tensor([False, True, False]))
    assert torch.allclose(gt_reg[0][1], torch.tensor([0.5, 0.5]))
    assert torch.allclose(gt_reg[1][1], torch.tensor([0.5, 0.5]))


def test_physical_grid_training_assignment_uses_physical_dense_centers():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_dense_axis_meta()]

    losses = head.forward_train(
        feat_list,
        mask_list,
        gt_segments=[torch.tensor([[1.5, 2.5]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        metas=metas,
    )
    debug = head.collect_debug_state()

    assert set(losses) == {"cls_loss", "reg_loss"}
    assert debug["physical_grid_actionformer_enabled"] is True
    assert debug["physical_grid_actionformer_valid_points"] == 3
    assert debug["physical_grid_actionformer_center_min"] == 0.0
    assert debug["physical_grid_actionformer_center_max"] == 5.0
    assert debug["physical_grid_actionformer_axis_delta_max"] == 3.0


def test_c3_candidate_config_and_launcher_are_fail_closed():
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_c3_physical_grid_actionformer_precheck.py"
    )
    script = read("scripts/run_c3_physical_grid_actionformer_precheck.sh")
    tool = read("tools/bata/validate_c3_physical_grid_actionformer_precheck.py")

    assert cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled is True
    assert cfg.model.rpn_head.physical_grid_actionformer.required is True
    assert cfg.model.rpn_head.physical_grid_actionformer.strict is True
    for split in ("train", "val", "test"):
        load_steps = [step for step in cfg.dataset[split].pipeline if step.get("type") == "LoadFrames"]
        assert len(load_steps) == 1
        assert load_steps[0].remap_gt_to_selected_axis is False
    assert cfg.protocol_flags.precheck_only is True
    assert cfg.protocol_flags.uses_p2_head is False
    assert cfg.protocol_flags.uses_raw_prediction_cache is False
    assert cfg.protocol_flags.uses_teacher is False
    assert cfg.protocol_flags.uses_test_gt is False
    assert cfg.protocol_flags.remote_sync_allowed is False
    assert cfg.protocol_flags.slurm_allowed is False

    assert "PRECHECK_ONLY" in script
    assert "tools/train.py" not in script
    assert "tools/test.py" not in script
    assert "pytest tests/test_c3_physical_grid_actionformer_candidate.py -q" in script
    assert "validate_c3_physical_grid_actionformer_precheck.py" in script
    assert "FORBIDDEN_CONFIG_TOKENS" in tool
    assert "for token in FORBIDDEN_CONFIG_TOKENS" in tool
    assert "P2" in tool
    assert "raw_prediction" in tool
    assert "teacher" in tool
    assert "test_gt" in tool


def test_c3_physical_grid_static_source_contracts_when_torch_is_unavailable():
    head = read("opentad/models/dense_heads/anchor_free_head.py")
    detector = read("opentad/models/detectors/actionformer.py")
    formatting = read("opentad/datasets/transforms/formatting.py")

    assert "physical_grid_actionformer=None" in head
    assert "self.physical_grid_enabled" in head
    assert "irregular_selected_positions" in head
    assert "selected_dense_indices" in head
    assert "selected_valid_len" in head
    assert 'meta["irregular_native_axis"] = True' in head
    assert 'meta["physical_grid_actionformer"] = True' in head
    assert "physical_masks[level_idx][batch_idx] = physical_masks[level_idx][batch_idx] & level_valid" in head
    assert "physical-grid ActionFormer requires dense-axis GT" in head
    assert "meta.get(\"irregular_native_axis\", None) is not True" in head
    assert "cb_dist_left = point[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])" in head
    assert "cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - point[:, 0, None]" in head
    assert "torch.cat(points, dim=1) if points[0].dim() == 3 else torch.cat(points, dim=0)" in head

    assert "metas=metas" in detector
    assert '"selected_dense_indices"' in formatting
    assert '"selected_valid_len"' in formatting
