import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _import_torch_or_skip():
    if sys.platform.startswith("win"):
        pytest.skip("torch tensor stability test runs on the Linux training environment")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch import unavailable in this environment: {exc}")
    return torch


def test_bvr_headv3_config_resolves_reg_log_distance_clamp():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(
        str(ROOT / "configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py")
    )

    assert cfg.model.rpn_head.type == "IrregularActionFormerHeadV3"
    assert float(cfg.model.rpn_head.max_reg_log_distance) == pytest.approx(6.0)
    assert cfg.model.rpn_head.regression_head_fp32 is True
    assert cfg.model.rpn_head.regression_loss_fp32 is True
    assert cfg.model.rpn_head.filter_invalid_regression_samples is True
    assert float(cfg.model.rpn_head.min_regression_segment_length) == pytest.approx(1e-6)


def test_regression_decode_clamps_huge_log_distance_and_backward_is_finite():
    torch = _import_torch_or_skip()

    from opentad.models.dense_heads.irregular_actionformer_head_v2 import IrregularActionFormerHeadV2

    head = object.__new__(IrregularActionFormerHeadV2)
    head.reg_denom_floor = 0.5
    head.max_reg_log_distance = 6.0

    points = [
        torch.tensor(
            [
                [
                    [10.0, 0.0, 10000.0, 1.0, 1.0],
                    [20.0, 0.0, 10000.0, 2.0, 2.0],
                    [30.0, 0.0, 10000.0, 4.0, 4.0],
                ]
            ],
            dtype=torch.float32,
        )
    ]
    reg_pred = [torch.full((1, 2, 3), 1000.0, dtype=torch.float32, requires_grad=True)]

    proposals = head.get_refined_proposals(points, reg_pred)
    assert proposals.dtype == torch.float32
    assert torch.isfinite(proposals).all()

    loss = proposals.square().mean()
    loss.backward()
    assert torch.isfinite(reg_pred[0].grad).all()


def test_headv3_regression_branch_outputs_fp32_inside_autocast():
    torch = _import_torch_or_skip()

    from opentad.models.dense_heads.irregular_actionformer_head_v3 import IrregularActionFormerHeadV3

    class PassModule(torch.nn.Module):
        def forward(self, feat, mask):
            return feat, mask

    head = object.__new__(IrregularActionFormerHeadV3)
    torch.nn.Module.__init__(head)
    head.regression_head_fp32 = True
    head.use_boundary_aux = False
    head.cls_convs = torch.nn.ModuleList([PassModule()])
    head.reg_convs = torch.nn.ModuleList([PassModule()])
    head.cls_head = torch.nn.Conv1d(4, 2, kernel_size=1)
    head.reg_head = torch.nn.Conv1d(4, 2, kernel_size=1)
    head.scale = torch.nn.ModuleList([torch.nn.Identity()])
    head._apply_geometry_modulation = lambda feat, mask, temporal_grid: feat

    feat = torch.randn(1, 4, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    temporal_grid = {}

    with torch.autocast(device_type=feat.device.type, dtype=torch.bfloat16):
        _, reg_pred, _ = head._forward_single_level(feat, mask, 0, temporal_grid)

    assert reg_pred.dtype == torch.float32
    assert torch.isfinite(reg_pred).all()


def test_headv3_regression_loss_filters_bad_fp16_samples_and_keeps_other_losses():
    torch = _import_torch_or_skip()

    from opentad.models.dense_heads.irregular_actionformer_head_v3 import IrregularActionFormerHeadV3
    from opentad.models.losses.iou_loss import DIOULoss

    head = object.__new__(IrregularActionFormerHeadV3)
    torch.nn.Module.__init__(head)
    head.train()
    head.regression_head_fp32 = True
    head.regression_loss_fp32 = True
    head.filter_invalid_regression_samples = True
    head.min_regression_segment_length = 1e-6
    head.loss_normalizer_momentum = 0.0
    head.loss_normalizer = torch.tensor(10.0)
    head.label_smoothing = 0.0
    head.num_classes = 2
    head.loss_weight = 1.0
    head.use_boundary_aux = True
    head.boundary_loss_weight = 1.0
    head.debug_enabled = True
    head.reg_loss = DIOULoss()

    def simple_loss(pred, target, reduction="sum"):
        loss = (pred.float() - target.float()).square()
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        return loss

    head.cls_loss = simple_loss
    head.boundary_loss = simple_loss

    pred_segments_source = torch.tensor(
        [[[0.0, 2.0], [0.0, 0.0], [5.0, 4.0], [float("nan"), 6.0], [10.0, float("inf")]]],
        dtype=torch.float16,
        requires_grad=True,
    )
    gt_segments_source = torch.tensor(
        [[[0.0, 2.0], [0.0, 2.0], [4.0, 6.0], [0.0, 1.0], [9.0, 11.0]]],
        dtype=torch.float16,
    )

    reg_pred = [torch.zeros(1, 2, 5, dtype=torch.float16)]

    def fake_refined_proposals(points, proposal_inputs):
        if proposal_inputs is reg_pred:
            return pred_segments_source
        return gt_segments_source

    head.get_refined_proposals = fake_refined_proposals
    head.prepare_targets = lambda points, gt_segments, gt_labels: (
        [torch.ones(5, 2)],
        [torch.ones(5, 2)],
        [torch.ones(5)],
        {},
    )
    head.prepare_boundary_targets = lambda points, gt_segments: [torch.ones(5, 2)]

    cls_pred = [torch.randn(1, 2, 5, requires_grad=True)]
    boundary_pred = [torch.randn(1, 2, 5, requires_grad=True)]
    mask_list = [torch.ones(1, 5, dtype=torch.bool)]
    points = [torch.zeros(1, 5, 5)]
    gt_segments = [torch.tensor([[0.0, 2.0]])]
    gt_labels = [torch.tensor([0])]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = head.losses(cls_pred, reg_pred, boundary_pred, mask_list, points, gt_segments, gt_labels)

    assert set(losses) == {"cls_loss", "reg_loss", "boundary_loss"}
    for loss_value in losses.values():
        assert torch.isfinite(loss_value)

    total_loss = sum(losses.values())
    total_loss.backward()

    assert pred_segments_source.grad is not None
    assert torch.isfinite(pred_segments_source.grad).all()
    assert cls_pred[0].grad is not None
    assert torch.isfinite(cls_pred[0].grad).all()
    assert boundary_pred[0].grad is not None
    assert torch.isfinite(boundary_pred[0].grad).all()

    debug_state = head.collect_debug_state()
    assert debug_state["head_v3_regression_loss_fp32_enabled"] is True
    assert debug_state["head_v3_invalid_regression_filter_enabled"] is True
    assert debug_state["head_v3_regression_samples_total_before_filter"] == 5
    assert debug_state["head_v3_regression_samples_kept_after_filter"] == 1
    assert debug_state["head_v3_bad_regression_samples_filtered"] == 4
    assert debug_state["head_v3_bad_regression_samples_filter_ratio"] == pytest.approx(0.8)
