import pytest
from mmengine.config import ConfigDict


def test_quality_head_fixture_uses_attribute_style_loss_config():
    loss_cfg = _make_loss_cfg()

    assert hasattr(loss_cfg, "cls_loss")
    assert hasattr(loss_cfg, "reg_loss")
    assert loss_cfg.cls_loss["type"] == "FocalLoss"
    assert loss_cfg.reg_loss["type"] == "DIOULoss"


def _make_loss_cfg():
    return ConfigDict(
        cls_loss=dict(type="FocalLoss"),
        reg_loss=dict(type="DIOULoss"),
    )


def _torch_and_head():
    try:
        import torch
        from opentad.models.dense_heads.anchor_free_head import AnchorFreeHead
    except OSError as exc:
        pytest.skip(f"torch import failed in this Windows environment: {exc}")
    except ModuleNotFoundError as exc:
        if exc.name == "nms_1d_cpu":
            pytest.skip(f"local OpenTAD NMS extension is unavailable: {exc}")
        raise
    return torch, AnchorFreeHead


def _make_head(quality_head_cfg=None):
    _, AnchorFreeHead = _torch_and_head()
    return AnchorFreeHead(
        num_classes=2,
        in_channels=4,
        feat_channels=4,
        num_convs=1,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1],
            regression_range=[(0, 10000)],
        ),
        loss=_make_loss_cfg(),
        center_sample="none",
        use_regress_range=False,
        quality_head_cfg=quality_head_cfg,
    )


def test_default_off_quality_path_keeps_prediction_shapes_and_scores():
    torch, _ = _torch_and_head()
    head = _make_head()
    points = [torch.tensor([[0.0, 0.0, 10000.0, 1.0], [1.0, 0.0, 10000.0, 1.0]])]
    reg_pred = [torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])]
    cls_pred = [torch.tensor([[[0.0, 1.0], [2.0, -2.0]]])]
    masks = [torch.tensor([[True, False]])]

    proposals, scores = head.get_valid_proposals_scores(points, reg_pred, cls_pred, masks)

    assert head.quality_head_enabled is False
    assert proposals[0].shape == (1, 2)
    assert scores[0].shape == (1, 2)
    assert torch.allclose(scores[0], cls_pred[0].permute(0, 2, 1).sigmoid()[0, :1])
    assert head.quality_qc_v2_enabled is False


def test_max_iou_quality_loss_is_finite_and_uses_all_valid_proposals():
    torch, _ = _torch_and_head()
    head = _make_head(
        dict(
            enabled=True,
            target_mode="max_iou",
            loss_weight=0.03,
            score_alpha=0.10,
            weight_init=0.0,
            bias_init=4.59511985013459,
        )
    )
    quality_pred = [torch.tensor([[[0.0, 1.0, -1.0]]], requires_grad=True)]
    valid_mask = torch.tensor([[True, True, True]])
    pos_mask = torch.tensor([[False, True, False]])
    all_pred_segments = torch.tensor([[[0.0, 1.0], [1.0, 3.0], [4.0, 5.0]]])
    gt_segments = [torch.tensor([[1.0, 3.0]])]
    target = head._max_iou_quality_target(all_pred_segments, valid_mask, gt_segments)

    quality_loss = head._quality_loss(
        quality_pred,
        valid_mask,
        pos_mask,
        pred_segments=all_pred_segments[pos_mask],
        target_segments=torch.tensor([[1.0, 3.0]]),
        all_pred_segments=all_pred_segments,
        gt_segments=gt_segments,
    )

    assert target.tolist() == [[0.0, 1.0, 0.0]]
    assert torch.isfinite(quality_loss)
    assert quality_loss.item() > 0


def test_quality_score_fusion_uses_low_alpha_model_score_only():
    torch, _ = _torch_and_head()
    head = _make_head(
        dict(
            enabled=True,
            target_mode="max_iou",
            loss_weight=0.03,
            score_alpha=0.10,
            weight_init=0.0,
            bias_init=4.59511985013459,
        )
    )
    points = [torch.tensor([[0.0, 0.0, 10000.0, 1.0], [1.0, 0.0, 10000.0, 1.0]])]
    reg_pred = [torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])]
    cls_pred = [torch.zeros(1, 2, 2)]
    masks = [torch.tensor([[True, True]])]
    quality_pred = [torch.tensor([[[4.59511985013459, 0.0]]])]

    _, fused_scores = head.get_valid_proposals_scores(points, reg_pred, cls_pred, masks, quality_pred=quality_pred)

    expected_quality = torch.sigmoid(torch.tensor([[4.59511985013459], [0.0]])).pow(0.10)
    assert torch.allclose(fused_scores[0], 0.5 * expected_quality.expand(-1, 2), atol=1e-6)
    assert fused_scores[0][0, 0] > fused_scores[0][1, 0]
    assert fused_scores[0][1, 0] == pytest.approx(0.5 * (0.5**0.10))


def test_sparse_irregular_qc_v2_returns_optional_deploy_visible_diagnostics():
    torch, _ = _torch_and_head()
    head = _make_head(
        dict(
            enabled=True,
            mode="sparse_irregular_qc_v2",
            target_mode="sparse_physical_iou_visibility",
            diagnostic_dump=True,
            loss_weight=0.03,
            score_alpha=0.10,
            weight_init=0.0,
            bias_init=4.59511985013459,
        )
    )
    points = [torch.tensor([[0.0, 0.0, 10000.0, 1.0], [1.0, 0.0, 10000.0, 1.0]])]
    reg_pred = [torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])]
    cls_pred = [torch.zeros(1, 2, 2)]
    masks = [torch.tensor([[True, True]])]
    quality_pred = [torch.tensor([[[4.59511985013459, 0.0]]])]
    metas = [
        dict(
            irregular_selected_positions=[0.0, 2.0],
            irregular_selected_valid_len=4.0,
            irregular_native_axis=False,
        )
    ]

    proposals, scores, diagnostics = head.get_valid_proposals_scores(
        points,
        reg_pred,
        cls_pred,
        masks,
        quality_pred=quality_pred,
        metas=metas,
    )

    assert head.quality_qc_v2_enabled is True
    assert proposals[0].shape == (2, 2)
    assert scores[0].shape == (2, 2)
    assert diagnostics[0]["diagnostic_available"] is True
    assert diagnostics[0]["coverage_available"] is True
    assert diagnostics[0]["cls_scores"].shape == (2, 2)
    assert diagnostics[0]["quality_scores"].shape == (2,)
    assert diagnostics[0]["selected_segments"].shape == (2, 2)
    assert diagnostics[0]["physical_segments"].shape == (2, 2)
    assert torch.allclose(diagnostics[0]["selected_segments"], torch.tensor([[0.0, 0.0], [0.0, 2.0]]))
    assert diagnostics[0]["selected_lengths"].tolist() == [0.0, 2.0]
    assert diagnostics[0]["physical_lengths"][1].item() > diagnostics[0]["selected_lengths"][1].item()
    assert "visibility_support" in diagnostics[0]
    assert "endpoint_support" in diagnostics[0]
    assert "gap_mean" in diagnostics[0]


def test_sparse_irregular_qc_v2_test_path_rejects_gt_targets():
    torch, _ = _torch_and_head()
    head = _make_head(
        dict(
            enabled=True,
            mode="sparse_irregular_qc_v2",
            target_mode="sparse_physical_iou_visibility",
            diagnostic_dump=True,
            loss_weight=0.03,
            score_alpha=0.10,
        )
    )
    feat = [torch.zeros(1, 4, 2)]
    masks = [torch.tensor([[True, True]])]

    with pytest.raises(AssertionError, match="GT"):
        head.forward_test(feat, masks, gt_segments=[torch.tensor([[0.0, 1.0]])])


def test_sparse_irregular_qc_v2_quality_target_uses_physical_geometry_and_visibility():
    torch, _ = _torch_and_head()
    head = _make_head(
        dict(
            enabled=True,
            mode="sparse_irregular_qc_v2",
            target_mode="sparse_physical_iou_visibility",
            loss_weight=0.03,
            score_alpha=0.10,
        )
    )
    pred_segments = torch.tensor([[[0.0, 2.0], [1.0, 2.0]]])
    valid_mask = torch.tensor([[True, True]])
    gt_segments = [torch.tensor([[0.0, 2.0]])]
    metas = [
        dict(
            irregular_selected_positions=[0.0, 1.0],
            irregular_selected_valid_len=4.0,
            irregular_native_axis=False,
        )
    ]

    target = head._sparse_irregular_qc_v2_quality_target(
        pred_segments,
        valid_mask,
        gt_segments,
        metas,
        dtype=pred_segments.dtype,
    )

    assert target.shape == (1, 2)
    assert torch.all(target >= 0)
    assert torch.all(target <= 1)
    assert target[0, 0] == pytest.approx(2.0 / 3.0)
    assert target[0, 1] < target[0, 0]
