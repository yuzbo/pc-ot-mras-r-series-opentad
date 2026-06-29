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
