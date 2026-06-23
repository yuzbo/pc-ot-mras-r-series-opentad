import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD_PATH = REPO_ROOT / "opentad/models/dense_heads/native_irregular_area_head_p2.py"


def _install_stub(name, module):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    return previous


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


def _load_head_class():
    stubs = {}
    previous = {}

    for name in (
        "opentad",
        "opentad.models",
        "opentad.models.dense_heads",
        "opentad.models.utils",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        stubs[name] = module

    bricks = types.ModuleType("opentad.models.bricks")
    bricks.ConvModule = object
    bricks.Scale = object
    stubs["opentad.models.bricks"] = bricks

    builder = types.ModuleType("opentad.models.builder")
    builder.HEADS = _Registry()
    builder.build_prior_generator = lambda cfg: None
    stubs["opentad.models.builder"] = builder

    sampling_contract = types.ModuleType("opentad.models.utils.sampling_contract")
    sampling_contract.validate_sampling_contract = lambda *args, **kwargs: None
    stubs["opentad.models.utils.sampling_contract"] = sampling_contract

    temporal_grid = types.ModuleType("opentad.models.utils.temporal_grid")
    for name in (
        "build_area_time_grid",
        "downsample_temporal_grid",
        "prepare_area_targets",
        "segment_area_integral",
        "temporal_grid_from_metas",
        "validate_area_time_grid",
        "validate_temporal_grid_alignment",
    ):
        setattr(temporal_grid, name, lambda *args, **kwargs: None)
    stubs["opentad.models.utils.temporal_grid"] = temporal_grid

    module_name = "opentad.models.dense_heads.native_irregular_area_head_p2_runtime_smoke"
    try:
        for name, module in stubs.items():
            previous[name] = _install_stub(name, module)
        spec = importlib.util.spec_from_file_location(module_name, HEAD_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.NativeIrregularAreaHeadP2
    finally:
        sys.modules.pop(module_name, None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _bare_head(**attrs):
    cls = _load_head_class()
    head = cls.__new__(cls)
    nn.Module.__init__(head)
    for key, value in attrs.items():
        setattr(head, key, value)
    return head


def _candidates(num_rows=4, feat_dim=5):
    pair_features = torch.linspace(-0.5, 0.7, steps=num_rows * feat_dim, dtype=torch.float32).view(num_rows, feat_dim)
    return {
        "hand_score": torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)[:num_rows],
        "pair_start_score": torch.full((num_rows,), 0.5, dtype=torch.float32),
        "pair_end_score": torch.full((num_rows,), 0.6, dtype=torch.float32),
        "area_integral": torch.full((num_rows,), 0.7, dtype=torch.float32),
        "pair_features": pair_features.clone().requires_grad_(True),
        "pair_start": torch.tensor([0.0, 2.0, 5.0, 9.0], dtype=torch.float32)[:num_rows],
        "pair_end": torch.tensor([1.0, 4.0, 8.0, 12.0], dtype=torch.float32)[:num_rows],
        "class_id": torch.tensor([0, 0, 1, 1], dtype=torch.long)[:num_rows],
    }


def test_disabled_quality_calibration_preserves_base_score_runtime():
    head = _bare_head(
        score_fusion_mode="hybrid_sum",
        enable_pair_scorer=True,
        enable_quality_calibration=False,
        pair_scorer=nn.Linear(5, 1),
    )
    candidates = _candidates()
    with torch.no_grad():
        head.pair_scorer.weight.fill_(0.1)
        head.pair_scorer.bias.fill_(0.0)
    expected = (
        0.5 * candidates["hand_score"]
        + 0.5 * torch.sigmoid(head.pair_scorer(candidates["pair_features"]).squeeze(-1))
    ).clamp(0.0, 1.0)

    scored = head._score_pair_candidates(candidates)

    assert torch.allclose(scored, expected)


def test_enabled_quality_calibration_scores_are_finite_and_bounded():
    head = _bare_head(
        score_fusion_mode="hand_geometric",
        enable_pair_scorer=False,
        enable_quality_calibration=True,
        quality_score_eps=1e-6,
        quality_base_delta=1.0,
        quality_score_beta=1.0,
        quality_boundary_gamma=1.0,
        quality_calibrator=nn.Linear(5, 2),
    )
    candidates = _candidates()
    scored = head._score_pair_candidates(candidates)

    assert scored.shape == candidates["hand_score"].shape
    assert torch.isfinite(scored).all()
    assert torch.all((0.0 <= scored) & (scored <= 1.0))
    assert scored.sum().backward() is None
    assert candidates["pair_features"].grad is not None
    assert torch.isfinite(candidates["pair_features"].grad).all()


@pytest.mark.parametrize(
    ("gt_segments", "gt_labels"),
    [
        (torch.tensor([[0.0, 1.0], [5.0, 8.0]], dtype=torch.float32), torch.tensor([0, 1], dtype=torch.long)),
        (torch.empty((0, 2), dtype=torch.float32), torch.empty((0,), dtype=torch.long)),
    ],
)
def test_quality_calibration_losses_forward_backward_runtime(gt_segments, gt_labels):
    candidates = _candidates()
    head = _bare_head(
        num_classes=2,
        max_pairs_per_class=4,
        quality_calibration_loss_weight=0.5,
        quality_boundary_loss_weight=0.25,
        quality_rank_loss_weight=0.05,
        quality_boundary_tau=1.0,
        quality_rank_positive_iou=0.7,
        quality_rank_negative_iou=0.3,
        quality_rank_margin=0.25,
        quality_rank_sample_size=4,
        quality_calibrator=nn.Linear(5, 2),
    )
    head._collect_area_level_pairs = lambda preds, grid, level_idx, batch_idx: candidates
    preds = {"area_logits": [torch.zeros(1, 2, 3, dtype=torch.float32, requires_grad=True)]}

    losses = head._quality_calibration_losses(
        preds,
        area_grids=[{}],
        gt_segments=[gt_segments],
        gt_labels=[gt_labels],
        normalizer=torch.tensor(1.0),
    )

    assert set(losses) == {"quality_calibration_loss", "quality_boundary_loss", "quality_rank_loss"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert candidates["pair_features"].grad is not None
    assert torch.isfinite(candidates["pair_features"].grad).all()


def test_quality_calibration_losses_no_candidates_runtime():
    head = _bare_head(
        num_classes=2,
        max_pairs_per_class=4,
        quality_calibration_loss_weight=0.5,
        quality_boundary_loss_weight=0.25,
        quality_rank_loss_weight=0.05,
        quality_calibrator=nn.Linear(5, 2),
    )
    head._collect_area_level_pairs = lambda preds, grid, level_idx, batch_idx: {}
    preds = {"area_logits": [torch.zeros(1, 2, 3, dtype=torch.float32, requires_grad=True)]}

    losses = head._quality_calibration_losses(
        preds,
        area_grids=[{}],
        gt_segments=[torch.empty((0, 2), dtype=torch.float32)],
        gt_labels=[torch.empty((0,), dtype=torch.long)],
        normalizer=torch.tensor(1.0),
    )

    assert torch.isfinite(sum(losses.values()))
    assert all(value.item() == 0.0 for value in losses.values())


def test_quality_rank_loss_samples_low_positive_and_high_negative_logits_runtime():
    head = _bare_head(
        quality_rank_positive_iou=0.7,
        quality_rank_negative_iou=0.3,
        quality_rank_margin=0.25,
        quality_rank_sample_size=1,
    )
    quality_logit = torch.tensor([5.0, -2.0, -5.0, 2.0], dtype=torch.float32, requires_grad=True)
    quality_target = torch.tensor([0.9, 0.95, 0.1, 0.05], dtype=torch.float32)

    loss = head._quality_rank_loss(quality_logit, quality_target)

    assert loss.item() == pytest.approx(4.25)
    loss.backward()
    assert quality_logit.grad.tolist() == [0.0, -1.0, 0.0, 1.0]


def test_quality_eval_guard_runtime():
    cls = _load_head_class()
    cls._reject_quality_eval_gt_kwargs({})
    with pytest.raises(ValueError, match="gt_segments"):
        cls._reject_quality_eval_gt_kwargs({"gt_segments": torch.empty((0, 2))})
    with pytest.raises(ValueError, match="oracle_targets"):
        cls._reject_quality_eval_gt_kwargs({"oracle_targets": torch.zeros(1)})
