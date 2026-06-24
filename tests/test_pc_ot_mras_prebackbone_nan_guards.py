from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_prebackbone_selector_module():
    for name in (
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        "opentad.models.builder",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    builder.build_selector = lambda cfg: cfg
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        SELECTOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _import_torch_or_skip():
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else f"exit {probe.returncode}"
        pytest.skip(f"torch unavailable in this process: {detail}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local DLL state.
        pytest.skip(f"torch unavailable in this process: {exc}")
    return torch


def _selector_shell(module, *, target_len: int):
    selector_cls = module.PCOTMRASPreBackboneFrameSelector
    selector = selector_cls.__new__(selector_cls)
    selector.target_len = int(target_len)
    selector.residual_count = None
    selector.protected_uniform_count = 0
    selector.coverage_guard_count = 0
    selector.straight_through_detector_loss = True
    selector.straight_through_downstream = True
    selector.residual_slot_role = "learned_residual"
    selector.st_surrogate_mode = "mean_proxy"
    selector.aux_gt_acquisition_loss_weight = 1.0
    selector.aux_duplicate_cap_loss_weight = 0.0
    selector.aux_duplicate_column_cap = 1.5
    selector.aux_value_loss_weight = 0.0
    selector.aux_risk_loss_weight = 0.0
    selector.aux_uncertainty_loss_weight = 0.0
    selector.aux_redundancy_loss_weight = 0.0
    selector.aux_role_entropy_loss_weight = 0.0
    selector.reader_regularizer_loss_weight = 0.0
    return selector


def test_rseries_hybrid_reader_outputs_full_masked_contract_and_gradients():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    reader_cls = getattr(module, "PCOTMRASRSeriesHybridFrameScout", None)
    assert reader_cls is not None, "PCOTMRASRSeriesHybridFrameScout must exist for C3-RS-Hybrid-ST"

    batch, time, channels, slots = 2, 8, 6, 4
    torch.manual_seed(20260624)
    reader = reader_cls(
        in_dim=channels,
        hidden_dim=16,
        num_slots=slots,
        temporal_layers=2,
        temporal_kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
        slot_mlp_layers=2,
    )
    reader.train()
    features = torch.randn(batch, time, channels, requires_grad=True)
    valid = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    time_coords = torch.linspace(0.0, 1.0, steps=time).unsqueeze(0).expand(batch, -1).clone()

    outputs = reader(features, valid, time_coords=time_coords)

    required = {
        "slot_logits",
        "acquisition_matrix",
        "value_logits",
        "risk_logits",
        "uncertainty_logits",
        "redundancy_logits",
        "role_logits",
        "regularizers",
    }
    assert required.issubset(outputs)
    assert outputs["slot_logits"].shape == (batch, slots, time)
    assert outputs["acquisition_matrix"].shape == (batch, slots, time)
    for key in ("value_logits", "risk_logits", "uncertainty_logits", "redundancy_logits"):
        assert outputs[key].shape == (batch, time), key
    assert outputs["role_logits"].shape[:2] == (batch, time)

    valid_slot_mask = valid[:, None, :].expand_as(outputs["slot_logits"])
    assert torch.isfinite(outputs["slot_logits"][valid_slot_mask]).all()
    assert torch.isfinite(outputs["acquisition_matrix"]).all()
    assert torch.all(outputs["acquisition_matrix"][~valid_slot_mask] == 0.0)
    assert torch.all(outputs["slot_logits"][~valid_slot_mask] < -1.0e20)
    for key in ("value_logits", "risk_logits", "uncertainty_logits", "redundancy_logits"):
        assert torch.isfinite(outputs[key][valid]).all(), key
        assert torch.all(outputs[key][~valid] == 0.0), key
    assert torch.all(outputs["role_logits"][~valid] == 0.0)
    row_sums = outputs["acquisition_matrix"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-5)

    regularizers = outputs["regularizers"]
    assert {"order_regularizer", "width_regularizer", "total_regularizer"}.issubset(regularizers)
    assert all(torch.isfinite(value).all() for value in regularizers.values())

    loss = outputs["acquisition_matrix"][valid_slot_mask].sum()
    loss = loss + outputs["slot_logits"][valid_slot_mask].mean() * 0.01
    loss = loss + outputs["value_logits"][valid].mean() * 0.01
    loss = loss + outputs["risk_logits"][valid].mean() * 0.01
    loss = loss + outputs["uncertainty_logits"][valid].mean() * 0.01
    loss = loss + outputs["redundancy_logits"][valid].mean() * 0.01
    loss = loss + outputs["role_logits"][valid].mean() * 0.01
    loss = loss + regularizers["total_regularizer"] * 0.01
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0.0
    trainable_grads = [param.grad for param in reader.parameters() if param.requires_grad]
    assert trainable_grads
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0.0 for grad in trainable_grads)


@pytest.mark.parametrize("dtype,scale", [("float16", 512.0), ("float32", 1.0e20)])
def test_masked_slot_transport_is_float32_finite_for_half_and_big_logits(dtype, scale):
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    logits_dtype = getattr(torch, dtype)
    logits = torch.tensor(
        [
            [
                [scale, scale - 12.0, -scale, -scale],
                [-scale, scale - 7.0, scale, -scale],
            ]
        ],
        dtype=logits_dtype,
    )
    valid = torch.tensor([[True, True, True, False]], dtype=torch.bool)

    masked_logits, acquisition_matrix = module._masked_slot_transport(logits, valid)

    assert masked_logits.dtype == torch.float32
    assert acquisition_matrix.dtype == torch.float32
    assert torch.isfinite(masked_logits[valid[:, None, :].expand_as(masked_logits)]).all()
    assert torch.isfinite(acquisition_matrix).all()
    assert torch.all(acquisition_matrix[..., 3] == 0.0)
    assert torch.allclose(acquisition_matrix.sum(dim=-1), torch.ones((1, 2)), atol=1.0e-6)


def test_sparse_transport_plan_fail_fast_on_nonfinite_logits_and_matrix():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector = _selector_shell(module, target_len=2)
    valid = torch.tensor([[True, True, True]], dtype=torch.bool)
    candidate_valid = valid.clone()
    candidate_dense_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)

    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 1] = float("nan")
    with pytest.raises(ValueError, match="slot_logits must be finite"):
        selector._sparse_transport_plan(
            {"slot_logits": logits},
            valid,
            candidate_valid=candidate_valid,
            candidate_dense_indices=candidate_dense_indices,
            training=True,
        )

    matrix = torch.full((1, 2, 3), 1.0 / 3.0)
    matrix[0, 1, 2] = float("nan")
    with pytest.raises(ValueError, match="acquisition matrix must be finite"):
        selector._sparse_transport_plan(
            {"acquisition_matrix": matrix},
            valid,
            candidate_valid=candidate_valid,
            candidate_dense_indices=candidate_dense_indices,
            training=True,
        )


def test_sparse_transport_plan_uses_real_frame_indices_and_sorts_selected_positions():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector = _selector_shell(module, target_len=3)
    valid = torch.tensor([[True, True, True, True, True, True, True]], dtype=torch.bool)
    candidate_valid = torch.tensor([[True, True, True, True, False]], dtype=torch.bool)
    candidate_dense_indices = torch.tensor([[0, 2, 4, 6, 6]], dtype=torch.long)
    logits = torch.tensor([[[0.0, 0.0, 8.0, 0.0, -9.0], [5.0, 0.0, 0.0, 0.0, -9.0], [0.0, 6.0, 0.0, 0.0, -9.0]]])
    matrix = torch.softmax(logits, dim=-1).masked_fill(~candidate_valid[:, None, :], 0.0)
    matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    plan = selector._sparse_transport_plan(
        {"slot_logits": logits, "acquisition_matrix": matrix},
        valid,
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        training=True,
    )

    assert plan["indices"].shape == (1, 3, 1)
    assert plan["selected_positions"].tolist() == [[0.0, 2.0, 4.0]]
    selected = plan["indices"][0, :, 0]
    assert torch.equal(selected, torch.tensor([0, 2, 4], dtype=torch.long))
    assert bool(valid[0, selected].all().item())
    assert torch.all(plan["selected_positions"][:, 1:] >= plan["selected_positions"][:, :-1])


def test_duplicate_aware_union_loss_keeps_positive_column_gradient_under_slot_collapse():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector = _selector_shell(module, target_len=4)
    matrix = torch.tensor(
        [
            [
                [0.95, 0.05 / 3.0, 0.05 / 3.0, 0.05 / 3.0],
                [0.95, 0.05 / 3.0, 0.05 / 3.0, 0.05 / 3.0],
                [0.95, 0.05 / 3.0, 0.05 / 3.0, 0.05 / 3.0],
                [0.95, 0.05 / 3.0, 0.05 / 3.0, 0.05 / 3.0],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    candidate_dense_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    gt_segments = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]

    losses = selector._losses(
        reader_outputs={"acquisition_matrix": matrix},
        valid_mask=valid,
        candidate_dense_indices=candidate_dense_indices,
        gt_segments=gt_segments,
    )
    loss = losses["selector_gt_acquisition_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()
    assert matrix.grad[0, :, 0].abs().sum().item() > 0.0


def test_st_surrogate_keeps_hard_real_frame_values_and_opens_soft_gradient_path():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector = _selector_shell(module, target_len=3)
    valid = torch.tensor([[True, True, True, True, True]], dtype=torch.bool)
    candidate_valid = valid.clone()
    candidate_dense_indices = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    logits = torch.tensor([[[8.0, 0.0, 0.0, 0.0, 0.0], [0.0, 7.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0, 0.0]]], requires_grad=True)
    matrix = torch.softmax(logits, dim=-1)
    inputs = torch.arange(1 * 1 * 5 * 2 * 2, dtype=torch.float32).view(1, 1, 5, 2, 2).requires_grad_(True)

    plan = selector._sparse_transport_plan(
        {"slot_logits": logits, "acquisition_matrix": matrix},
        valid,
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        training=True,
    )
    hard_out = selector._apply_sparse_transport(inputs, plan["indices"], plan["weights"], transport_weights=None)
    st_out = selector._apply_sparse_transport(
        inputs,
        plan["indices"],
        plan["weights"],
        transport_weights=plan["transport_weights"],
    )

    assert torch.allclose(st_out.detach(), hard_out.detach(), atol=0.0, rtol=0.0)
    st_out.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum().item() > 0.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0
    assert plan["st_active_row_counts"] == [3]
