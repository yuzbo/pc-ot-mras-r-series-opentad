import subprocess
import sys

import pytest


torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
if torch_probe.returncode != 0:
    pytest.skip("torch unavailable", allow_module_level=True)

import torch

from pc_ot_mras_test_utils import load_pc_ot_mras_classes


PCOTMRASReader, PCOTMRASDetectorBridge = load_pc_ot_mras_classes()


def _source_tokens():
    torch.manual_seed(20260619)
    source = torch.randn(2, 5, 3, requires_grad=True)
    dense_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    return source, dense_mask


def _legal_reader_outputs(source_tokens, dense_mask, *, requires_grad=False):
    batch, dense_len, channels = source_tokens.shape
    matrix = source_tokens.new_zeros((batch, 3, dense_len))
    matrix[0, 0, 0:2] = torch.tensor([0.30, 0.50], dtype=matrix.dtype, device=matrix.device)
    matrix[0, 1, 1:4] = torch.tensor([0.10, 0.20, 0.30], dtype=matrix.dtype, device=matrix.device)
    matrix[0, 2, 2:5] = torch.tensor([0.20, 0.20, 0.40], dtype=matrix.dtype, device=matrix.device)
    matrix[1, 0, 0:2] = torch.tensor([0.25, 0.25], dtype=matrix.dtype, device=matrix.device)
    matrix[1, 1, 1:3] = torch.tensor([0.30, 0.40], dtype=matrix.dtype, device=matrix.device)
    selected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool, device=matrix.device)
    if requires_grad:
        matrix.requires_grad_()
    gates = torch.tensor(
        [
            [0.80, 0.60, 0.80],
            [0.50, 0.70, 0.20],
        ],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    return {
        "acquisition_matrix": matrix,
        "selected_tokens": source_tokens.new_zeros((batch, 3, channels)),
        "selected_mask": selected_mask,
        "valid_mask": dense_mask,
        "selected_times": torch.tensor(
            [[0.10, 0.45, 0.80], [0.15, 0.60, 0.00]],
            dtype=matrix.dtype,
            device=matrix.device,
        ),
        "centers": torch.tensor(
            [[0.10, 0.45, 0.80], [0.15, 0.60, 0.00]],
            dtype=matrix.dtype,
            device=matrix.device,
        ),
        "widths": torch.full((batch, 3), 0.20, dtype=matrix.dtype, device=matrix.device),
        "gates": gates,
        "valid_lengths": dense_mask.long().sum(dim=1),
    }


def test_acquisition_contract_accepts_legal_soft_matrix_and_recomputes_tokens():
    source_tokens, dense_mask = _source_tokens()
    reader_outputs = _legal_reader_outputs(source_tokens, dense_mask, requires_grad=True)
    bridge = PCOTMRASDetectorBridge(in_channels=3, out_channels=4)

    feats, masks, aux = bridge(source_tokens=source_tokens, reader_outputs=reader_outputs, return_aux=True)

    assert feats[0].shape == (2, 4, 3)
    assert torch.equal(masks[0], reader_outputs["selected_mask"])
    assert aux["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert aux["selected_tokens_source_verified"] is True
    assert torch.allclose(
        aux["selected_tokens"],
        torch.bmm(reader_outputs["acquisition_matrix"], source_tokens),
        atol=1e-6,
    )

    loss = feats[0].square().mean() + aux["selected_tokens"].square().mean()
    loss.backward()
    assert source_tokens.grad is not None
    assert torch.isfinite(source_tokens.grad).all()
    assert reader_outputs["acquisition_matrix"].grad is not None
    assert torch.isfinite(reader_outputs["acquisition_matrix"].grad).all()


@pytest.mark.parametrize(
    "bad_value, message",
    [
        (-1.0e-4, "non-negative"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
    ],
)
def test_acquisition_contract_rejects_negative_nan_and_inf(bad_value, message):
    source_tokens, dense_mask = _source_tokens()
    reader_outputs = _legal_reader_outputs(source_tokens, dense_mask)
    reader_outputs["acquisition_matrix"] = reader_outputs["acquisition_matrix"].clone()
    reader_outputs["acquisition_matrix"][0, 0, 0] = bad_value
    bridge = PCOTMRASDetectorBridge(in_channels=3, out_channels=4)

    with pytest.raises(ValueError, match=message):
        bridge(source_tokens=source_tokens, reader_outputs=reader_outputs, return_aux=True)


def test_acquisition_contract_rejects_all_zero_selected_row():
    source_tokens, dense_mask = _source_tokens()
    reader_outputs = _legal_reader_outputs(source_tokens, dense_mask)
    reader_outputs["acquisition_matrix"] = reader_outputs["acquisition_matrix"].clone()
    reader_outputs["acquisition_matrix"][0, 1, :] = 0.0
    bridge = PCOTMRASDetectorBridge(in_channels=3, out_channels=4)

    with pytest.raises(ValueError, match="row mass"):
        bridge(source_tokens=source_tokens, reader_outputs=reader_outputs, return_aux=True)


def test_acquisition_contract_rejects_row_mass_above_gate_bound():
    source_tokens, dense_mask = _source_tokens()
    reader_outputs = _legal_reader_outputs(source_tokens, dense_mask)
    reader_outputs["acquisition_matrix"] = reader_outputs["acquisition_matrix"].clone()
    reader_outputs["acquisition_matrix"][0, 0, :] = 0.0
    reader_outputs["acquisition_matrix"][0, 0, 0] = 1.20
    reader_outputs["gates"] = reader_outputs["gates"].clone()
    reader_outputs["gates"][0, 0] = 1.20
    bridge = PCOTMRASDetectorBridge(in_channels=3, out_channels=4)

    with pytest.raises(ValueError, match="row mass.*<= 1.0"):
        bridge(source_tokens=source_tokens, reader_outputs=reader_outputs, return_aux=True)


def test_acquisition_contract_rejects_padded_dense_token_contamination():
    source_tokens, dense_mask = _source_tokens()
    reader_outputs = _legal_reader_outputs(source_tokens, dense_mask)
    reader_outputs["acquisition_matrix"] = reader_outputs["acquisition_matrix"].clone()
    reader_outputs["acquisition_matrix"][1, 0, 4] = 1.0e-3
    bridge = PCOTMRASDetectorBridge(in_channels=3, out_channels=4)

    with pytest.raises(ValueError, match="invalid/padded positions"):
        bridge(source_tokens=source_tokens, reader_outputs=reader_outputs, return_aux=True)


def test_acquisition_contract_rejects_padded_selected_slot_mass():
    source_tokens, dense_mask = _source_tokens()
    reader_outputs = _legal_reader_outputs(source_tokens, dense_mask)
    reader_outputs["acquisition_matrix"] = reader_outputs["acquisition_matrix"].clone()
    reader_outputs["acquisition_matrix"][1, 2, 0] = 1.0e-3
    bridge = PCOTMRASDetectorBridge(in_channels=3, out_channels=4)

    with pytest.raises(ValueError, match="invalid/padded selected slots"):
        bridge(source_tokens=source_tokens, reader_outputs=reader_outputs, return_aux=True)


def test_acquisition_contract_preserves_reader_bridge_gradient_path():
    torch.manual_seed(20260620)
    reader = PCOTMRASReader(in_dim=3, hidden_dim=8, num_slots=4, num_blocks=1, num_roles=6)
    bridge = PCOTMRASDetectorBridge(in_channels=8, out_channels=5)
    features = torch.randn(2, 6, 3, requires_grad=True)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[1, 4:] = False
    coords = torch.linspace(0.0, 1.0, steps=6).unsqueeze(0).expand(2, -1).clone()
    coords[1, 4:] = 0.0

    reader_outputs = reader(features, valid, coords)
    feats, masks, aux = bridge(
        source_tokens=reader_outputs["browser_memory"],
        reader_outputs=reader_outputs,
        return_aux=True,
    )
    loss = (
        feats[0].square().mean()
        + aux["selected_times"].mean()
        + reader_outputs["acquisition_matrix"].square().mean()
        + reader_outputs["regularizers"]["total_regularizer"]
    )
    loss.backward()

    assert aux["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert torch.equal(masks[0], reader_outputs["selected_mask"])
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    for name, grad in {
        "key_proj": reader.key_proj.weight.grad,
        "gate_head": reader.gate_head.weight.grad,
        "center_inc_head": reader.center_inc_head.weight.grad,
        "bridge_token_proj": bridge.token_proj.weight.grad,
    }.items():
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum().item() > 0, name
