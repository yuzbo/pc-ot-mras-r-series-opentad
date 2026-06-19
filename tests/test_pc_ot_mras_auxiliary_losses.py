import importlib.util
import subprocess
import sys
from pathlib import Path

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
    pytest.skip(
        "torch unavailable in this process: "
        + (torch_probe.stderr.strip().splitlines()[-1] if torch_probe.stderr.strip() else f"exit {torch_probe.returncode}"),
        allow_module_level=True,
    )

import torch

from pc_ot_mras_test_utils import load_pc_ot_mras_classes


ROOT = Path(__file__).resolve().parents[1]
PCOTMRASReader, _PCOTMRASDetectorBridge = load_pc_ot_mras_classes()


def _load_aux_loss():
    path = ROOT / "opentad" / "models" / "losses" / "pc_ot_mras_auxiliary_losses.py"
    spec = importlib.util.spec_from_file_location("pc_ot_mras_auxiliary_losses_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pc_ot_mras_auxiliary_losses


pc_ot_mras_auxiliary_losses = _load_aux_loss()


def _reader_outputs():
    torch.manual_seed(20260619)
    reader = PCOTMRASReader(in_dim=5, hidden_dim=16, num_slots=6, num_blocks=1, num_roles=6)
    features = torch.randn(2, 12, 5, requires_grad=True)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[1, 8:] = False
    outputs = reader(features, valid)
    return reader, features, outputs


def test_pc_ot_mras_auxiliary_losses_accept_batched_gt_segments_and_backpropagate():
    reader, features, outputs = _reader_outputs()
    gt_segments = [
        torch.tensor([[1.0, 5.0], [6.0, 9.0]], dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.float32),
    ]

    losses = pc_ot_mras_auxiliary_losses(
        outputs,
        gt_segments,
        weights=dict(
            body=0.02,
            start=0.05,
            end=0.05,
            boundary=0.05,
            uncertainty=0.01,
            redundancy=0.005,
            process=0.01,
            pair=0.05,
            allocation=0.02,
            regularizer=0.01,
        ),
    )

    assert {
        "pc_ot_mras_aux_body_loss",
        "pc_ot_mras_aux_start_loss",
        "pc_ot_mras_aux_end_loss",
        "pc_ot_mras_aux_boundary_loss",
        "pc_ot_mras_aux_uncertainty_loss",
        "pc_ot_mras_aux_redundancy_loss",
        "pc_ot_mras_aux_process_loss",
        "pc_ot_mras_aux_pair_loss",
        "pc_ot_mras_aux_allocation_loss",
        "pc_ot_mras_aux_regularizer_loss",
    } == set(losses)
    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value).item()

    sum(losses.values()).backward()
    for param in (
        reader.body_head.weight,
        reader.start_head.weight,
        reader.end_head.weight,
        reader.boundary_head.weight,
        reader.uncertainty_head.weight,
        reader.redundancy_head.weight,
        reader.process_head.weight,
        reader.gate_head.weight,
        reader.pair_scorer[-1].weight,
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum().item() > 0
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_pc_ot_mras_auxiliary_losses_reject_bad_gt_contracts():
    _reader, _features, outputs = _reader_outputs()

    with pytest.raises(ValueError, match="gt_segments length"):
        pc_ot_mras_auxiliary_losses(outputs, [torch.tensor([[1.0, 3.0]])])

    with pytest.raises(ValueError, match=r"gt_segments entries must be \[N,2\]"):
        pc_ot_mras_auxiliary_losses(outputs, [torch.tensor([1.0, 3.0]), torch.empty((0, 2))])

    with pytest.raises(ValueError, match="boundary_sigma must be positive"):
        pc_ot_mras_auxiliary_losses(outputs, [torch.empty((0, 2)), torch.empty((0, 2))], boundary_sigma=0.0)
