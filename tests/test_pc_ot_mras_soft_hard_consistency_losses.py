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


def _load_soft_hard_loss():
    path = ROOT / "opentad" / "models" / "losses" / "pc_ot_mras_soft_hard_consistency_losses.py"
    spec = importlib.util.spec_from_file_location("pc_ot_mras_soft_hard_consistency_losses_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pc_ot_mras_soft_hard_consistency_losses


pc_ot_mras_soft_hard_consistency_losses = _load_soft_hard_loss()


def _reader_outputs():
    torch.manual_seed(20260620)
    reader = PCOTMRASReader(in_dim=5, hidden_dim=16, num_slots=5, num_blocks=1, num_roles=6)
    features = torch.randn(2, 10, 5, requires_grad=True)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 7:] = False
    outputs = reader(features, valid)
    return reader, features, outputs


def test_soft_hard_consistency_losses_are_train_only_surrogates_and_backpropagate():
    reader, features, outputs = _reader_outputs()

    losses = pc_ot_mras_soft_hard_consistency_losses(outputs)

    assert {
        "pc_ot_mras_soft_hard_slot_allocation_loss",
        "pc_ot_mras_soft_hard_global_acquisition_loss",
        "pc_ot_mras_soft_hard_selected_time_loss",
        "pc_ot_mras_soft_hard_gate_confidence_loss",
        "pc_ot_mras_soft_hard_duplicate_mass_loss",
    } == set(losses)
    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value).item()

    sum(losses.values()).backward()

    for name, param in {
        "key_proj": reader.key_proj.weight,
        "gate_head": reader.gate_head.weight,
        "center_inc_head": reader.center_inc_head.weight,
        "width_head": reader.width_head.weight,
    }.items():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0, name
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0


def test_soft_hard_consistency_zero_weights_returns_graph_safe_zero_loss():
    _reader, _features, outputs = _reader_outputs()

    losses = pc_ot_mras_soft_hard_consistency_losses(
        outputs,
        weights=dict(
            slot_allocation=0.0,
            global_acquisition=0.0,
            selected_time=0.0,
            gate_confidence=0.0,
            duplicate_mass=0.0,
        ),
    )

    assert set(losses) == {"pc_ot_mras_soft_hard_zero_loss"}
    assert losses["pc_ot_mras_soft_hard_zero_loss"].ndim == 0


def test_soft_hard_consistency_rejects_bad_reader_contracts():
    _reader, _features, outputs = _reader_outputs()

    bad = dict(outputs)
    bad["valid_mask"] = bad["valid_mask"].clone()
    bad["valid_mask"][0, 4] = False
    bad["valid_mask"][0, 5] = True
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        pc_ot_mras_soft_hard_consistency_losses(bad)

    bad = dict(outputs)
    bad["acquisition_matrix"] = bad["acquisition_matrix"].clone()
    bad["acquisition_matrix"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pc_ot_mras_soft_hard_consistency_losses(bad)

    bad = dict(outputs)
    bad["selected_times"] = bad["selected_times"][:, :-1]
    with pytest.raises(ValueError, match=r"selected_times must be \[B,K\]"):
        pc_ot_mras_soft_hard_consistency_losses(bad)
