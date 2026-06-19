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


def test_fake_detector_loss_reaches_acquisition_parameters():
    torch.manual_seed(111)
    reader = PCOTMRASReader(in_dim=6, hidden_dim=18, num_slots=7, num_blocks=2, num_roles=6)
    bridge = PCOTMRASDetectorBridge(in_channels=18, out_channels=12)
    features = torch.randn(2, 11, 6, requires_grad=True)
    valid = torch.ones(2, 11, dtype=torch.bool)
    valid[1, 9:] = False
    coords = torch.linspace(0.0, 1.0, steps=11).unsqueeze(0).expand(2, -1).clone()
    coords[1, 9:] = 0.0

    out = reader(features, valid, coords)
    feats, masks, aux = bridge(out, return_aux=True)
    pair_weight = torch.arange(11, dtype=features.dtype).view(1, 1, 11).to(features.device)
    detector_style_loss = (
        feats[0][masks[0].unsqueeze(1).expand_as(feats[0])].mean()
        + out["selected_times"].mean()
        + out["allocation"].square().mean()
        + (out["pair_prob"] * pair_weight).sum(dim=(1, 2)).mean()
        + out["regularizers"]["total_regularizer"]
    )
    detector_style_loss.backward()

    checks = {
        "query_embed": reader.query_embed.grad,
        "key_proj": reader.key_proj.weight.grad,
        "center_inc_head": reader.center_inc_head.weight.grad,
        "center_shift_head": reader.center_shift_head.weight.grad,
        "width_head": reader.width_head.weight.grad,
        "gate_head": reader.gate_head.weight.grad,
        "role_bias_head": reader.role_bias_head.weight.grad,
        "process_head": reader.process_head.weight.grad,
        "start_head": reader.start_head.weight.grad,
        "end_head": reader.end_head.weight.grad,
        "boundary_head": reader.boundary_head.weight.grad,
        "uncertainty_head": reader.uncertainty_head.weight.grad,
        "pair_scorer": reader.pair_scorer[-1].weight.grad,
        "bridge_proj": bridge.token_proj.weight.grad,
    }
    for name, grad in checks.items():
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum().item() > 0, name

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
