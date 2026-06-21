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


PCOTMRASReader, _PCOTMRASDetectorBridge = load_pc_ot_mras_classes()


def test_pc_ot_mras_pair_distribution_masks_illegal_pairs():
    torch.manual_seed(131)
    reader = PCOTMRASReader(in_dim=4, hidden_dim=12, num_slots=4, num_blocks=1)
    features = torch.randn(2, 7, 4)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[1, 5:] = False

    out = reader(features, valid)
    pair_prob = out["pair_prob"]
    pair_mask = out["pair_valid_mask"]

    assert pair_prob.shape == (2, 7, 7)
    assert torch.all(pair_prob[~pair_mask] == 0)
    for i in range(7):
        assert torch.all(pair_prob[:, i, : i + 1] == 0)
    assert pair_prob[0].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert pair_prob[1].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.all(pair_prob[1, 5:, :] == 0)
    assert torch.all(pair_prob[1, :, 5:] == 0)


def test_pc_ot_mras_pair_distribution_single_timestep_all_invalid():
    torch.manual_seed(133)
    reader = PCOTMRASReader(in_dim=4, hidden_dim=12, num_slots=4, num_blocks=1)
    features = torch.randn(2, 1, 4)
    valid = torch.ones(2, 1, dtype=torch.bool)

    out = reader(features, valid)

    assert out["pair_prob"].shape == (2, 1, 1)
    assert out["pair_valid_mask"].shape == (2, 1, 1)
    assert not out["pair_valid_mask"].any().item()
    assert torch.all(out["pair_prob"] == 0)
    assert torch.isfinite(out["pair_prob"]).all()
    assert torch.isfinite(out["regularizers"]["pair_entropy_loss"]).all()


def test_pc_ot_mras_pair_distribution_accepts_amp_softmax_dtype(monkeypatch):
    torch.manual_seed(135)
    reader = PCOTMRASReader(in_dim=4, hidden_dim=12, num_slots=4, num_blocks=1)
    module = sys.modules["opentad.models.selectors.pc_ot_mras_reader"]
    original_softmax = module._masked_softmax

    def half_softmax(logits, mask, dim):
        return original_softmax(logits, mask, dim).to(dtype=torch.float16)

    monkeypatch.setattr(module, "_masked_softmax", half_softmax)
    h = torch.randn(2, 6, 12, dtype=torch.float32)
    valid = torch.ones(2, 6, dtype=torch.bool)
    coords = torch.linspace(0.0, 1.0, 6, dtype=torch.float32)[None, :].expand(2, -1)
    dense = {
        "start_logits": torch.randn(2, 6, dtype=torch.float32),
        "end_logits": torch.randn(2, 6, dtype=torch.float32),
    }

    _logits, pair_prob, pair_mask = reader._pair_distribution(h, valid, coords, dense)

    assert pair_prob.dtype == torch.float32
    assert torch.all(pair_prob[~pair_mask] == 0)
    assert torch.allclose(pair_prob.sum(dim=(1, 2)), torch.ones(2), atol=1e-3)


def test_pc_ot_mras_regularizers_keep_entropy_finite_for_half_zero_probs():
    reader = PCOTMRASReader(in_dim=4, hidden_dim=12, num_slots=4, num_blocks=1)
    allocation = torch.zeros(2, 4, 6, dtype=torch.float16)
    allocation[:, :, 0] = 1.0
    pair_prob = torch.zeros(2, 6, 6, dtype=torch.float16)
    centers = torch.linspace(0.1, 0.9, 4, dtype=torch.float32)[None, :].expand(2, -1)
    widths = torch.full((2, 4), 0.1, dtype=torch.float32)
    gates = torch.full((2, 4), 0.5, dtype=torch.float32)

    regularizers = reader._regularizers(
        allocation=allocation,
        centers=centers,
        widths=widths,
        gates=gates,
        pair_prob=pair_prob,
    )

    assert torch.isfinite(regularizers["entropy_loss"]).all()
    assert torch.isfinite(regularizers["pair_entropy_loss"]).all()
    assert torch.isfinite(regularizers["total_regularizer"]).all()


def test_process_boundary_logits_influence_allocation():
    torch.manual_seed(137)
    reader = PCOTMRASReader(in_dim=4, hidden_dim=10, num_slots=3, num_blocks=1)
    features = torch.randn(1, 8, 4)
    valid = torch.ones(1, 8, dtype=torch.bool)
    out = reader(features, valid)
    dense = {key: out[key].clone() for key in (
        "start_logits",
        "end_logits",
        "boundary_logits",
        "body_logits",
        "uncertainty_logits",
        "redundancy_logits",
        "process_logits",
    )}
    dense["boundary_logits"][:, 4] = dense["boundary_logits"][:, 4] + 25.0
    logits, allocation = reader._allocation(
        h=out["browser_memory"],
        valid=out["valid_mask"],
        time_coords=out["time_coords"],
        slot_state=out["slot_state"],
        centers=out["centers"],
        widths=out["widths"],
        dense=dense,
    )

    assert logits.shape == out["allocation_logits"].shape
    assert allocation[:, :, 4].mean().item() != pytest.approx(out["allocation"][:, :, 4].mean().item())
