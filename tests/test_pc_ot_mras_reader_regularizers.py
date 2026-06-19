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


def test_pc_ot_mras_regularizers_are_finite_and_expose_collapse_terms():
    torch.manual_seed(121)
    reader = PCOTMRASReader(in_dim=4, hidden_dim=12, num_slots=5, num_blocks=1, column_cap=0.25)
    features = torch.randn(1, 9, 4)
    valid = torch.ones(1, 9, dtype=torch.bool)

    out = reader(features, valid)
    regs = out["regularizers"]

    expected = {
        "order_loss",
        "diversity_loss",
        "entropy_loss",
        "column_cap_loss",
        "budget_loss",
        "width_loss",
        "pair_entropy_loss",
        "total_regularizer",
    }
    assert expected.issubset(regs.keys())
    for key in expected:
        assert torch.is_tensor(regs[key]), key
        assert regs[key].ndim == 0, key
        assert torch.isfinite(regs[key]).all(), key
    assert regs["column_cap_loss"].item() >= 0
    assert regs["entropy_loss"].item() > 0
