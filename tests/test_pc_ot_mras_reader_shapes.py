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


PCOTMRASReader, PCOTMRASDetectorBridge = load_pc_ot_mras_classes()
LowCostAcquisitionBrowser = sys.modules[
    "opentad.models.selectors.lowcost_acquisition_browser"
].LowCostAcquisitionBrowser
ROOT = Path(__file__).resolve().parents[1]


def _inputs(batch=2, time=12, dim=5):
    torch.manual_seed(101)
    features = torch.randn(batch, time, dim)
    valid = torch.ones(batch, time, dtype=torch.bool)
    valid[1, 8:] = False
    coords = torch.linspace(0.0, 1.0, steps=time).unsqueeze(0).expand(batch, -1).clone()
    coords[1, 8:] = 0.0
    return features, valid, coords


def test_pc_ot_mras_reader_shapes_masks_and_ordered_slots():
    reader = PCOTMRASReader(in_dim=5, hidden_dim=16, num_slots=6, num_blocks=2, num_roles=6)
    features, valid, coords = _inputs()

    out = reader(features, valid, coords)

    assert out["allocation"].shape == (2, 6, 12)
    assert out["allocation_logits"].shape == (2, 6, 12)
    assert out["selected_tokens"].shape == (2, 6, 16)
    assert out["selected_times"].shape == (2, 6)
    assert out["centers"].shape == (2, 6)
    assert out["widths"].shape == (2, 6)
    assert out["gates"].shape == (2, 6)
    assert out["process_logits"].shape == (2, 12, 7)
    assert out["body_logits"].shape == (2, 12)
    assert out["start_logits"].shape == (2, 12)
    assert out["end_logits"].shape == (2, 12)
    assert out["boundary_logits"].shape == (2, 12)
    assert out["uncertainty_logits"].shape == (2, 12)
    assert out["redundancy_logits"].shape == (2, 12)
    assert out["pair_logits"].shape == (2, 12, 12)
    assert out["pair_prob"].shape == (2, 12, 12)
    assert out["pair_valid_mask"].shape == (2, 12, 12)
    assert torch.is_tensor(out["regularizers"]["total_regularizer"])
    assert out["regularizers"]["total_regularizer"].ndim == 0
    assert torch.isfinite(out["regularizers"]["total_regularizer"])
    assert out["role_ids"].shape == (2, 6)
    assert out["round_ids"].shape == (2, 6)
    assert torch.all(out["allocation"] >= 0)
    assert torch.all(out["acquisition_matrix"] >= 0)
    assert torch.allclose(out["allocation"].sum(dim=-1), torch.ones(2, 6), atol=1e-5)
    assert torch.allclose(out["acquisition_matrix"].sum(dim=-1), out["gates"], atol=1e-5)
    assert torch.allclose(out["acquisition_matrix"], out["allocation"] * out["gates"].unsqueeze(-1), atol=1e-6)
    assert torch.all(out["allocation"][1, :, 8:] == 0)
    assert torch.all(out["acquisition_matrix"][1, :, 8:] == 0)
    assert torch.all(out["browser_memory"][1, 8:, :] == 0)
    assert torch.all(out["selected_mask"])
    assert torch.all(out["centers"][:, 1:] >= out["centers"][:, :-1])
    assert torch.all(out["widths"] >= reader.cfg.min_width)
    assert torch.all(out["widths"] <= reader.cfg.max_width)
    assert torch.isfinite(out["selected_tokens"]).all()
    assert out["valid_lengths"].tolist() == [12, 8]
    assert "value_logits" not in out
    assert "risk_logits" not in out


def test_pc_ot_mras_reader_value_heads_are_explicit_opt_in():
    reader = PCOTMRASReader(
        in_dim=5,
        hidden_dim=16,
        num_slots=6,
        num_blocks=1,
        num_roles=6,
        enable_value_heads=True,
    )
    features, valid, coords = _inputs()

    out = reader(features, valid, coords)

    assert reader.cfg.enable_value_heads is True
    assert out["value_logits"].shape == (2, 12)
    assert out["risk_logits"].shape == (2, 12)
    assert torch.isfinite(out["value_logits"][:, :8]).all()
    assert torch.isfinite(out["risk_logits"][:, :8]).all()
    assert torch.all(out["allocation"][:, :, :] >= 0)
    assert torch.allclose(out["allocation"].sum(dim=-1), torch.ones(2, 6), atol=1e-5)


def test_pc_ot_mras_reader_can_skip_pair_distribution_for_reader_only_runs():
    reader = PCOTMRASReader(
        in_dim=5,
        hidden_dim=16,
        num_slots=6,
        num_blocks=1,
        num_roles=6,
        emit_pair_distribution=False,
    )
    features, valid, coords = _inputs()

    out = reader(features, valid, coords)

    assert "pair_logits" not in out
    assert "pair_prob" not in out
    assert "pair_valid_mask" not in out
    assert torch.is_tensor(out["regularizers"]["total_regularizer"])
    assert torch.isfinite(out["regularizers"]["total_regularizer"])
    assert out["regularizers"]["pair_entropy_loss"].item() == pytest.approx(0.0, abs=1e-7)


def test_pc_ot_mras_dense_heads_mask_half_logits_with_output_dtype_sentinel():
    class HalfHead(torch.nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.out_dim = out_dim

        def forward(self, h):
            return torch.zeros((*h.shape[:2], self.out_dim), dtype=torch.float16, device=h.device)

    reader = PCOTMRASReader(
        in_dim=5,
        hidden_dim=8,
        num_slots=4,
        num_blocks=1,
        num_roles=6,
        enable_value_heads=True,
    )
    reader.process_head = HalfHead(reader.cfg.num_process_states)
    for name in (
        "start_head",
        "end_head",
        "boundary_head",
        "body_head",
        "uncertainty_head",
        "redundancy_head",
        "value_head",
        "risk_head",
    ):
        setattr(reader, name, HalfHead(1))

    h = torch.randn(2, 6, 8, dtype=torch.float32)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    out = reader._dense_heads(h, valid)

    expected = torch.tensor(torch.finfo(torch.float16).min / 4.0, dtype=torch.float16)
    for key, logits in out.items():
        mask = valid.unsqueeze(-1).expand_as(logits) if logits.ndim == 3 else valid
        assert logits.dtype == torch.float16
        assert torch.isfinite(logits).all()
        assert torch.all(logits[~mask] == expected)
        assert torch.all(logits[mask] == 0)


def test_lowcost_browser_masks_half_logits_with_output_dtype_sentinel():
    class HalfHead(torch.nn.Module):
        def forward(self, h):
            return torch.zeros((*h.shape[:2], 1), dtype=torch.float16, device=h.device)

    browser = LowCostAcquisitionBrowser(in_dim=5, hidden_dim=8, num_blocks=1)
    for name in ("acq_head", "start_head", "end_head", "boundary_head"):
        setattr(browser, name, HalfHead())

    features = torch.randn(2, 6, 5, dtype=torch.float32)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    out = browser(features, valid)

    expected = torch.tensor(torch.finfo(torch.float16).min / 4.0, dtype=torch.float16)
    for key in ("acq_logits", "start_logits", "end_logits", "boundary_logits"):
        logits = out[key]
        assert logits.dtype == torch.float16
        assert torch.isfinite(logits).all()
        assert torch.all(logits[~valid] == expected)
        assert torch.all(logits[valid] == 0)


def test_pc_ot_mras_reader_centers_are_monotonic_without_posthoc_sort():
    source = (ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py").read_text(encoding="utf-8")
    method_source = source.split("def _ordered_centers_and_widths", 1)[1].split("def _dense_heads", 1)[0]
    assert "torch.sort" not in method_source
    assert ".sort(" not in method_source

    reader = PCOTMRASReader(in_dim=5, hidden_dim=16, num_slots=12, num_blocks=1, num_roles=6)
    features, valid, coords = _inputs(time=12, dim=5)
    out = reader(features, valid, coords)
    assert torch.all(out["centers"][:, 1:] > out["centers"][:, :-1])


def test_pc_ot_mras_detector_bridge_preserves_continuous_gradient_path():
    reader = PCOTMRASReader(in_dim=5, hidden_dim=16, num_slots=6, num_blocks=1, num_roles=6)
    bridge = PCOTMRASDetectorBridge(in_channels=16, out_channels=10)
    features, valid, coords = _inputs()
    features.requires_grad_(True)

    out = reader(features, valid, coords)
    feats, masks, aux = bridge(out, return_aux=True)

    assert feats[0].shape == (2, 10, 6)
    assert masks[0].shape == (2, 6)
    assert aux["selected_times"].shape == (2, 6)
    loss = feats[0].mean() + aux["selected_times"].mean() + aux["gates"].mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert reader.key_proj.weight.grad is not None
    assert torch.isfinite(reader.key_proj.weight.grad).all()


def test_pc_ot_mras_reader_rejects_invalid_inputs():
    reader = PCOTMRASReader(in_dim=3, hidden_dim=8, num_slots=4, num_blocks=1)
    features = torch.zeros(1, 5, 3)

    bad_features = features.clone()
    bad_features[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="lowcost_features must be finite"):
        reader(bad_features, torch.ones(1, 5, dtype=torch.bool))

    with pytest.raises(ValueError, match="binary"):
        reader(features, torch.tensor([[1.0, 1.0, 0.5, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="contiguous valid prefix"):
        reader(features, torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool))

    with pytest.raises(ValueError, match="time_coords"):
        reader(features, torch.ones(1, 5, dtype=torch.bool), torch.zeros(1, 4))

    with pytest.raises(ValueError, match="strictly increasing"):
        reader(features, torch.ones(1, 5, dtype=torch.bool), torch.tensor([[0.0, 0.25, 0.20, 0.75, 1.0]]))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reader(features, torch.ones(1, 5, dtype=torch.bool), torch.tensor([[0.0, 0.25, 0.50, 0.75, 1.10]]))
