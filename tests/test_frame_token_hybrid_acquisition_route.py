from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "opentad" / "models" / "selectors" / "frame_token_hybrid_acquisition_route.py"


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


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


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_route_module():
    sys.modules.pop("opentad.models.selectors.frame_token_hybrid_acquisition_route", None)
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.frame_token_hybrid_acquisition_route",
        ROUTE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_hybrid_frames(torch, *, batch: int = 1, dense_len: int = 32):
    values = torch.zeros(dense_len, dtype=torch.float32)
    values[:5] = torch.linspace(0.0, 4.0, 5)
    values[5:19] = 8.0
    values[19:24] = torch.linspace(8.0, 14.0, 5)
    values[24:] = 14.0
    frames = values.view(1, 1, dense_len, 1, 1).expand(batch, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"frame-token-hybrid-{idx}"} for idx in range(batch)]
    return frames, masks, metas


def test_frame_token_hybrid_generates_stable_gap_span_tokens_without_forbidden_payloads():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=32,
        target_dense_len=32,
        anchor_stride=8,
        boundary_radius=1,
        stable_gap_min_len=6,
        stable_epsilon=0.05,
    )
    inputs, masks, metas = _make_hybrid_frames(torch)

    outputs = selector.forward_test(inputs, masks, metas)

    plan = outputs["metas"][0]["frame_token_hybrid_acquisition_plan"]
    span_tokens = plan["span_tokens"]
    assert plan["route_label"] == "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"
    assert plan["meta_key"] == "frame_token_hybrid_acquisition_plan"
    assert plan["uses_gt"] is False
    assert plan["uses_teacher"] is False
    assert plan["uses_oracle"] is False
    assert plan["uses_raw_prediction_cache"] is False
    assert span_tokens, plan
    first_span = span_tokens[0]
    assert {"span_start", "span_end", "role", "visibility", "compression_confidence"} <= set(first_span)
    assert first_span["role"] == "stable_gap_span_token"
    assert first_span["span_start"] <= 6
    assert first_span["span_end"] >= 17
    assert 0.0 < first_span["visibility"] < 1.0
    assert first_span["compression_confidence"] >= 0.9

    forbidden_fragments = ("gt", "teacher", "oracle", "cache", "prediction", "result", "checkpoint")
    span_text = repr(span_tokens).lower()
    for token in forbidden_fragments:
        assert token not in span_text


def test_frame_token_hybrid_bridge_preserves_observed_raw_positions_exactly():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=32,
        target_dense_len=32,
        anchor_stride=8,
        boundary_radius=1,
        stable_gap_min_len=6,
    )
    inputs, masks, metas = _make_hybrid_frames(torch)

    outputs = selector.forward_test(inputs, masks, metas)

    dense = outputs["inputs"]
    observed = outputs["metas"][0]["frame_token_hybrid_observed_raw_positions"]
    assert dense.shape == inputs.shape
    assert observed == sorted(set(observed))
    for position in observed:
        assert torch.equal(dense[0, :, position], inputs[0, :, position]), position
    assert "frame_token_hybrid_bridge" in outputs["metas"][0]
    bridge_meta = outputs["metas"][0]["frame_token_hybrid_bridge"]
    assert bridge_meta["dense_completion_conditioning_keys"] == [
        "span_start",
        "span_end",
        "role",
        "visibility",
        "compression_confidence",
    ]


def test_frame_token_hybrid_dense_completion_does_not_forward_unobserved_raw_frames():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=16,
        target_dense_len=16,
        anchor_stride=8,
        boundary_radius=0,
        boundary_epsilon=99999.0,
        stable_gap_min_len=99,
    )
    values = torch.zeros(16, dtype=torch.float32)
    values[0] = 0.0
    values[4] = 1234.0
    values[8] = 80.0
    values[15] = 150.0
    inputs = values.view(1, 1, 16, 1, 1).expand(1, 3, 16, 2, 2).contiguous()
    masks = torch.ones((1, 16), dtype=torch.bool)

    outputs = selector.forward_test(inputs, masks, [{"sample_id": "raw-leak-check"}])

    observed = outputs["metas"][0]["frame_token_hybrid_observed_raw_positions"]
    assert 4 not in observed
    expected_position_4 = inputs[0, :, 0] * 0.5 + inputs[0, :, 8] * 0.5
    assert torch.allclose(outputs["inputs"][0, :, 4], expected_position_4)
    assert not torch.equal(outputs["inputs"][0, :, 4], inputs[0, :, 4])


def test_frame_token_hybrid_dense_completion_zeroes_invalid_mask_suffix():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=16,
        target_dense_len=16,
        anchor_stride=8,
        boundary_radius=0,
        boundary_epsilon=99999.0,
        stable_gap_min_len=99,
    )
    values = torch.arange(16, dtype=torch.float32)
    inputs = values.view(1, 1, 16, 1, 1).expand(1, 3, 16, 2, 2).contiguous()
    masks = torch.zeros((1, 16), dtype=torch.bool)
    masks[:, :12] = True

    outputs = selector.forward_test(inputs, masks, [{"sample_id": "invalid-mask-check"}])

    assert outputs["masks"][0, :12].all()
    assert not outputs["masks"][0, 12:].any()
    assert torch.count_nonzero(outputs["inputs"][0, :, 12:]) == 0


def test_frame_token_hybrid_outputs_actionformer_compatible_dense_axis_and_train_passthrough():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(dense_window_size=768, target_dense_len=768)
    inputs, masks, metas = _make_hybrid_frames(torch, batch=2, dense_len=768)
    gt_segments = [torch.tensor([[10.0, 20.0]], dtype=torch.float32) for _idx in range(2)]
    gt_labels = [torch.tensor([1], dtype=torch.long) for _idx in range(2)]

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, [{"sample_id": "dense-axis-0"}, {"sample_id": "dense-axis-1"}])

    assert train_outputs["inputs"].shape == (2, 3, 768, 2, 2)
    assert train_outputs["masks"].shape == (2, 768)
    assert train_outputs["masks"].dtype == torch.bool
    assert train_outputs["masks"].all()
    assert test_outputs["inputs"].shape == (2, 3, 768, 2, 2)
    assert train_outputs["gt_segments"] is gt_segments
    assert train_outputs["gt_labels"] is gt_labels
    assert train_outputs["metas"][0]["frame_token_hybrid_acquisition_plan"]["output_dense_axis_len"] == 768


@pytest.mark.parametrize(
    "bad_meta",
    [
        {"gt_segments": [[1, 2]]},
        {"teacher_logits": "forbidden"},
        {"oracle_boundary": [1, 2]},
        {"raw_prediction_cache": "must-not-use"},
        {"result_detection": "must-not-use"},
        {"checkpoint_path": "must-not-use"},
    ],
)
def test_frame_token_hybrid_forward_test_rejects_forbidden_deploy_meta_payloads(bad_meta):
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(dense_window_size=32, target_dense_len=32)
    inputs, masks, _metas = _make_hybrid_frames(torch)

    with pytest.raises(ValueError, match="forbidden test-time meta"):
        selector.forward_test(inputs, masks, [bad_meta])
