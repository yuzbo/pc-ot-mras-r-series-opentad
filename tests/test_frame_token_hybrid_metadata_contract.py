from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "opentad" / "models" / "selectors" / "frame_token_hybrid_acquisition_route.py"
PREVIEW_TRANSFORM_PATH = ROOT / "opentad" / "datasets" / "transforms" / "frame_token_hybrid.py"


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
    import torch

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


def _load_preview_transform_module():
    sys.modules.pop("opentad.datasets.transforms.frame_token_hybrid", None)
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.datasets", ROOT / "opentad" / "datasets")
    _ensure_package("opentad.datasets.transforms", ROOT / "opentad" / "datasets" / "transforms")
    builder = types.ModuleType("opentad.datasets.builder")
    builder.PIPELINES = _Registry()
    sys.modules["opentad.datasets.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.datasets.transforms.frame_token_hybrid",
        PREVIEW_TRANSFORM_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preview_probe_metadata_drives_selection_instead_of_dense_input_signal():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=16,
        target_dense_len=16,
        anchor_stride=8,
        boundary_radius=0,
        boundary_epsilon=0.1,
        stable_gap_min_len=6,
        stable_epsilon=0.01,
        require_preview_signal=True,
    )
    values = torch.zeros(16, dtype=torch.float32)
    values[4:] = 1000.0
    inputs = values.view(1, 1, 16, 1, 1).expand(1, 3, 16, 2, 2).contiguous()
    masks = torch.ones((1, 16), dtype=torch.bool)
    metas = [
        {
            "sample_id": "preview-probe-contract",
            "frame_token_hybrid_preview_signal": [0.0] * 16,
            "frame_token_hybrid_preview_positions": list(range(16)),
            "frame_token_hybrid_preview_source": "frame_inds_temporal_gap_preview_probe",
        }
    ]

    outputs = selector.forward_test(inputs, masks, metas)

    meta = outputs["metas"][0]
    plan = meta["frame_token_hybrid_acquisition_plan"]
    assert plan["selection_decision_source"] == "deploy_preview_probe_metadata"
    assert plan["selection_surface"] == "preview_probe_visible_pre_backbone_bridge"
    assert plan["boundary_positions"] == []
    assert 4 not in meta["frame_token_hybrid_observed_raw_positions"]
    assert plan["preview_probe"]["signal_meta_key"] == "frame_token_hybrid_preview_signal"
    assert plan["preview_probe"]["positions_meta_key"] == "frame_token_hybrid_preview_positions"
    assert plan["preview_probe"]["source_meta_key"] == "frame_token_hybrid_preview_source"
    assert plan["preview_probe"]["source"] == "frame_inds_temporal_gap_preview_probe"
    assert plan["preview_probe"]["observation_count"] == 16

    observed_mask = meta["frame_token_hybrid_observed_raw_mask"]
    span_mask = meta["frame_token_hybrid_span_derived_mask"]
    completion_mask = meta["frame_token_hybrid_dense_completion_mask"]
    assert len(observed_mask) == 16
    assert len(span_mask) == 16
    assert len(completion_mask) == 16
    assert observed_mask[0] is True
    assert observed_mask[8] is True
    assert observed_mask[15] is True
    assert observed_mask[4] is False
    assert span_mask[4] is True
    assert completion_mask[4] is True
    assert completion_mask[8] is False

    accounting = plan["compute_accounting"]
    assert accounting["observed_raw_frame_count"] == len(meta["frame_token_hybrid_observed_raw_positions"])
    assert accounting["compressed_span_token_count"] == len(plan["span_tokens"])
    assert accounting["dense_completion_position_count"] == sum(completion_mask)
    assert accounting["actual_decode_saving_in_current_actionformer_pipeline"] is False
    assert accounting["raw_decode_saving_claim_allowed"] is False
    assert accounting["pre_decode_loader_hook_reviewed"] is False


def test_low_cost_preview_probe_transform_writes_deploy_visible_metadata_without_decode_or_leakage():
    transform_module = _load_preview_transform_module()
    transform = transform_module.FrameTokenHybridPreviewProbe(
        signal_meta_key="frame_token_hybrid_preview_signal",
        positions_meta_key="frame_token_hybrid_preview_positions",
        source_meta_key="frame_token_hybrid_preview_source",
    )
    results = {
        "video_name": "video_validation_0000051",
        "frame_inds": [[0], [8], [16], [24], [80], [88], [96], [104]],
        "masks": [1, 1, 1, 1, 1, 1, 0, 0],
        "gt_segments": [[1.0, 2.0]],
    }

    output = transform(results)

    assert output["frame_token_hybrid_preview_positions"] == [0, 1, 2, 3, 4, 5]
    assert len(output["frame_token_hybrid_preview_signal"]) == 6
    assert output["frame_token_hybrid_preview_source"] == "frame_inds_temporal_gap_preview_probe"
    assert "frame_token_hybrid_preview_uses_gt" not in output
    assert "frame_token_hybrid_preview_uses_teacher" not in output
    assert "frame_token_hybrid_preview_uses_cache" not in output
    text = repr(
        [
            output["frame_token_hybrid_preview_signal"],
            output["frame_token_hybrid_preview_positions"],
            output["frame_token_hybrid_preview_source"],
        ]
    ).lower()
    for forbidden in ("gt", "teacher", "oracle", "cache", "prediction", "result", "checkpoint"):
        assert forbidden not in text


def test_require_preview_probe_signal_fails_closed_without_deploy_visible_signal():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=16,
        target_dense_len=16,
        require_preview_signal=True,
    )
    inputs = torch.zeros((1, 3, 16, 2, 2), dtype=torch.float32)
    masks = torch.ones((1, 16), dtype=torch.bool)

    with pytest.raises(ValueError, match="requires deploy-visible preview/probe signal"):
        selector.forward_test(inputs, masks, [{"sample_id": "missing-preview"}])


def test_preview_probe_metadata_rejects_forbidden_leakage_values():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=16,
        target_dense_len=16,
        require_preview_signal=True,
    )
    inputs = torch.zeros((1, 3, 16, 2, 2), dtype=torch.float32)
    masks = torch.ones((1, 16), dtype=torch.bool)
    metas = [
        {
            "sample_id": "leakage-check",
            "frame_token_hybrid_preview_signal": [0.0] * 16,
            "frame_token_hybrid_preview_positions": list(range(16)),
            "frame_token_hybrid_preview_source": "raw_prediction_cache",
        }
    ]

    with pytest.raises(ValueError, match="forbidden test-time meta"):
        selector.forward_test(inputs, masks, metas)
