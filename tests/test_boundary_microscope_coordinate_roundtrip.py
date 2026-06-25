from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUTE_LABEL = "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"


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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_lightweight_runtime():
    for name in tuple(sys.modules):
        if name.startswith("opentad.models.selectors.boundary_microscope_acquisition_route"):
            sys.modules.pop(name, None)
        if name.startswith("opentad.models.dense_heads.anchor_free_head"):
            sys.modules.pop(name, None)
        if name.startswith("opentad.models.utils.temporal_grid"):
            sys.modules.pop(name, None)
        if name.startswith("opentad.models.utils.post_processing.utils"):
            sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.dense_heads", ROOT / "opentad" / "models" / "dense_heads")
    _ensure_package("opentad.models.utils", ROOT / "opentad" / "models" / "utils")
    _ensure_package(
        "opentad.models.utils.post_processing",
        ROOT / "opentad" / "models" / "utils" / "post_processing",
    )

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    builder.HEADS = _Registry()
    builder.build_prior_generator = lambda cfg: None
    builder.build_loss = lambda cfg: None
    sys.modules["opentad.models.builder"] = builder

    bricks = types.ModuleType("opentad.models.bricks")
    bricks.ConvModule = object
    bricks.Scale = object
    sys.modules["opentad.models.bricks"] = bricks

    route_module = _load_module(
        "opentad.models.selectors.boundary_microscope_acquisition_route",
        ROOT / "opentad" / "models" / "selectors" / "boundary_microscope_acquisition_route.py",
    )
    _load_module(
        "opentad.models.utils.temporal_grid",
        ROOT / "opentad" / "models" / "utils" / "temporal_grid.py",
    )
    head_module = _load_module(
        "opentad.models.dense_heads.anchor_free_head",
        ROOT / "opentad" / "models" / "dense_heads" / "anchor_free_head.py",
    )
    post_utils = _load_module(
        "opentad.models.utils.post_processing.utils",
        ROOT / "opentad" / "models" / "utils" / "post_processing" / "utils.py",
    )
    return route_module, head_module, post_utils


def _make_boundary_frames(torch):
    dense_len = 64
    signal = torch.zeros(dense_len, dtype=torch.float32)
    signal[8:22] = 4.0
    signal[39:50] = 3.0
    inputs = signal.view(1, 1, dense_len, 1, 1).expand(1, 3, dense_len, 3, 3).contiguous()
    masks = torch.ones((1, dense_len), dtype=torch.bool)
    metas = [
        {
            "video_name": "boundary-microscope-roundtrip",
            "fps": 2.0,
            "snippet_stride": 4.0,
            "offset_frames": 0.0,
            "window_start_frame": 10.0,
            "duration": 200.0,
        }
    ]
    return inputs, masks, metas


def test_boundary_microscope_physical_grid_head_uses_selected_dense_positions_for_roundtrip():
    torch = _import_torch_or_skip()
    route_module, head_module, post_utils = _install_lightweight_runtime()
    selector = route_module.BoundaryMicroscopeAcquisitionRoute(
        target_len=32,
        dense_window_size=64,
        microscope_radius=2,
        microscope_stride=1,
        anchor_stride=16,
        max_dense_gap=8,
        route_label=ROUTE_LABEL,
    )
    inputs, masks, metas = _make_boundary_frames(torch)

    selected_outputs = selector.forward_test(inputs, masks, metas)
    selected_meta = selected_outputs["metas"][0]
    selected_positions = torch.as_tensor(
        selected_meta["irregular_selected_positions"],
        dtype=torch.float32,
    )
    selected_mask = selected_outputs["masks"]
    selected_len = int(selected_mask.sum().item())
    assert selected_len == selected_positions.numel()
    assert selected_len <= selected_outputs["selected_positions"].shape[1]
    assert selected_meta["boundary_microscope_detector_input_positions"] == selected_outputs[
        "selected_positions"
    ][0].tolist()
    assert selected_meta["boundary_microscope_true_observation_positions"] == selected_positions.tolist()
    assert 8.0 in selected_positions.tolist()
    assert 22.0 in selected_positions.tolist()

    head = object.__new__(head_module.AnchorFreeHead)
    head.physical_grid_actionformer = True
    head.temporal_grid_positions_key = "irregular_selected_positions"
    head.temporal_grid_valid_len_key = "irregular_selected_valid_len"
    head.temporal_grid_required = True
    head.temporal_grid_strict = True
    head.prior_generator = types.SimpleNamespace(regression_range=[(0, 10000)])

    output_len = int(selected_outputs["inputs"].shape[2])
    assert output_len == 64
    assert selected_mask.shape == (1, output_len)
    assert selected_mask[0, :selected_len].all()
    assert not selected_mask[0, selected_len:].any()

    feat_list = (torch.zeros((1, 4, output_len), dtype=torch.float32),)
    mask_list = (selected_mask,)
    points = head._points_for_feature_geometry(feat_list, mask_list, metas=selected_outputs["metas"], mark_native_axis=True)

    assert selected_meta["irregular_native_axis"] is True
    assert selected_meta["physical_grid_actionformer"] is True
    assert torch.equal(points[0][0, :selected_len, 0], selected_positions)
    assert torch.equal(
        points[0][0, selected_len:, 0],
        torch.full((output_len - selected_len,), float(selected_positions[-1])),
    )

    start_slot = int((selected_positions == 8.0).nonzero(as_tuple=True)[0][0].item())
    stride = points[0][0, start_slot, 3].clamp_min(1.0e-4)
    reg_pred = [torch.zeros((1, 2, output_len), dtype=torch.float32)]
    reg_pred[0][0, 1, start_slot] = (22.0 - 8.0) / stride
    proposals = head.get_refined_proposals(points, reg_pred)

    assert torch.allclose(proposals[0, start_slot], torch.tensor([8.0, 22.0]), atol=1.0e-4)
    seconds = post_utils.convert_to_seconds(proposals[0, start_slot : start_slot + 1].clone(), selected_meta)
    assert torch.allclose(seconds[0], torch.tensor([21.0, 49.0]), atol=1.0e-4)
