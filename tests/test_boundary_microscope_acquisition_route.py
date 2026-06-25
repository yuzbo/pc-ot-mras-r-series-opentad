from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "opentad" / "models" / "selectors" / "boundary_microscope_acquisition_route.py"


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
    sys.modules.pop("opentad.models.selectors.boundary_microscope_acquisition_route", None)
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.boundary_microscope_acquisition_route",
        ROUTE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_boundary_frames(torch, *, batch: int = 2, dense_len: int = 64):
    signal = torch.zeros(dense_len, dtype=torch.float32)
    signal[8:22] = 4.0
    signal[39:50] = 3.0
    frames = signal.view(1, 1, dense_len, 1, 1).expand(batch, 3, dense_len, 3, 3).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"boundary-microscope-{idx}"} for idx in range(batch)]
    gt_segments = [
        torch.tensor([[8.0, 22.0], [39.0, 50.0]], dtype=torch.float32)
        for _idx in range(batch)
    ]
    gt_labels = [torch.tensor([1, 2], dtype=torch.long) for _idx in range(batch)]
    return frames, masks, metas, gt_segments, gt_labels


def _make_boundary_frames_6d(torch, *, batch: int = 2, dense_len: int = 64):
    signal = torch.zeros(dense_len, dtype=torch.float32)
    signal[8:22] = 4.0
    signal[39:50] = 3.0
    batch_offset = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1, 1, 1) * 1000.0
    view_offset = torch.arange(2, dtype=torch.float32).view(1, 2, 1, 1, 1, 1) * 100.0
    channel_offset = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1, 1) * 10.0
    height_offset = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 1, 3, 1)
    width_offset = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 1, 1, 3) * 0.1
    frames = (
        signal.view(1, 1, 1, dense_len, 1, 1)
        + batch_offset
        + view_offset
        + channel_offset
        + height_offset
        + width_offset
    ).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"boundary-microscope-6d-{idx}"} for idx in range(batch)]
    gt_segments = [
        torch.tensor([[8.0, 22.0], [39.0, 50.0]], dtype=torch.float32)
        for _idx in range(batch)
    ]
    gt_labels = [torch.tensor([1, 2], dtype=torch.long) for _idx in range(batch)]
    return frames, masks, metas, gt_segments, gt_labels


def test_boundary_microscope_selects_dense_packets_around_scanned_hazards():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(
        target_len=32,
        dense_window_size=64,
        microscope_radius=2,
        microscope_stride=1,
        anchor_stride=16,
        max_dense_gap=8,
    )
    inputs, masks, metas, _gt_segments, _gt_labels = _make_boundary_frames(torch)

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["boundary_microscope_selected_dense_indices"]
    plan = outputs["metas"][0]["boundary_microscope_acquisition_plan"]
    assert outputs["inputs"].shape == (2, 3, 64, 3, 3)
    assert outputs["masks"].shape == (2, 64)
    assert outputs["masks"].dtype == torch.bool
    assert outputs["masks"][0, : len(selected)].all()
    assert not outputs["masks"][0, len(selected) :].any()
    assert int(outputs["masks"][0].sum().item()) == len(selected)
    assert selected == sorted(set(selected))
    assert len(selected) <= 32
    for boundary in (8, 22, 39, 50):
        assert any(abs(pos - boundary) <= 2 for pos in selected), (boundary, selected)
    assert plan["route_label"] == "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
    assert plan["meta_key"] == "boundary_microscope_acquisition_plan"
    assert plan["uses_gt"] is False
    assert plan["uses_teacher"] is False
    assert plan["uses_raw_prediction_cache"] is False
    assert "start_hazard_positions" in plan
    assert "end_hazard_positions" in plan
    assert plan["raw_input_temporal_len"] == 64
    assert plan["true_observation_count"] == len(selected)
    assert plan["padding_count"] == 64 - len(selected)
    assert plan["padding_slots_are_invalid"] is True


def test_boundary_microscope_supports_videomae_6d_raw_layout_and_temporal_geometry():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(
        target_len=32,
        dense_window_size=64,
        microscope_radius=2,
        microscope_stride=1,
        anchor_stride=16,
        max_dense_gap=8,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_boundary_frames_6d(torch)

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    selected = outputs["metas"][0]["boundary_microscope_selected_dense_indices"]
    selected_len = len(selected)
    assert outputs["inputs"].shape == (2, 2, 3, 64, 3, 3)
    assert outputs["masks"].shape == (2, 64)
    assert outputs["masks"].dtype == torch.bool
    assert outputs["masks"][:, :selected_len].all()
    assert not outputs["masks"][:, selected_len:].any()
    assert outputs["selected_positions"].shape == (2, 64)
    assert outputs["irregular_selected_positions"].shape == (2, 64)
    assert torch.equal(outputs["selected_positions"], outputs["irregular_selected_positions"])
    assert torch.equal(outputs["selected_output_valid_lengths"], torch.tensor([selected_len, selected_len]))
    assert torch.equal(outputs["irregular_selected_output_valid_len"], outputs["selected_output_valid_lengths"])
    assert torch.equal(outputs["irregular_selected_valid_len"], torch.tensor([64, 64]))
    assert outputs["selected_positions"][0, :selected_len].tolist() == [float(pos) for pos in selected]
    assert outputs["selected_positions"][0, selected_len:].tolist() == [float(selected[-1])] * (64 - selected_len)
    assert outputs["metas"][0]["irregular_selected_positions"] == [float(pos) for pos in selected]
    assert outputs["metas"][0]["boundary_microscope_true_observation_positions"] == [float(pos) for pos in selected]
    assert outputs["metas"][0]["boundary_microscope_detector_input_positions"] == outputs[
        "selected_positions"
    ][0].tolist()
    assert outputs["metas"][0]["irregular_selected_output_valid_len"] == float(selected_len)
    assert outputs["metas"][0]["irregular_selected_valid_len"] == 64.0
    assert outputs["metas"][0]["boundary_microscope_raw_input_temporal_len"] == 64
    assert outputs["metas"][0]["boundary_microscope_padding_count"] == 64 - selected_len
    assert outputs["metas"][0]["boundary_microscope_padding_slots_are_invalid"] is True
    plan = outputs["metas"][0]["boundary_microscope_acquisition_plan"]
    assert plan["input_layout"] == "[B,N,C,T,H,W]"
    assert plan["input_temporal_axis"] == 3
    assert plan["budget_contract"] == "videomae_dense_raw_len_with_prefix_true_observation_mask"
    assert plan["padding_not_gt_teacher_or_cache"] is True
    assert plan["detector_input_positions_len"] == 64
    assert plan["irregular_meta_positions_are_true_observation_prefix"] is True
    assert int(outputs["selected_output_valid_lengths"][0].item()) <= outputs["selected_positions"].shape[1]
    gather = outputs["selected_positions"].long()
    expected = torch.stack(
        [inputs[batch_idx, :, :, gather[batch_idx], :, :] for batch_idx in range(inputs.shape[0])],
        dim=0,
    )
    assert torch.equal(outputs["inputs"], expected)
    for boundary in (8, 22, 39, 50):
        assert any(abs(pos - boundary) <= 2 for pos in selected), (boundary, selected)
    assert outputs["gt_segments"] is gt_segments
    assert outputs["gt_labels"] is gt_labels


def test_boundary_microscope_keeps_interior_background_anchors_and_max_gap():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(
        target_len=40,
        dense_window_size=64,
        microscope_radius=1,
        microscope_stride=1,
        anchor_stride=20,
        max_dense_gap=6,
    )
    inputs, masks, metas, _gt_segments, _gt_labels = _make_boundary_frames(torch, batch=1)

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["boundary_microscope_selected_dense_indices"]
    roles = outputs["metas"][0]["boundary_microscope_selected_roles"]
    gaps = [right - left for left, right in zip(selected[:-1], selected[1:])]
    assert max(gaps) <= 6
    assert any(role in {"interior_anchor", "background_anchor", "gap_guard_anchor"} for role in roles)
    assert any(12 <= pos <= 18 for pos, role in zip(selected, roles) if role == "interior_anchor")
    assert any(pos <= 4 or pos >= 55 for pos, role in zip(selected, roles) if role == "background_anchor")


def test_boundary_microscope_forward_train_and_test_shapes_match_prefix_masks():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(
        target_len=36,
        dense_window_size=64,
        microscope_radius=2,
        anchor_stride=12,
        max_dense_gap=8,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_boundary_frames(torch, batch=1)

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, [{"sample_id": "shape-test"}])

    assert train_outputs["inputs"].shape[:2] == (1, 3)
    assert train_outputs["inputs"].shape[2] == 64
    assert train_outputs["masks"].shape == (1, 64)
    assert int(train_outputs["masks"].sum().item()) == len(
        train_outputs["metas"][0]["boundary_microscope_selected_dense_indices"]
    )
    assert test_outputs["inputs"].shape[:2] == (1, 3)
    assert test_outputs["inputs"].shape[2] == 64
    assert test_outputs["masks"].shape == (1, 64)
    assert int(test_outputs["masks"].sum().item()) == len(
        test_outputs["metas"][0]["boundary_microscope_selected_dense_indices"]
    )
    assert train_outputs["gt_segments"] is gt_segments
    assert train_outputs["gt_labels"] is gt_labels
    assert "boundary_microscope_acquisition_plan" in train_outputs["metas"][0]


def test_boundary_microscope_raw_videomae_path_pads_to_dense_window_len_for_768_window():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(
        target_len=384,
        dense_window_size=768,
        microscope_radius=3,
        microscope_stride=1,
        anchor_stride=24,
        max_dense_gap=8,
        max_start_hazards=4,
        max_end_hazards=4,
    )
    dense_len = 768
    signal = torch.zeros(dense_len, dtype=torch.float32)
    signal[96:160] = 2.0
    signal[420:480] = 3.0
    batch_offsets = torch.arange(2, dtype=torch.float32).view(2, 1, 1, 1, 1, 1) * 1000.0
    inputs = signal.view(1, 1, 1, dense_len, 1, 1).expand(2, 1, 3, dense_len, 2, 2).contiguous()
    inputs = inputs + batch_offsets
    masks = torch.ones((2, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"fixed-raw-len-{idx}"} for idx in range(2)]

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["boundary_microscope_selected_dense_indices"]
    real_count = len(selected)
    assert real_count < 384
    assert outputs["inputs"].shape == (2, 1, 3, 768, 2, 2)
    assert outputs["masks"].shape == (2, 768)
    assert outputs["selected_positions"].shape == (2, 768)
    assert outputs["irregular_selected_positions"].shape == (2, 768)
    assert int(outputs["masks"][0].sum().item()) == real_count
    assert outputs["masks"][0, :real_count].all()
    assert not outputs["masks"][0, real_count:].any()
    assert outputs["selected_positions"][0, :real_count].tolist() == [float(pos) for pos in selected]
    assert outputs["selected_positions"][0, real_count:].tolist() == [float(selected[-1])] * (768 - real_count)
    assert outputs["metas"][0]["irregular_selected_positions"] == [float(pos) for pos in selected]
    assert outputs["metas"][0]["boundary_microscope_true_observation_positions"] == [float(pos) for pos in selected]
    assert outputs["metas"][0]["boundary_microscope_detector_input_positions"] == outputs[
        "selected_positions"
    ][0].tolist()
    assert outputs["metas"][0]["boundary_microscope_raw_input_temporal_len"] == 768
    assert outputs["metas"][0]["boundary_microscope_padding_count"] == 768 - real_count
    assert outputs["metas"][0]["boundary_microscope_padding_slots_are_invalid"] is True
    plan = outputs["metas"][0]["boundary_microscope_acquisition_plan"]
    assert plan["route_label"] == "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
    assert plan["selection_surface"] == "pre_backbone_raw_frame"
    assert plan["raw_input_temporal_len"] == 768
    assert plan["true_observation_count"] == real_count
    assert plan["padding_count"] == 768 - real_count
    assert plan["padding_slots_are_invalid"] is True
    assert plan["padding_not_gt_teacher_or_cache"] is True
    assert plan["detector_input_positions_len"] == 768
    assert plan["irregular_meta_positions_are_true_observation_prefix"] is True
    serialized = repr(outputs["metas"][0]).lower()
    for forbidden in ("c3-pro", "pc_ot_mras_prebackbone", "bh-sdc", "event-surprise", "event surprise", "frame-token", "frame_token", "combo"):
        assert forbidden not in serialized


def test_boundary_microscope_forward_test_rejects_forbidden_meta_shortcuts():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(target_len=16, dense_window_size=64)
    inputs, masks, _metas, _gt_segments, _gt_labels = _make_boundary_frames(torch, batch=1)

    with pytest.raises(ValueError, match="forbidden test-time meta"):
        selector.forward_test(inputs, masks, [{"raw_prediction_cache": "must-not-use"}])

    with pytest.raises(ValueError, match="forbidden test-time meta"):
        selector.forward_test(inputs, masks, [{"note": "teacher shortcut must-not-use"}])


def test_boundary_microscope_rejects_route_drift_and_dense_window_mismatch():
    torch = _import_torch_or_skip()
    module = _load_route_module()

    for route_label in (
        "C3-Pro",
        "BH-SDC",
        "EventSurpriseTemporalAcquisitionSelector",
        "frame-token-hybrid",
        "Boundary-Microscope+C3-combo",
    ):
        with pytest.raises(ValueError, match="route_label"):
            module.BoundaryMicroscopeAcquisitionRoute(
                dense_window_size=64,
                route_label=route_label,
            )
    with pytest.raises(ValueError, match="meta_key"):
        module.BoundaryMicroscopeAcquisitionRoute(
            dense_window_size=64,
            meta_key="c3_boundary_plan",
        )

    selector = module.BoundaryMicroscopeAcquisitionRoute(target_len=16, dense_window_size=63)
    inputs, masks, metas, _gt_segments, _gt_labels = _make_boundary_frames(torch, batch=1)
    with pytest.raises(ValueError, match="dense_window_size"):
        selector.forward_test(inputs, masks, metas)


def test_boundary_microscope_exposes_live_selector_cache_guard():
    _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(dense_window_size=64)

    assert selector.forbid_raw_prediction_cache is True


def test_boundary_microscope_metadata_does_not_drift_to_other_routes():
    torch = _import_torch_or_skip()
    module = _load_route_module()
    selector = module.BoundaryMicroscopeAcquisitionRoute(target_len=32, dense_window_size=64)
    inputs, masks, metas, _gt_segments, _gt_labels = _make_boundary_frames(torch, batch=1)

    outputs = selector.forward_test(inputs, masks, metas)

    serialized_meta = repr(outputs["metas"][0]).lower()
    for forbidden in ("bh-sdc", "event-surprise", "event surprise", "frame-token", "frame_token", "combo"):
        assert forbidden not in serialized_meta
