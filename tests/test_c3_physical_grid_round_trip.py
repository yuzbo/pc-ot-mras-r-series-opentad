import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "tests" / "test_c3_physical_grid_actionformer_candidate.py"


def _load_helper():
    module_name = "c3_physical_grid_actionformer_candidate_helpers"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()
TORCH_AVAILABLE = helper.TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import numpy as np
    import torch

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch unavailable: dynamic C3 physical-grid round-trip evidence is environment-only",
)


def read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


def _ensure_package(name, path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_post_processing_utils():
    return _load_module(
        "c3_physical_grid_post_processing_utils",
        ROOT / "opentad" / "models" / "utils" / "post_processing" / "utils.py",
    )


def _load_loadframes_or_skip():
    if not TORCH_AVAILABLE:
        pytest.skip("torch unavailable: dynamic LoadFrames metadata check is environment-only")

    pandas = pytest.importorskip("pandas")
    assert pandas is not None

    package = "c3_physical_grid_dataset_runtime"
    _ensure_package(package, ROOT / "opentad")
    _ensure_package(f"{package}.datasets", ROOT / "opentad" / "datasets")
    _ensure_package(f"{package}.datasets.transforms", ROOT / "opentad" / "datasets" / "transforms")

    builder = types.ModuleType(f"{package}.datasets.builder")
    builder.PIPELINES = _Registry()
    sys.modules[f"{package}.datasets.builder"] = builder

    boundary = types.ModuleType(f"{package}.datasets.transforms.boundary_acquisition")
    boundary.load_value_transport_selection_ledger = lambda *args, **kwargs: {}
    sys.modules[f"{package}.datasets.transforms.boundary_acquisition"] = boundary

    module = _load_module(
        f"{package}.datasets.transforms.end_to_end",
        ROOT / "opentad" / "datasets" / "transforms" / "end_to_end.py",
    )
    return module.LoadFrames


def _head_with_zero_decode():
    head = helper._make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    return head


def _feat_mask_for_four_selected_plus_tail():
    return [torch.zeros(1, 2, 5)], [torch.tensor([[True, True, True, True, True]])]


def _round_trip_meta(**extra):
    meta = {
        "video_name": "synthetic_c3_round_trip",
        "fps": 25.0,
        "duration": 100.0,
        "snippet_stride": 2.0,
        "offset_frames": 5.0,
        "window_start_frame": 100.0,
        "irregular_selected_positions": [0.0, 3.0, 4.0, 10.0],
        "selected_dense_indices": [0, 3, 4, 10],
        "selected_valid_len": 4,
        "irregular_selected_valid_len": 12.0,
        "irregular_dense_valid_len": 12.0,
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
        "pc_ot_mras_prebackbone_remap_gt_to_selected_axis": False,
    }
    meta.update(extra)
    return meta


@requires_torch
def test_loadframes_records_nonuniform_dense_indices_as_native_physical_metadata():
    LoadFrames = _load_loadframes_or_skip()
    transform = LoadFrames(method="random_fixed_subsample", remap_gt_to_selected_axis=False, scale_factor=1)
    results = {}

    transform._set_irregular_axis_meta(results, np.asarray([0, 3, 4, 10], dtype=np.int64), valid_len=12)

    assert results["selected_dense_indices"].tolist() == [0.0, 3.0, 4.0, 10.0]
    assert results["irregular_selected_positions"].tolist() == [0.0, 3.0, 4.0, 10.0]
    assert results["selected_valid_len"] == 4
    assert results["irregular_selected_valid_len"] == 12.0
    assert results["irregular_dense_valid_len"] == 12.0
    assert results["remap_gt_to_selected_axis"] is False
    assert results["gt_remapped_to_selected_axis"] is False
    assert results["irregular_native_axis"] is True


@requires_torch
def test_physical_grid_train_targets_assign_on_dense_physical_gt_not_selected_axis():
    head = _head_with_zero_decode()
    feat_list, mask_list = _feat_mask_for_four_selected_plus_tail()
    meta = _round_trip_meta()
    points = head.prior_generator(feat_list)

    physical_points, physical_masks = head._build_physical_points_and_masks(
        points,
        mask_list,
        metas=[meta],
        train_mode=True,
    )
    gt_cls, gt_reg = head.prepare_targets(
        physical_points,
        gt_segments=[torch.tensor([[9.5, 10.5]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
    )

    assert physical_masks[0].tolist() == [[True, True, True, True, False]]
    assert torch.allclose(physical_points[0][0, :, 0], torch.tensor([0.0, 3.0, 4.0, 10.0, 12.0]))
    assert torch.equal(gt_cls[0].sum(dim=1) > 0, torch.tensor([False, False, False, True, False]))
    assert gt_cls[0][3, 1].item() == 1.0
    assert torch.allclose(gt_reg[0][3], torch.tensor([0.125, 0.125]))
    assert meta["irregular_native_axis"] is True
    assert meta["physical_grid_actionformer"] is True
    assert meta["physical_grid_dense_valid_len"] == 12.0


@requires_torch
@pytest.mark.parametrize(
    "bad_meta,match",
    [
        (_round_trip_meta(remap_gt_to_selected_axis=True), "selected-axis GT remap is forbidden"),
        (_round_trip_meta(irregular_native_axis=False), "irregular_native_axis must be explicitly True"),
        (
            {
                "video_name": "missing_positions",
                "irregular_native_axis": True,
                "remap_gt_to_selected_axis": False,
            },
            "requires irregular_selected_positions or selected_dense_indices",
        ),
    ],
)
def test_physical_grid_train_path_fails_closed_for_selected_axis_or_missing_metadata(bad_meta, match):
    head = _head_with_zero_decode()
    feat_list, mask_list = _feat_mask_for_four_selected_plus_tail()

    with pytest.raises(ValueError, match=match):
        head.forward_train(
            feat_list,
            mask_list,
            gt_segments=[torch.tensor([[9.5, 10.5]], dtype=torch.float32)],
            gt_labels=[torch.tensor([1], dtype=torch.long)],
            metas=[bad_meta],
        )


@requires_torch
def test_inference_decode_and_seconds_conversion_round_trip_stays_on_physical_dense_axis():
    post_utils = _load_post_processing_utils()
    head = _head_with_zero_decode()
    feat_list, mask_list = _feat_mask_for_four_selected_plus_tail()
    meta = _round_trip_meta()

    proposals, scores = head.forward_test(feat_list, mask_list, metas=[meta])

    expected_dense = torch.tensor(
        [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(proposals[0], expected_dense)
    assert scores[0].shape == (4, 2)
    assert meta["irregular_native_axis"] is True
    assert meta["physical_grid_actionformer"] is True

    seconds = post_utils.convert_to_seconds(proposals[0].clone(), meta)
    expected_seconds = (expected_dense * meta["snippet_stride"] + meta["window_start_frame"] + meta["offset_frames"]) / meta["fps"]
    assert torch.allclose(seconds, expected_seconds)

    double_remap_meta = dict(meta)
    double_remap_meta["irregular_native_axis"] = False
    double_remapped_seconds = post_utils.convert_to_seconds(proposals[0].clone(), double_remap_meta)
    assert not torch.allclose(double_remapped_seconds, expected_seconds)


def test_precheck_launcher_runs_round_trip_evidence_without_train_or_test_entrypoints():
    script = read("scripts/run_c3_physical_grid_actionformer_precheck.sh")
    forbidden_commands = (
        "tools/train.py",
        "tools/test.py",
        "sbatch",
        "srun",
        "ssh ",
        "scp ",
        "rsync",
    )

    assert "tests/test_c3_physical_grid_round_trip.py" in script
    for token in forbidden_commands:
        assert token not in script
    assert "PRECHECK_ONLY" in script
    assert "remote sync" in script or "remote_sync" in script
    assert "Slurm" in script or "slurm" in script
