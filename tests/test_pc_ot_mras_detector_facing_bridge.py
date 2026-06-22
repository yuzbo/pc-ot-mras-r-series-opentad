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
    pytest.skip("torch unavailable", allow_module_level=True)

import torch

from pc_ot_mras_test_utils import load_pc_ot_mras_classes


_PCOTMRASReader, PCOTMRASDetectorBridge = load_pc_ot_mras_classes()
ROOT = Path(__file__).resolve().parents[1]


def _load_temporal_grid_from_metas():
    spec = importlib.util.spec_from_file_location(
        "pc_ot_mras_temporal_grid_for_test",
        ROOT / "opentad" / "models" / "utils" / "temporal_grid.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.temporal_grid_from_metas


temporal_grid_from_metas = _load_temporal_grid_from_metas()


def _source_features():
    torch.manual_seed(20260618)
    features = torch.randn(2, 4, 6, requires_grad=True)
    masks = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    return features, masks


def _reader_outputs(features, masks):
    batch, channels, time = features.shape
    slots = 3
    allocation = torch.zeros(batch, slots, time, dtype=features.dtype, device=features.device)
    allocation[0, 0, 0:2] = torch.tensor([0.70, 0.30], dtype=features.dtype, device=features.device)
    allocation[0, 1, 2:5] = torch.tensor([0.20, 0.50, 0.30], dtype=features.dtype, device=features.device)
    allocation[0, 2, 4:6] = torch.tensor([0.25, 0.75], dtype=features.dtype, device=features.device)
    allocation[1, 0, 0:2] = torch.tensor([0.60, 0.40], dtype=features.dtype, device=features.device)
    allocation[1, 1, 1:4] = torch.tensor([0.20, 0.20, 0.60], dtype=features.dtype, device=features.device)
    allocation[1, 2, 2:4] = torch.tensor([0.50, 0.50], dtype=features.dtype, device=features.device)
    selected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool, device=features.device)
    allocation = allocation * selected_mask.unsqueeze(-1).to(dtype=allocation.dtype)
    allocation.requires_grad_()
    gates = torch.tensor(
        [
            [0.80, 0.65, 0.50],
            [0.75, 0.55, 0.20],
        ],
        dtype=features.dtype,
        device=features.device,
    )
    matrix = allocation * gates[:, :, None]
    matrix.retain_grad()
    selected_tokens = torch.randn(batch, slots, channels, dtype=features.dtype, device=features.device, requires_grad=True)
    base = torch.tensor(
        [
            [0.15, 0.50, 0.85],
            [0.10, 0.45, 0.00],
        ],
        dtype=features.dtype,
        device=features.device,
    )
    return {
        "acquisition_matrix": matrix,
        "allocation": allocation,
        "selected_tokens": selected_tokens,
        "selected_times": base,
        "centers": base + 0.01,
        "widths": torch.full((batch, slots), 0.20, dtype=features.dtype, device=features.device),
        "gates": gates,
        "selected_mask": selected_mask,
        "valid_mask": masks,
        "valid_lengths": masks.long().sum(dim=1),
        "validLengths": masks.long().sum(dim=1),
    }


def _metas(reader_outputs):
    return [
        {"sample_id": "synthetic-0", "pc_ot_mras_reader_outputs": reader_outputs},
        {"sample_id": "synthetic-1"},
    ]


def _assert_prefix(mask):
    valid_count = mask.long().sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < valid_count[:, None]
    assert torch.equal(mask, expected)


def test_actionformer_style_neck_call_returns_tuple_and_continuous_aux():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)

    feat_out, mask_out, meta_out = bridge((features,), (masks,), metas=_metas(reader_outputs))

    assert isinstance(feat_out, tuple)
    assert isinstance(mask_out, tuple)
    assert isinstance(meta_out, list)
    assert feat_out[0].shape == (2, 5, 3)
    assert mask_out[0].shape == (2, 3)
    assert mask_out[0].dtype == torch.bool
    _assert_prefix(mask_out[0])
    assert torch.all(feat_out[0][1, :, 2] == 0)

    aux = meta_out[0]["pc_ot_mras_bridge"]
    aux_1 = meta_out[1]["pc_ot_mras_bridge"]
    assert aux_1 is not aux
    assert aux["continuous_axis"] is True
    assert aux["uses_hard_gather"] is False
    assert aux["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert aux["selected_tokens_source_verified"] is True
    assert aux["selected_tokens_masked"] is True
    assert "selected_times" in aux
    assert "centers" in aux
    assert "selected_dense_positions" in aux
    assert "dense_valid_len_tensor" in aux
    assert "widths" in aux
    assert "gates" in aux
    assert "acquisition_matrix" in aux
    assert aux["selected_tokens"].shape == (3, 4)
    assert aux_1["selected_tokens"].shape == (3, 4)
    assert aux_1["selected_mask"].shape == (3,)
    assert aux["selected_dense_positions"].shape == (3,)
    assert aux_1["selected_dense_positions"].shape == (3,)
    assert torch.allclose(aux["selected_dense_positions"][:3], torch.tensor(meta_out[0]["irregular_selected_positions"]))
    assert torch.allclose(aux_1["selected_dense_positions"][:2], torch.tensor(meta_out[1]["irregular_selected_positions"]))
    assert torch.allclose(aux["dense_valid_len_tensor"], torch.tensor(6.0))
    assert torch.allclose(aux_1["dense_valid_len_tensor"], torch.tensor(4.0))
    assert aux["temporal_tensor_metadata_mode"] == "selected_dense_positions_from_centers"
    assert aux_1["temporal_tensor_metadata_mode"] == "selected_dense_positions_from_centers"
    assert aux_1["acquisition_matrix"].shape == (3, 6)
    assert torch.all(aux_1["selected_times"][2] == 0)
    assert torch.all(aux_1["centers"][2] == 0)
    assert torch.all(aux_1["widths"][2] == 0)
    assert torch.all(aux_1["gates"][2] == 0)
    assert torch.all(aux_1["acquisition_matrix"][2] == 0)
    assert torch.all(aux_1["allocation"][2] == 0)
    assert meta_out[0]["irregular_selected_count"] == 3
    assert meta_out[0]["irregular_dense_valid_len"] == 6
    assert meta_out[0]["irregular_selected_valid_len"] == 6
    assert meta_out[0]["irregular_native_axis"] is True
    assert meta_out[0]["pc_ot_mras_temporal_meta_mode"] == "ordered_slot_centers_continuous"
    assert len(meta_out[0]["irregular_selected_positions"]) == 3
    assert all(0.0 <= pos < 6.0 for pos in meta_out[0]["irregular_selected_positions"])
    assert all(
        left < right
        for left, right in zip(
            meta_out[0]["irregular_selected_positions"],
            meta_out[0]["irregular_selected_positions"][1:],
        )
    )
    assert meta_out[1]["irregular_selected_count"] == 2
    assert meta_out[1]["irregular_dense_valid_len"] == 4
    assert meta_out[1]["irregular_selected_valid_len"] == 4
    assert meta_out[1]["irregular_native_axis"] is True
    assert len(meta_out[1]["irregular_selected_positions"]) == 2
    assert all(0.0 <= pos < 4.0 for pos in meta_out[1]["irregular_selected_positions"])

    grid = temporal_grid_from_metas(meta_out, mask_out[0], required=True, strict=True)
    assert torch.equal(grid["valid_mask"], mask_out[0])
    assert torch.allclose(
        grid["center"][0, :3],
        torch.tensor(meta_out[0]["irregular_selected_positions"], dtype=torch.float32, device=grid["center"].device),
    )
    assert torch.allclose(
        grid["center"][1, :2],
        torch.tensor(meta_out[1]["irregular_selected_positions"], dtype=torch.float32, device=grid["center"].device),
    )
    assert torch.allclose(grid["dense_valid_len"], torch.tensor([6.0, 4.0], device=grid["dense_valid_len"].device))

    expected_tokens = torch.bmm(reader_outputs["acquisition_matrix"], features.transpose(1, 2).contiguous())
    expected_masked = expected_tokens.masked_fill(~reader_outputs["selected_mask"].unsqueeze(-1), 0.0)
    assert torch.allclose(aux["selected_tokens"], expected_masked[0], atol=1e-6)
    assert torch.allclose(aux_1["selected_tokens"], expected_masked[1], atol=1e-6)
    assert torch.all(aux_1["selected_tokens"][2] == 0)
    assert not torch.allclose(aux["selected_tokens"], reader_outputs["selected_tokens"][0], atol=1e-3)

    loss = feat_out[0].square().mean() + aux["selected_tokens"].square().mean() + aux_1["selected_tokens"].square().mean()
    loss.backward()
    assert bridge.token_proj.weight.grad is not None
    assert torch.isfinite(bridge.token_proj.weight.grad).all()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0
    assert reader_outputs["acquisition_matrix"].grad is not None
    assert torch.isfinite(reader_outputs["acquisition_matrix"].grad).all()
    assert reader_outputs["acquisition_matrix"].grad.abs().sum().item() > 0


def test_standard_neck_accepts_real_reader_outputs_with_valid_lengths_and_temporal_grid():
    reader = _PCOTMRASReader(in_dim=4, hidden_dim=8, num_slots=3, num_blocks=1, num_roles=6)
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()

    reader_outputs = reader(features.transpose(1, 2).contiguous(), masks)
    assert "valid_lengths" in reader_outputs
    feat_out, mask_out, meta_out = bridge((features,), (masks,), metas=_metas(reader_outputs))

    assert feat_out[0].shape == (2, 5, 3)
    assert torch.equal(mask_out[0], reader_outputs["selected_mask"])
    aux = meta_out[0]["pc_ot_mras_bridge"]
    assert aux["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert aux["selected_tokens_source_verified"] is True
    assert aux["uses_hard_gather"] is False

    grid = temporal_grid_from_metas(meta_out, mask_out[0], required=True, strict=True)
    assert torch.equal(grid["valid_mask"], mask_out[0])
    assert torch.allclose(grid["dense_valid_len"], torch.tensor([6.0, 4.0], device=grid["dense_valid_len"].device))


def test_standard_neck_allocation_key_reports_matching_token_source():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5, allocation_key="allocation")
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)

    feat_out, mask_out, meta_out = bridge((features,), (masks,), metas=_metas(reader_outputs))

    aux = meta_out[0]["pc_ot_mras_bridge"]
    aux_1 = meta_out[1]["pc_ot_mras_bridge"]
    expected_tokens = torch.bmm(reader_outputs["allocation"], features.transpose(1, 2).contiguous())
    expected_masked = expected_tokens.masked_fill(~reader_outputs["selected_mask"].unsqueeze(-1), 0.0)
    assert feat_out[0].shape == (2, 5, 3)
    assert torch.equal(mask_out[0], reader_outputs["selected_mask"])
    assert aux["selected_tokens_source"] == "recomputed_from_allocation"
    assert aux["selected_tokens_source_verified"] is True
    assert torch.allclose(aux["selected_tokens"], expected_masked[0], atol=1e-6)
    assert torch.allclose(aux_1["selected_tokens"], expected_masked[1], atol=1e-6)


def test_metadata_position_source_selected_times_changes_temporal_positions_only_when_enabled():
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)
    default_bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    selected_times_bridge = PCOTMRASDetectorBridge(
        in_channels=4,
        out_channels=5,
        metadata_position_source="selected_times",
    )

    _default_feats, default_masks, default_meta = default_bridge((features,), (masks,), metas=_metas(reader_outputs))
    _selected_feats, selected_masks, selected_meta = selected_times_bridge(
        (features,),
        (masks,),
        metas=_metas(reader_outputs),
    )

    assert torch.equal(default_masks[0], selected_masks[0])
    default_aux = default_meta[0]["pc_ot_mras_bridge"]
    selected_aux = selected_meta[0]["pc_ot_mras_bridge"]
    expected_centers = reader_outputs["centers"][0] * float(masks[0].sum().item())
    expected_selected_times = reader_outputs["selected_times"][0] * float(masks[0].sum().item() - 1)

    assert default_aux["temporal_tensor_metadata_mode"] == "selected_dense_positions_from_centers"
    assert selected_aux["temporal_tensor_metadata_mode"] == "selected_dense_positions_from_selected_times"
    assert selected_aux["selected_times_dense_position_scale"] == "valid_len_minus_one_token_index"
    assert torch.allclose(default_aux["selected_dense_positions"], expected_centers, atol=1e-6)
    assert torch.allclose(selected_aux["selected_dense_positions"], expected_selected_times, atol=1e-6)
    assert not torch.allclose(default_aux["selected_dense_positions"], selected_aux["selected_dense_positions"])
    assert default_meta[0]["irregular_selected_positions"] == pytest.approx(
        default_aux["selected_dense_positions"].detach().cpu().tolist()
    )
    assert selected_meta[0]["irregular_selected_positions"] == pytest.approx(
        selected_aux["selected_dense_positions"].detach().cpu().tolist()
    )
    assert default_meta[0]["pc_ot_mras_temporal_meta_mode"] == "ordered_slot_centers_continuous"
    assert selected_meta[0]["pc_ot_mras_temporal_meta_mode"] == "selected_times_continuous"
    assert selected_meta[0]["pc_ot_mras_selected_times_dense_position_scale"] == "valid_len_minus_one_token_index"


def test_metadata_position_source_selected_times_fails_closed_when_nonmonotonic():
    features, masks = _source_features()
    reader_outputs = dict(_reader_outputs(features, masks))
    reader_outputs["selected_times"] = reader_outputs["selected_times"].clone()
    reader_outputs["selected_times"][0] = torch.tensor(
        [0.60, 0.40, 0.90],
        dtype=features.dtype,
        device=features.device,
    )
    selected_times_bridge = PCOTMRASDetectorBridge(
        in_channels=4,
        out_channels=5,
        metadata_position_source="selected_times",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        selected_times_bridge((features,), (masks,), metas=_metas(reader_outputs))


def test_metadata_position_source_selected_times_sort_jitter_repairs_nonmonotonic():
    features, masks = _source_features()
    reader_outputs = dict(_reader_outputs(features, masks))
    reader_outputs["selected_times"] = reader_outputs["selected_times"].clone()
    reader_outputs["selected_times"][0] = torch.tensor(
        [0.60, 0.40, 0.40],
        dtype=features.dtype,
        device=features.device,
    )
    selected_times_bridge = PCOTMRASDetectorBridge(
        in_channels=4,
        out_channels=5,
        metadata_position_source="selected_times",
        metadata_position_repair="sort_jitter",
    )

    _selected_feats, selected_masks, selected_meta = selected_times_bridge(
        (features,),
        (masks,),
        metas=_metas(reader_outputs),
    )

    assert torch.equal(selected_masks[0], reader_outputs["selected_mask"])
    aux = selected_meta[0]["pc_ot_mras_bridge"]
    positions = torch.tensor(selected_meta[0]["irregular_selected_positions"])
    assert bool(((positions[1:] - positions[:-1]) > 0).all().item())
    assert selected_meta[0]["pc_ot_mras_temporal_meta_mode"] == "selected_times_continuous"
    assert selected_meta[0]["metadata_position_repair_mode"] == "sort_jitter"
    assert aux["metadata_position_repair_mode"] == "sort_jitter"
    assert int(aux["selected_dense_position_pre_repair_strict_violation_count"][0].item()) > 0
    assert int(aux["selected_dense_position_jitter_repair_count"][0].item()) > 0
    assert int(aux["selected_dense_position_strict_violation_count"][0].item()) == 0
    assert selected_meta[0]["pc_ot_mras_selected_dense_position_strict_violation_count"] == 0


def test_no_gate_scale_recomputes_tokens_from_allocation_without_changing_default():
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)
    default_bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    no_gate_bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5, no_gate_scale=True)

    _default_feats, _default_masks, default_meta = default_bridge((features,), (masks,), metas=_metas(reader_outputs))
    _no_gate_feats, _no_gate_masks, no_gate_meta = no_gate_bridge((features,), (masks,), metas=_metas(reader_outputs))

    default_aux = default_meta[0]["pc_ot_mras_bridge"]
    no_gate_aux = no_gate_meta[0]["pc_ot_mras_bridge"]
    expected_default = torch.bmm(reader_outputs["acquisition_matrix"], features.transpose(1, 2).contiguous())
    expected_no_gate = torch.bmm(reader_outputs["allocation"], features.transpose(1, 2).contiguous())

    assert default_bridge.no_gate_scale is False
    assert default_bridge.allocation_key == "acquisition_matrix"
    assert no_gate_bridge.no_gate_scale is True
    assert no_gate_bridge.allocation_key == "allocation"
    assert default_aux["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert no_gate_aux["selected_tokens_source"] == "recomputed_from_allocation"
    assert torch.allclose(default_aux["selected_tokens"], expected_default[0], atol=1e-6)
    assert torch.allclose(no_gate_aux["selected_tokens"], expected_no_gate[0], atol=1e-6)
    assert not torch.allclose(default_aux["selected_tokens"], no_gate_aux["selected_tokens"])


def test_legacy_selected_tokens_path_still_backprops_to_reader_selected_tokens():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features.detach(), masks)

    feat_out, mask_out, aux = bridge(reader_outputs=reader_outputs, return_aux=True)
    assert feat_out[0].shape == (2, 5, 3)
    _assert_prefix(mask_out[0])
    assert aux["uses_hard_gather"] is False
    assert aux["selected_tokens_source"] == "reader_outputs_selected_tokens_unverified"
    assert aux["selected_tokens_source_verified"] is False
    assert aux["selected_tokens_masked"] is True
    assert torch.all(aux["selected_tokens"][1, 2] == 0)

    loss = feat_out[0].sum() + aux["selected_tokens"].sum()
    loss.backward()
    assert bridge.token_proj.weight.grad is not None
    assert reader_outputs["selected_tokens"].grad is not None
    assert torch.isfinite(reader_outputs["selected_tokens"].grad).all()
    assert reader_outputs["selected_tokens"].grad.abs().sum().item() > 0


def test_standard_neck_missing_metas_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    with pytest.raises(ValueError, match="metas must be provided"):
        bridge((features,), (masks,), metas=None)


def test_standard_neck_missing_reader_outputs_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    with pytest.raises(ValueError, match="pc_ot_mras_reader_outputs"):
        bridge((features,), (masks,), metas=[{}, {}])


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "gt_segments",
        "ground_truth_segments",
        "teacher_logits",
        "oracle_scores",
        "prediction_cache",
        "checkpoint_path",
        "result_detection",
        "raw_predictions",
        "gtSegments",
        "groundTruthSegments",
        "teacherLogits",
        "oracleScores",
        "predictionCache",
        "checkpointPath",
        "resultDetection",
        "rawPredictions",
        "gtsegments",
        "groundtruthsegments",
        "teacherlogits",
        "oraclescores",
        "predictioncache",
        "checkpointpath",
        "resultdetection",
        "rawpredictions",
        "source_teacherlogits",
        "source_gtsegments",
        "metadata_checkpointpath",
        "payloadteacherlogits",
        "metadatagtsegments",
        "xpredictioncache",
    ],
)
def test_standard_neck_forbidden_reader_payload_fails_closed(forbidden_key):
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)
    reader_outputs[forbidden_key] = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="forbidden reader payload"):
        bridge((features,), (masks,), metas=_metas(reader_outputs))


def test_standard_neck_nested_prefixed_forbidden_reader_payload_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)
    reader_outputs["diagnostics"] = {
        "source_teacherlogits": torch.zeros(2, 3),
    }

    with pytest.raises(ValueError, match="forbidden reader payload"):
        bridge((features,), (masks,), metas=_metas(reader_outputs))


def test_standard_neck_non_binary_mask_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)
    bad_masks = masks.float()
    bad_masks[0, 2] = 0.5
    with pytest.raises(ValueError, match="binary"):
        bridge((features,), (bad_masks,), metas=_metas(reader_outputs))


def test_standard_neck_shape_mismatch_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)
    reader_outputs["acquisition_matrix"] = reader_outputs["acquisition_matrix"][:, :, :-1]
    reader_outputs["allocation"] = reader_outputs["allocation"][:, :, :-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        bridge((features,), (masks,), metas=_metas(reader_outputs))


def test_standard_neck_metas_batch_mismatch_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)

    with pytest.raises(ValueError, match="metas length must match batch size"):
        bridge((features,), (masks,), metas=[{"sample_id": "only", "pc_ot_mras_reader_outputs": reader_outputs}])


def test_standard_neck_padding_allocation_mass_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=4, out_channels=5)
    features, masks = _source_features()
    reader_outputs = _reader_outputs(features, masks)
    reader_outputs["acquisition_matrix"] = reader_outputs["acquisition_matrix"].clone()
    reader_outputs["allocation"] = reader_outputs["acquisition_matrix"]
    reader_outputs["acquisition_matrix"][1, 0, 4] = 0.25

    with pytest.raises(ValueError, match="must not allocate mass to invalid/padded positions"):
        bridge((features,), (masks,), metas=_metas(reader_outputs))
