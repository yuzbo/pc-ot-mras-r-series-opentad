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


ROOT = Path(__file__).resolve().parents[1]
_PCOTMRASReader, PCOTMRASDetectorBridge = load_pc_ot_mras_classes()


def _load_function(module_name, file_name, function_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "opentad" / "models" / "utils" / file_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


temporal_grid_from_metas = _load_function(
    "pc_ot_mras_temporal_grid_contract_test",
    "temporal_grid.py",
    "temporal_grid_from_metas",
)
validate_sampling_contract = _load_function(
    "pc_ot_mras_sampling_contract_test",
    "sampling_contract.py",
    "validate_sampling_contract",
)


def _assert_prefix(mask):
    valid_count = mask.long().sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < valid_count[:, None]
    assert torch.equal(mask, expected)


def _expected_lengths(start_len, levels):
    lengths = [start_len]
    for _ in range(1, levels):
        lengths.append((lengths[-1] + 1) // 2)
    return lengths


def _synthetic_reader_outputs(features, dense_mask, selected_mask):
    batch, channels, dense_len = features.shape
    slots = selected_mask.shape[1]
    acquisition = torch.zeros(batch, slots, dense_len, dtype=features.dtype, device=features.device)
    acquisition[0, 0, 0] = 1.0
    acquisition[0, 1, 2:4] = torch.tensor([0.25, 0.75], dtype=features.dtype, device=features.device)
    acquisition[0, 2, 6] = 1.0
    acquisition[0, 3, 8:10] = torch.tensor([0.40, 0.60], dtype=features.dtype, device=features.device)
    acquisition[1, 0, 0:2] = torch.tensor([0.60, 0.40], dtype=features.dtype, device=features.device)
    acquisition[1, 1, 3] = 1.0
    acquisition[1, 2, 6:8] = torch.tensor([0.20, 0.80], dtype=features.dtype, device=features.device)

    centers = torch.tensor(
        [
            [0.10, 0.40, 0.65, 0.90],
            [0.125, 0.50, 0.875, 0.00],
        ],
        dtype=features.dtype,
        device=features.device,
    )
    return {
        "acquisition_matrix": acquisition,
        "allocation": acquisition,
        "selected_mask": selected_mask,
        "valid_mask": dense_mask,
        "selected_times": centers,
        "centers": centers,
        "widths": torch.full((batch, slots), 0.20, dtype=features.dtype, device=features.device),
        "gates": torch.ones(batch, slots, dtype=features.dtype, device=features.device),
        "selected_tokens": torch.zeros(batch, slots, channels, dtype=features.dtype, device=features.device),
        "valid_lengths": dense_mask.long().sum(dim=1),
    }


def _run_synthetic_bridge(output_strides=(1, 2, 4, 8)):
    bridge = PCOTMRASDetectorBridge(
        in_channels=3,
        out_channels=4,
        add_time_features=False,
        output_strides=output_strides,
    )
    features = torch.arange(2 * 3 * 10, dtype=torch.float32).reshape(2, 3, 10)
    dense_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    selected_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    reader_outputs = _synthetic_reader_outputs(features, dense_mask, selected_mask)
    metas = [
        {"sample_id": "pc_ot_mras_temporal_contract|0", "pc_ot_mras_reader_outputs": reader_outputs},
        {"sample_id": "pc_ot_mras_temporal_contract|1"},
    ]
    return bridge((features,), (dense_mask,), metas=metas)


def test_pc_ot_mras_metadata_separates_dense_len_selected_count_and_physical_positions():
    feats, masks, meta_out = _run_synthetic_bridge()

    assert [feat.shape[-1] for feat in feats] == _expected_lengths(4, 4)
    assert [mask.shape[-1] for mask in masks] == _expected_lengths(4, 4)
    assert [int(mask[0].sum().item()) for mask in masks] == [4, 2, 1, 1]
    assert [int(mask[1].sum().item()) for mask in masks] == [3, 2, 1, 1]
    for mask in masks:
        _assert_prefix(mask)

    assert meta_out[0]["pc_ot_mras_bridge"]["output_strides"] == "(1, 2, 4, 8)"
    assert meta_out[1]["pc_ot_mras_bridge"]["output_strides"] == "(1, 2, 4, 8)"

    assert meta_out[0]["irregular_selected_count"] == 4
    assert meta_out[1]["irregular_selected_count"] == 3
    assert meta_out[0]["irregular_dense_valid_len"] == 10
    assert meta_out[1]["irregular_dense_valid_len"] == 8
    assert meta_out[0]["irregular_selected_valid_len"] == 10
    assert meta_out[1]["irregular_selected_valid_len"] == 8
    assert meta_out[0]["irregular_selected_valid_len"] != meta_out[0]["irregular_selected_count"]
    assert meta_out[1]["irregular_selected_valid_len"] != meta_out[1]["irregular_selected_count"]
    assert meta_out[0]["irregular_selected_valid_len_semantics"] == "carried_forward_dense_valid_len_alias"
    assert meta_out[1]["irregular_selected_valid_len_semantics"] == "carried_forward_dense_valid_len_alias"

    assert meta_out[0]["irregular_selected_positions"] == pytest.approx([1.0, 4.0, 6.5, 9.0])
    assert meta_out[1]["irregular_selected_positions"] == pytest.approx([1.0, 4.0, 7.0])
    assert meta_out[0]["irregular_selected_positions"] != [0.0, 1.0, 2.0, 3.0]
    assert meta_out[1]["irregular_selected_positions"] != [0.0, 1.0, 2.0]

    validate_sampling_contract(meta_out, masks[0], split="train")
    grid = temporal_grid_from_metas(meta_out, masks[0], required=True, strict=True)
    assert torch.equal(grid["valid_mask"], masks[0])
    assert torch.allclose(grid["dense_valid_len"], torch.tensor([10.0, 8.0], device=grid["dense_valid_len"].device))
    assert torch.allclose(grid["center"][0, :4], torch.tensor([1.0, 4.0, 6.5, 9.0]))
    assert torch.allclose(grid["center"][1, :3], torch.tensor([1.0, 4.0, 7.0]))


def test_pc_ot_mras_metadata_rejects_selected_valid_len_alias_mismatch():
    _feats, masks, meta_out = _run_synthetic_bridge()
    bad = [dict(item) for item in meta_out]
    bad[0]["irregular_selected_valid_len"] = bad[0]["irregular_selected_count"]

    with pytest.raises(ValueError, match="irregular_dense_valid_len must match"):
        validate_sampling_contract(bad, masks[0], split="train")
    with pytest.raises(ValueError, match="irregular_dense_valid_len must match"):
        temporal_grid_from_metas(bad, masks[0], required=True, strict=True)


def test_pc_ot_mras_bridge_level0_metadata_follows_selected_mask_prefix_only():
    _feats, masks, meta_out = _run_synthetic_bridge()
    aux_0 = meta_out[0]["pc_ot_mras_bridge"]
    aux_1 = meta_out[1]["pc_ot_mras_bridge"]

    assert torch.equal(aux_0["selected_mask"], masks[0][0])
    assert torch.equal(aux_1["selected_mask"], masks[0][1])
    assert aux_0["centers"].shape == (4,)
    assert aux_1["centers"].shape == (4,)
    assert torch.all(aux_1["centers"][3:] == 0)
    assert torch.all(aux_1["widths"][3:] == 0)
    assert torch.all(aux_1["gates"][3:] == 0)
    assert torch.all(aux_1["acquisition_matrix"][3:] == 0)

    assert len(meta_out[0]["irregular_selected_positions"]) == int(masks[0][0].sum().item())
    assert len(meta_out[1]["irregular_selected_positions"]) == int(masks[0][1].sum().item())
    assert max(meta_out[0]["irregular_selected_positions"]) < meta_out[0]["irregular_dense_valid_len"]
    assert max(meta_out[1]["irregular_selected_positions"]) < meta_out[1]["irregular_dense_valid_len"]
