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


_PCOTMRASReader, PCOTMRASDetectorBridge = load_pc_ot_mras_classes()


def _set_identity_projection(bridge):
    with torch.no_grad():
        bridge.token_proj.weight.zero_()
        bridge.token_proj.weight.copy_(torch.eye(bridge.out_channels, bridge.in_channels))
        bridge.token_proj.bias.zero_()


def _assert_prefix(mask):
    valid_count = mask.long().sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < valid_count[:, None]
    assert torch.equal(mask, expected)


def _manual_selected_reader_outputs():
    selected_tokens = torch.tensor(
        [
            [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0], [5.0, 15.0]],
            [[21.0, 31.0], [22.0, 32.0], [23.0, 33.0], [999.0, 999.0], [777.0, 777.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    return {
        "selected_tokens": selected_tokens,
        "selected_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.bool,
        ),
    }


def _source_reader_outputs(acquisition_matrix, dense_mask, selected_mask):
    batch, slots, dense_len = acquisition_matrix.shape
    base = torch.tensor(
        [
            [0.05, 0.25, 0.45, 0.65, 0.85],
            [0.10, 0.30, 0.50, 0.70, 0.00],
        ],
        dtype=acquisition_matrix.dtype,
        device=acquisition_matrix.device,
    )
    return {
        "acquisition_matrix": acquisition_matrix,
        "selected_mask": selected_mask,
        "valid_mask": dense_mask,
        "selected_times": base[:batch, :slots],
        "centers": base[:batch, :slots],
        "widths": torch.full((batch, slots), 0.20, dtype=acquisition_matrix.dtype, device=acquisition_matrix.device),
        "gates": torch.ones(batch, slots, dtype=acquisition_matrix.dtype, device=acquisition_matrix.device),
        "valid_lengths": dense_mask.long().sum(dim=1),
    }


def _acquisition_matrix():
    matrix = torch.zeros(2, 5, 7, dtype=torch.float32)
    matrix[0, 0, 0:2] = torch.tensor([0.70, 0.30])
    matrix[0, 1, 1:3] = torch.tensor([0.40, 0.60])
    matrix[0, 2, 3] = 1.0
    matrix[0, 3, 4:6] = torch.tensor([0.25, 0.75])
    matrix[0, 4, 6] = 1.0
    matrix[1, 0, 0:2] = torch.tensor([0.50, 0.50])
    matrix[1, 1, 1:3] = torch.tensor([0.25, 0.75])
    matrix[1, 2, 2:4] = torch.tensor([0.60, 0.40])
    matrix[1, 3, 3:5] = torch.tensor([0.35, 0.65])
    matrix.requires_grad_()
    return matrix


def _multilevel_source_inputs():
    level0_features = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
    level0_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    level1_features = torch.tensor(
        [
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
            ],
            [
                [40.0, 41.0, 42.0, 0.0],
                [50.0, 51.0, 52.0, 0.0],
                [60.0, 61.0, 62.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    level1_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    acquisition_matrix = torch.zeros(2, 3, 4, dtype=torch.float32)
    acquisition_matrix[0, 0, 0:2] = torch.tensor([0.75, 0.25])
    acquisition_matrix[0, 1, 1:3] = torch.tensor([0.40, 0.60])
    acquisition_matrix[0, 2, 2:4] = torch.tensor([0.30, 0.70])
    acquisition_matrix[1, 0, 0:2] = torch.tensor([0.60, 0.40])
    acquisition_matrix[1, 1, 1:3] = torch.tensor([0.20, 0.80])
    selected_mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=torch.bool,
    )
    reader_outputs = _source_reader_outputs(acquisition_matrix, level1_mask, selected_mask)
    metas = [
        {"sample_id": "pc-ot-mras-r13-source-level-0", "pc_ot_mras_reader_outputs": reader_outputs},
        {"sample_id": "pc-ot-mras-r13-source-level-1"},
    ]
    return (level0_features, level1_features), (level0_mask, level1_mask), reader_outputs, metas


def _expected_lengths(start_len, levels):
    lengths = [start_len]
    for _ in range(1, levels):
        lengths.append((lengths[-1] + 1) // 2)
    return lengths


def test_default_bridge_still_returns_single_level():
    bridge = PCOTMRASDetectorBridge(in_channels=2, out_channels=2, add_time_features=False, norm=False)
    _set_identity_projection(bridge)
    reader_outputs = _manual_selected_reader_outputs()

    feats, masks, aux = bridge(reader_outputs=reader_outputs, return_aux=True)

    assert len(feats) == 1
    assert len(masks) == 1
    assert feats[0].shape == (2, 2, 5)
    assert masks[0].shape == (2, 5)
    assert torch.equal(masks[0], reader_outputs["selected_mask"])
    assert torch.all(feats[0][1, :, 3:] == 0)
    assert aux["output_strides"] == "(1,)"
    assert aux["uses_hard_gather"] is False


def test_multiscale_outputs_have_p2_lengths_prefix_masks_and_zero_padding():
    bridge = PCOTMRASDetectorBridge(
        in_channels=2,
        out_channels=2,
        add_time_features=False,
        norm=False,
        output_strides=[1, 2, 4, 8, 16, 32],
    )
    _set_identity_projection(bridge)
    reader_outputs = _manual_selected_reader_outputs()

    feats, masks, aux = bridge(reader_outputs=reader_outputs, return_aux=True)

    assert aux["output_strides"] == "(1, 2, 4, 8, 16, 32)"
    assert [feat.shape[-1] for feat in feats] == _expected_lengths(5, 6)
    assert [mask.shape[-1] for mask in masks] == _expected_lengths(5, 6)
    assert [int(mask[0].sum().item()) for mask in masks] == [5, 3, 2, 1, 1, 1]
    assert [int(mask[1].sum().item()) for mask in masks] == [3, 2, 1, 1, 1, 1]
    for mask in masks:
        _assert_prefix(mask)

    assert torch.all(feats[0][1, :, 3:] == 0)
    assert torch.all(feats[1][1, :, 2] == 0)
    assert torch.all(feats[2][0, :, 1] == feats[1][0, :, 2])

    level0 = reader_outputs["selected_tokens"].transpose(1, 2)
    assert torch.allclose(feats[1][1, :, 1], level0[1, :, 2], atol=1e-6)
    assert not torch.allclose(feats[1][1, :, 1], 0.5 * level0[1, :, 2], atol=1e-6)

    for feat, mask in zip(feats, masks):
        for batch_idx in range(mask.shape[0]):
            count = int(mask[batch_idx].sum().item())
            if count < mask.shape[1]:
                assert torch.all(feat[batch_idx, :, count:] == 0)


def test_multiscale_all_levels_backprop_to_bridge_source_and_acquisition():
    torch.manual_seed(20260619)
    bridge = PCOTMRASDetectorBridge(
        in_channels=3,
        out_channels=4,
        add_time_features=False,
        output_strides=[1, 2, 4, 8, 16, 32],
    )
    source_tokens = torch.randn(2, 7, 3, requires_grad=True)
    dense_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    selected_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    acquisition_matrix = _acquisition_matrix()
    reader_outputs = _source_reader_outputs(acquisition_matrix, dense_mask, selected_mask)

    feats, masks, aux = bridge(source_tokens=source_tokens, reader_outputs=reader_outputs, return_aux=True)
    loss = sum(feat.square().mean() for feat in feats)
    loss.backward()

    assert len(feats) == 6
    assert [feat.shape[-1] for feat in feats] == _expected_lengths(5, 6)
    assert aux["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert aux["selected_tokens_source_verified"] is True
    assert bridge.token_proj.weight.grad is not None
    assert torch.isfinite(bridge.token_proj.weight.grad).all()
    assert bridge.token_proj.weight.grad.abs().sum().item() > 0
    assert source_tokens.grad is not None
    assert torch.isfinite(source_tokens.grad).all()
    assert source_tokens.grad.abs().sum().item() > 0
    assert acquisition_matrix.grad is not None
    assert torch.isfinite(acquisition_matrix.grad).all()
    assert acquisition_matrix.grad.abs().sum().item() > 0
    for mask in masks:
        _assert_prefix(mask)


def test_standard_neck_path_returns_multiscale_outputs_and_writes_metas():
    bridge = PCOTMRASDetectorBridge(
        in_channels=3,
        out_channels=4,
        add_time_features=False,
        output_strides=[1, 2, 4, 8, 16, 32],
    )
    features = torch.randn(2, 3, 7)
    dense_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    selected_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    reader_outputs = _source_reader_outputs(_acquisition_matrix(), dense_mask, selected_mask)
    metas = [
        {"sample_id": "pc-ot-mras-r11-0", "pc_ot_mras_reader_outputs": reader_outputs},
        {"sample_id": "pc-ot-mras-r11-1"},
    ]

    feats, masks, meta_out = bridge((features,), (dense_mask,), metas=metas)

    assert len(feats) == 6
    assert len(masks) == 6
    assert [feat.shape[-1] for feat in feats] == _expected_lengths(5, 6)
    assert [mask.shape[-1] for mask in masks] == _expected_lengths(5, 6)
    assert [int(mask[0].sum().item()) for mask in masks] == [5, 3, 2, 1, 1, 1]
    assert [int(mask[1].sum().item()) for mask in masks] == [4, 2, 1, 1, 1, 1]
    for mask in masks:
        _assert_prefix(mask)

    assert isinstance(meta_out, list)
    assert meta_out[0]["pc_ot_mras_bridge"]["output_strides"] == "(1, 2, 4, 8, 16, 32)"
    assert meta_out[0]["irregular_selected_count"] == 5
    assert meta_out[1]["irregular_selected_count"] == 4
    assert meta_out[0]["irregular_dense_valid_len"] == 7
    assert meta_out[1]["irregular_dense_valid_len"] == 5
    assert meta_out[0]["irregular_selected_valid_len"] == 7
    assert meta_out[1]["irregular_selected_valid_len"] == 5
    assert meta_out[0]["irregular_selected_valid_len_semantics"] == "carried_forward_dense_valid_len_alias"
    assert meta_out[1]["irregular_selected_valid_len_semantics"] == "carried_forward_dense_valid_len_alias"
    assert meta_out[0]["pc_ot_mras_bridge"]["source_feature_level"] == 0
    assert meta_out[0]["pc_ot_mras_bridge"]["source_feature_num_levels"] == 1
    assert meta_out[0]["pc_ot_mras_bridge"]["source_feature_level_explicit"] is False
    assert meta_out[0]["pc_ot_mras_bridge"]["source_feature_level_selection"] == "implicit_single_level"


def test_standard_neck_path_uses_explicit_multilevel_source_feature_level():
    bridge = PCOTMRASDetectorBridge(
        in_channels=3,
        out_channels=3,
        add_time_features=False,
        norm=False,
        source_feature_level=1,
    )
    _set_identity_projection(bridge)
    feat_list, mask_list, reader_outputs, metas = _multilevel_source_inputs()

    feats, masks, meta_out = bridge(feat_list, mask_list, metas=metas)

    expected_tokens = torch.bmm(reader_outputs["acquisition_matrix"], feat_list[1].transpose(1, 2).contiguous())
    expected_tokens = expected_tokens.masked_fill(~reader_outputs["selected_mask"].unsqueeze(-1), 0.0)
    assert torch.allclose(feats[0].transpose(1, 2), expected_tokens, atol=1e-6)
    assert torch.equal(masks[0], reader_outputs["selected_mask"])

    aux_0 = meta_out[0]["pc_ot_mras_bridge"]
    aux_1 = meta_out[1]["pc_ot_mras_bridge"]
    assert aux_0["source_feature_level"] == 1
    assert aux_0["source_feature_num_levels"] == 2
    assert aux_0["source_feature_level_explicit"] is True
    assert aux_0["source_feature_level_selection"] == "explicit"
    assert aux_1["source_feature_level"] == 1
    assert aux_1["source_feature_num_levels"] == 2
    assert aux_1["source_feature_level_explicit"] is True
    assert torch.allclose(aux_0["selected_tokens"], expected_tokens[0], atol=1e-6)
    assert torch.allclose(aux_1["selected_tokens"], expected_tokens[1], atol=1e-6)


def test_standard_neck_multilevel_without_source_feature_level_fails_closed():
    bridge = PCOTMRASDetectorBridge(in_channels=3, out_channels=3, add_time_features=False, norm=False)
    feat_list, mask_list, _reader_outputs, metas = _multilevel_source_inputs()

    with pytest.raises(ValueError, match="set source_feature_level explicitly"):
        bridge(feat_list, mask_list, metas=metas)


def test_standard_neck_invalid_source_feature_level_fails_closed():
    bridge = PCOTMRASDetectorBridge(
        in_channels=3,
        out_channels=3,
        add_time_features=False,
        norm=False,
        source_feature_level=2,
    )
    feat_list, mask_list, _reader_outputs, metas = _multilevel_source_inputs()

    with pytest.raises(ValueError, match="source_feature_level out of range"):
        bridge(feat_list, mask_list, metas=metas)


@pytest.mark.parametrize("bad_level", [True, 0.0, 1.5, "0"])
def test_standard_neck_rejects_non_integer_source_feature_level(bad_level):
    with pytest.raises(ValueError, match="source_feature_level must be a non-negative integer or None"):
        PCOTMRASDetectorBridge(
            in_channels=3,
            out_channels=3,
            add_time_features=False,
            norm=False,
            source_feature_level=bad_level,
        )
