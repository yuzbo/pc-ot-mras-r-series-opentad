import copy
import importlib.util
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r12_p2_optin_adapter_local.py"
OUTPUT_STRIDES = [1, 2, 4, 8, 16, 32]
R13_BRIDGE_SOURCE_FEATURE_LEVEL = 0


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
import torch.nn as nn


def _load_helper_module():
    helper_path = ROOT / "tests" / "test_pc_ot_mras_actionformer_multiscale_p2_smoke.py"
    spec = importlib.util.spec_from_file_location("pc_ot_mras_r12_actionformer_p2_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = _load_helper_module()
ActionFormer = HELPERS.ActionFormer
READER_OUTPUTS_META_KEY = HELPERS.READER_OUTPUTS_META_KEY


class SyntheticActionFormerPyramidProjection(nn.Module):
    """Minimal six-level ActionFormer-style projection for config-faithful smoke."""

    def __init__(self, channels=4, max_seq_len=32, output_levels=6):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(channels)
        self.output_levels = int(output_levels)
        self.n_mha_win_size = 1
        self.arch = (0, 0, self.output_levels - 1)
        self.max_seq_len = int(max_seq_len)
        self.proj = nn.Conv1d(self.channels, self.out_channels, kernel_size=1, bias=False)
        nn.init.eye_(self.proj.weight[:, :, 0])

    @staticmethod
    def _assert_prefix(mask):
        valid_count = mask.long().sum(dim=1)
        expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < valid_count[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("SyntheticActionFormerPyramidProjection expects prefix masks")

    @staticmethod
    def _downsample_prefix(features, mask):
        even_feat = features[:, :, 0::2]
        odd_feat = features[:, :, 1::2]
        even_valid = mask[:, 0::2]
        odd_valid = mask[:, 1::2]
        if odd_feat.shape[-1] < even_feat.shape[-1]:
            pad_len = even_feat.shape[-1] - odd_feat.shape[-1]
            odd_feat = torch.cat(
                (odd_feat, features.new_zeros((features.shape[0], features.shape[1], pad_len))),
                dim=-1,
            )
            odd_valid = torch.cat((odd_valid, mask.new_zeros((mask.shape[0], pad_len))), dim=1)

        even_weight = even_valid[:, None].to(dtype=features.dtype)
        odd_weight = odd_valid[:, None].to(dtype=features.dtype)
        pooled_mask = even_valid | odd_valid
        denom = (even_weight + odd_weight).clamp_min(1.0)
        pooled = (even_feat * even_weight + odd_feat * odd_weight) / denom
        return pooled.masked_fill(~pooled_mask[:, None], 0.0).contiguous(), pooled_mask

    def forward(self, x, masks):
        if x.ndim != 3:
            raise ValueError(f"projection input must be [B,C,T], got {tuple(x.shape)}")
        if masks.ndim != 2 or masks.shape != (x.shape[0], x.shape[-1]):
            raise ValueError("projection mask must be [B,T] and match feature length")
        masks = masks.bool()
        self._assert_prefix(masks)
        x = self.proj(x).masked_fill(~masks[:, None], 0.0)
        feats = [x]
        out_masks = [masks]
        for _ in range(1, self.output_levels):
            next_feat, next_mask = self._downsample_prefix(feats[-1], out_masks[-1])
            feats.append(next_feat)
            out_masks.append(next_mask)
        return tuple(feats), tuple(out_masks)


def _register_r13_projection_surrogate():
    builder = sys.modules["opentad.models.builder"]
    builder.PROJECTIONS.register_module()(SyntheticActionFormerPyramidProjection)


_register_r13_projection_surrogate()


def _load_mmengine_config():
    mmengine_config = pytest.importorskip("mmengine.config")
    assert CONFIG.exists()
    return mmengine_config.Config.fromfile(str(CONFIG))


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items() if key != "_delete_"}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


def _assert_r13_bridge_source_level(mapping):
    assert "source_feature_level" in mapping, (
        "R13 formal PC-OT-MRAS config must explicitly set "
        "model.neck.source_feature_level so multi-level projection tuples do "
        "not rely on the old single-level bridge assumption"
    )
    assert mapping["source_feature_level"] == R13_BRIDGE_SOURCE_FEATURE_LEVEL
    return mapping["source_feature_level"]


def _tiny_model_cfg_from_formal_config(cfg):
    model_cfg = _plain(copy.deepcopy(cfg.model))
    formal_projection = model_cfg["projection"]
    projection_levels = 1 + int(formal_projection["arch"][-1])
    assert projection_levels == len(OUTPUT_STRIDES)
    assert model_cfg["pc_ot_mras_reader_feature_level"] == R13_BRIDGE_SOURCE_FEATURE_LEVEL
    _assert_r13_bridge_source_level(model_cfg["neck"])
    model_cfg.pop("backbone", None)
    model_cfg["projection"] = dict(
        type="SyntheticActionFormerPyramidProjection",
        channels=4,
        max_seq_len=32,
        output_levels=projection_levels,
    )
    model_cfg["pc_ot_mras_reader"].update(
        in_dim=4,
        hidden_dim=8,
        num_slots=8,
        num_blocks=1,
        dropout=0.0,
    )
    model_cfg["neck"].update(
        in_channels=4,
        out_channels=4,
    )
    model_cfg["rpn_head"].update(
        num_classes=3,
        in_channels=4,
        feat_channels=4,
        num_convs=0,
    )
    model_cfg["rpn_head"]["area_head"].update(
        max_boundaries_per_side=4,
        max_pairs_per_class=6,
    )
    return model_cfg


def _assert_no_forbidden_runtime_literal(text):
    lower = text.lower()
    for literal in (
        "tools/train.py",
        "tools/test.py",
        "#sbatch",
        "sbatch ",
        "srun ",
        "ssh ",
        "scp ",
        "/root/",
        "result_detection.json",
        "load_from_raw_predictions=true",
        "save_raw_prediction=true",
    ):
        assert literal not in lower


def test_r12_pc_ot_mras_opt_in_config_contract_is_parseable():
    cfg = _load_mmengine_config()
    text = CONFIG.read_text(encoding="utf-8")
    _assert_no_forbidden_runtime_literal(text)

    gate = cfg.r12_pc_ot_mras_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R12_config_opt_in_local_smoke"
    assert gate.reviewed_predecessor == "R11c3"
    assert gate.default_off is True
    assert gate.explicit_config_opt_in is True
    assert gate.local_synthetic_gate_only is True
    assert gate.allow_detector_training is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.allow_real_dataset is False
    assert gate.allow_checkpoint is False
    assert gate.allow_tools_test is False
    assert "synthetic_cpu_forward_loss_smoke" in gate.allowed_checks
    assert "tools_test_or_map" in gate.forbidden_checks

    model = cfg.model
    assert model.type == "ActionFormer"
    assert model.projection.type == "Conv1DTransformerProj"
    assert 1 + model.projection.arch[-1] == len(OUTPUT_STRIDES)
    assert model.pc_ot_mras_reader_feature_level == 0
    assert model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert model.pc_ot_mras_reader.in_dim == model.projection.out_channels == 512
    assert model.pc_ot_mras_reader.hidden_dim == 96
    assert model.pc_ot_mras_reader.num_slots == 384
    assert model.pc_ot_mras_reader.num_roles == 6

    assert model.neck.type == "PCOTMRASDetectorBridge"
    assert list(model.neck.output_strides) == OUTPUT_STRIDES
    source_feature_level = _assert_r13_bridge_source_level(model.neck)
    assert source_feature_level == model.pc_ot_mras_reader_feature_level
    assert model.neck.in_channels == model.projection.out_channels
    assert model.neck.out_channels == model.rpn_head.in_channels == 512
    assert model.neck.allocation_key == "acquisition_matrix"
    assert model.neck.add_time_features is True

    assert model.rpn_head.type == "NativeIrregularAreaHeadP2"
    assert list(model.rpn_head.prior_generator.strides) == OUTPUT_STRIDES
    assert [tuple(item) for item in model.rpn_head.prior_generator.regression_range] == [
        (0, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 10000),
    ]
    assert model.rpn_head.temporal_grid.required is True
    assert model.rpn_head.temporal_grid.decode_axis == "dense"
    assert model.rpn_head.temporal_grid.positions_key == "irregular_selected_positions"
    assert model.rpn_head.temporal_grid.valid_len_key == "irregular_selected_valid_len"
    assert model.rpn_head.area_head.observation_half_width == "cell_support"
    assert model.rpn_head.area_head.use_regression_range_assignment is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


def test_r12_pc_ot_mras_tiny_config_copy_preserves_formal_topology():
    cfg = _load_mmengine_config()
    tiny = _tiny_model_cfg_from_formal_config(cfg)

    assert tiny["type"] == "ActionFormer"
    assert tiny["projection"]["type"] == "SyntheticActionFormerPyramidProjection"
    assert tiny["projection"]["output_levels"] == len(OUTPUT_STRIDES)
    assert tiny["pc_ot_mras_reader"]["type"] == "PCOTMRASReader"
    assert tiny["neck"]["type"] == "PCOTMRASDetectorBridge"
    assert tiny["rpn_head"]["type"] == "NativeIrregularAreaHeadP2"
    assert tiny["neck"]["output_strides"] == OUTPUT_STRIDES
    assert tiny["pc_ot_mras_reader_feature_level"] == R13_BRIDGE_SOURCE_FEATURE_LEVEL
    assert _assert_r13_bridge_source_level(tiny["neck"]) == R13_BRIDGE_SOURCE_FEATURE_LEVEL
    assert tiny["rpn_head"]["prior_generator"]["strides"] == OUTPUT_STRIDES
    assert tiny["rpn_head"]["temporal_grid"]["decode_axis"] == "dense"
    assert tiny["pc_ot_mras_reader"]["in_dim"] == tiny["projection"]["channels"] == tiny["neck"]["in_channels"]
    assert tiny["neck"]["out_channels"] == tiny["rpn_head"]["in_channels"]
    assert tiny["rpn_head"]["feat_channels"] == tiny["neck"]["out_channels"]
    assert "backbone" not in tiny


def _assert_r13_multilevel_projection_neck_input(captured, split):
    assert captured["split"] == split
    feat_list = captured["feat_list"]
    mask_list = captured["mask_list"]
    assert len(feat_list) == len(mask_list) == len(OUTPUT_STRIDES)
    assert [feat.shape[-1] for feat in feat_list] == HELPERS._expected_lengths(32, len(OUTPUT_STRIDES))
    assert [mask.shape[-1] for mask in mask_list] == HELPERS._expected_lengths(32, len(OUTPUT_STRIDES))
    for feat, mask in zip(feat_list, mask_list):
        assert feat.shape[0] == 1
        assert feat.shape[1] == 4
        assert mask.dtype == torch.bool
        HELPERS._assert_prefix(mask)
        assert torch.isfinite(feat).all()


def test_r12_pc_ot_mras_actionformer_reader_bridge_p2_synthetic_forward_loss_smoke():
    cfg = _load_mmengine_config()
    model_kwargs = _tiny_model_cfg_from_formal_config(cfg)
    assert model_kwargs.pop("type") == "ActionFormer"
    model = ActionFormer(**model_kwargs)
    inputs, masks, metas, gt_segments, gt_labels = HELPERS._inputs()
    captured = {}

    original_neck_forward = model.neck.forward
    original_forward_train = model.rpn_head.forward_train
    original_forward_test = model.rpn_head.forward_test

    def capture_neck_forward(feat_list, mask_list, metas=None, **kwargs):
        split = "train" if model.training else "test"
        captured[f"neck_{split}"] = {
            "split": split,
            "feat_list": feat_list,
            "mask_list": mask_list,
            "metas": metas,
        }
        return original_neck_forward(feat_list, mask_list, metas=metas, **kwargs)

    def capture_forward_train(feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        captured["train"] = {
            "split": "train",
            "feat_list": feat_list,
            "mask_list": mask_list,
            "metas": metas,
        }
        return original_forward_train(
            feat_list,
            mask_list,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            metas=metas,
            **kwargs,
        )

    def capture_forward_test(feat_list, mask_list, metas=None, **kwargs):
        captured["test"] = {
            "split": "test",
            "feat_list": feat_list,
            "mask_list": mask_list,
            "metas": metas,
        }
        return original_forward_test(feat_list, mask_list, metas=metas, **kwargs)

    model.neck.forward = capture_neck_forward
    model.rpn_head.forward_train = capture_forward_train
    model.rpn_head.forward_test = capture_forward_test

    model.train()
    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    assert set(losses) == {
        "area_loss",
        "start_gap_loss",
        "end_gap_loss",
        "start_offset_loss",
        "end_offset_loss",
        "boundary_uncertainty_loss",
        "cost",
    }
    assert all(torch.isfinite(value).item() for value in losses.values())
    assert READER_OUTPUTS_META_KEY not in metas[0]
    _assert_r13_multilevel_projection_neck_input(captured["neck_train"], "train")
    HELPERS._assert_multiscale_p2_call(captured["train"], "train")

    losses["cost"].backward()
    for param in (
        inputs,
        model.projection.proj.weight,
        model.pc_ot_mras_reader.input_proj.weight,
        model.pc_ot_mras_reader.gate_head.weight,
        model.neck.token_proj.weight,
        model.rpn_head.area_head.weight,
        model.rpn_head.start_gap_head.weight,
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum().item() > 0

    model.eval()
    with torch.no_grad():
        proposals, scores = model.forward_test(inputs.detach(), masks, metas=metas)

    _assert_r13_multilevel_projection_neck_input(captured["neck_test"], "test")
    HELPERS._assert_multiscale_p2_call(captured["test"], "test")
    assert READER_OUTPUTS_META_KEY not in metas[0]
    assert len(proposals) == 1
    assert len(scores) == 1
    assert proposals[0].ndim == 2
    assert proposals[0].shape[1] == 2
    assert scores[0].ndim == 2
    assert scores[0].shape[1] == 3
    assert torch.isfinite(proposals[0]).all()
    assert torch.isfinite(scores[0]).all()
    if proposals[0].numel():
        assert torch.all(proposals[0][:, 1] > proposals[0][:, 0])
