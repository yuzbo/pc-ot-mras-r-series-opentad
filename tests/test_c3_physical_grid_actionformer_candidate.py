import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGE = "c3_physical_grid_actionformer_runtime"


torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
TORCH_AVAILABLE = torch_probe.returncode == 0

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


def read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def load_mmengine_config_or_skip(rel_path):
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(ROOT / rel_path))


def test_torch_runtime_is_available_for_c3_precheck():
    if not TORCH_AVAILABLE:
        detail = torch_probe.stderr.strip() or torch_probe.stdout.strip() or f"exit {torch_probe.returncode}"
        pytest.fail(f"torch unavailable: C3 physical-grid precheck is an environment blocker, not a pass ({detail})")


class _Registry:
    def __init__(self):
        self._items = {}

    def register_module(self):
        def _decorator(cls):
            self._items[cls.__name__] = cls
            return cls

        return _decorator

    def build(self, cfg):
        cfg = dict(cfg)
        type_name = cfg.pop("type")
        return self._items[type_name](**cfg)


if TORCH_AVAILABLE:

    class _DummyLoss(nn.Module):
        def forward(self, inputs, targets, reduction="none", **kwargs):
            loss = inputs.sum() * 0.0
            if reduction in {"mean", "sum"}:
                return loss
            return inputs.new_zeros(inputs.shape[:-1])

    class _Scale(nn.Module):
        def __init__(self, init_value=1.0):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(float(init_value)))

        def forward(self, x):
            return x * self.scale

    class _ConvModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("physical-grid head tests use num_convs=0")


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


def _install_head_runtime_or_skip():
    if not TORCH_AVAILABLE:
        pytest.skip(
            "torch unavailable in this process: "
            + (
                torch_probe.stderr.strip().splitlines()[-1]
                if torch_probe.stderr.strip()
                else f"exit {torch_probe.returncode}"
            )
        )

    _ensure_package(RUNTIME_PACKAGE, ROOT / "opentad")
    _ensure_package(f"{RUNTIME_PACKAGE}.models", ROOT / "opentad" / "models")
    _ensure_package(f"{RUNTIME_PACKAGE}.models.dense_heads", ROOT / "opentad" / "models" / "dense_heads")
    _ensure_package(f"{RUNTIME_PACKAGE}.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package(
        f"{RUNTIME_PACKAGE}.models.dense_heads.prior_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator",
    )

    builder = types.ModuleType(f"{RUNTIME_PACKAGE}.models.builder")
    builder.HEADS = _Registry()
    builder.SELECTORS = _Registry()
    builder.PRIOR_GENERATORS = _Registry()
    builder.LOSSES = _Registry()
    builder.build_prior_generator = lambda cfg: builder.PRIOR_GENERATORS.build(cfg)
    builder.build_selector = lambda cfg: builder.SELECTORS.build(cfg)
    builder.build_loss = lambda cfg: _DummyLoss()
    sys.modules[f"{RUNTIME_PACKAGE}.models.builder"] = builder

    bricks = types.ModuleType(f"{RUNTIME_PACKAGE}.models.bricks")
    bricks.ConvModule = _ConvModule
    bricks.Scale = _Scale
    sys.modules[f"{RUNTIME_PACKAGE}.models.bricks"] = bricks

    _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.prior_generator.point_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator" / "point_generator.py",
    )
    _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.anchor_free_head",
        ROOT / "opentad" / "models" / "dense_heads" / "anchor_free_head.py",
    )
    actionformer_head = _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.actionformer_head",
        ROOT / "opentad" / "models" / "dense_heads" / "actionformer_head.py",
    )
    return actionformer_head.ActionFormerHead


def _install_prebackbone_selector_or_skip():
    _install_head_runtime_or_skip()
    selector_module = _load_module(
        f"{RUNTIME_PACKAGE}.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py",
    )
    return selector_module.PCOTMRASPreBackboneFrameSelector


def _make_head(**kwargs):
    ActionFormerHead = _install_head_runtime_or_skip()
    head = ActionFormerHead(
        num_classes=2,
        in_channels=2,
        feat_channels=2,
        num_convs=0,
        prior_generator=dict(type="PointGenerator", strides=[1], regression_range=[(0, 10000)]),
        loss=types.SimpleNamespace(cls_loss=dict(type="DummyLoss"), reg_loss=dict(type="DummyLoss")),
        **kwargs,
    )
    with torch.no_grad():
        head.cls_head.weight.zero_()
        head.cls_head.bias.zero_()
        head.reg_head.weight.zero_()
        head.reg_head.bias.zero_()
        head.scale[0].scale.fill_(1.0)
    return head


def _features_and_mask():
    return [torch.zeros(1, 2, 4)], [torch.tensor([[True, True, True, False]])]


def _irregular_meta(**extra):
    meta = {
        "video_name": "synthetic",
        "irregular_selected_positions": [0.0, 2.0, 5.0],
        "selected_dense_indices": [0, 2, 5],
        "selected_valid_len": 3,
        "irregular_selected_valid_len": 10.0,
        "irregular_native_axis": False,
        "remap_gt_to_selected_axis": False,
    }
    meta.update(extra)
    return meta


def _dense_axis_meta(**extra):
    return _irregular_meta(irregular_native_axis=True, remap_gt_to_selected_axis=False, **extra)


def test_physical_grid_opt_in_decodes_on_dense_positions_and_masks_padded_tail():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta()]

    proposals, scores = head.forward_test(feat_list, mask_list, metas=metas)

    expected_centers = torch.tensor([0.0, 2.0, 5.0])
    assert torch.allclose(proposals[0][:, 0], expected_centers)
    assert torch.allclose(proposals[0][:, 1], expected_centers)
    assert proposals[0].shape == (3, 2)
    assert scores[0].shape == (3, 2)
    assert metas[0]["irregular_native_axis"] is True
    assert metas[0]["physical_grid_actionformer"] is True


def test_physical_grid_default_keeps_selected_axis_with_same_metadata():
    head = _make_head()
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta()]

    proposals, _scores = head.forward_test(feat_list, mask_list, metas=metas)

    assert torch.allclose(proposals[0][:, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(proposals[0][:, 1], torch.tensor([0.0, 1.0, 2.0]))
    assert metas[0]["irregular_native_axis"] is False
    assert "physical_grid_actionformer" not in metas[0]


def test_physical_grid_missing_metadata_fails_closed():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()

    with pytest.raises(ValueError, match="physical-grid ActionFormer requires"):
        head.forward_test(feat_list, mask_list, metas=[{"video_name": "missing"}])


def test_physical_grid_rejects_selected_axis_gt_remap_metadata():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta(remap_gt_to_selected_axis=True)]

    with pytest.raises(ValueError, match="dense-axis GT"):
        head.forward_train(
            feat_list,
            mask_list,
            gt_segments=[torch.tensor([[1.5, 2.5]], dtype=torch.float32)],
            gt_labels=[torch.tensor([1], dtype=torch.long)],
            metas=metas,
        )


def test_physical_grid_training_rejects_selected_axis_gt_native_axis_false():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta(irregular_native_axis=False, remap_gt_to_selected_axis=False)]

    with pytest.raises(ValueError, match="dense-axis GT"):
        head.forward_train(
            feat_list,
            mask_list,
            gt_segments=[torch.tensor([[1.5, 2.5]], dtype=torch.float32)],
            gt_labels=[torch.tensor([1], dtype=torch.long)],
            metas=metas,
        )


def test_prepare_targets_batched_physical_points_use_per_sample_centers_without_assigner():
    head = _make_head(physical_grid_actionformer=dict(enabled=False))
    points = [
        torch.tensor(
            [
                [[0.0, 0.0, 100.0, 1.0], [2.0, 0.0, 100.0, 1.0], [5.0, 0.0, 100.0, 1.0]],
                [[10.0, 0.0, 100.0, 1.0], [12.0, 0.0, 100.0, 1.0], [15.0, 0.0, 100.0, 1.0]],
            ],
            dtype=torch.float32,
        )
    ]
    gt_cls, gt_reg = head.prepare_targets(
        points,
        gt_segments=[
            torch.tensor([[1.5, 2.5]], dtype=torch.float32),
            torch.tensor([[11.5, 12.5]], dtype=torch.float32),
        ],
        gt_labels=[torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)],
    )

    assert torch.equal(gt_cls[0].argmax(dim=1), torch.tensor([0, 0, 0]))
    assert torch.equal(gt_cls[0].sum(dim=1) > 0, torch.tensor([False, True, False]))
    assert torch.equal(gt_cls[1].argmax(dim=1), torch.tensor([0, 1, 0]))
    assert torch.equal(gt_cls[1].sum(dim=1) > 0, torch.tensor([False, True, False]))
    assert torch.allclose(gt_reg[0][1], torch.tensor([0.5, 0.5]))
    assert torch.allclose(gt_reg[1][1], torch.tensor([0.5, 0.5]))


def test_physical_grid_training_assignment_uses_physical_dense_centers():
    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_dense_axis_meta()]

    losses = head.forward_train(
        feat_list,
        mask_list,
        gt_segments=[torch.tensor([[1.5, 2.5]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        metas=metas,
    )
    debug = head.collect_debug_state()

    assert set(losses) == {"cls_loss", "reg_loss"}
    assert debug["physical_grid_actionformer_enabled"] is True
    assert debug["physical_grid_actionformer_valid_points"] == 3
    assert debug["physical_grid_actionformer_center_min"] == 0.0
    assert debug["physical_grid_actionformer_center_max"] == 5.0
    assert debug["physical_grid_actionformer_axis_delta_reference"] == "selected_slot_ordinal"
    assert debug["physical_grid_actionformer_axis_delta_max"] == 3.0


def test_prebackbone_selector_dense_axis_meta_feeds_physical_grid_train_path():
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    selector = PCOTMRASPreBackboneFrameSelector(
        reader=dict(type="PCOTMRASBoundaryDifficultyTemporalFrameScout", in_dim=4, hidden_dim=4, num_slots=4),
        target_len=4,
        dense_window_size=12,
        descriptor_dim=4,
        remap_gt_to_selected_axis=False,
    )
    metas = selector._write_selected_axis_meta(
        metas=[{"video_name": "selector-physical-grid"}],
        selected_positions=torch.tensor([[0, 3, 4, 10]], dtype=torch.long),
        valid_lengths=torch.tensor([12], dtype=torch.long),
        selected_output_valid_lengths=torch.tensor([4], dtype=torch.long),
        training=True,
    )

    assert metas[0]["pc_ot_mras_prebackbone_remap_gt_to_selected_axis"] is False
    assert metas[0]["irregular_native_axis"] is True
    assert metas[0]["irregular_selected_positions"] == [0.0, 3.0, 4.0, 10.0]
    assert metas[0]["irregular_dense_valid_len"] == 12
    assert metas[0]["irregular_selected_count"] == 4
    assert metas[0]["selected_valid_len"] == 4
    assert metas[0]["irregular_selected_valid_len"] == 12.0

    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list = [torch.zeros(1, 2, 4)]
    mask_list = [torch.tensor([[True, True, True, True]])]
    losses = head.forward_train(
        feat_list,
        mask_list,
        gt_segments=[torch.tensor([[2.5, 4.5]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        metas=metas,
    )
    debug = head.collect_debug_state()

    assert set(losses) == {"cls_loss", "reg_loss"}
    assert debug["physical_grid_actionformer_valid_points"] == 4
    assert debug["physical_grid_actionformer_selected_count"] == 4
    assert debug["physical_grid_actionformer_dense_valid_len_max"] == 12.0
    assert debug["physical_grid_actionformer_center_max"] == 10.0


def test_prebackbone_selector_forward_train_meta_feeds_physical_grid_train_path():
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    selector = PCOTMRASPreBackboneFrameSelector(
        reader=dict(
            type="PCOTMRASBoundaryDifficultyTemporalFrameScout",
            in_dim=4,
            hidden_dim=4,
            num_slots=4,
            temporal_layers=1,
            dilations=(1,),
            dropout=0.0,
        ),
        target_len=4,
        dense_window_size=12,
        descriptor_dim=4,
        remap_gt_to_selected_axis=False,
    )
    inputs = torch.arange(1 * 3 * 12 * 2 * 2, dtype=torch.float32).reshape(1, 3, 12, 2, 2)
    masks = torch.ones(1, 12, dtype=torch.bool)
    selected = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "selector-forward-train"}],
        gt_segments=[torch.tensor([[2.5, 4.5]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
    )
    metas = selected["metas"]

    assert selected["inputs"].shape[2] == 4
    assert selected["masks"].shape == (1, 4)
    assert selected["masks"].all()
    assert selected["gt_segments"][0].shape == (1, 2)
    assert torch.allclose(selected["gt_segments"][0], torch.tensor([[2.5, 4.5]], dtype=torch.float32))
    assert metas[0]["pc_ot_mras_prebackbone_remap_gt_to_selected_axis"] is False
    assert metas[0]["irregular_native_axis"] is True
    assert metas[0]["irregular_dense_valid_len"] == 12
    assert metas[0]["irregular_selected_count"] == 4
    assert metas[0]["selected_valid_len"] == 4
    assert metas[0]["irregular_selected_valid_len"] == 12.0
    assert len(metas[0]["irregular_selected_positions"]) == 4

    head = _make_head(physical_grid_actionformer=dict(enabled=True, required=True, strict=True))
    feat_list = [torch.zeros(1, 2, 4)]
    mask_list = [selected["masks"]]
    losses = head.forward_train(
        feat_list,
        mask_list,
        gt_segments=selected["gt_segments"],
        gt_labels=selected["gt_labels"],
        metas=metas,
    )
    debug = head.collect_debug_state()

    assert set(losses) == {"cls_loss", "reg_loss"}
    assert debug["physical_grid_actionformer_valid_points"] == 4
    assert debug["physical_grid_actionformer_selected_count"] == 4
    assert debug["physical_grid_actionformer_dense_valid_len_max"] == 12.0


def test_c3_candidate_config_and_launcher_are_fail_closed():
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/input_random_fixed_50pct_c3_physical_grid_actionformer_precheck.py"
    )
    script = read("scripts/run_c3_physical_grid_actionformer_precheck.sh")
    tool = read("tools/bata/validate_c3_physical_grid_actionformer_precheck.py")

    assert cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled is True
    assert cfg.model.rpn_head.physical_grid_actionformer.required is True
    assert cfg.model.rpn_head.physical_grid_actionformer.strict is True
    for split in ("train", "val", "test"):
        load_steps = [step for step in cfg.dataset[split].pipeline if step.get("type") == "LoadFrames"]
        assert len(load_steps) == 1
        assert load_steps[0].remap_gt_to_selected_axis is False
    assert cfg.protocol_flags.precheck_only is True
    assert cfg.protocol_flags.uses_p2_head is False
    assert cfg.protocol_flags.uses_raw_prediction_cache is False
    assert cfg.protocol_flags.uses_teacher is False
    assert cfg.protocol_flags.uses_test_gt is False
    assert cfg.protocol_flags.remote_sync_allowed is False
    assert cfg.protocol_flags.slurm_allowed is False

    assert "PRECHECK_ONLY" in script
    assert "tools/train.py" not in script
    assert "tools/test.py" not in script
    assert "pytest tests/test_c3_physical_grid_actionformer_candidate.py -q" in script
    assert "validate_c3_physical_grid_actionformer_precheck.py" in script
    assert "FORBIDDEN_CONFIG_TOKENS" in tool
    assert "for token in FORBIDDEN_CONFIG_TOKENS" in tool
    assert "P2" in tool
    assert "raw_prediction" in tool
    assert "teacher" in tool
    assert "test_gt" in tool


def test_c3_physical_grid_static_source_contracts_when_torch_is_unavailable():
    head = read("opentad/models/dense_heads/anchor_free_head.py")
    detector = read("opentad/models/detectors/actionformer.py")
    formatting = read("opentad/datasets/transforms/formatting.py")

    assert "physical_grid_actionformer=None" in head
    assert "self.physical_grid_enabled" in head
    assert "irregular_selected_positions" in head
    assert "selected_dense_indices" in head
    assert "selected_valid_len" in head
    assert "def _physical_selected_count_from_meta" in head
    assert 'for key in ("selected_valid_len", "irregular_selected_count")' in head
    assert "selected_count = self._physical_selected_count_from_meta(meta, positions)" in head
    assert "physical_grid_selected_count" in head
    assert "slot_index = torch.arange(point.shape[0], device=base_device)" in head
    assert "level_valid = slot_index < int(selected_count)" in head
    assert "level_valid = physical_center <" not in head
    assert '"physical_grid_actionformer_axis_delta_reference": "selected_slot_ordinal"' in head
    assert "debug_axis_delta.append((kept_centers - selected_center[kept]).abs().detach())" not in head
    assert "debug_axis_delta.append((kept_centers - slot_ordinal[kept]).abs().detach())" in head
    selected_count_fn = head.split("def _physical_selected_count_from_meta", 1)[1].split(
        "def _physical_positions_from_meta", 1
    )[0]
    assert "irregular_selected_valid_len" not in selected_count_fn
    assert "irregular_dense_valid_len" not in selected_count_fn
    assert 'meta["irregular_native_axis"] = True' in head
    assert 'meta["physical_grid_actionformer"] = True' in head
    assert "physical_masks[level_idx][batch_idx] = physical_masks[level_idx][batch_idx] & level_valid" in head
    assert "physical-grid ActionFormer requires dense-axis GT" in head
    assert "meta.get(\"irregular_native_axis\", None) is not True" in head
    assert "cb_dist_left = point[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])" in head
    assert "cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - point[:, 0, None]" in head
    assert "torch.cat(points, dim=1) if points[0].dim() == 3 else torch.cat(points, dim=0)" in head

    selector = read("opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py")
    assert "meta[\"irregular_native_axis\"] = not bool(self.remap_gt_to_selected_axis)" in selector
    assert "meta[\"pc_ot_mras_prebackbone_remap_gt_to_selected_axis\"] = bool(self.remap_gt_to_selected_axis)" in selector
    assert "meta[\"irregular_dense_valid_len\"] = int(valid_cpu[idx])" in selector
    assert "meta[\"irregular_selected_count\"] = int(selected_valid_cpu[idx])" in selector
    assert "meta[\"selected_valid_len\"] = int(selected_valid_cpu[idx])" in selector
    assert (
        "meta[\"irregular_selected_valid_len_semantics\"] = \"carried_forward_dense_valid_len_alias\""
        in selector
    )

    assert "metas=metas" in detector
    assert '"selected_dense_indices"' in formatting
    assert '"selected_valid_len"' in formatting


def test_coarse_actionness_scout_has_no_learned_boundary_heads():
    selector = read("opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py")
    class_source = selector.split("class PCOTMRASCoarseActionnessFrameScout", 1)[1].split(
        "class PCOTMRASPreBackboneFrameSelector",
        1,
    )[0]

    assert "self.action_head" in class_source
    assert "start_head" not in class_source
    assert "end_head" not in class_source
    assert "boundary_head" not in class_source
    assert '"start_logits"' not in class_source
    assert '"end_logits"' not in class_source
    assert '"boundary_logits"' not in class_source


def test_coarse_actionness_scout_outputs_only_binary_derived_sampling_signals():
    _install_prebackbone_selector_or_skip()
    selector_module = sys.modules[f"{RUNTIME_PACKAGE}.models.selectors.pc_ot_mras_prebackbone_frame_selector"]
    scout = selector_module.PCOTMRASCoarseActionnessFrameScout(
        in_dim=4,
        hidden_dim=4,
        num_slots=6,
        temporal_layers=1,
        temporal_kernel_size=3,
        dilations=(1,),
        dropout=0.0,
    )
    features = torch.randn(1, 8, 4)
    valid = torch.ones(1, 8, dtype=torch.bool)
    out = scout(features, valid)

    assert out["action_logits"].shape == (1, 8)
    assert out["actionness_logits"].shape == (1, 8)
    assert out["uncertainty_score"].shape == (1, 8)
    assert out["change_score"].shape == (1, 8)
    assert out["frame_selection_logits"].shape == (1, 8)
    assert "start_logits" not in out
    assert "end_logits" not in out
    assert "boundary_logits" not in out
    assert "risk_logits" not in out
    assert torch.isfinite(out["frame_selection_logits"]).all()


def test_coarse_actionness_uncertainty_plan_uses_classification_uncertainty_roles():
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    selector = PCOTMRASPreBackboneFrameSelector(
        reader=dict(
            type="PCOTMRASCoarseActionnessFrameScout",
            in_dim=4,
            hidden_dim=4,
            num_slots=6,
            temporal_layers=1,
            temporal_kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        ),
        target_len=6,
        dense_window_size=12,
        descriptor_dim=4,
        selection_strategy="coarse_actionness_uncertainty",
        coarse_uniform_count=1,
        coarse_action_count=2,
        coarse_uncertainty_count=1,
        coarse_change_count=1,
        coarse_background_count=1,
        max_dense_gap=0,
        max_gap_guard_count=0,
        remap_gt_to_selected_axis=False,
        straight_through_detector_loss=False,
    )
    candidate_valid = torch.ones(1, 12, dtype=torch.bool)
    candidate_dense_indices = torch.arange(12, dtype=torch.long).unsqueeze(0)
    valid = candidate_valid.clone()
    action_logits = torch.tensor([[-5.0, -4.0, -3.0, 0.0, 5.0, 4.0, -0.1, 0.1, -4.0, 0.0, -5.0, 3.0]])

    plan = selector._coarse_actionness_uncertainty_transport_plan(
        reader_outputs={"actionness_logits": action_logits},
        valid=valid,
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        training=False,
    )
    roles = plan["selected_roles"][0][:6]

    assert plan["selected_positions"].shape == (1, 6)
    assert plan["selected_output_valid_lengths"].tolist() == [6]
    assert "coarse_uniform" in roles
    assert "coarse_action" in roles
    assert "coarse_uncertainty" in roles
    assert "coarse_change" in roles
    assert "coarse_background" in roles
    assert plan["coarse_policy_meta"][0]["uses_learned_boundary_head"] is False
    assert plan["coarse_policy_meta"][0]["uses_gt"] is False


def test_coarse_actionness_candidate_points_expose_components_roles_and_mixed_fill(tmp_path, monkeypatch):
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    selector = PCOTMRASPreBackboneFrameSelector(
        reader=dict(
            type="PCOTMRASCoarseActionnessFrameScout",
            in_dim=4,
            hidden_dim=4,
            num_slots=6,
            temporal_layers=1,
            temporal_kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        ),
        target_len=6,
        dense_window_size=8,
        descriptor_dim=4,
        selection_strategy="coarse_actionness_uncertainty",
        coarse_uniform_count=0,
        coarse_action_count=2,
        coarse_uncertainty_count=1,
        coarse_change_count=1,
        coarse_background_count=0,
        max_dense_gap=0,
        max_gap_guard_count=0,
        remap_gt_to_selected_axis=False,
        straight_through_detector_loss=False,
    )
    candidate_valid = torch.ones(1, 8, dtype=torch.bool)
    candidate_dense_indices = torch.arange(8, dtype=torch.long).unsqueeze(0)
    valid = candidate_valid.clone()
    action_logits = torch.tensor([[0.0, 6.0, -6.0, 0.0, 6.0, -6.0, -6.0, -6.0]])

    scores = selector._coarse_actionness_scores(
        reader_outputs={"actionness_logits": action_logits},
        candidate_valid=candidate_valid,
    )
    for key in ("p_action", "entropy", "p_change", "margin"):
        assert key in scores
        assert scores[key].shape == candidate_valid.shape

    plan = selector._coarse_actionness_uncertainty_transport_plan(
        reader_outputs={"actionness_logits": action_logits},
        valid=valid,
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        training=False,
    )
    policy = plan["coarse_policy_meta"][0]
    candidate_points = policy["candidate_points"]
    selected_roles = plan["selected_roles"][0][:6]

    assert "coarse_mixed_fill" in selected_roles
    assert "dense_fill" not in selected_roles
    assert len(candidate_points) == 8
    assert any(len(point["eligible_roles"]) >= 2 for point in candidate_points)
    mixed_eligible = [point for point in candidate_points if "coarse_mixed_fill" in point["eligible_roles"]]
    assert 0 < len(mixed_eligible) < len(candidate_points)
    for point in candidate_points:
        assert set(
            [
                "candidate_idx",
                "dense_index",
                "valid",
                "final_role",
                "source_score_role",
                "eligible_roles",
                "components",
            ]
        ).issubset(point)
        assert point["candidate_idx"] == point["dense_index"]
        assert point["valid"] is True
        components = point["components"]
        for key in ("p_action", "entropy", "p_change", "margin"):
            assert key in components
            assert 0.0 <= components[key] <= 1.0

    dump_path = tmp_path / "selector_metadata.jsonl"
    monkeypatch.setenv("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL", str(dump_path))
    meta = {"pc_ot_mras_prebackbone_coarse_actionness_policy": policy}
    selector._append_metadata_dump_row(
        meta=meta,
        batch_idx=0,
        selected_dense_indices=[int(item) for item in plan["selected_positions"][0][:6].tolist()],
        valid_len=8,
        gt_segments=None,
        reader_outputs={"actionness_logits": action_logits},
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        training=False,
    )
    row = json.loads(dump_path.read_text(encoding="utf-8").strip())
    components = row["selector_score_components"]
    for key in ("p_action", "entropy", "p_change", "margin"):
        assert key in components
        assert len(components[key]) == 8
    assert row["selector_candidate_points"] == candidate_points


def test_exact_uniform_c3_config_plan_uses_only_uniform_roles_with_dense_coverage():
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py"
    )
    selector_cfg = dict(cfg.model.frame_selector)
    selector_cfg.pop("type")
    selector = PCOTMRASPreBackboneFrameSelector(**selector_cfg)
    dense_len = int(selector.dense_window_size)
    candidate_valid = torch.ones(1, dense_len, dtype=torch.bool)
    candidate_dense_indices = torch.arange(dense_len, dtype=torch.long).unsqueeze(0)
    valid = candidate_valid.clone()
    action_logits = torch.linspace(-4.0, 4.0, dense_len, dtype=torch.float32).unsqueeze(0)

    plan = selector._coarse_actionness_uncertainty_transport_plan(
        reader_outputs={"actionness_logits": action_logits},
        valid=valid,
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        training=False,
    )
    positions = [int(item) for item in plan["selected_positions"][0].tolist()]
    gaps = [right - left for left, right in zip(positions[:-1], positions[1:])]
    role_counts = plan["coarse_policy_meta"][0]["role_counts"]

    assert plan["selected_output_valid_lengths"].tolist() == [384]
    assert set(plan["selected_roles"][0]) == {"coarse_uniform"}
    assert role_counts == {"coarse_uniform": 384}
    assert positions == sorted(positions)
    assert len(set(positions)) == 384
    assert max(gaps) <= 3
    assert plan["coarse_policy_meta"][0]["configured_quota"]["coarse_action"] == 0
    assert plan["st_active_row_counts"] == [0]


def test_uniform_biased_c3_config_plan_preserves_actionness_bias_under_max_gap3():
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer_n16r4.py"
    )
    selector_cfg = dict(cfg.model.frame_selector)
    selector_cfg.pop("type")
    selector = PCOTMRASPreBackboneFrameSelector(**selector_cfg)
    dense_len = int(selector.dense_window_size)
    candidate_valid = torch.ones(1, dense_len, dtype=torch.bool)
    candidate_dense_indices = torch.arange(dense_len, dtype=torch.long).unsqueeze(0)
    valid = candidate_valid.clone()
    action_logits = torch.full((1, dense_len), -4.0, dtype=torch.float32)
    action_logits[:, 1::8] = 6.0
    action_logits[:, 3::8] = 0.0

    plan = selector._coarse_actionness_uncertainty_transport_plan(
        reader_outputs={"actionness_logits": action_logits},
        valid=valid,
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        training=False,
    )
    positions = [int(item) for item in plan["selected_positions"][0].tolist()]
    gaps = [right - left for left, right in zip(positions[:-1], positions[1:])]
    role_counts = plan["coarse_policy_meta"][0]["role_counts"]

    assert plan["selected_output_valid_lengths"].tolist() == [384]
    assert positions == sorted(positions)
    assert len(set(positions)) == 384
    assert max(gaps) <= 3
    assert role_counts["coarse_uniform"] == 288
    assert role_counts.get("coarse_action", 0) > 0
    assert role_counts.get("coarse_uncertainty", 0) > 0
    assert role_counts.get("coarse_max_gap_guard", 0) > 0
    assert "coarse_change" not in role_counts
    assert "coarse_background" not in role_counts
    assert plan["coarse_policy_meta"][0]["configured_quota"]["coarse_action"] == 72
    assert plan["max_gap_guard_meta"][0]["enabled"] is True
    assert plan["max_gap_guard_meta"][0]["max_dense_gap"] == 3
    assert plan["max_gap_guard_meta"][0]["max_gap_guard_count"] == 12


def test_coarse_actionness_selector_forward_train_writes_policy_metadata_and_action_loss():
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    selector = PCOTMRASPreBackboneFrameSelector(
        reader=dict(
            type="PCOTMRASCoarseActionnessFrameScout",
            in_dim=4,
            hidden_dim=4,
            num_slots=4,
            temporal_layers=1,
            temporal_kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        ),
        target_len=4,
        dense_window_size=12,
        descriptor_dim=4,
        selection_strategy="coarse_actionness_uncertainty",
        coarse_uniform_count=1,
        coarse_action_count=1,
        coarse_uncertainty_count=1,
        coarse_change_count=1,
        coarse_background_count=0,
        max_dense_gap=0,
        max_gap_guard_count=0,
        remap_gt_to_selected_axis=False,
        aux_gt_acquisition_loss_weight=0.1,
        reader_regularizer_loss_weight=0.0,
    )
    inputs = torch.arange(1 * 3 * 12 * 2 * 2, dtype=torch.float32).reshape(1, 3, 12, 2, 2)
    masks = torch.ones(1, 12, dtype=torch.bool)
    selected = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "coarse-actionness"}],
        gt_segments=[torch.tensor([[2.0, 7.0]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
    )
    meta = selected["metas"][0]

    assert selected["inputs"].shape[2] == 4
    assert selected["masks"].shape == (1, 4)
    assert "selector_gt_actionness_loss" in selected["losses"]
    assert meta["pc_ot_mras_prebackbone_selection_strategy"] == "coarse_actionness_uncertainty"
    assert meta["pc_ot_mras_prebackbone_hard_selection_source"] == "classification_probability_uncertainty_change"
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["enabled"] is True
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["uses_learned_boundary_head"] is False
    assert meta["pc_ot_mras_prebackbone_protocol_flags"]["uses_test_gt"] is False


def test_coarse_actionness_selector_forward_test_writes_deploy_safe_policy_metadata_without_gt(tmp_path, monkeypatch):
    PCOTMRASPreBackboneFrameSelector = _install_prebackbone_selector_or_skip()
    selector = PCOTMRASPreBackboneFrameSelector(
        reader=dict(
            type="PCOTMRASCoarseActionnessFrameScout",
            in_dim=4,
            hidden_dim=4,
            num_slots=4,
            temporal_layers=1,
            temporal_kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        ),
        target_len=4,
        dense_window_size=8,
        descriptor_dim=4,
        selection_strategy="coarse_actionness_uncertainty",
        coarse_uniform_count=1,
        coarse_action_count=1,
        coarse_uncertainty_count=1,
        coarse_change_count=1,
        coarse_background_count=0,
        max_dense_gap=0,
        max_gap_guard_count=0,
        remap_gt_to_selected_axis=False,
        straight_through_detector_loss=False,
    )
    inputs = torch.arange(1 * 3 * 8 * 2 * 2, dtype=torch.float32).reshape(1, 3, 8, 2, 2)
    masks = torch.ones(1, 8, dtype=torch.bool)
    dump_path = tmp_path / "selector_metadata.jsonl"
    monkeypatch.setenv("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL", str(dump_path))

    selected = selector.forward_test(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "coarse-actionness-forward-test"}],
    )
    meta = selected["metas"][0]

    assert selected["inputs"].shape[2] == 4
    assert selected["masks"].shape == (1, 4)
    assert selected["masks"].all()
    assert selected["inputs"].shape[2] * 2 == inputs.shape[2]
    assert meta["irregular_native_axis"] is True
    assert meta["irregular_dense_valid_len"] == 8
    assert meta["irregular_selected_count"] == 4
    assert meta["selected_valid_len"] == 4
    assert meta["pc_ot_mras_prebackbone_selection_strategy"] == "coarse_actionness_uncertainty"
    assert meta["pc_ot_mras_prebackbone_hard_selection_source"] == "classification_probability_uncertainty_change"
    assert len(meta["irregular_selected_positions"]) == 4
    assert len(meta["pc_ot_mras_prebackbone_selected_dense_indices"]) == 4
    assert all(0 <= pos < 8 for pos in meta["pc_ot_mras_prebackbone_selected_dense_indices"])
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["enabled"] is True
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["protocol"] == "binary_actionness_uncertainty_v0"
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["uses_gt"] is False
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["uses_teacher"] is False
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["uses_raw_prediction_cache"] is False
    assert meta["pc_ot_mras_prebackbone_coarse_actionness_policy"]["uses_learned_boundary_head"] is False
    assert meta["pc_ot_mras_prebackbone_protocol_flags"]["uses_gt"] is False
    assert meta["pc_ot_mras_prebackbone_protocol_flags"]["uses_test_gt"] is False
    assert meta["pc_ot_mras_prebackbone_protocol_flags"]["uses_teacher"] is False
    assert meta["pc_ot_mras_prebackbone_protocol_flags"]["uses_raw_prediction_cache"] is False
    assert meta["pc_ot_mras_prebackbone_protocol_flags"]["uses_learned_boundary_head"] is False
    diagnostics = meta["pc_ot_mras_prebackbone_reader_diagnostics"]
    assert diagnostics["action"]["available"] is True
    assert diagnostics["boundary"]["available"] is False
    row = json.loads(dump_path.read_text(encoding="utf-8").strip())
    components = row["selector_score_components"]
    for key in ("actionness_logits", "p_action", "uncertainty", "change", "background", "mixed"):
        assert key in components
        assert len(components[key]) == 8
    for key in ("p_action", "uncertainty", "change", "background", "mixed"):
        assert all(0.0 <= value <= 1.0 for value in components[key])


def test_coarse_actionness_uncertainty_config_keeps_boundary_head_out_of_selector():
    cfg = load_mmengine_config_or_skip(
        "configs/adatad/thumos/pc_ot_mras_coarse_actionness_uncertainty_c3_physical_grid_actionformer_n16r4.py"
    )

    assert cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.route_family == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.experiment_scope.selection_strategy == "coarse_actionness_uncertainty"
    assert cfg.experiment_scope.uses_learned_boundary_head is False
    assert cfg.protocol_flags.uses_learned_boundary_head is False
    assert cfg.model.frame_selector.selection_strategy == "coarse_actionness_uncertainty"
    assert cfg.model.frame_selector.reader.type == "PCOTMRASCoarseActionnessFrameScout"
    assert cfg.model.frame_selector.aux_frame_score_boundary_loss_weight == 0.0
    assert cfg.model.frame_selector.aux_risk_loss_weight == 0.0
    assert cfg.model.frame_selector.aux_uncertainty_loss_weight == 0.0
    assert cfg.protocol_flags.remote_sync_allowed is True
    assert cfg.protocol_flags.slurm_allowed is True
    assert cfg.protocol_flags.tools_test_allowed is False
    assert cfg.protocol_flags.tools_train_allowed is True
    assert cfg.protocol_flags.metric_claim_allowed is False
    assert cfg.protocol_flags.paper_claim_allowed is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.route == cfg.route_id
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.stage == cfg.stage_id
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_slurm is True
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_tools_test is False
