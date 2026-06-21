import importlib.util
import subprocess
import sys
import types
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
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
READER_OUTPUTS_META_KEY = "pc_ot_mras_reader_outputs"


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
        if type_name not in self._items:
            raise KeyError(type_name)
        return self._items[type_name](**cfg)


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


class _SyntheticScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return x * self.scale


class _SyntheticAffineDropPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return x * self.scale


def _install_actionformer_runtime():
    for name in (
        "opentad.ctf_bdi_role_constants",
        "opentad.models.builder",
        "opentad.models.utils.post_processing",
        "opentad.models.backbones",
        "opentad.models.selectors.lowcost_acquisition_browser",
        "opentad.models.selectors.pc_ot_mras_reader",
        "opentad.models.losses.pc_ot_mras_auxiliary_losses",
        "opentad.models.losses.pc_ot_mras_soft_hard_consistency_losses",
        "opentad.models.losses.pc_ot_mras_value_distillation_losses",
        "opentad.models.necks.pc_ot_mras_detector_bridge",
        "opentad.models.detectors.base",
        "opentad.models.detectors.single_stage",
        "opentad.models.detectors.actionformer",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.detectors", ROOT / "opentad" / "models" / "detectors")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.necks", ROOT / "opentad" / "models" / "necks")
    _ensure_package("opentad.models.losses", ROOT / "opentad" / "models" / "losses")
    _ensure_package("opentad.models.bricks", ROOT / "opentad" / "models" / "bricks")
    _ensure_package("opentad.models.utils", ROOT / "opentad" / "models" / "utils")

    bricks = sys.modules["opentad.models.bricks"]
    bricks.Scale = _SyntheticScale
    bricks.AffineDropPath = _SyntheticAffineDropPath

    backbones = types.ModuleType("opentad.models.backbones")
    backbones.BackboneWrapper = lambda cfg: (_ for _ in ()).throw(
        RuntimeError("PC-OT-MRAS ActionFormer smoke must not build a backbone")
    )
    sys.modules["opentad.models.backbones"] = backbones

    post_processing = types.ModuleType("opentad.models.utils.post_processing")
    post_processing.load_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("PC-OT-MRAS ActionFormer smoke must not load predictions")
    )
    post_processing.save_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("PC-OT-MRAS ActionFormer smoke must not save predictions")
    )
    post_processing.batched_nms = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("PC-OT-MRAS ActionFormer smoke must not run NMS")
    )
    post_processing.convert_to_seconds = lambda segments, meta: segments
    sys.modules["opentad.models.utils.post_processing"] = post_processing

    builder = types.ModuleType("opentad.models.builder")
    builder.MODELS = _Registry()
    builder.SELECTORS = builder.MODELS
    builder.NECKS = builder.MODELS
    builder.HEADS = builder.MODELS
    builder.DETECTORS = builder.MODELS
    builder.PROJECTIONS = builder.MODELS
    builder.TOKEN_COMPRESSORS = builder.MODELS
    builder.ROI_EXTRACTORS = builder.MODELS
    builder.PRIOR_GENERATORS = builder.MODELS
    builder.PROPOSAL_GENERATORS = builder.MODELS
    builder.TRANSFORMERS = builder.MODELS
    builder.LOSSES = builder.MODELS
    builder.MATCHERS = builder.MODELS
    builder.build_backbone = lambda cfg: backbones.BackboneWrapper(cfg)
    builder.build_projection = lambda cfg: builder.PROJECTIONS.build(cfg)
    builder.build_selector = lambda cfg: builder.SELECTORS.build(cfg)
    builder.build_token_compressor = lambda cfg: builder.TOKEN_COMPRESSORS.build(cfg)
    builder.build_neck = lambda cfg: builder.NECKS.build(cfg)
    builder.build_head = lambda cfg: builder.HEADS.build(cfg)
    sys.modules["opentad.models.builder"] = builder

    class SyntheticIdentityProjection(nn.Module):
        def __init__(self, channels=4, max_seq_len=8):
            super().__init__()
            self.n_mha_win_size = 1
            self.arch = (0,)
            self.max_seq_len = int(max_seq_len)
            self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=False)
            nn.init.eye_(self.proj.weight[:, :, 0])

        def forward(self, x, masks):
            return (self.proj(x),), (masks,)

    class SyntheticFlatProjection(nn.Module):
        def __init__(self, channels=4, max_seq_len=8):
            super().__init__()
            self.n_mha_win_size = 1
            self.arch = (0,)
            self.max_seq_len = int(max_seq_len)
            self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=False)
            nn.init.eye_(self.proj.weight[:, :, 0])

        def forward(self, x, masks):
            return self.proj(x), masks

    class SyntheticRPNHead(nn.Module):
        def __init__(self, in_channels=4, num_classes=3):
            super().__init__()
            self.prior_generator = types.SimpleNamespace(strides=[1])
            self.num_classes = int(num_classes)
            self.score = nn.Conv1d(int(in_channels), self.num_classes, kernel_size=1)
            self.last_train_metas = None
            self.last_test_metas = None

        def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None):
            self.last_train_metas = metas
            feat = feat_list[0]
            mask = mask_list[0]
            valid_feat = feat.masked_fill(~mask[:, None, :], 0.0)
            logits = self.score(valid_feat)
            return {"detector_loss": logits.square().mean() + valid_feat.square().mean()}

        def forward_test(self, feat_list, mask_list, metas=None):
            self.last_test_metas = metas
            feat = feat_list[0]
            mask = mask_list[0]
            batch, _, time = feat.shape
            proposals = []
            scores = []
            for batch_idx in range(batch):
                valid_count = int(mask[batch_idx].long().sum().item())
                starts = torch.arange(valid_count, device=feat.device, dtype=feat.dtype)
                if valid_count <= 0:
                    proposals.append(torch.empty((0, 2), device=feat.device, dtype=feat.dtype))
                    scores.append(torch.empty((0, self.num_classes), device=feat.device, dtype=feat.dtype))
                    continue
                proposals.append(torch.stack([starts, starts + 1.0], dim=-1))
                score = self.score(feat[batch_idx : batch_idx + 1])[:, :, :valid_count]
                scores.append(score.squeeze(0).transpose(0, 1).sigmoid())
            return proposals, scores

    builder.PROJECTIONS.register_module()(SyntheticIdentityProjection)
    builder.PROJECTIONS.register_module()(SyntheticFlatProjection)
    builder.HEADS.register_module()(SyntheticRPNHead)

    _load_module("opentad.ctf_bdi_role_constants", ROOT / "opentad" / "ctf_bdi_role_constants.py")
    _load_module(
        "opentad.models.selectors.lowcost_acquisition_browser",
        ROOT / "opentad" / "models" / "selectors" / "lowcost_acquisition_browser.py",
    )
    reader_module = _load_module(
        "opentad.models.selectors.pc_ot_mras_reader",
        ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py",
    )
    bridge_module = _load_module(
        "opentad.models.necks.pc_ot_mras_detector_bridge",
        ROOT / "opentad" / "models" / "necks" / "pc_ot_mras_detector_bridge.py",
    )
    _load_module("opentad.models.detectors.base", ROOT / "opentad" / "models" / "detectors" / "base.py")
    _load_module(
        "opentad.models.detectors.single_stage",
        ROOT / "opentad" / "models" / "detectors" / "single_stage.py",
    )
    actionformer_module = _load_module(
        "opentad.models.detectors.actionformer",
        ROOT / "opentad" / "models" / "detectors" / "actionformer.py",
    )
    return actionformer_module.ActionFormer, reader_module.PCOTMRASReader, bridge_module.PCOTMRASDetectorBridge


ActionFormer, PCOTMRASReader, PCOTMRASDetectorBridge = _install_actionformer_runtime()


def _model(
    projection_type="SyntheticIdentityProjection",
    pc_ot_mras_reader_aux_loss=None,
    pc_ot_mras_reader_soft_hard_loss=None,
    pc_ot_mras_reader_value_loss=None,
    pc_ot_mras_reader_eval_override=None,
    enable_value_heads=False,
):
    return ActionFormer(
        projection=dict(type=projection_type, channels=4, max_seq_len=8),
        neck=dict(type="PCOTMRASDetectorBridge", in_channels=4, out_channels=4),
        rpn_head=dict(type="SyntheticRPNHead", in_channels=4, num_classes=3),
        pc_ot_mras_reader=dict(
            type="PCOTMRASReader",
            in_dim=4,
            hidden_dim=8,
            num_slots=3,
            num_blocks=1,
            num_roles=6,
            enable_value_heads=enable_value_heads,
        ),
        pc_ot_mras_reader_aux_loss=pc_ot_mras_reader_aux_loss,
        pc_ot_mras_reader_soft_hard_loss=pc_ot_mras_reader_soft_hard_loss,
        pc_ot_mras_reader_value_loss=pc_ot_mras_reader_value_loss,
        pc_ot_mras_reader_eval_override=pc_ot_mras_reader_eval_override,
    )


def _inputs():
    torch.manual_seed(20260619)
    inputs = torch.randn(1, 4, 8, requires_grad=True)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [{"sample_id": "pc_ot_actionformer_forward_selector|0"}]
    gt_segments = [torch.tensor([[1.0, 6.5]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]
    return inputs, masks, metas, gt_segments, gt_labels


def _hybrid_value_target(time=8, slots=3):
    value = [0.0] * time
    risk = [0.0] * time
    redundancy = [0.0] * time
    value[3] = 1.0
    value[4] = 0.75
    risk[7] = 1.0
    redundancy[0] = 1.0
    return dict(
        schema_version="pc_ot_mras_value_targets_v0",
        target_family="pc_ot_mras_voi_distill_hybrid_v0",
        split="train",
        source_split="train",
        training_only=True,
        forbidden_splits=["val", "validation", "test"],
        counterfactual_utility_ready=True,
        diagnostic_only=False,
        synthetic_unit_test=True,
        target_source="unit_counterfactual_value_distill_v0",
        unit="local_dense_index",
        reader_axis="projection_level0_after_pad",
        dense_len=time,
        target_len=slots,
        valid_len=time,
        index_base=0,
        sample_id="pc_ot_actionformer_forward_selector|0",
        uses_gt=True,
        uses_teacher=False,
        uses_cache=False,
        uses_prediction_cache=False,
        uses_raw_prediction=False,
        uses_raw_predictions=False,
        contains_raw_predictions=False,
        contains_detection_results=False,
        value_scores=value,
        risk_scores=risk,
        redundancy_scores=redundancy,
        valid_mask=[1] * time,
        operations=[
            dict(
                operation_id="unit|swap0|0->3",
                operation_type="delete_add_swap",
                add_index=3,
                delete_index=0,
                utility=1.0,
                label_positive=True,
                label_negative=False,
                label_uncertain=False,
                split="train",
                training_only=True,
                target_source="unit_counterfactual_value_distill_v0",
                counterfactual_utility_ready=True,
                diagnostic_only=False,
                dense_len=time,
                target_len=slots,
                uses_gt=True,
                uses_teacher=False,
                uses_cache=False,
                uses_prediction_cache=False,
                uses_raw_prediction=False,
            )
        ],
    )


def test_pc_ot_mras_reader_runs_inside_actionformer_after_projection_before_neck_and_backprops():
    model = _model()
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    calls = []
    original_forward = model.pc_ot_mras_reader.forward

    def spy_forward(lowcost_features, valid_mask):
        calls.append((tuple(lowcost_features.shape), tuple(valid_mask.shape), bool(lowcost_features.requires_grad)))
        return original_forward(lowcost_features, valid_mask)

    model.pc_ot_mras_reader.forward = spy_forward

    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert "cost" in losses
    assert "detector_loss" in losses
    assert not any(key.startswith("pc_ot_mras_aux_") for key in losses)
    assert torch.isfinite(losses["cost"]).item()
    assert calls == [((1, 8, 4), (1, 8), True)]
    losses["cost"].backward()

    assert READER_OUTPUTS_META_KEY not in metas[0]
    train_metas = model.rpn_head.last_train_metas
    assert isinstance(train_metas, list)
    assert READER_OUTPUTS_META_KEY in train_metas[0]
    assert "pc_ot_mras_bridge" in train_metas[0]
    assert train_metas[0]["pc_ot_mras_bridge"]["uses_hard_gather"] is False
    assert train_metas[0]["pc_ot_mras_bridge"]["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert train_metas[0]["pc_ot_mras_bridge"]["selected_tokens_source_verified"] is True

    reader_outputs = train_metas[0][READER_OUTPUTS_META_KEY]
    assert reader_outputs["allocation"].shape == (1, 3, 8)
    assert reader_outputs["acquisition_matrix"].shape == (1, 3, 8)
    assert reader_outputs["selected_mask"].shape == (1, 3)
    assert torch.equal(reader_outputs["valid_mask"], masks)
    assert torch.all(reader_outputs["allocation"][:, :, :] >= 0)

    assert model.pc_ot_mras_reader.input_proj.weight.grad is not None
    assert torch.isfinite(model.pc_ot_mras_reader.input_proj.weight.grad).all()
    assert model.pc_ot_mras_reader.input_proj.weight.grad.abs().sum().item() > 0
    assert model.pc_ot_mras_reader.query_embed.grad is not None
    assert torch.isfinite(model.pc_ot_mras_reader.query_embed.grad).all()
    assert model.pc_ot_mras_reader.query_embed.grad.abs().sum().item() > 0
    for reader_param in (
        model.pc_ot_mras_reader.key_proj.weight,
        model.pc_ot_mras_reader.center_inc_head.weight,
        model.pc_ot_mras_reader.width_head.weight,
        model.pc_ot_mras_reader.gate_head.weight,
        model.pc_ot_mras_reader.boundary_head.weight,
    ):
        assert reader_param.grad is not None
        assert torch.isfinite(reader_param.grad).all()
        assert reader_param.grad.abs().sum().item() > 0
    assert model.projection.proj.weight.grad is not None
    assert torch.isfinite(model.projection.proj.weight.grad).all()
    assert model.projection.proj.weight.grad.abs().sum().item() > 0
    assert model.neck.token_proj.weight.grad is not None
    assert torch.isfinite(model.neck.token_proj.weight.grad).all()
    assert model.neck.time_proj.weight.grad is not None
    assert torch.isfinite(model.neck.time_proj.weight.grad).all()
    assert model.neck.time_proj.weight.grad.abs().sum().item() > 0
    assert model.rpn_head.score.weight.grad is not None
    assert torch.isfinite(model.rpn_head.score.weight.grad).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum().item() > 0


def test_actionformer_eval_override_bypasses_reader_and_emits_exact_uniform_payload():
    model = _model(pc_ot_mras_reader_eval_override=dict(enabled=True, mode="exact_uniform", num_slots=3))
    model.eval()
    inputs = torch.randn(1, 4, 8)
    masks = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    metas = [{"sample_id": "reader_disabled_eval_override|0"}]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("eval override must bypass the learned reader")

    model.pc_ot_mras_reader.forward = fail_if_called
    with torch.no_grad():
        model.forward_test(inputs, masks, metas=metas)

    test_metas = model.rpn_head.last_test_metas
    assert READER_OUTPUTS_META_KEY in test_metas[0]
    reader_outputs = test_metas[0][READER_OUTPUTS_META_KEY]
    assert reader_outputs["schema_version"] == "pc_ot_mras_eval_override_reader_outputs_v0"
    assert reader_outputs["override_mode"] == "exact_uniform"
    assert torch.equal(reader_outputs["valid_mask"], masks)
    assert reader_outputs["allocation"].shape == (1, 3, 8)
    assert reader_outputs["selected_mask"].tolist() == [[True, True, True]]
    assert torch.equal(reader_outputs["allocation"][0].argmax(dim=-1), torch.tensor([0, 2, 4]))
    assert torch.allclose(reader_outputs["allocation"].sum(dim=-1), torch.ones(1, 3))
    assert torch.all(reader_outputs["allocation"][0, :, 5:] == 0)
    assert torch.allclose(reader_outputs["selected_times"][0], torch.tensor([0.0, 0.5, 1.0]))
    assert test_metas[0]["pc_ot_mras_bridge"]["selected_tokens_source"] == "recomputed_from_acquisition_matrix"


def test_actionformer_eval_override_uses_prefix_selected_mask_for_short_inputs():
    model = _model(pc_ot_mras_reader_eval_override=dict(enabled=True, mode="exact_uniform", num_slots=6))
    model.eval()
    features = torch.randn(1, 4, 8)
    masks = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)

    reader_outputs = model._pc_ot_mras_eval_override_outputs(features, masks)

    assert reader_outputs["selected_mask"].tolist() == [[True, True, True, False, False, False]]
    assert torch.equal(reader_outputs["allocation"][0, :3].argmax(dim=-1), torch.tensor([0, 1, 2]))
    assert torch.all(reader_outputs["allocation"][0, 3:] == 0)
    assert torch.all(reader_outputs["centers"][0, 3:] == 0)
    assert torch.all(reader_outputs["widths"][0, 3:] == 0)


def test_pc_ot_mras_reader_aux_loss_is_train_only_opt_in_and_backprops_to_reader_heads():
    model = _model(
        pc_ot_mras_reader_aux_loss=dict(
            enabled=True,
            weights=dict(
                body=0.02,
                start=0.05,
                end=0.05,
                boundary=0.05,
                uncertainty=0.01,
                redundancy=0.005,
                process=0.01,
                pair=0.05,
                allocation=0.02,
                regularizer=0.01,
            ),
            boundary_sigma=1.5,
            short_action_len=4.0,
            adjacent_gap=2.0,
        )
    )
    inputs, masks, metas, gt_segments, gt_labels = _inputs()

    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    aux_keys = {key for key in losses if key.startswith("pc_ot_mras_aux_")}
    expected_aux_keys = {
        "pc_ot_mras_aux_body_loss",
        "pc_ot_mras_aux_start_loss",
        "pc_ot_mras_aux_end_loss",
        "pc_ot_mras_aux_boundary_loss",
        "pc_ot_mras_aux_uncertainty_loss",
        "pc_ot_mras_aux_redundancy_loss",
        "pc_ot_mras_aux_process_loss",
        "pc_ot_mras_aux_pair_loss",
        "pc_ot_mras_aux_allocation_loss",
        "pc_ot_mras_aux_regularizer_loss",
    }
    assert expected_aux_keys <= aux_keys
    assert all(torch.isfinite(losses[key]).item() for key in expected_aux_keys)
    expected_cost = sum(value for key, value in losses.items() if key != "cost")
    assert torch.allclose(losses["cost"], expected_cost)

    losses["cost"].backward()

    for reader_param in (
        model.pc_ot_mras_reader.start_head.weight,
        model.pc_ot_mras_reader.end_head.weight,
        model.pc_ot_mras_reader.boundary_head.weight,
        model.pc_ot_mras_reader.body_head.weight,
        model.pc_ot_mras_reader.uncertainty_head.weight,
        model.pc_ot_mras_reader.redundancy_head.weight,
        model.pc_ot_mras_reader.process_head.weight,
        model.pc_ot_mras_reader.pair_scorer[-1].weight,
    ):
        assert reader_param.grad is not None
        assert torch.isfinite(reader_param.grad).all()
        assert reader_param.grad.abs().sum().item() > 0


def test_pc_ot_mras_reader_aux_loss_disabled_config_preserves_default_loss_surface():
    model = _model(pc_ot_mras_reader_aux_loss=dict(enabled=False, weights=dict(start=1.0)))
    inputs, masks, metas, gt_segments, gt_labels = _inputs()

    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert "detector_loss" in losses
    assert not any(key.startswith("pc_ot_mras_aux_") for key in losses)


def test_actionformer_soft_hard_consistency_loss_is_train_only_and_backprops_to_reader():
    model = _model(
        pc_ot_mras_reader_soft_hard_loss=dict(
            enabled=True,
            weights=dict(
                slot_allocation=0.02,
                global_acquisition=0.02,
                selected_time=0.01,
                gate_confidence=0.005,
                duplicate_mass=0.005,
            ),
        )
    )
    inputs, masks, metas, gt_segments, gt_labels = _inputs()

    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    soft_hard_keys = {key for key in losses if key.startswith("pc_ot_mras_soft_hard_")}
    assert {
        "pc_ot_mras_soft_hard_slot_allocation_loss",
        "pc_ot_mras_soft_hard_global_acquisition_loss",
        "pc_ot_mras_soft_hard_selected_time_loss",
        "pc_ot_mras_soft_hard_gate_confidence_loss",
        "pc_ot_mras_soft_hard_duplicate_mass_loss",
    } <= soft_hard_keys
    expected_cost = sum(value for key, value in losses.items() if key != "cost")
    assert torch.allclose(losses["cost"], expected_cost)

    losses["cost"].backward()
    for name, param in {
        "key_proj": model.pc_ot_mras_reader.key_proj.weight,
        "gate_head": model.pc_ot_mras_reader.gate_head.weight,
        "center_inc_head": model.pc_ot_mras_reader.center_inc_head.weight,
    }.items():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0, name


def test_actionformer_soft_hard_consistency_loss_does_not_run_in_forward_test():
    model = _model(pc_ot_mras_reader_soft_hard_loss=dict(enabled=True))
    inputs, masks, metas, gt_segments, gt_labels = _inputs()

    def fail_if_called(_metas):
        raise AssertionError("soft-hard consistency loss must not run during forward_test")

    model._pc_ot_mras_reader_soft_hard_losses = fail_if_called
    model.eval()

    with torch.no_grad():
        proposals, scores = model.forward_test(inputs, masks, metas=metas)

    assert len(proposals) == 1
    assert len(scores) == 1
    assert READER_OUTPUTS_META_KEY in model.rpn_head.last_test_metas[0]


def test_actionformer_value_loss_train_only_and_strips_targets_before_head():
    model = _model(
        enable_value_heads=True,
        pc_ot_mras_reader_value_loss=dict(
            enabled=True,
            require_targets=True,
            target_source="unit_counterfactual_value_distill_v0",
            allow_train_gt=True,
            allow_teacher_targets=False,
        ),
    )
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    metas[0]["pc_ot_mras_value_targets"] = _hybrid_value_target()

    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    value_keys = {key for key in losses if key.startswith("pc_ot_mras_value_")}
    assert {
        "pc_ot_mras_value_dense_value_loss",
        "pc_ot_mras_value_dense_risk_loss",
        "pc_ot_mras_value_pair_operation_loss",
    } <= value_keys
    assert all(torch.isfinite(losses[key]).item() for key in value_keys)
    expected_cost = sum(value for key, value in losses.items() if key != "cost")
    assert torch.allclose(losses["cost"], expected_cost)
    assert "pc_ot_mras_value_targets" in metas[0]

    train_metas = model.rpn_head.last_train_metas
    assert isinstance(train_metas, list)
    assert READER_OUTPUTS_META_KEY in train_metas[0]
    assert "pc_ot_mras_bridge" in train_metas[0]
    forbidden = {
        "pc_ot_mras_value_targets",
        "value_transport_targets",
        "teacher_value_targets",
        "pc_ot_mras_value_manifest",
    }
    assert forbidden.isdisjoint(train_metas[0])

    losses["cost"].backward()
    for name, param in {
        "value_head": model.pc_ot_mras_reader.value_head.weight,
        "risk_head": model.pc_ot_mras_reader.risk_head.weight,
        "redundancy_head": model.pc_ot_mras_reader.redundancy_head.weight,
        "gate_head": model.pc_ot_mras_reader.gate_head.weight,
        "key_proj": model.pc_ot_mras_reader.key_proj.weight,
    }.items():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0, name


def test_pc_ot_mras_reader_aux_loss_merge_refuses_cost_key_collision():
    model = _model(pc_ot_mras_reader_aux_loss=dict(enabled=True, weights=dict(start=0.0)))
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    model._pc_ot_mras_reader_auxiliary_losses = lambda _metas, _segments: {"cost": inputs.sum() * 0.0}

    with pytest.raises(ValueError, match="must not return a cost key"):
        model.forward_train(
            inputs,
            masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
        )


def test_pc_ot_mras_reader_runs_inside_actionformer_forward_test_after_projection():
    model = _model()
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    model.eval()
    calls = []
    original_forward = model.pc_ot_mras_reader.forward

    def spy_forward(lowcost_features, valid_mask):
        calls.append((tuple(lowcost_features.shape), tuple(valid_mask.shape)))
        return original_forward(lowcost_features, valid_mask)

    model.pc_ot_mras_reader.forward = spy_forward

    with torch.no_grad():
        proposals, scores = model.forward_test(inputs, masks, metas=metas)

    assert calls == [((1, 8, 4), (1, 8))]
    assert READER_OUTPUTS_META_KEY not in metas[0]
    assert len(proposals) == 1
    assert len(scores) == 1
    assert proposals[0].shape == (3, 2)
    assert scores[0].shape == (3, 3)
    assert torch.isfinite(proposals[0]).all()
    assert torch.isfinite(scores[0]).all()
    assert torch.all(proposals[0][:, 1] > proposals[0][:, 0])
    test_metas = model.rpn_head.last_test_metas
    assert READER_OUTPUTS_META_KEY in test_metas[0]
    assert test_metas[0]["pc_ot_mras_bridge"]["continuous_axis"] is True
    assert test_metas[0]["irregular_native_axis"] is True
    assert len(test_metas[0]["irregular_selected_positions"]) == 3


def test_pc_ot_mras_reader_aux_loss_opt_in_does_not_run_in_forward_test():
    model = _model(pc_ot_mras_reader_aux_loss=dict(enabled=True, weights=dict(start=0.05)))
    inputs, masks, metas, gt_segments, gt_labels = _inputs()

    def fail_if_called(_metas, _segments):
        raise AssertionError("reader aux loss must not run during forward_test")

    model._pc_ot_mras_reader_auxiliary_losses = fail_if_called
    model.eval()

    with torch.no_grad():
        proposals, scores = model.forward_test(inputs, masks, metas=metas)

    assert len(proposals) == 1
    assert len(scores) == 1
    assert READER_OUTPUTS_META_KEY in model.rpn_head.last_test_metas[0]


def test_actionformer_value_loss_does_not_run_in_forward_test_and_rejects_targets():
    model = _model(
        enable_value_heads=True,
        pc_ot_mras_reader_value_loss=dict(
            enabled=True,
            require_targets=True,
            target_source="unit_counterfactual_value_distill_v0",
            allow_train_gt=True,
        ),
    )
    inputs, masks, metas, gt_segments, gt_labels = _inputs()

    def fail_if_called(_metas):
        raise AssertionError("reader value loss must not run during forward_test")

    model._pc_ot_mras_reader_value_losses = fail_if_called
    model.eval()

    with torch.no_grad():
        proposals, scores = model.forward_test(inputs, masks, metas=metas)

    assert len(proposals) == 1
    assert len(scores) == 1
    assert READER_OUTPUTS_META_KEY in model.rpn_head.last_test_metas[0]

    metas_with_target = [{"sample_id": "pc_ot_actionformer_forward_selector|0", "pc_ot_mras_value_targets": _hybrid_value_target()}]
    with pytest.raises(ValueError, match="forward_test forbids train-only PC-OT-MRAS value targets"):
        model.forward_test(inputs, masks, metas=metas_with_target)


def test_actionformer_pc_ot_mras_reader_hook_refuses_external_reader_output_injection():
    model = _model()
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    metas[0][READER_OUTPUTS_META_KEY] = {"selected_tokens": torch.zeros(1, 3, 4)}

    with pytest.raises(ValueError, match="refusing external reader-output injection"):
        model.forward_train(
            inputs,
            masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
        )


def test_actionformer_pc_ot_mras_reader_hook_refuses_external_reader_output_injection_in_forward_test():
    model = _model()
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    metas[0][READER_OUTPUTS_META_KEY] = {"selected_tokens": torch.zeros(1, 3, 4)}

    with pytest.raises(ValueError, match="refusing external reader-output injection"):
        model.forward_test(inputs, masks, metas=metas)


def test_actionformer_pc_ot_mras_reader_forbids_raw_prediction_load_bypass():
    model = _model()
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    infer_cfg = types.SimpleNamespace(
        load_from_raw_predictions=True,
        save_raw_prediction=False,
        folder="must_not_be_used",
    )

    with pytest.raises(ValueError, match="PC-OT-MRAS forbids raw-prediction load/save"):
        model.forward_detection(inputs, masks, metas=metas, infer_cfg=infer_cfg, post_cfg=types.SimpleNamespace())


def test_actionformer_pc_ot_mras_reader_forbids_raw_prediction_save_cache():
    model = _model()
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    infer_cfg = types.SimpleNamespace(
        load_from_raw_predictions=False,
        save_raw_prediction=True,
        folder="must_not_be_used",
    )

    with pytest.raises(ValueError, match="PC-OT-MRAS forbids raw-prediction load/save"):
        model.forward_detection(inputs, masks, metas=metas, infer_cfg=infer_cfg, post_cfg=types.SimpleNamespace())


def test_actionformer_pc_ot_mras_reader_hook_rejects_non_tuple_projection_output():
    model = _model()
    features = torch.randn(1, 4, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="projection outputs as feature/mask tuples"):
        model._inject_pc_ot_mras_reader_outputs(features, masks, metas=[{}])


def test_actionformer_pc_ot_mras_reader_hook_rejects_non_tuple_projection_in_forward_path():
    model = _model(projection_type="SyntheticFlatProjection")
    inputs, masks, metas, gt_segments, gt_labels = _inputs()

    with pytest.raises(ValueError, match="projection outputs as feature/mask tuples"):
        model.forward_train(
            inputs,
            masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
        )


def test_actionformer_pc_ot_mras_reader_parameters_are_in_optimizer_groups():
    model = _model()

    groups = model.get_optim_groups({"weight_decay": 0.05, "lr": 1.0e-4})
    grouped_ids = {id(param) for group in groups for param in group["params"]}
    trainable = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("backbone")
    }

    assert id(model.pc_ot_mras_reader.query_embed) in grouped_ids
    assert {id(param) for param in trainable.values()} == grouped_ids
