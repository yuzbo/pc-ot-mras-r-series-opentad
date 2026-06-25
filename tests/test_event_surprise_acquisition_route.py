from __future__ import annotations

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
MODULE_PATH = ROOT / "opentad" / "models" / "selectors" / "event_surprise_acquisition_route.py"
INIT_PATH = ROOT / "opentad" / "models" / "selectors" / "__init__.py"
ACTIONFORMER_PATH = ROOT / "opentad" / "models" / "detectors" / "actionformer.py"
BASE_DETECTOR_PATH = ROOT / "opentad" / "models" / "detectors" / "base.py"
SINGLE_STAGE_PATH = ROOT / "opentad" / "models" / "detectors" / "single_stage.py"


class _Registry:
    def __init__(self):
        self.classes = {}

    def register_module(self):
        def _decorator(cls):
            self.classes[cls.__name__] = cls
            return cls

        return _decorator

    def build(self, cfg):
        cfg = dict(cfg)
        cls = self.classes[cfg.pop("type")]
        return cls(**cfg)


def _ensure_package(name, path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_event_module():
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.event_surprise_acquisition_route",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, builder.SELECTORS


def _install_event_actionformer_runtime():
    for name in (
        "opentad.models.builder",
        "opentad.models.detectors.base",
        "opentad.models.detectors.single_stage",
        "opentad.models.detectors.actionformer",
        "opentad.models.selectors.event_surprise_acquisition_route",
        "opentad.models.utils.post_processing",
        "opentad.models.bricks",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.detectors", ROOT / "opentad" / "models" / "detectors")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.utils", ROOT / "opentad" / "models" / "utils")

    builder = types.ModuleType("opentad.models.builder")
    builder.MODELS = _Registry()
    builder.SELECTORS = builder.MODELS
    builder.DETECTORS = builder.MODELS
    builder.PROJECTIONS = builder.MODELS
    builder.HEADS = builder.MODELS
    builder.NECKS = builder.MODELS
    builder.TOKEN_COMPRESSORS = builder.MODELS
    builder.build_backbone = lambda cfg: (_ for _ in ()).throw(RuntimeError("no backbone in smoke"))
    builder.build_projection = lambda cfg: builder.PROJECTIONS.build(cfg)
    builder.build_head = lambda cfg: builder.HEADS.build(cfg)
    builder.build_neck = lambda cfg: builder.NECKS.build(cfg)
    builder.build_selector = lambda cfg: builder.SELECTORS.build(cfg)
    builder.build_token_compressor = lambda cfg: builder.TOKEN_COMPRESSORS.build(cfg)
    sys.modules["opentad.models.builder"] = builder

    bricks = types.ModuleType("opentad.models.bricks")

    class _Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, x):
            return x * self.scale

    bricks.Scale = _Scale
    bricks.AffineDropPath = _Scale
    sys.modules["opentad.models.bricks"] = bricks

    post_processing = types.ModuleType("opentad.models.utils.post_processing")
    post_processing.load_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no cache"))
    post_processing.save_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no cache"))
    post_processing.batched_nms = lambda *args, **kwargs: args[:3]
    post_processing.convert_to_seconds = lambda segments, meta: segments
    sys.modules["opentad.models.utils.post_processing"] = post_processing

    class EventIdentityProjection(nn.Module):
        def __init__(self, channels=4, max_seq_len=6):
            super().__init__()
            self.n_mha_win_size = 1
            self.arch = (0,)
            self.max_seq_len = int(max_seq_len)
            self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=False)
            nn.init.eye_(self.proj.weight[:, :, 0])

        def forward(self, x, masks):
            return (self.proj(x),), (masks,)

    class EventSelectedAxisHead(nn.Module):
        def __init__(self, in_channels=4, num_classes=2):
            super().__init__()
            self.prior_generator = types.SimpleNamespace(strides=[1])
            self.num_classes = int(num_classes)
            self.score = nn.Conv1d(int(in_channels), self.num_classes, kernel_size=1)
            self.last_test_metas = None

        def forward_test(self, feat_list, mask_list, metas=None):
            self.last_test_metas = metas
            feat = feat_list[0]
            mask = mask_list[0]
            proposals = []
            scores = []
            for batch_idx in range(int(feat.shape[0])):
                count = int(mask[batch_idx].long().sum().item())
                axis = torch.arange(count, device=feat.device, dtype=feat.dtype)
                proposals.append(torch.stack([axis, (axis + 1.0).clamp(max=max(count - 1, 0))], dim=-1))
                scores.append(torch.ones((count, self.num_classes), device=feat.device, dtype=feat.dtype))
            return proposals, scores

    builder.PROJECTIONS.register_module()(EventIdentityProjection)
    builder.HEADS.register_module()(EventSelectedAxisHead)

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.event_surprise_acquisition_route",
        MODULE_PATH,
    )
    event_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = event_module
    spec.loader.exec_module(event_module)
    for module_name, path in (
        ("opentad.models.detectors.base", BASE_DETECTOR_PATH),
        ("opentad.models.detectors.single_stage", SINGLE_STAGE_PATH),
        ("opentad.models.detectors.actionformer", ACTIONFORMER_PATH),
    ):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return builder


def _features():
    features = torch.zeros(2, 12, 4)
    features[:, :, 0] = torch.linspace(0.0, 1.0, 12)
    features[0, 5:, 1] = 6.0
    features[0, 8:, 2] = -4.0
    features[1, 3:5, 1] = 3.0
    features[1, 6:9, 2] = -5.0
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    return features, valid


def _assert_prefix(mask):
    valid_count = mask.long().sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < valid_count[:, None]
    assert torch.equal(mask, expected)


def test_event_surprise_selector_registers_with_fake_registry_and_builds():
    module, registry = _load_event_module()

    selector = registry.build(
        dict(
            type="EventSurpriseTemporalAcquisitionSelector",
            in_dim=4,
            target_len=6,
            max_gap=3,
            coverage_anchor_count=3,
        )
    )

    assert isinstance(selector, module.EventSurpriseTemporalAcquisitionSelector)
    assert selector.route_label == "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
    assert selector.meta_key == "event_surprise_acquisition_plan"
    assert selector.forbid_raw_prediction_cache is True


def test_forward_train_and_test_shapes_sorted_unique_indices_and_prefix_mask():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=4,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        surprise_topk=3,
    )
    features, valid = _features()

    train_out = selector.forward_train(features, valid)
    test_out = selector.forward_test(features, valid, metas=[{"video_id": "deploy-visible"}])

    for out in (train_out, test_out):
        assert out["acquisition_matrix"].shape == (2, 6, 12)
        assert out["selected_dense_indices"].shape == (2, 6)
        assert out["selected_mask"].shape == (2, 6)
        assert out["selected_tokens"].shape == (2, 6, 4)
        assert out["valid_mask"].shape == (2, 12)
        assert out["selected_times"].shape == (2, 6)
        assert out["selected_positions"].shape == (2, 6)
        assert out["centers"].shape == (2, 6)
        assert out["widths"].shape == (2, 6)
        assert out["gates"].shape == (2, 6)
        assert out["route_label"] == "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
        assert out["meta_key"] == "event_surprise_acquisition_plan"
        _assert_prefix(out["selected_mask"])

        for batch_idx in range(out["selected_dense_indices"].shape[0]):
            count = int(out["selected_mask"][batch_idx].sum().item())
            selected = out["selected_dense_indices"][batch_idx, :count]
            assert torch.equal(selected, torch.unique(selected))
            assert torch.equal(selected, selected.sort().values)
            gaps = selected[1:] - selected[:-1]
            assert gaps.numel() == 0 or int(gaps.max().item()) <= 3

    assert torch.equal(train_out["selected_dense_indices"], test_out["selected_dense_indices"])
    assert 5 in train_out["selected_dense_indices"][0].tolist()
    assert 8 in train_out["selected_dense_indices"][0].tolist()


def test_actionformer_keyword_path_returns_selected_inputs_masks_metas_and_remaps_gt():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=4,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        surprise_topk=3,
        input_layout="bct",
        remap_gt_to_selected_axis=True,
    )
    features, valid = _features()
    inputs = features.transpose(1, 2).contiguous()
    metas = [{"video_id": "deploy-visible-0"}, {"video_id": "deploy-visible-1"}]
    gt_segments = [
        torch.tensor([[4.0, 9.0]], dtype=torch.float32),
        torch.tensor([[2.0, 7.0]], dtype=torch.float32),
    ]
    gt_labels = [torch.tensor([1], dtype=torch.long), torch.tensor([2], dtype=torch.long)]

    train_out = selector.forward_train(
        inputs=inputs,
        masks=valid,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    test_out = selector.forward_test(inputs=inputs, masks=valid, metas=metas)

    assert train_out["inputs"].shape == (2, 4, 6)
    assert test_out["inputs"].shape == (2, 4, 6)
    assert train_out["masks"].shape == (2, 6)
    assert train_out["masks"].dtype == torch.bool
    assert train_out["gt_segments"][0].shape == (1, 2)
    assert not torch.equal(train_out["gt_segments"][0], gt_segments[0])
    assert torch.equal(train_out["gt_labels"][0], gt_labels[0])
    for meta in train_out["metas"]:
        assert "event_surprise_acquisition_plan" in meta
        assert meta["event_surprise_protocol_flags"]["uses_test_gt"] is False
        assert meta["event_surprise_protocol_flags"]["uses_teacher"] is False
        assert meta["event_surprise_protocol_flags"]["uses_raw_prediction_cache"] is False
        assert meta["event_surprise_protocol_flags"]["decode_saving_claim_allowed"] is False
        assert meta["event_surprise_protocol_flags"]["runtime_flops_claim_allowed"] is False
        assert meta["event_surprise_protocol_flags"]["route_isolated_from_c3"] is True
        assert meta["event_surprise_selected_dense_indices"] == sorted(meta["event_surprise_selected_dense_indices"])
        assert meta["irregular_selected_positions"] == [float(x) for x in meta["event_surprise_selected_dense_indices"]]
        assert meta["event_surprise_inference_mapping"]["selected_axis_to_dense_axis"] is True
        assert meta["event_surprise_inference_mapping"]["output_axis"] == "dense_window"


def test_selected_axis_dense_axis_roundtrip_for_gt_and_inference_proposals():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=4,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        remap_gt_to_selected_axis=True,
    )
    selected_dense_positions = torch.tensor([0, 2, 4, 6, 8, 10], dtype=torch.float32)
    dense_segments = torch.tensor([[1.0, 5.0], [7.0, 9.0]], dtype=torch.float32)

    selected_segments, keep = selector._remap_one_gt(dense_segments, selected_dense_positions)
    dense_roundtrip = selector.selected_axis_segments_to_dense_axis(
        selected_segments,
        selected_dense_positions,
    )

    assert keep.tolist() == [True, True]
    assert torch.allclose(dense_roundtrip, dense_segments, atol=1.0e-5)


def test_actionformer_inference_mapping_maps_head_selected_axis_proposals_to_dense_axis():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=4,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        surprise_topk=3,
        input_layout="bct",
    )
    features, valid = _features()
    inputs = features.transpose(1, 2).contiguous()
    selector_out = selector.forward_test(inputs=inputs, masks=valid, metas=[{"video_id": "v0"}, {"video_id": "v1"}])
    head_selected_axis = [
        torch.tensor([[0.0, 1.0], [2.5, 4.0]], dtype=torch.float32),
        torch.tensor([[1.0, 3.0]], dtype=torch.float32),
    ]

    mapped = selector.map_selected_axis_predictions_to_dense_axis(
        head_selected_axis,
        selector_out["event_surprise_selector_outputs"],
        selector_out["metas"],
    )

    for batch_idx, proposal in enumerate(mapped):
        selected = selector_out["event_surprise_selector_outputs"]["selected_dense_indices"][batch_idx]
        count = int(selector_out["event_surprise_selector_outputs"]["selected_mask"][batch_idx].sum().item())
        assert proposal.shape == head_selected_axis[batch_idx].shape
        assert float(proposal.min().item()) >= float(selected[:count].min().item())
        assert float(proposal.max().item()) <= float(selected[:count].max().item())
        assert selector_out["metas"][batch_idx]["event_surprise_inference_mapping"]["mapping_applied"] is True


def test_build_actionformer_and_forward_smoke_maps_selected_axis_predictions_to_dense_axis():
    builder = _install_event_actionformer_runtime()
    model = builder.DETECTORS.build(
        dict(
            type="ActionFormer",
            projection=dict(type="EventIdentityProjection", channels=4, max_seq_len=6),
            rpn_head=dict(type="EventSelectedAxisHead", in_channels=4, num_classes=2),
            frame_selector=dict(
                type="EventSurpriseTemporalAcquisitionSelector",
                in_dim=4,
                target_len=6,
                max_gap=3,
                coverage_anchor_count=3,
                surprise_topk=3,
                input_layout="bct",
            ),
        )
    )
    features, valid = _features()
    inputs = features.transpose(1, 2).contiguous()
    metas = [{"video_id": "event-smoke-0"}, {"video_id": "event-smoke-1"}]

    proposals, scores = model.forward_test(inputs, valid, metas=metas)

    assert len(proposals) == 2
    assert len(scores) == 2
    output_metas = model.rpn_head.last_test_metas
    for batch_idx, proposal in enumerate(proposals):
        selected = output_metas[batch_idx]["event_surprise_selected_dense_indices"]
        assert output_metas[batch_idx]["event_surprise_inference_mapping"]["mapping_applied"] is True
        assert proposal.shape[-1] == 2
        assert float(proposal.min().item()) >= float(min(selected))
        assert float(proposal.max().item()) <= float(max(selected))


def test_5d_preview_is_marked_as_loaded_dense_prototype_without_compute_saving_claims():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=4,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        input_layout="bcthw",
    )
    inputs = torch.randn(1, 4, 12, 2, 2)
    masks = torch.ones(1, 12, dtype=torch.bool)

    out = selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_id": "deploy-visible"}])
    flags = out["metas"][0]["event_surprise_protocol_flags"]

    assert out["inputs"].shape == (1, 4, 6, 2, 2)
    assert flags["preview_feature_source"] == "loaded_dense_detector_input_prototype"
    assert flags["decode_saving_claim_allowed"] is False
    assert flags["runtime_flops_claim_allowed"] is False


def test_6d_single_clip_raw_inputs_keep_dense_backbone_temporal_contract_and_metadata():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=3,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        input_layout="bct",
        remap_gt_to_selected_axis=True,
    )
    temporal_values = torch.arange(12, dtype=torch.float32).view(1, 1, 1, 12, 1, 1)
    inputs = temporal_values.expand(2, 1, 3, 12, 2, 2).clone()
    masks = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    metas = [{"video_id": "event-6d-0"}, {"video_id": "event-6d-1"}]
    gt_segments = [
        torch.tensor([[2.0, 8.0]], dtype=torch.float32),
        torch.tensor([[1.0, 7.0]], dtype=torch.float32),
    ]
    gt_labels = [torch.tensor([1], dtype=torch.long), torch.tensor([2], dtype=torch.long)]

    out = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert out["inputs"].shape == inputs.shape
    assert torch.equal(out["inputs"], inputs)
    assert torch.equal(out["masks"], masks)
    assert out["masks"].dtype == torch.bool
    assert torch.equal(out["gt_segments"][0], gt_segments[0])
    assert torch.equal(out["gt_segments"][1], gt_segments[1])
    assert torch.equal(out["gt_labels"][0], gt_labels[0])
    assert torch.equal(out["gt_labels"][1], gt_labels[1])
    assert out["event_surprise_selector_outputs"]["detector_input_axis"] == "dense_window"
    assert out["event_surprise_selector_outputs"]["selected_axis_predictions_require_mapping"] is False
    for batch_idx, meta in enumerate(out["metas"]):
        selected = out["event_surprise_selector_outputs"]["selected_dense_indices"][batch_idx]
        count = int(out["event_surprise_selector_outputs"]["selected_mask"][batch_idx].sum().item())
        selected_prefix = selected[:count]
        assert meta["event_surprise_selected_dense_indices"] == selected_prefix.tolist()
        assert meta["irregular_selected_positions"] == [float(item) for item in selected_prefix.tolist()]
        assert meta["event_surprise_remap_gt_to_selected_axis"] is False
        assert meta["event_surprise_protocol_flags"]["preview_feature_source"] == "loaded_dense_detector_input_prototype"
        assert meta["event_surprise_protocol_flags"]["detector_input_axis"] == "dense_window"
        assert meta["event_surprise_inference_mapping"]["selected_axis_to_dense_axis"] is False
        assert meta["event_surprise_inference_mapping"]["input_axis"] == "dense_window"
        assert meta["event_surprise_acquisition_plan"]["selected_count"] == count
        assert meta["event_surprise_acquisition_plan"]["remap_gt_to_selected_axis"] is False

    dense_axis_proposals = [
        torch.tensor([[1.0, 3.0], [4.0, 8.0]], dtype=torch.float32),
        torch.tensor([[0.5, 6.0]], dtype=torch.float32),
    ]
    mapped = selector.map_selected_axis_predictions_to_dense_axis(
        dense_axis_proposals,
        out["event_surprise_selector_outputs"],
        out["metas"],
    )

    for original, mapped_proposal in zip(dense_axis_proposals, mapped):
        assert torch.equal(mapped_proposal, original)
    for meta in out["metas"]:
        assert meta["event_surprise_inference_mapping"]["mapping_applied"] is False
        assert meta["event_surprise_inference_mapping"]["mapping_skipped_reason"] == "dense_detector_axis"


def test_6d_multi_clip_raw_inputs_fail_closed_until_flattening_semantics_are_defined():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(in_dim=3, target_len=6, input_layout="bct")
    inputs = torch.randn(1, 2, 3, 12, 2, 2)
    masks = torch.ones(1, 12, dtype=torch.bool)

    with pytest.raises(ValueError, match="6D detector inputs require a singleton clip dimension"):
        selector.forward_test(inputs=inputs, masks=masks, metas=[{"video_id": "event-6d-n2"}])


def test_remap_gt_filters_segments_and_labels_with_one_keep_mask():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=4,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        remap_gt_to_selected_axis=True,
    )
    selected_dense_indices = torch.tensor([[0, 2, 4, 6, 8, 10]], dtype=torch.long)
    selected_mask = torch.ones((1, 6), dtype=torch.bool)
    gt_segments = [torch.tensor([[1.0, 5.0], [6.0, 6.0], [7.0, 9.0]], dtype=torch.float32)]
    gt_labels = [[3, 4, 5]]

    mapped_segments, mapped_labels = selector._remap_gt_batch(
        gt_segments,
        gt_labels,
        selected_dense_indices,
        selected_mask,
    )

    assert mapped_segments[0].shape == (2, 2)
    assert mapped_labels[0] == [3, 5]


def test_remap_gt_rejects_segment_label_count_mismatch_before_mapping():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(
        in_dim=4,
        target_len=6,
        max_gap=3,
        coverage_anchor_count=3,
        remap_gt_to_selected_axis=True,
    )

    with pytest.raises(ValueError, match="gt segment/label count mismatch"):
        selector._remap_gt_batch(
            [torch.tensor([[1.0, 5.0], [7.0, 9.0]], dtype=torch.float32)],
            [torch.tensor([3], dtype=torch.long)],
            torch.tensor([[0, 2, 4, 6, 8, 10]], dtype=torch.long),
            torch.ones((1, 6), dtype=torch.bool),
        )


def test_event_surprise_route_is_exported_without_other_new_route_requirements():
    text = INIT_PATH.read_text(encoding="utf-8")

    assert "EventSurpriseTemporalAcquisitionSelector" in text
    assert "EVENT_SURPRISE_ROUTE_LABEL" in text
    assert "EVENT_SURPRISE_META_KEY" in text


@pytest.mark.parametrize(
    "bad_meta",
    [
        {"gt_segments": [[0, 1]]},
        {"teacher_score": [0.5]},
        {"raw_prediction_cache": "forbidden"},
        {"safe": {"oracle_hint": True}},
    ],
)
def test_forward_test_rejects_forbidden_test_meta_payloads(bad_meta):
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(in_dim=4, target_len=6)
    features, valid = _features()

    with pytest.raises(ValueError, match="forbidden deploy meta"):
        selector.forward_test(features, valid, metas=[bad_meta])


def test_forward_test_rejects_unknown_test_meta_key_even_without_forbidden_words():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(in_dim=4, target_len=6)
    features, valid = _features()

    with pytest.raises(ValueError, match="unexpected deploy meta key"):
        selector.forward_test(features, valid, metas=[{"video_id": "safe", "harmless_extra": 1}])


def test_forward_test_accepts_allowlisted_deploy_visible_meta_keys():
    module, _registry = _load_event_module()
    selector = module.EventSurpriseTemporalAcquisitionSelector(in_dim=4, target_len=6)
    features, valid = _features()

    out = selector.forward_test(
        features,
        valid,
        metas=[
            {
                "video_name": "video_test_000001",
                "video_id": "video_test_000001",
                "sample_id": "sample-1",
                "data_path": "thumos14/test/video_test_000001.mp4",
                "fps": 30.0,
                "duration": 120.0,
                "snippet_stride": 4.0,
                "window_start_frame": 0.0,
                "resize_length": 768,
                "window_size": 768,
                "offset_frames": 0.0,
            },
            {"video_id": "video_test_000002"},
        ],
    )

    assert out["route_label"] == "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
