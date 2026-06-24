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


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "opentad" / "models" / "selectors" / "event_surprise_acquisition_route.py"
INIT_PATH = ROOT / "opentad" / "models" / "selectors" / "__init__.py"


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
        assert meta["event_surprise_protocol_flags"]["route_isolated_from_c3"] is True
        assert meta["event_surprise_selected_dense_indices"] == sorted(meta["event_surprise_selected_dense_indices"])
        assert meta["irregular_selected_positions"] == [float(x) for x in meta["event_surprise_selected_dense_indices"]]


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
