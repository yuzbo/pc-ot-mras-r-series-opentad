import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


class _FakeOptimizer:
    def __init__(self, param_groups, **kwargs):
        if "backbone" in kwargs:
            raise TypeError("unexpected keyword argument 'backbone'")
        self.param_groups = list(param_groups)
        self.kwargs = dict(kwargs)


class _Param:
    def __init__(self, requires_grad=True):
        self.requires_grad = bool(requires_grad)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


class _Backbone:
    def __init__(self, freeze_backbone=True, trainable_adapter=False, trainable_side=False):
        self.freeze_backbone = freeze_backbone
        self._params = [
            ("model.backbone.stem.weight", _Param(requires_grad=not freeze_backbone)),
            ("model.backbone.stem.bias", _Param(requires_grad=not freeze_backbone)),
            ("model.backbone.adapter.weight", _Param(requires_grad=(not freeze_backbone) or trainable_adapter)),
            ("model.backbone.adapter.bias", _Param(requires_grad=(not freeze_backbone) or trainable_adapter)),
            ("model.backbone.side.weight", _Param(requires_grad=trainable_side)),
        ]

    def named_parameters(self):
        return list(self._params)

    def parameters(self):
        return [param for _, param in self._params]


class _Detector:
    def __init__(self, backbone):
        self.backbone = backbone
        self.head_weight = _Param(requires_grad=True)
        self.head_bias = _Param(requires_grad=True)

    def named_parameters(self):
        for name, param in self.backbone.named_parameters():
            yield f"backbone.{name}", param
        yield "head.weight", self.head_weight
        yield "head.bias", self.head_bias


class _DDPWrap:
    def __init__(self, module):
        self.module = module


def _load_optimizer_module(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.optim = types.SimpleNamespace(
        AdamW=_FakeOptimizer,
        Adam=_FakeOptimizer,
        SGD=_FakeOptimizer,
    )

    fake_opentad = types.ModuleType("opentad")
    fake_opentad.__path__ = [str(ROOT / "opentad")]
    fake_cores = types.ModuleType("opentad.cores")
    fake_cores.__path__ = [str(ROOT / "opentad" / "cores")]
    fake_layer_decay = types.ModuleType("opentad.cores.layer_decay_optimizer")
    fake_layer_decay.build_vit_optimizer = lambda cfg, model, logger: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "opentad", fake_opentad)
    monkeypatch.setitem(sys.modules, "opentad.cores", fake_cores)
    monkeypatch.setitem(sys.modules, "opentad.cores.layer_decay_optimizer", fake_layer_decay)

    spec = importlib.util.spec_from_file_location(
        "opentad.cores.optimizer",
        ROOT / "opentad" / "cores" / "optimizer.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "opentad.cores.optimizer", module)
    spec.loader.exec_module(module)
    return module


def _optimizer_cfg(custom_name="adapter"):
    return {
        "type": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.05,
        "backbone": {
            "lr": 0.0,
            "weight_decay": 0.0,
            "custom": [{"name": custom_name, "lr": 2e-4, "weight_decay": 0.05}],
            "exclude": ["backbone"],
        },
    }


def _optimizer_param_ids(optimizer):
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def _params_by_name(model):
    return {name: param for name, param in model.module.named_parameters()}


def test_frozen_backbone_without_trainable_params_pops_backbone_cfg_and_uses_detector_params(monkeypatch):
    optimizer_module = _load_optimizer_module(monkeypatch)
    model = _DDPWrap(_Detector(_Backbone(freeze_backbone=True, trainable_adapter=False)))
    optimizer = optimizer_module.build_optimizer(_optimizer_cfg(), model, _Logger())
    param_ids = _optimizer_param_ids(optimizer)
    params = _params_by_name(model)

    assert id(params["head.weight"]) in param_ids
    assert id(params["head.bias"]) in param_ids
    assert all(id(param) not in param_ids for name, param in params.items() if name.startswith("backbone."))
    assert "backbone" not in optimizer.kwargs


def test_frozen_backbone_trainable_adapter_params_use_backbone_custom_group(monkeypatch):
    optimizer_module = _load_optimizer_module(monkeypatch)
    model = _DDPWrap(_Detector(_Backbone(freeze_backbone=True, trainable_adapter=True)))
    optimizer = optimizer_module.build_optimizer(_optimizer_cfg(), model, _Logger())
    param_ids = _optimizer_param_ids(optimizer)
    params = _params_by_name(model)
    adapter_ids = {
        id(params["backbone.model.backbone.adapter.weight"]),
        id(params["backbone.model.backbone.adapter.bias"]),
    }
    stem_ids = {
        id(params["backbone.model.backbone.stem.weight"]),
        id(params["backbone.model.backbone.stem.bias"]),
    }

    assert adapter_ids.issubset(param_ids)
    assert stem_ids.isdisjoint(param_ids)
    assert id(params["head.weight"]) in param_ids
    adapter_group = [
        group
        for group in optimizer.param_groups
        if adapter_ids.issubset({id(param) for param in group["params"]})
    ]
    assert len(adapter_group) == 1
    assert adapter_group[0]["lr"] == pytest.approx(2e-4)
    assert adapter_group[0]["weight_decay"] == pytest.approx(0.05)


def test_frozen_backbone_trainable_params_mismatched_by_custom_config_fail_closed(monkeypatch):
    optimizer_module = _load_optimizer_module(monkeypatch)
    model = _DDPWrap(_Detector(_Backbone(freeze_backbone=True, trainable_adapter=True)))

    with pytest.raises(AssertionError, match="matched no non-empty parameter group"):
        optimizer_module.build_optimizer(_optimizer_cfg(custom_name="not_adapter"), model, _Logger())


def test_unfrozen_backbone_preserves_backbone_and_detector_optimizer_groups(monkeypatch):
    optimizer_module = _load_optimizer_module(monkeypatch)
    model = _DDPWrap(_Detector(_Backbone(freeze_backbone=False)))
    optimizer = optimizer_module.build_optimizer(_optimizer_cfg(), model, _Logger())
    param_ids = _optimizer_param_ids(optimizer)
    params = _params_by_name(model)

    assert id(params["backbone.model.backbone.adapter.weight"]) in param_ids
    assert id(params["backbone.model.backbone.adapter.bias"]) in param_ids
    assert id(params["head.weight"]) in param_ids
    assert id(params["head.bias"]) in param_ids
    assert "backbone" not in optimizer.kwargs
