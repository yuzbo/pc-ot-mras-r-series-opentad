import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENGINE_PATH = ROOT / "opentad" / "cores" / "train_engine.py"


class _Autocast:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _AverageMeter:
    def __init__(self):
        self.values = []
        self.avg = 0.0

    def update(self, value):
        self.values.append(float(value))
        self.avg = sum(self.values) / len(self.values)


class _Loss:
    def __init__(self, value, owner):
        self.value = float(value)
        self.owner = owner
        self.data = self

    def backward(self):
        self.owner.backward_calls += 1

    def clone(self):
        return self

    def detach(self):
        return self

    def div_(self, value):
        self.value /= float(value)
        return self

    def item(self):
        return self.value


class _Finite:
    def all(self):
        return self

    def item(self):
        return True


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        if args:
            message = message % args
        self.messages.append(str(message))


class _ToyLoader:
    def __init__(self, length):
        self.length = int(length)

    def __len__(self):
        return self.length

    def __iter__(self):
        for idx in range(self.length):
            yield {"x": idx + 1}


class _ToyModel:
    def __init__(self):
        self.module = types.SimpleNamespace()
        self.train_calls = 0
        self.forward_calls = 0
        self.backward_calls = 0

    def train(self):
        self.train_calls += 1

    def __call__(self, x, return_loss=False):
        assert return_loss is True
        self.forward_calls += 1
        cost = _Loss(x, self)
        return {"cost": cost, "aux_loss": _Loss(x * 0.5, self)}


class _ToyOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0
        self.steps = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.steps += 1


class _ToyScheduler:
    def __init__(self):
        self.steps = 0

    def get_last_lr(self):
        return [1.0e-3]

    def step(self):
        self.steps += 1


def _load_train_engine_with_fake_runtime(monkeypatch):
    fake_torch = types.SimpleNamespace(
        float16="float16",
        is_tensor=lambda value: isinstance(value, _Loss),
        as_tensor=lambda value: value,
        isfinite=lambda value: _Finite(),
        cuda=types.SimpleNamespace(
            amp=types.SimpleNamespace(autocast=_Autocast),
            max_memory_allocated=lambda: 0,
        ),
        nn=types.SimpleNamespace(
            utils=types.SimpleNamespace(clip_grad_norm_=lambda *args, **kwargs: None),
        ),
    )
    fake_tqdm = types.SimpleNamespace(tqdm=lambda iterable, disable=False: iterable)
    fake_misc = types.SimpleNamespace(
        AverageMeter=_AverageMeter,
        reduce_loss=lambda losses: losses,
    )
    fake_utils = types.ModuleType("opentad.utils")
    fake_utils.misc = fake_misc
    fake_opentad = types.ModuleType("opentad")
    fake_opentad.utils = fake_utils

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setitem(sys.modules, "opentad", fake_opentad)
    monkeypatch.setitem(sys.modules, "opentad.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "opentad.utils.misc", fake_misc)

    module_name = "train_engine_fake_runtime_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, TRAIN_ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_one_epoch_stops_after_max_train_iters(monkeypatch):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel()
    optimizer = _ToyOptimizer()
    scheduler = _ToyScheduler()
    logger = _Logger()

    train_engine.train_one_epoch(
        _ToyLoader(length=5),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=logger,
        logging_interval=1,
        max_train_iters=2,
    )

    assert model.train_calls == 1
    assert model.forward_calls == 2
    assert model.backward_calls == 2
    assert optimizer.zero_grad_calls == 2
    assert optimizer.steps == 2
    assert scheduler.steps == 2
    assert any("max_train_iters=2 reached" in message for message in logger.messages)


@pytest.mark.parametrize("max_train_iters", [0, -1])
def test_train_one_epoch_rejects_non_positive_max_train_iters(monkeypatch, max_train_iters):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel()
    optimizer = _ToyOptimizer()
    scheduler = _ToyScheduler()

    with pytest.raises(ValueError, match="max_train_iters must be positive"):
        train_engine.train_one_epoch(
            _ToyLoader(length=5),
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=_Logger(),
            max_train_iters=max_train_iters,
        )

    assert model.train_calls == 0
    assert model.forward_calls == 0
    assert scheduler.steps == 0


def test_train_one_epoch_tracks_nonfinite_gradients_without_hard_error_text():
    text = TRAIN_ENGINE_PATH.read_text(encoding="utf-8")

    assert "NONFINITE_GRAD_SKIP" in text
    assert "_zero_grad_for_skip(optimizer)" in text
    assert "training produced non-finite parameter gradient" not in text


def _import_torch_or_skip():
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else f"exit {probe.returncode}"
        pytest.skip(f"torch unavailable in this process: {detail}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local DLL state.
        pytest.skip(f"torch unavailable in this process: {exc}")
    return torch


def test_grad_nonfinite_inspection_reports_parameter_name_without_throwing():
    torch = _import_torch_or_skip()
    sys.modules.pop("train_engine_real_runtime_under_test", None)
    spec = importlib.util.spec_from_file_location("train_engine_real_runtime_under_test", TRAIN_ENGINE_PATH)
    train_engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_engine)

    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("nan"), float("inf")]])
    model.bias.grad = torch.zeros_like(model.bias)

    total_norm, details = train_engine._inspect_grad_norm(model)

    assert torch.isfinite(total_norm).all()
    assert any("weight: shape=(1, 2)" in detail and "nan=1 inf=1" in detail for detail in details)
