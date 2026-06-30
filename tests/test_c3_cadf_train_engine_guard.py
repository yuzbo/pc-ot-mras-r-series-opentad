import types

import pytest
import torch

from opentad.cores.train_engine import train_one_epoch


class _OneBatchLoader:
    def __iter__(self):
        yield {}

    def __len__(self):
        return 1


class _NanCostModel:
    def __init__(self):
        self.module = types.SimpleNamespace()
        self.train_called = False

    def train(self):
        self.train_called = True

    def __call__(self, **kwargs):
        return {
            "cost": torch.tensor(float("nan"), requires_grad=True),
            "loss_c3_actionness": torch.tensor(float("nan"), requires_grad=True),
        }

    def parameters(self):
        return []


class _CountingOptimizer:
    def __init__(self):
        self.zero_grad_count = 0
        self.step_count = 0

    def zero_grad(self):
        self.zero_grad_count += 1

    def step(self):
        self.step_count += 1


class _CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def get_last_lr(self):
        return [1e-4]

    def step(self):
        self.step_count += 1


class _ListLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def test_cadf_train_engine_nonfinite_guard_skips_optimizer_step_before_threshold():
    model = _NanCostModel()
    optimizer = _CountingOptimizer()
    scheduler = _CountingScheduler()
    logger = _ListLogger()

    train_one_epoch(
        _OneBatchLoader(),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=logger,
        max_train_iters=1,
        nonfinite_loss_guard=dict(enabled=True, max_skips=1, max_consecutive_skips=1),
    )

    assert model.train_called is True
    assert optimizer.step_count == 0
    assert scheduler.step_count == 0
    assert any("non-finite loss" in message for level, message in logger.messages if level == "warning")


def test_cadf_train_engine_nonfinite_guard_fails_closed_before_optimizer_step():
    model = _NanCostModel()
    optimizer = _CountingOptimizer()
    scheduler = _CountingScheduler()
    logger = _ListLogger()

    with pytest.raises(RuntimeError, match="Non-finite training loss exceeded guard threshold"):
        train_one_epoch(
            _OneBatchLoader(),
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=logger,
            max_train_iters=1,
            nonfinite_loss_guard=dict(enabled=True, max_skips=0, max_consecutive_skips=0),
        )

    assert optimizer.step_count == 0
    assert scheduler.step_count == 0
