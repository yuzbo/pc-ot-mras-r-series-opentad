import inspect

from opentad.cores.train_engine import train_one_epoch


def test_train_one_epoch_exposes_max_train_iters_for_clean_smoke():
    signature = inspect.signature(train_one_epoch)

    assert "max_train_iters" in signature.parameters
    assert signature.parameters["max_train_iters"].default is None
