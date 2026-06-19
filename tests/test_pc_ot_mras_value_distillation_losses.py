import importlib.util
import subprocess
import sys
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

from pc_ot_mras_test_utils import load_pc_ot_mras_classes


ROOT = Path(__file__).resolve().parents[1]
PCOTMRASReader, _PCOTMRASDetectorBridge = load_pc_ot_mras_classes()


def _load_value_loss_module():
    path = ROOT / "opentad" / "models" / "losses" / "pc_ot_mras_value_distillation_losses.py"
    spec = importlib.util.spec_from_file_location("pc_ot_mras_value_distillation_losses_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


value_loss_module = _load_value_loss_module()
pc_ot_mras_value_distillation_losses = value_loss_module.pc_ot_mras_value_distillation_losses


def _hybrid_value_target(time=12, slots=6):
    value = [0.0] * time
    risk = [0.0] * time
    redundancy = [0.0] * time
    value[3] = 1.0
    value[4] = 0.8
    risk[min(time - 1, 10)] = 1.0
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
        sample_id="unit|0",
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


def _reader_outputs(time=12, slots=6):
    torch.manual_seed(20260619)
    reader = PCOTMRASReader(
        in_dim=5,
        hidden_dim=16,
        num_slots=slots,
        num_blocks=1,
        num_roles=6,
        enable_value_heads=True,
    )
    features = torch.randn(1, time, 5, requires_grad=True)
    valid = torch.ones(1, time, dtype=torch.bool)
    return reader, features, reader(features, valid)


def test_pc_ot_mras_value_loss_accepts_hybrid_targets_and_backprops():
    reader, features, outputs = _reader_outputs()

    losses = pc_ot_mras_value_distillation_losses(
        outputs,
        metas=[{"pc_ot_mras_value_targets": _hybrid_value_target()}],
        require_targets=True,
        target_source="unit_counterfactual_value_distill_v0",
        allow_train_gt=True,
        allow_teacher_targets=False,
    )

    expected = {
        "pc_ot_mras_value_dense_value_loss",
        "pc_ot_mras_value_dense_risk_loss",
        "pc_ot_mras_value_dense_redundancy_loss",
        "pc_ot_mras_value_acquisition_loss",
        "pc_ot_mras_value_allocation_loss",
        "pc_ot_mras_value_gate_loss",
        "pc_ot_mras_value_pair_operation_loss",
    }
    assert expected <= set(losses)
    assert all(value.ndim == 0 and torch.isfinite(value).item() for value in losses.values())

    sum(losses.values()).backward()
    for name, param in {
        "value_head": reader.value_head.weight,
        "risk_head": reader.risk_head.weight,
        "redundancy_head": reader.redundancy_head.weight,
        "gate_head": reader.gate_head.weight,
        "key_proj": reader.key_proj.weight,
        "input_proj": reader.input_proj.weight,
    }.items():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0, name
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda target: target.update(split="validation"), "train-split only"),
        (lambda target: target.pop("counterfactual_utility_ready"), "counterfactual_utility_ready"),
        (lambda target: target.update(counterfactual_utility_ready=False), "counterfactual_utility_ready"),
        (lambda target: target.update(diagnostic_only=True), "diagnostic_only"),
        (lambda target: target.update(uses_prediction_cache=True), "cache/raw-prediction"),
        (lambda target: target.update(uses_teacher=True), "teacher-derived"),
        (lambda target: target.update(target_source="wrong_source"), "target_source"),
    ],
)
def test_pc_ot_mras_value_loss_rejects_bad_provenance(mutate, match):
    _reader, _features, outputs = _reader_outputs()
    target = _hybrid_value_target()
    mutate(target)

    with pytest.raises(ValueError, match=match):
        pc_ot_mras_value_distillation_losses(
            outputs,
            metas=[{"pc_ot_mras_value_targets": target}],
            require_targets=True,
            target_source="unit_counterfactual_value_distill_v0",
            allow_train_gt=True,
            allow_teacher_targets=False,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda target: target.update(value_scores=[0.0] * 11), "value_scores"),
        (lambda target: target.update(risk_scores=[0.0, float("nan")] + [0.0] * 10), "risk_scores"),
        (lambda target: target.update(valid_mask=[1, 0, 1] + [0] * 9), "contiguous valid prefix"),
        (lambda target: target["operations"][0].update(add_index=99), "out of range"),
        (lambda target: target["operations"][0].update(label_negative=True), "both positive and negative"),
        (lambda target: target.update(target_len=5), "target_len"),
    ],
)
def test_pc_ot_mras_value_loss_rejects_bad_shapes_and_operations(mutate, match):
    _reader, _features, outputs = _reader_outputs()
    target = _hybrid_value_target()
    mutate(target)

    with pytest.raises(ValueError, match=match):
        pc_ot_mras_value_distillation_losses(
            outputs,
            metas=[{"pc_ot_mras_value_targets": target}],
            require_targets=True,
            target_source="unit_counterfactual_value_distill_v0",
            allow_train_gt=True,
        )


def test_pc_ot_mras_value_loss_rejects_legacy_value_transport_targets():
    _reader, _features, outputs = _reader_outputs()

    with pytest.raises(ValueError, match="legacy PC-OT-MRAS value target keys"):
        pc_ot_mras_value_distillation_losses(
            outputs,
            metas=[{"value_transport_targets": {"legacy": True}}],
            require_targets=True,
            target_source="unit_counterfactual_value_distill_v0",
        )


def test_pc_ot_mras_value_loss_missing_targets_can_only_return_zero_when_optional():
    _reader, _features, outputs = _reader_outputs()

    with pytest.raises(ValueError, match="missing required pc_ot_mras_value_targets"):
        pc_ot_mras_value_distillation_losses(outputs, metas=[{}], require_targets=True)

    losses = pc_ot_mras_value_distillation_losses(outputs, metas=[{}], require_targets=False)
    assert set(losses) == {"pc_ot_mras_value_zero_loss"}
    assert losses["pc_ot_mras_value_zero_loss"].ndim == 0
    assert torch.isfinite(losses["pc_ot_mras_value_zero_loss"]).item()
