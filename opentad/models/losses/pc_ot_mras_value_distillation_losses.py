from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, Optional

import torch
import torch.nn.functional as F


TARGET_KEY = "pc_ot_mras_value_targets"
LEGACY_TARGET_KEYS = (
    "value_transport_targets",
    "teacher_value_targets",
    "pc_ot_mras_value_manifest",
)

_DEFAULT_WEIGHTS = {
    "dense_value": 0.02,
    "dense_risk": 0.01,
    "dense_redundancy": 0.005,
    "acquisition": 0.05,
    "allocation": 0.01,
    "gate": 0.01,
    "pair_operation": 0.03,
}

_FORBIDDEN_TRUE_FLAGS = (
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_raw_predictions",
    "contains_raw_predictions",
    "contains_detection_results",
)

_REQUIRED_EXPLICIT_KEYS = (
    "schema_version",
    "target_family",
    "split",
    "source_split",
    "target_source",
    "unit",
    "reader_axis",
    "sample_id",
    "dense_len",
    "target_len",
    "valid_len",
)


def _require_tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"reader_outputs['{key}'] must be a tensor")
    if torch.is_complex(value):
        raise ValueError(f"reader_outputs['{key}'] must be real-valued")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"reader_outputs['{key}'] must be finite")
    return value


def _valid_mask(reader_outputs: Mapping[str, object]) -> torch.Tensor:
    valid = _require_tensor(reader_outputs, "valid_mask")
    if valid.ndim != 2:
        raise ValueError(f"valid_mask must be [B,T], got {tuple(valid.shape)}")
    if valid.dtype != torch.bool and not bool(((valid == 0) | (valid == 1)).all().item()):
        raise ValueError("valid_mask must contain binary 0/1 values")
    valid = valid.bool()
    counts = valid.long().sum(dim=1)
    if bool((counts <= 0).any().item()):
        raise ValueError("valid_mask must contain at least one valid position per sample")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < counts[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError("valid_mask must be a contiguous valid prefix")
    return valid


def _zero_like_loss(reader_outputs: Mapping[str, object]) -> torch.Tensor:
    for key in ("value_logits", "risk_logits", "redundancy_logits", "allocation_logits", "gate_logits"):
        value = reader_outputs.get(key)
        if torch.is_tensor(value):
            return value.sum() * 0.0
    raise ValueError("reader_outputs must contain at least one trainable PC-OT-MRAS value logit tensor")


def _as_bool(mapping: Mapping[str, object], key: str) -> bool:
    if key not in mapping:
        raise ValueError(f"value target must explicitly declare {key}")
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"value target {key} must be a bool")
    return bool(value)


def _require_nonempty(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"value target must explicitly declare {key}")
    return mapping[key]


def _is_synthetic_unit_target(target: Mapping[str, object]) -> bool:
    source = str(target.get("target_source", ""))
    return bool(target.get("synthetic_unit_test", False)) or source.startswith("unit_")


def _target_vector(
    target: Mapping[str, object],
    key: str,
    *,
    dense_len: int,
    device: torch.device,
) -> torch.Tensor:
    if key not in target:
        raise ValueError(f"value target missing {key}")
    value = target[key]
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim != 1 or int(tensor.numel()) != int(dense_len):
        raise ValueError(f"{key} must be [dense_len={dense_len}], got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{key} must be finite")
    eps = 1.0e-6
    if not bool(((tensor >= -eps) & (tensor <= 1.0 + eps)).all().item()):
        raise ValueError(f"{key} must be normalized to [0, 1]")
    return tensor.clamp(0.0, 1.0)


def _target_valid_mask(
    target: Mapping[str, object],
    *,
    dense_len: int,
    valid_len: int,
    device: torch.device,
) -> torch.Tensor:
    if "valid_mask" not in target:
        raise ValueError("value target missing valid_mask")
    raw = target["valid_mask"]
    if torch.is_tensor(raw):
        mask = raw.detach().to(device=device)
    else:
        mask = torch.as_tensor(raw, device=device)
    if mask.ndim != 1 or int(mask.numel()) != int(dense_len):
        raise ValueError(f"valid_mask target must be [dense_len={dense_len}], got {tuple(mask.shape)}")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("value target valid_mask must contain binary 0/1 values")
    mask = mask.bool()
    prefix = torch.arange(dense_len, device=device) < int(valid_len)
    if not torch.equal(mask, prefix):
        raise ValueError("value target valid_mask must be a contiguous valid prefix")
    if int(mask.long().sum().item()) != int(valid_len):
        raise ValueError("value target valid_mask sum must equal valid_len")
    return mask


def _validate_target_provenance(
    target: Mapping[str, object],
    *,
    dense_len: int,
    target_len: int,
    expected_target_source: Optional[str],
    allow_train_gt: bool,
    allow_teacher_targets: bool,
) -> None:
    for key in _REQUIRED_EXPLICIT_KEYS:
        _require_nonempty(target, key)
    if target["schema_version"] != "pc_ot_mras_value_targets_v0":
        raise ValueError("unsupported pc_ot_mras value target schema_version")
    if target["target_family"] != "pc_ot_mras_voi_distill_hybrid_v0":
        raise ValueError("unsupported pc_ot_mras value target target_family")
    if expected_target_source is not None and str(target["target_source"]) != str(expected_target_source):
        raise ValueError("value target target_source does not match configured source")
    if str(target["split"]) != "train" or str(target.get("source_split", "train")) != "train":
        raise ValueError("pc_ot_mras value targets are train-split only")
    if _as_bool(target, "training_only") is not True:
        raise ValueError("pc_ot_mras value targets must be training_only=True")
    if _as_bool(target, "counterfactual_utility_ready") is not True:
        raise ValueError("pc_ot_mras value targets must be marked counterfactual_utility_ready=True")
    if _as_bool(target, "diagnostic_only") is True:
        raise ValueError("diagnostic_only value targets are not accepted for R20A")
    if str(target["unit"]) != "local_dense_index":
        raise ValueError("pc_ot_mras value target unit must be local_dense_index")
    if str(target["reader_axis"]) != "projection_level0_after_pad":
        raise ValueError("pc_ot_mras value target reader_axis must be projection_level0_after_pad")
    if int(target["dense_len"]) != int(dense_len):
        raise ValueError("value target dense_len must match reader dense axis")
    if int(target["target_len"]) != int(target_len):
        raise ValueError("value target target_len must match reader acquisition slots")
    if int(target["valid_len"]) <= 0 or int(target["valid_len"]) > int(dense_len):
        raise ValueError("value target valid_len must be inside dense_len")
    if int(target.get("index_base", 0)) != 0:
        raise ValueError("value target index_base must be 0")
    forbidden_splits = {str(item).lower() for item in target.get("forbidden_splits", [])}
    if not {"val", "validation", "test"}.issubset(forbidden_splits):
        raise ValueError("value target must forbid val, validation, and test splits")
    for key in _FORBIDDEN_TRUE_FLAGS:
        if _as_bool(target, key) is True:
            raise ValueError(f"value target forbids cache/raw-prediction flag {key}=True")
    uses_gt = _as_bool(target, "uses_gt")
    if uses_gt and not allow_train_gt:
        raise ValueError("train-GT-derived value targets require allow_train_gt=True")
    uses_teacher = _as_bool(target, "uses_teacher")
    if uses_teacher and not allow_teacher_targets:
        raise ValueError("teacher-derived value targets require allow_teacher_targets=True")
    if uses_teacher:
        for key in ("teacher_kind", "teacher_config_sha256", "teacher_checkpoint_sha256"):
            _require_nonempty(target, key)
    if not _is_synthetic_unit_target(target):
        _require_nonempty(target, "target_artifact_sha256")
        _require_nonempty(target, "manifest_sha256")


def _validate_operation(
    op: Mapping[str, object],
    *,
    dense_len: int,
    target_len: int,
    valid_len: int,
    expected_target_source: Optional[str],
    allow_train_gt: bool,
    allow_teacher_targets: bool,
) -> None:
    if str(op.get("split")) != "train":
        raise ValueError("operation targets are train-split only")
    if bool(op.get("training_only")) is not True:
        raise ValueError("operation targets must be training_only=True")
    if bool(op.get("counterfactual_utility_ready")) is not True:
        raise ValueError("operation targets must be counterfactual_utility_ready=True")
    if bool(op.get("diagnostic_only")) is True:
        raise ValueError("diagnostic_only operation targets are not accepted")
    if expected_target_source is not None and str(op.get("target_source")) != str(expected_target_source):
        raise ValueError("operation target_source does not match configured source")
    if int(op.get("dense_len", -1)) != int(dense_len) or int(op.get("target_len", -1)) != int(target_len):
        raise ValueError("operation dense_len/target_len must match target")
    if bool(op.get("uses_gt", False)) and not allow_train_gt:
        raise ValueError("train-GT-derived operation targets require allow_train_gt=True")
    if bool(op.get("uses_teacher", False)) and not allow_teacher_targets:
        raise ValueError("teacher-derived operation targets require allow_teacher_targets=True")
    for key in ("uses_cache", "uses_prediction_cache", "uses_raw_prediction"):
        if bool(op.get(key, False)):
            raise ValueError(f"operation target forbids {key}=True")
    add = int(op.get("add_index", -1))
    delete = int(op.get("delete_index", -1))
    if add < 0 or delete < 0 or add >= int(dense_len) or delete >= int(dense_len):
        raise ValueError("operation add/delete index out of range")
    if add >= int(valid_len) or delete >= int(valid_len):
        raise ValueError("operation add/delete index must be inside valid prefix")
    positive = bool(op.get("label_positive", False))
    negative = bool(op.get("label_negative", False))
    if positive and negative:
        raise ValueError("operation cannot be both positive and negative")


def _collect_targets(
    metas: Sequence[Mapping[str, object]],
    *,
    valid: torch.Tensor,
    target_len: int,
    require_targets: bool,
    expected_target_source: Optional[str],
    allow_train_gt: bool,
    allow_teacher_targets: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], list[tuple[int, Mapping[str, object]]]]:
    batch, dense_len = valid.shape
    if len(metas) != int(batch):
        raise ValueError(f"metas length {len(metas)} must match reader batch size {batch}")

    values = []
    risks = []
    redundancies = []
    masks = []
    operations: list[tuple[int, Mapping[str, object]]] = []
    seen = 0
    for batch_idx, meta in enumerate(metas):
        if not isinstance(meta, Mapping):
            raise ValueError(f"metas[{batch_idx}] must be a mapping")
        present_legacy = [key for key in LEGACY_TARGET_KEYS if key in meta]
        if present_legacy:
            raise ValueError(f"legacy PC-OT-MRAS value target keys are not accepted: {present_legacy}")
        if TARGET_KEY not in meta:
            if require_targets:
                raise ValueError(f"metas[{batch_idx}] missing required {TARGET_KEY}")
            continue
        target = meta[TARGET_KEY]
        if not isinstance(target, Mapping):
            raise ValueError(f"metas[{batch_idx}][{TARGET_KEY!r}] must be a mapping")
        seen += 1
        _validate_target_provenance(
            target,
            dense_len=int(dense_len),
            target_len=int(target_len),
            expected_target_source=expected_target_source,
            allow_train_gt=allow_train_gt,
            allow_teacher_targets=allow_teacher_targets,
        )
        valid_len = int(target["valid_len"])
        target_mask = _target_valid_mask(target, dense_len=int(dense_len), valid_len=valid_len, device=valid.device)
        if not torch.equal(target_mask, valid[batch_idx]):
            raise ValueError("value target valid_mask must match reader valid_mask")
        values.append(_target_vector(target, "value_scores", dense_len=int(dense_len), device=valid.device))
        risks.append(_target_vector(target, "risk_scores", dense_len=int(dense_len), device=valid.device))
        redundancies.append(_target_vector(target, "redundancy_scores", dense_len=int(dense_len), device=valid.device))
        masks.append(target_mask)
        for op in target.get("operations", []) or []:
            if not isinstance(op, Mapping):
                raise ValueError("operation target entries must be mappings")
            _validate_operation(
                op,
                dense_len=int(dense_len),
                target_len=int(target_len),
                valid_len=valid_len,
                expected_target_source=expected_target_source,
                allow_train_gt=allow_train_gt,
                allow_teacher_targets=allow_teacher_targets,
            )
            operations.append((batch_idx, op))

    if seen == 0:
        return None, None, None, None, []
    if seen != int(batch):
        raise ValueError("pc_ot_mras value targets must be present for the whole batch or absent for all samples")
    return torch.stack(values), torch.stack(risks), torch.stack(redundancies), torch.stack(masks), operations


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("value BCE logits, target, and valid mask must share shape [B,T]")
    if torch.is_complex(logits) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("value BCE logits must be finite and real-valued")
    valid_f = valid.to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target.to(device=logits.device, dtype=logits.dtype),
        reduction="none",
    )
    return (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def _masked_distribution(scores: torch.Tensor, valid: torch.Tensor, eps: float) -> torch.Tensor:
    score = scores.to(dtype=torch.float32).masked_fill(~valid, 0.0).clamp_min(0.0)
    mass = score.sum(dim=-1, keepdim=True)
    uniform = valid.to(dtype=torch.float32) / valid.to(dtype=torch.float32).sum(dim=-1, keepdim=True).clamp_min(1.0)
    dist = score / mass.clamp_min(eps)
    return torch.where(mass > eps, dist, uniform)


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg = float(torch.finfo(logits.dtype).min / 4.0)
    return F.log_softmax(logits.masked_fill(~mask, neg), dim=dim).masked_fill(~mask, 0.0)


def _validate_reader_shapes(
    reader_outputs: Mapping[str, object],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acquisition = _require_tensor(reader_outputs, "acquisition_matrix")
    allocation = _require_tensor(reader_outputs, "allocation")
    gate_logits = _require_tensor(reader_outputs, "gate_logits")
    if acquisition.ndim != 3 or acquisition.shape[0] != valid.shape[0] or acquisition.shape[2] != valid.shape[1]:
        raise ValueError("acquisition_matrix must be [B,K,T] and match valid_mask")
    if allocation.shape != acquisition.shape:
        raise ValueError("allocation must share shape with acquisition_matrix")
    if gate_logits.ndim != 2 or gate_logits.shape != acquisition.shape[:2]:
        raise ValueError("gate_logits must be [B,K] and match acquisition_matrix")
    if acquisition.device != valid.device or allocation.device != valid.device or gate_logits.device != valid.device:
        raise ValueError("reader outputs must share device with valid_mask")
    if not bool(torch.isfinite(acquisition).all().item()) or not bool(torch.isfinite(allocation).all().item()):
        raise ValueError("acquisition_matrix and allocation must be finite")
    if not bool(torch.isfinite(gate_logits).all().item()):
        raise ValueError("gate_logits must be finite")
    return acquisition, allocation, gate_logits


def pc_ot_mras_value_distillation_losses(
    reader_outputs: Mapping[str, object],
    metas: Sequence[Mapping[str, object]],
    *,
    weights: Optional[Mapping[str, float]] = None,
    require_targets: bool = True,
    target_source: Optional[str] = None,
    allow_train_gt: bool = True,
    allow_teacher_targets: bool = False,
    pair_temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Train-only R20 value-of-information distillation for PC-OT-MRAS."""

    if not isinstance(reader_outputs, Mapping):
        raise ValueError("reader_outputs must be a mapping")
    if not isinstance(metas, (list, tuple)):
        raise ValueError("metas must be a list/tuple of mappings")
    valid = _valid_mask(reader_outputs)
    acquisition, allocation, gate_logits = _validate_reader_shapes(reader_outputs, valid)
    batch, target_len, dense_len = acquisition.shape
    if batch != valid.shape[0] or dense_len != valid.shape[1]:
        raise ValueError("reader batch/dense shapes are inconsistent")

    value_target, risk_target, red_target, target_valid, operations = _collect_targets(
        metas,
        valid=valid,
        target_len=int(target_len),
        require_targets=bool(require_targets),
        expected_target_source=target_source,
        allow_train_gt=bool(allow_train_gt),
        allow_teacher_targets=bool(allow_teacher_targets),
    )
    if value_target is None:
        return {"pc_ot_mras_value_zero_loss": _zero_like_loss(reader_outputs)}
    target_valid = target_valid.to(device=valid.device)

    merged_weights = dict(_DEFAULT_WEIGHTS)
    merged_weights.update(dict(weights or {}))
    losses: Dict[str, torch.Tensor] = {}
    eps = torch.finfo(acquisition.dtype).eps
    value_dist = _masked_distribution(value_target, target_valid, float(eps)).to(device=acquisition.device)

    if float(merged_weights.get("dense_value", 0.0)) != 0.0:
        losses["pc_ot_mras_value_dense_value_loss"] = (
            _masked_bce(_require_tensor(reader_outputs, "value_logits"), value_target, target_valid)
            * float(merged_weights["dense_value"])
        )

    if float(merged_weights.get("dense_risk", 0.0)) != 0.0:
        losses["pc_ot_mras_value_dense_risk_loss"] = (
            _masked_bce(_require_tensor(reader_outputs, "risk_logits"), risk_target, target_valid)
            * float(merged_weights["dense_risk"])
        )

    if float(merged_weights.get("dense_redundancy", 0.0)) != 0.0:
        losses["pc_ot_mras_value_dense_redundancy_loss"] = (
            _masked_bce(_require_tensor(reader_outputs, "redundancy_logits"), red_target, target_valid)
            * float(merged_weights["dense_redundancy"])
        )

    col_mass = acquisition.sum(dim=1).masked_fill(~valid, 0.0)
    col_dist = col_mass / col_mass.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    if float(merged_weights.get("acquisition", 0.0)) != 0.0:
        acquisition_ce = -(value_dist.to(dtype=col_dist.dtype) * col_dist.clamp_min(float(eps)).log()).sum(dim=-1).mean()
        losses["pc_ot_mras_value_acquisition_loss"] = acquisition_ce * float(merged_weights["acquisition"])

    if float(merged_weights.get("allocation", 0.0)) != 0.0:
        allocation_logits = _require_tensor(reader_outputs, "allocation_logits")
        if allocation_logits.shape != acquisition.shape:
            raise ValueError("allocation_logits must share shape with acquisition_matrix")
        alloc_mask = valid[:, None, :].expand_as(allocation_logits)
        log_prob = _masked_log_softmax(allocation_logits, alloc_mask, dim=-1)
        allocation_ce = -(value_dist[:, None, :].to(dtype=log_prob.dtype) * log_prob).sum(dim=-1).mean()
        losses["pc_ot_mras_value_allocation_loss"] = allocation_ce * float(merged_weights["allocation"])

    if float(merged_weights.get("gate", 0.0)) != 0.0:
        gate_target = torch.einsum(
            "bkt,bt->bk",
            allocation.detach().to(dtype=torch.float32),
            (value_target - risk_target).clamp(0.0, 1.0),
        ).clamp(0.0, 1.0)
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits,
            gate_target.to(device=gate_logits.device, dtype=gate_logits.dtype),
            reduction="mean",
        )
        losses["pc_ot_mras_value_gate_loss"] = gate_loss * float(merged_weights["gate"])

    if float(merged_weights.get("pair_operation", 0.0)) != 0.0 and operations:
        value_logits = reader_outputs.get("value_logits")
        risk_logits = reader_outputs.get("risk_logits")
        red_logits = reader_outputs.get("redundancy_logits")
        pair_terms = []
        temperature = max(float(pair_temperature), 1.0e-6)
        for batch_idx, op in operations:
            add = int(op.get("add_index", -1))
            delete = int(op.get("delete_index", -1))
            if not bool(valid[batch_idx, add].item()) or not bool(valid[batch_idx, delete].item()):
                raise ValueError("operation add/delete index must be inside valid prefix")
            positive = bool(op.get("label_positive", False))
            negative = bool(op.get("label_negative", False))
            uncertain = bool(op.get("label_uncertain", False))
            if uncertain or (not positive and not negative):
                continue
            transport_score = torch.log(col_mass[batch_idx, add].clamp_min(float(eps))) - torch.log(
                col_mass[batch_idx, delete].clamp_min(float(eps))
            )
            score = transport_score
            if torch.is_tensor(value_logits):
                score = score + value_logits[batch_idx, add]
            if torch.is_tensor(red_logits):
                score = score + red_logits[batch_idx, delete]
            if torch.is_tensor(risk_logits):
                score = score - risk_logits[batch_idx, add]
            label = score.new_tensor(1.0 if positive else 0.0)
            utility = abs(float(op.get("utility", 1.0)))
            weight = score.new_tensor(max(0.25, min(4.0, utility)))
            pair_terms.append(F.binary_cross_entropy_with_logits(score / temperature, label, reduction="none") * weight)
        if pair_terms:
            losses["pc_ot_mras_value_pair_operation_loss"] = (
                torch.stack(pair_terms).mean() * float(merged_weights["pair_operation"])
            )

    if not losses:
        losses["pc_ot_mras_value_zero_loss"] = _zero_like_loss(reader_outputs)
    return losses


__all__ = [
    "TARGET_KEY",
    "LEGACY_TARGET_KEYS",
    "pc_ot_mras_value_distillation_losses",
]
