from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..builder import NECKS


_READER_OUTPUTS_META_KEY = "pc_ot_mras_reader_outputs"
_BRIDGE_META_KEY = "pc_ot_mras_bridge"
_SLOT_ALIGNED_AUX_KEYS = frozenset(
    {
        "selected_tokens",
        "selected_times",
        "selected_positions",
        "centers",
        "widths",
        "gates",
        "allocation",
        "acquisition_matrix",
    }
)
_BLOCKED_META_TOKENS = frozenset(
    {
        "gt",
        "oracle",
        "teacher",
        "cache",
        "prediction",
        "predictions",
        "pred",
        "checkpoint",
        "ckpt",
        "result",
    }
)
_BLOCKED_META_PHRASES = frozenset(
    {
        "ground_truth",
        "groundtruth",
        "raw_prediction",
        "raw_predictions",
        "rawprediction",
        "rawpredictions",
    }
)
_BLOCKED_META_COMPACT_PHRASES = frozenset(
    {
        "groundtruth",
        "rawprediction",
        "rawpredictions",
        "gtsegment",
        "gtsegments",
        "gtlabel",
        "gtlabels",
        "gtannotation",
        "gtannotations",
    }
)
_TRANSPORT_NEGATIVE_TOL = 0.0
_TRANSPORT_PAD_MASS_TOL = 1.0e-7
_TRANSPORT_MIN_ROW_MASS = 1.0e-7
_TRANSPORT_ROW_MASS_ATOL = 1.0e-5
_TRANSPORT_ROW_MASS_RTOL = 1.0e-5
_TRANSPORT_MAX_ROW_MASS = 1.0


def _binary_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{name} must be [B,K], got {tuple(mask.shape)}")
    if torch.is_complex(mask):
        raise ValueError(f"{name} must contain binary 0/1 values")
    if torch.is_floating_point(mask) and not bool(torch.isfinite(mask).all().item()):
        raise ValueError(f"{name} must be finite")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError(f"{name} must be binary")
    return mask.bool()


def _prefix_binary_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    valid = _binary_mask(mask, name=name)
    valid_count = valid.long().sum(dim=1)
    if bool((valid_count <= 0).any().item()):
        raise ValueError(f"{name} must contain at least one valid position per sample")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError(f"{name} must be a contiguous valid prefix")
    return valid


def _require_tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    if key not in mapping:
        raise ValueError(f"reader_outputs missing '{key}'")
    value = mapping[key]
    if not torch.is_tensor(value):
        raise ValueError(f"reader_outputs['{key}'] must be a tensor")
    return value


def _normalized_text(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or ""))
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text)


def _contains_blocked_fragment(value: object) -> bool:
    normalized = _normalized_text(value)
    tokens = [token for token in re.split(r"_+", normalized.strip("_")) if token]
    if any(token in _BLOCKED_META_TOKENS for token in tokens):
        return True
    if "ground" in tokens and "truth" in tokens:
        return True
    compact = "".join(tokens)
    if any(phrase in compact for phrase in _BLOCKED_META_COMPACT_PHRASES):
        return True
    if any(token != "gt" and token in compact for token in _BLOCKED_META_TOKENS):
        return True
    if compact.startswith("gt"):
        return True
    return any(phrase in normalized for phrase in _BLOCKED_META_PHRASES)


def _validate_reader_payload(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key or "")
            if _contains_blocked_fragment(key_text):
                raise ValueError(f"{location}.{key_text} contains forbidden reader payload")
            _validate_reader_payload(item, location=f"{location}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_reader_payload(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and _contains_blocked_fragment(value):
        raise ValueError(f"{location} contains forbidden reader payload")


def _optional_slot_tensor(
    reader_outputs: Mapping[str, object],
    key: str,
    *,
    expected_shape: torch.Size,
    expected_device: torch.device,
) -> Optional[torch.Tensor]:
    value = reader_outputs.get(key)
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise ValueError(f"reader_outputs['{key}'] must be a tensor when provided")
    if value.device != expected_device:
        raise ValueError(f"{key} must be on the same device as source_tokens")
    if value.shape != expected_shape:
        raise ValueError(f"{key} shape mismatch: {tuple(value.shape)} vs {tuple(expected_shape)}")
    if torch.is_complex(value):
        raise ValueError(f"{key} must be real-valued")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{key} must be finite")
    return value


def _validate_transport_matrix_contract(
    matrix: torch.Tensor,
    *,
    name: str,
    dense_valid_mask: torch.Tensor,
    selected_mask: Optional[torch.Tensor],
    gates: Optional[torch.Tensor],
) -> None:
    if matrix.ndim != 3:
        raise ValueError(f"{name} must be [B,K,T], got {tuple(matrix.shape)}")
    if matrix.shape[1] <= 0 or matrix.shape[2] <= 0:
        raise ValueError(f"{name} must have non-empty selected and dense axes")
    if dense_valid_mask.shape != (matrix.shape[0], matrix.shape[2]):
        raise ValueError(f"{name} and valid_mask shape mismatch")
    if torch.is_complex(matrix):
        raise ValueError(f"{name} must be real-valued")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"{name} must be finite")
    if bool((matrix < _TRANSPORT_NEGATIVE_TOL).any().item()):
        raise ValueError(f"{name} must be non-negative")

    if selected_mask is None:
        selected_mask = torch.ones(matrix.shape[:2], dtype=torch.bool, device=matrix.device)
    elif selected_mask.shape != matrix.shape[:2]:
        raise ValueError(f"{name} and selected_mask shape mismatch")

    invalid_dense_mass = matrix.masked_select(~dense_valid_mask[:, None, :].expand_as(matrix)).abs()
    if invalid_dense_mass.numel() and bool((invalid_dense_mass > _TRANSPORT_PAD_MASS_TOL).any().item()):
        raise ValueError(f"{name} must not allocate mass to invalid/padded positions")

    row_mass = matrix.sum(dim=-1)
    invalid_row_mass = row_mass.masked_select(~selected_mask).abs()
    if invalid_row_mass.numel() and bool((invalid_row_mass > _TRANSPORT_PAD_MASS_TOL).any().item()):
        raise ValueError(f"{name} must not allocate row mass to invalid/padded selected slots")

    selected_row_mass = row_mass.masked_select(selected_mask)
    if selected_row_mass.numel() == 0:
        raise ValueError(f"{name} must contain at least one selected row")
    if bool((selected_row_mass <= _TRANSPORT_MIN_ROW_MASS).any().item()):
        raise ValueError(f"{name} selected rows must have row mass > {_TRANSPORT_MIN_ROW_MASS}")
    if bool((selected_row_mass > _TRANSPORT_MAX_ROW_MASS + _TRANSPORT_ROW_MASS_ATOL).any().item()):
        raise ValueError(f"{name} selected row mass must be <= {_TRANSPORT_MAX_ROW_MASS}")

    if name == "allocation":
        ones = torch.ones_like(selected_row_mass)
        if not bool(torch.allclose(
            selected_row_mass,
            ones,
            atol=_TRANSPORT_ROW_MASS_ATOL,
            rtol=_TRANSPORT_ROW_MASS_RTOL,
        )):
            raise ValueError("allocation selected row mass must be close to 1")
        return

    if gates is None:
        return
    if gates.shape != matrix.shape[:2]:
        raise ValueError(f"{name} and gates shape mismatch")
    valid_gates = gates.to(dtype=row_mass.dtype).masked_select(selected_mask)
    if bool((valid_gates < -_TRANSPORT_ROW_MASS_ATOL).any().item()):
        raise ValueError("gates selected rows must be non-negative")
    if bool((valid_gates > _TRANSPORT_MAX_ROW_MASS + _TRANSPORT_ROW_MASS_ATOL).any().item()):
        raise ValueError(f"gates selected rows must be <= {_TRANSPORT_MAX_ROW_MASS}")
    if not bool(torch.allclose(
        selected_row_mass,
        valid_gates,
        atol=_TRANSPORT_ROW_MASS_ATOL,
        rtol=_TRANSPORT_ROW_MASS_RTOL,
    )):
        raise ValueError(f"{name} selected row mass must match gates")


@NECKS.register_module()
class PCOTMRASDetectorBridge(nn.Module):
    """Wrap PC-OT-MRAS reader output as detector-neck continuous tokens.

    The bridge never performs hard gather from integer positions. When source
    tokens are provided it recomputes the selected tokens by multiplying the
    reader acquisition matrix with dense tokens; otherwise it consumes the
    reader's already continuous selected_tokens field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_time_features: bool = True,
        time_feature_dim: int = 4,
        norm: bool = True,
        allocation_key: str = "acquisition_matrix",
        no_gate_scale: bool = False,
        metadata_position_source: str = "centers",
        metadata_position_repair: str = "fail",
        output_strides: Tuple[int, ...] = (1,),
        source_feature_level: Optional[int] = None,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(out_channels) <= 0:
            raise ValueError("out_channels must be positive")
        if int(time_feature_dim) <= 0:
            raise ValueError("time_feature_dim must be positive")
        if allocation_key not in {"allocation", "acquisition_matrix"}:
            raise ValueError("allocation_key must be 'allocation' or 'acquisition_matrix'")
        if metadata_position_source not in {"centers", "selected_times", "selected_positions"}:
            raise ValueError("metadata_position_source must be 'centers', 'selected_times', or 'selected_positions'")
        if metadata_position_repair not in {"fail", "sort_jitter"}:
            raise ValueError("metadata_position_repair must be 'fail' or 'sort_jitter'")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.add_time_features = bool(add_time_features)
        self.time_feature_dim = int(time_feature_dim)
        self.allocation_key = "allocation" if bool(no_gate_scale) else str(allocation_key)
        self.no_gate_scale = bool(no_gate_scale)
        self.metadata_position_source = str(metadata_position_source)
        self.metadata_position_repair = str(metadata_position_repair)
        self.output_strides = self._normalize_output_strides(output_strides)
        self.source_feature_level = self._normalize_source_feature_level(source_feature_level)

        self.token_proj = nn.Linear(self.in_channels, self.out_channels)
        self.time_proj = nn.Linear(self.time_feature_dim, self.out_channels) if self.add_time_features else None
        self.norm = nn.LayerNorm(self.out_channels) if bool(norm) else nn.Identity()

    @staticmethod
    def _normalize_output_strides(output_strides: Tuple[int, ...]) -> tuple[int, ...]:
        if output_strides is None:
            output_strides = (1,)
        try:
            strides = tuple(int(stride) for stride in output_strides)
        except TypeError as exc:
            raise ValueError("output_strides must be an iterable of positive integer strides") from exc
        if not strides:
            raise ValueError("output_strides must contain at least one stride")
        if any(stride <= 0 for stride in strides):
            raise ValueError("output_strides must contain only positive integer strides")
        if strides[0] != 1:
            raise ValueError("output_strides must start with stride 1 for the selected-token level")
        for prev, current in zip(strides, strides[1:]):
            if current != prev * 2:
                raise ValueError("output_strides must be a dyadic chain like (1, 2, 4, ...)")
        return strides

    @staticmethod
    def _normalize_source_feature_level(source_feature_level: Optional[int]) -> Optional[int]:
        if source_feature_level is None:
            return None
        if isinstance(source_feature_level, bool) or not isinstance(source_feature_level, int):
            raise ValueError("source_feature_level must be a non-negative integer or None")
        level = source_feature_level
        if level < 0:
            raise ValueError("source_feature_level must be a non-negative integer or None")
        return level

    @staticmethod
    def _is_neck_call(source_tokens: object, reader_outputs: object) -> bool:
        return isinstance(source_tokens, (tuple, list)) and isinstance(reader_outputs, (tuple, list))

    def _resolve_standard_source_level(
        self,
        feat_list: object,
        mask_list: object,
    ) -> tuple[int, bool, int]:
        if not isinstance(feat_list, (tuple, list)):
            raise ValueError("PC-OT-MRAS bridge standard neck path expects feature tensors as a tuple/list")
        if not isinstance(mask_list, (tuple, list)):
            raise ValueError("PC-OT-MRAS bridge standard neck path expects mask tensors as a tuple/list")
        if len(feat_list) <= 0:
            raise ValueError("PC-OT-MRAS bridge standard neck path expects at least one feature tensor")
        if len(mask_list) <= 0:
            raise ValueError("PC-OT-MRAS bridge standard neck path expects at least one mask tensor")
        if len(feat_list) != len(mask_list):
            raise ValueError("PC-OT-MRAS bridge standard neck path expects feature/mask level counts to match")

        num_levels = len(feat_list)
        if self.source_feature_level is None:
            if num_levels != 1:
                raise ValueError(
                    "PC-OT-MRAS bridge standard neck path received multiple feature levels; "
                    "set source_feature_level explicitly"
                )
            return 0, False, num_levels

        level = int(self.source_feature_level)
        if level >= num_levels:
            raise ValueError(
                "source_feature_level out of range for PC-OT-MRAS bridge standard neck path: "
                f"got {level}, num_levels={num_levels}"
            )
        return level, True, num_levels

    def _validate_neck_inputs(
        self,
        feat_list: object,
        mask_list: object,
    ) -> tuple[torch.Tensor, torch.Tensor, int, bool, int]:
        source_level, source_level_explicit, num_levels = self._resolve_standard_source_level(feat_list, mask_list)
        features = feat_list[source_level]
        masks = mask_list[source_level]
        if not torch.is_tensor(features):
            raise ValueError(f"features[{source_level}] must be a tensor")
        if not torch.is_tensor(masks):
            raise ValueError(f"masks[{source_level}] must be a tensor")
        if features.ndim != 3:
            raise ValueError(f"features[{source_level}] must be [B,C,T], got {tuple(features.shape)}")
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"expected features[{source_level}] channel dim {self.in_channels}, got {features.shape[1]}"
            )
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError(f"features[{source_level}] must be finite")
        if masks.device != features.device:
            raise ValueError(f"masks[{source_level}] must be on the same device as features[{source_level}]")
        valid = _prefix_binary_mask(masks, name=f"masks[{source_level}]")
        if valid.shape != (features.shape[0], features.shape[2]):
            raise ValueError(
                f"features[{source_level}] and masks[{source_level}] temporal shape mismatch: "
                f"{tuple(features.shape)} vs {tuple(valid.shape)}"
            )
        return features, valid, source_level, source_level_explicit, num_levels

    @staticmethod
    def _reader_outputs_from_metas(metas: object) -> Mapping[str, object]:
        if metas is None:
            raise ValueError("metas must be provided for the PC-OT-MRAS standard neck path")
        if isinstance(metas, Mapping):
            if _READER_OUTPUTS_META_KEY not in metas:
                raise ValueError(f"metas missing '{_READER_OUTPUTS_META_KEY}'")
            value = metas[_READER_OUTPUTS_META_KEY]
            if not isinstance(value, Mapping):
                raise ValueError(f"metas['{_READER_OUTPUTS_META_KEY}'] must be a mapping")
            _validate_reader_payload(value, location=_READER_OUTPUTS_META_KEY)
            return value
        if isinstance(metas, (tuple, list)):
            holders = [meta for meta in metas if isinstance(meta, Mapping) and _READER_OUTPUTS_META_KEY in meta]
            if not holders:
                raise ValueError(f"metas missing '{_READER_OUTPUTS_META_KEY}'")
            value = holders[0][_READER_OUTPUTS_META_KEY]
            if not isinstance(value, Mapping):
                raise ValueError(f"metas[*]['{_READER_OUTPUTS_META_KEY}'] must be a mapping")
            for holder in holders[1:]:
                if holder[_READER_OUTPUTS_META_KEY] is not value:
                    raise ValueError(f"metas[*]['{_READER_OUTPUTS_META_KEY}'] must point to one batch-level mapping")
            _validate_reader_payload(value, location=_READER_OUTPUTS_META_KEY)
            return value
        raise ValueError("metas must be a mapping or a list/tuple of mappings")

    @staticmethod
    def _slice_aux_for_sample(aux: Dict[str, object], *, batch_size: int, batch_idx: int) -> Dict[str, object]:
        sample_aux: Dict[str, object] = {}
        for key, value in aux.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                sample_aux[key] = value[batch_idx]
            else:
                sample_aux[key] = value
        return sample_aux

    @staticmethod
    def _aux_batch_size(aux: Dict[str, object]) -> Optional[int]:
        selected_mask = aux.get("selected_mask")
        if torch.is_tensor(selected_mask) and selected_mask.ndim == 2:
            return int(selected_mask.shape[0])
        selected_tokens = aux.get("selected_tokens")
        if torch.is_tensor(selected_tokens) and selected_tokens.ndim == 3:
            return int(selected_tokens.shape[0])
        return None

    @staticmethod
    def _temporal_meta_from_sample_aux(aux: Dict[str, object]) -> dict[str, object]:
        selected_mask = aux.get("selected_mask")
        centers = aux.get("centers")
        valid_mask = aux.get("valid_mask")
        if not (torch.is_tensor(selected_mask) and torch.is_tensor(centers) and torch.is_tensor(valid_mask)):
            return {}
        if selected_mask.ndim != 1 or centers.ndim != 1 or valid_mask.ndim != 1:
            return {}
        valid_count = int(selected_mask.long().sum().item())
        if valid_count <= 0:
            return {}
        dense_valid_len = int(valid_mask.long().sum().item())
        if dense_valid_len <= 0:
            raise ValueError("valid_mask must contain at least one valid dense position")
        selected_dense_positions = aux.get("selected_dense_positions")
        if torch.is_tensor(selected_dense_positions):
            if selected_dense_positions.ndim != 1 or selected_dense_positions.shape[0] < valid_count:
                raise ValueError("selected_dense_positions must cover selected_mask for PC-OT-MRAS temporal metadata")
            positions = selected_dense_positions[:valid_count].to(dtype=torch.float32)
        else:
            valid_centers = centers[:valid_count].to(dtype=torch.float32)
            if not bool(torch.isfinite(valid_centers).all().item()):
                raise ValueError("centers must be finite for PC-OT-MRAS temporal metadata")
            positions = valid_centers * float(dense_valid_len)
        positions = positions.clamp(min=0.0, max=float(dense_valid_len) - 1.0e-4)
        if not bool(torch.isfinite(positions).all().item()):
            raise ValueError("PC-OT-MRAS temporal metadata positions must be finite")
        if valid_count > 1 and bool(((positions[1:] - positions[:-1]) <= 0).any().item()):
            raise ValueError("PC-OT-MRAS temporal metadata positions must be strictly increasing")
        tensor_mode = aux.get("temporal_tensor_metadata_mode")
        if tensor_mode == "selected_dense_positions_from_selected_times":
            temporal_meta_mode = "selected_times_continuous"
        elif tensor_mode == "selected_dense_positions_from_selected_positions":
            temporal_meta_mode = "selected_positions_continuous"
        else:
            temporal_meta_mode = "ordered_slot_centers_continuous"
        meta = {
            "irregular_selected_positions": [float(item) for item in positions.detach().cpu().tolist()],
            "irregular_dense_valid_len": int(dense_valid_len),
            "irregular_selected_valid_len": int(dense_valid_len),
            "irregular_selected_valid_len_semantics": "carried_forward_dense_valid_len_alias",
            "irregular_selected_count": int(valid_count),
            "irregular_native_axis": True,
            "pc_ot_mras_temporal_meta_mode": temporal_meta_mode,
        }
        scale = aux.get("selected_times_dense_position_scale")
        if isinstance(scale, str):
            meta["pc_ot_mras_selected_times_dense_position_scale"] = scale
        strict_violation_count = aux.get("selected_dense_position_strict_violation_count")
        if torch.is_tensor(strict_violation_count) and strict_violation_count.ndim == 0:
            meta["pc_ot_mras_selected_dense_position_strict_violation_count"] = int(
                strict_violation_count.detach().cpu().item()
            )
        pre_repair_strict_violation_count = aux.get("selected_dense_position_pre_repair_strict_violation_count")
        if torch.is_tensor(pre_repair_strict_violation_count) and pre_repair_strict_violation_count.ndim == 0:
            meta["pc_ot_mras_selected_dense_position_pre_repair_strict_violation_count"] = int(
                pre_repair_strict_violation_count.detach().cpu().item()
            )
        jitter_repair_count = aux.get("selected_dense_position_jitter_repair_count")
        if torch.is_tensor(jitter_repair_count) and jitter_repair_count.ndim == 0:
            meta["pc_ot_mras_selected_dense_position_jitter_repair_count"] = int(
                jitter_repair_count.detach().cpu().item()
            )
        repair_mode = aux.get("metadata_position_repair_mode")
        if isinstance(repair_mode, str):
            meta["metadata_position_repair_mode"] = repair_mode
        return meta

    @staticmethod
    def _write_bridge_metadata(metas: object, aux: Dict[str, object]) -> object:
        batch_size = PCOTMRASDetectorBridge._aux_batch_size(aux)
        if isinstance(metas, Mapping):
            sample_aux = (
                PCOTMRASDetectorBridge._slice_aux_for_sample(aux, batch_size=batch_size, batch_idx=0)
                if batch_size == 1
                else aux
            )
            out = dict(metas)
            out[_BRIDGE_META_KEY] = sample_aux
            out.update(PCOTMRASDetectorBridge._temporal_meta_from_sample_aux(sample_aux))
            return out
        if isinstance(metas, list):
            if batch_size is None:
                batch_size = len(metas)
            if len(metas) != batch_size:
                raise ValueError("metas length must match bridge aux batch size")
            out = []
            for batch_idx, meta in enumerate(metas):
                if not isinstance(meta, Mapping):
                    raise ValueError("each metas item must be a mapping")
                item = dict(meta)
                sample_aux = PCOTMRASDetectorBridge._slice_aux_for_sample(
                    aux,
                    batch_size=batch_size,
                    batch_idx=batch_idx,
                )
                item[_BRIDGE_META_KEY] = sample_aux
                item.update(PCOTMRASDetectorBridge._temporal_meta_from_sample_aux(sample_aux))
                out.append(item)
            return out
        if isinstance(metas, tuple):
            if batch_size is None:
                batch_size = len(metas)
            if len(metas) != batch_size:
                raise ValueError("metas length must match bridge aux batch size")
            out = []
            for batch_idx, meta in enumerate(metas):
                if not isinstance(meta, Mapping):
                    raise ValueError("each metas item must be a mapping")
                item = dict(meta)
                sample_aux = PCOTMRASDetectorBridge._slice_aux_for_sample(
                    aux,
                    batch_size=batch_size,
                    batch_idx=batch_idx,
                )
                item[_BRIDGE_META_KEY] = sample_aux
                item.update(PCOTMRASDetectorBridge._temporal_meta_from_sample_aux(sample_aux))
                out.append(item)
            return tuple(out)
        raise ValueError("metas must be a mapping or a list/tuple of mappings")

    @staticmethod
    def _validate_standard_metas_batch_size(metas: object, *, batch_size: int) -> None:
        if metas is None:
            raise ValueError("metas must be provided for the PC-OT-MRAS standard neck path")
        if isinstance(metas, Mapping):
            if int(batch_size) != 1:
                raise ValueError("mapping metas are only valid for batch size 1 in the standard neck path")
            return
        if isinstance(metas, (list, tuple)):
            if len(metas) != int(batch_size):
                raise ValueError("metas length must match batch size in the standard neck path")
            return
        raise ValueError("metas must be a mapping or a list/tuple of mappings")

    @staticmethod
    def _validate_reader_tensor_tree(value: object, *, expected_device: torch.device, location: str) -> None:
        if torch.is_tensor(value):
            if value.device != expected_device:
                raise ValueError(f"{location} must be on the same device as selected source features")
            if torch.is_complex(value):
                raise ValueError(f"{location} must be real-valued")
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{location} must be finite")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                _location = f"{location}.{str(key or '')}"
                PCOTMRASDetectorBridge._validate_reader_tensor_tree(
                    item,
                    expected_device=expected_device,
                    location=_location,
                )
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                PCOTMRASDetectorBridge._validate_reader_tensor_tree(
                    item,
                    expected_device=expected_device,
                    location=f"{location}[{index}]",
                )

    def _continuous_tokens(
        self,
        source_tokens: Optional[torch.Tensor],
        reader_outputs: Mapping[str, object],
    ) -> tuple[torch.Tensor, str, bool]:
        if source_tokens is None:
            selected_tokens = _require_tensor(reader_outputs, "selected_tokens")
            if selected_tokens.ndim != 3:
                raise ValueError(f"selected_tokens must be [B,K,C], got {tuple(selected_tokens.shape)}")
            if selected_tokens.shape[-1] != self.in_channels:
                raise ValueError(f"expected selected token dim {self.in_channels}, got {selected_tokens.shape[-1]}")
            if torch.is_complex(selected_tokens):
                raise ValueError("selected_tokens must be real-valued")
            if not bool(torch.isfinite(selected_tokens).all().item()):
                raise ValueError("selected_tokens must be finite")
            return selected_tokens, "reader_outputs_selected_tokens_unverified", False

        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must be [B,T,C], got {tuple(source_tokens.shape)}")
        if source_tokens.shape[-1] != self.in_channels:
            raise ValueError(f"expected source token dim {self.in_channels}, got {source_tokens.shape[-1]}")
        if torch.is_complex(source_tokens):
            raise ValueError("source_tokens must be real-valued")
        if not bool(torch.isfinite(source_tokens).all().item()):
            raise ValueError("source_tokens must be finite")

        allocation = _require_tensor(reader_outputs, self.allocation_key)
        if allocation.ndim != 3:
            raise ValueError(f"{self.allocation_key} must be [B,K,T], got {tuple(allocation.shape)}")
        if allocation.device != source_tokens.device:
            raise ValueError(f"{self.allocation_key} must be on the same device as source_tokens")
        if allocation.shape[0] != source_tokens.shape[0] or allocation.shape[2] != source_tokens.shape[1]:
            raise ValueError(
                f"{self.allocation_key} and source_tokens shape mismatch: "
                f"{tuple(allocation.shape)} vs {tuple(source_tokens.shape)}"
            )
        if torch.is_complex(allocation):
            raise ValueError(f"{self.allocation_key} must be real-valued")
        if not bool(torch.isfinite(allocation).all().item()):
            raise ValueError(f"{self.allocation_key} must be finite")

        valid = reader_outputs.get("valid_mask")
        if valid is None:
            raise ValueError("reader_outputs['valid_mask'] is required when recomputing selected tokens")
        if not torch.is_tensor(valid):
            raise ValueError("reader_outputs['valid_mask'] must be a tensor when provided")
        if valid.device != source_tokens.device:
            raise ValueError("valid_mask must be on the same device as source_tokens")
        dense_valid = _prefix_binary_mask(valid, name="valid_mask")
        if dense_valid.shape != source_tokens.shape[:2]:
            raise ValueError("valid_mask shape mismatch")

        selected_mask = None
        raw_selected_mask = reader_outputs.get("selected_mask")
        if raw_selected_mask is not None:
            if not torch.is_tensor(raw_selected_mask):
                raise ValueError("reader_outputs['selected_mask'] must be a tensor when provided")
            if raw_selected_mask.device != source_tokens.device:
                raise ValueError("selected_mask must be on the same device as source_tokens")
            selected_mask = _prefix_binary_mask(raw_selected_mask, name="selected_mask")
            if selected_mask.shape != allocation.shape[:2]:
                raise ValueError("selected_mask shape mismatch")

        gates = _optional_slot_tensor(
            reader_outputs,
            "gates",
            expected_shape=allocation.shape[:2],
            expected_device=source_tokens.device,
        )
        _validate_transport_matrix_contract(
            allocation,
            name=self.allocation_key,
            dense_valid_mask=dense_valid,
            selected_mask=selected_mask,
            gates=gates if self.allocation_key == "acquisition_matrix" else None,
        )
        source_tokens = source_tokens.masked_fill(~dense_valid.unsqueeze(-1), 0.0)
        return torch.bmm(allocation.to(dtype=source_tokens.dtype), source_tokens), f"recomputed_from_{self.allocation_key}", True

    def _selected_mask(self, reader_outputs: Mapping[str, object], selected_tokens: torch.Tensor) -> torch.Tensor:
        raw_mask = reader_outputs.get("selected_mask")
        if raw_mask is not None:
            if not torch.is_tensor(raw_mask):
                raise ValueError("reader_outputs['selected_mask'] must be a tensor when provided")
            if raw_mask.device != selected_tokens.device:
                raise ValueError("selected_mask must be on the same device as selected_tokens")
            selected_mask = _prefix_binary_mask(raw_mask, name="selected_mask")
            if selected_mask.shape != selected_tokens.shape[:2]:
                raise ValueError("selected_mask shape mismatch")
            return selected_mask

        allocation = reader_outputs.get(self.allocation_key)
        if torch.is_tensor(allocation):
            if allocation.ndim != 3 or allocation.shape[:2] != selected_tokens.shape[:2]:
                raise ValueError(f"{self.allocation_key} shape mismatch")
            if allocation.device != selected_tokens.device:
                raise ValueError(f"{self.allocation_key} must be on the same device as selected_tokens")
            return _prefix_binary_mask(allocation.sum(dim=-1) > 0, name="selected_mask")
        return torch.ones(selected_tokens.shape[:2], dtype=torch.bool, device=selected_tokens.device)

    def _time_features(self, reader_outputs: Mapping[str, object], selected_tokens: torch.Tensor) -> torch.Tensor:
        required = ("selected_times", "centers", "widths", "gates")
        values = []
        for key in required:
            value = _require_tensor(reader_outputs, key)
            if value.shape != selected_tokens.shape[:2]:
                raise ValueError(f"{key} must be [B,K]")
            if value.device != selected_tokens.device:
                raise ValueError(f"{key} must be on the same device as selected_tokens")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{key} must be finite")
            values.append(value.to(dtype=selected_tokens.dtype))
        base = torch.stack(values, dim=-1)
        if self.time_feature_dim == 4:
            return base
        if self.time_feature_dim < 4:
            return base[..., : self.time_feature_dim]
        pad = torch.zeros(
            (*base.shape[:2], self.time_feature_dim - 4),
            device=base.device,
            dtype=base.dtype,
        )
        return torch.cat([base, pad], dim=-1)

    @staticmethod
    def _reorder_slot_aligned_tensor(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        view_shape = (order.shape[0], order.shape[1]) + (1,) * (value.ndim - 2)
        gather_index = order.reshape(view_shape).expand(-1, -1, *value.shape[2:])
        return torch.gather(value, dim=1, index=gather_index)

    def _repair_slot_order_for_temporal_metadata(
        self,
        reader_outputs: Mapping[str, object],
        selected_tokens: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> tuple[Mapping[str, object], torch.Tensor, torch.Tensor]:
        if self.metadata_position_repair != "sort_jitter":
            return reader_outputs, selected_tokens, selected_mask
        if self.metadata_position_source not in {"selected_times", "selected_positions"}:
            return reader_outputs, selected_tokens, selected_mask

        sort_source = reader_outputs.get(self.metadata_position_source)
        if not torch.is_tensor(sort_source):
            raise ValueError(f"{self.metadata_position_source} is required for PC-OT-MRAS temporal metadata repair")
        if sort_source.shape != selected_mask.shape:
            raise ValueError(f"{self.metadata_position_source} shape must match selected_mask for temporal metadata repair")
        if sort_source.device != selected_tokens.device:
            raise ValueError(f"{self.metadata_position_source} must be on the selected-token device")
        if not bool(torch.isfinite(sort_source).all().item()):
            raise ValueError(f"{self.metadata_position_source} must be finite for temporal metadata repair")

        sort_key = sort_source.to(dtype=torch.float32)
        invalid_sentinel = sort_key.new_full((), float("inf"))
        sort_key = torch.where(selected_mask, sort_key, invalid_sentinel)
        order = torch.argsort(sort_key, dim=1)

        selected_tokens = self._reorder_slot_aligned_tensor(selected_tokens, order)
        selected_mask = self._reorder_slot_aligned_tensor(selected_mask, order)
        repaired_outputs: Dict[str, object] = dict(reader_outputs)
        old_shape = tuple(order.shape)
        for key, value in reader_outputs.items():
            if torch.is_tensor(value) and value.ndim >= 2 and tuple(value.shape[:2]) == old_shape:
                repaired_outputs[key] = self._reorder_slot_aligned_tensor(value, order)
        repaired_outputs["metadata_position_repair_mode"] = "sort_jitter"
        repaired_outputs["metadata_position_sort_order"] = order
        return repaired_outputs, selected_tokens, selected_mask

    @staticmethod
    def _repair_dense_positions_with_jitter(
        dense_positions: torch.Tensor,
        selected_mask: torch.Tensor,
        max_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repaired = dense_positions.clone()
        repair_counts = torch.zeros((dense_positions.shape[0],), dtype=torch.long, device=dense_positions.device)
        for batch_idx in range(dense_positions.shape[0]):
            valid_count = int(selected_mask[batch_idx].long().sum().item())
            if valid_count <= 1:
                continue
            row = repaired[batch_idx, :valid_count].clone()
            max_allowed = float(max_position[batch_idx, 0].detach().cpu().item())
            eps = min(1.0e-3, max(max_allowed, 1.0) / max(float(valid_count) * 4.0, 1.0))
            count = 0
            for idx in range(1, valid_count):
                min_next = row[idx - 1] + eps
                if bool((row[idx] <= min_next).detach().cpu().item()):
                    row[idx] = min_next
                    count += 1
            if bool((row[-1] > max_allowed).detach().cpu().item()):
                shift = row[-1] - row.new_tensor(max_allowed)
                row = row - shift
                if bool((row[0] < 0.0).detach().cpu().item()):
                    row = torch.linspace(
                        0.0,
                        max_allowed,
                        steps=valid_count,
                        device=row.device,
                        dtype=row.dtype,
                    )
                    count = max(count, valid_count)
            row = row.clamp(min=0.0, max=max_allowed)
            repaired[batch_idx, :valid_count] = row
            repair_counts[batch_idx] = int(count)
        return repaired, repair_counts

    def _metadata(
        self,
        reader_outputs: Mapping[str, object],
        selected_tokens: torch.Tensor,
        selected_mask: torch.Tensor,
        *,
        selected_tokens_source: str,
        selected_tokens_source_verified: bool,
        source_feature_level: Optional[int] = None,
        source_feature_num_levels: Optional[int] = None,
        source_feature_level_explicit: Optional[bool] = None,
    ) -> Dict[str, object]:
        meta: Dict[str, object] = {
            "schema_version": "pc_ot_mras_continuous_bridge_v0",
            "continuous_axis": True,
            "uses_hard_gather": False,
            "selected_tokens_source": selected_tokens_source,
            "selected_tokens_source_verified": selected_tokens_source_verified,
            "selected_tokens_masked": True,
            "selected_tokens": selected_tokens,
            "selected_mask": selected_mask,
            "output_strides": str(self.output_strides),
        }
        if source_feature_level is not None:
            meta["source_feature_level"] = int(source_feature_level)
            meta["source_feature_num_levels"] = int(source_feature_num_levels)
            meta["source_feature_level_explicit"] = bool(source_feature_level_explicit)
            meta["source_feature_level_selection"] = (
                "explicit" if bool(source_feature_level_explicit) else "implicit_single_level"
            )
        centers = reader_outputs.get("centers")
        dense_valid_mask = reader_outputs.get("valid_mask")
        if torch.is_tensor(centers) and torch.is_tensor(dense_valid_mask):
            if centers.shape != selected_mask.shape:
                raise ValueError("centers shape must match selected_mask for PC-OT-MRAS temporal metadata")
            if centers.device != selected_tokens.device or dense_valid_mask.device != selected_tokens.device:
                raise ValueError("temporal metadata tensors must be on the selected-token device")
            dense_valid_mask = _prefix_binary_mask(dense_valid_mask, name="valid_mask")
            dense_valid_len = dense_valid_mask.long().sum(dim=1).to(device=centers.device, dtype=centers.dtype)
            if bool((dense_valid_len <= 0).any().item()):
                raise ValueError("valid_mask must contain at least one valid dense position")
            max_position = (dense_valid_len[:, None] - 1.0e-4).clamp_min(0.0)
            if self.metadata_position_source == "selected_times":
                selected_times = _require_tensor(reader_outputs, "selected_times")
                if selected_times.shape != selected_mask.shape:
                    raise ValueError("selected_times shape must match selected_mask for PC-OT-MRAS temporal metadata")
                if selected_times.device != selected_tokens.device:
                    raise ValueError("selected_times must be on the selected-token device")
                dense_position_scale = (dense_valid_len - 1.0).clamp_min(0.0).to(dtype=torch.float32)
                raw_dense_positions = selected_times.to(dtype=torch.float32) * dense_position_scale[:, None].to(
                    dtype=torch.float32
                )
                temporal_mode = "selected_dense_positions_from_selected_times"
                meta["selected_times_dense_position_scale"] = "valid_len_minus_one_token_index"
            elif self.metadata_position_source == "selected_positions":
                selected_positions = reader_outputs.get("selected_positions")
                if torch.is_tensor(selected_positions):
                    if selected_positions.shape != selected_mask.shape:
                        raise ValueError(
                            "selected_positions shape must match selected_mask for PC-OT-MRAS temporal metadata"
                        )
                    if selected_positions.device != selected_tokens.device:
                        raise ValueError("selected_positions must be on the selected-token device")
                    raw_dense_positions = selected_positions.to(dtype=torch.float32)
                else:
                    acquisition = _require_tensor(reader_outputs, "acquisition_matrix")
                    if acquisition.ndim != 3 or acquisition.shape[:2] != selected_mask.shape:
                        raise ValueError(
                            "acquisition_matrix shape must match selected_mask for PC-OT-MRAS temporal metadata"
                        )
                    if acquisition.device != selected_tokens.device:
                        raise ValueError("acquisition_matrix must be on the selected-token device")
                    dense_pos = torch.arange(acquisition.shape[-1], device=acquisition.device, dtype=torch.float32)
                    acquisition_f = acquisition.to(dtype=torch.float32)
                    row_mass = acquisition_f.sum(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
                    raw_dense_positions = torch.einsum("bkt,t->bk", acquisition_f, dense_pos) / row_mass
                temporal_mode = "selected_dense_positions_from_selected_positions"
            else:
                raw_dense_positions = centers.to(dtype=torch.float32) * dense_valid_len[:, None].to(dtype=torch.float32)
                temporal_mode = "selected_dense_positions_from_centers"
            if not bool(torch.isfinite(raw_dense_positions).all().item()):
                raise ValueError("PC-OT-MRAS temporal metadata positions must be finite")
            dense_positions = torch.minimum(
                torch.maximum(raw_dense_positions, centers.new_zeros((), dtype=torch.float32)),
                max_position.to(dtype=torch.float32),
            )
            dense_positions = dense_positions.masked_fill(~selected_mask, 0.0)
            if dense_positions.shape[1] > 1:
                position_deltas = dense_positions[:, 1:] - dense_positions[:, :-1]
                adjacent_valid = selected_mask[:, 1:] & selected_mask[:, :-1]
                strict_violations = (position_deltas <= 0.0) & adjacent_valid
                meta["selected_dense_position_pre_repair_strict_violation_count"] = strict_violations.long().sum(dim=1)
                if (
                    self.metadata_position_repair == "sort_jitter"
                    and self.metadata_position_source in {"selected_times", "selected_positions"}
                    and bool(strict_violations.any().item())
                ):
                    dense_positions, repair_counts = self._repair_dense_positions_with_jitter(
                        dense_positions,
                        selected_mask,
                        max_position.to(dtype=torch.float32),
                    )
                    meta["selected_dense_position_repair_mode"] = "sort_jitter"
                    meta["selected_dense_position_jitter_repair_count"] = repair_counts
                    position_deltas = dense_positions[:, 1:] - dense_positions[:, :-1]
                    strict_violations = (position_deltas <= 0.0) & adjacent_valid
                meta["selected_dense_position_delta"] = position_deltas.masked_fill(~adjacent_valid, 0.0)
                meta["selected_dense_position_strict_violation_count"] = strict_violations.long().sum(dim=1)
                meta["selected_dense_positions_strictly_increasing"] = strict_violations.long().sum(dim=1) == 0
            repair_mode = reader_outputs.get("metadata_position_repair_mode")
            if isinstance(repair_mode, str):
                meta["metadata_position_repair_mode"] = repair_mode
            meta["selected_dense_positions"] = dense_positions
            meta["dense_valid_len_tensor"] = dense_valid_len.to(dtype=torch.float32)
            meta["temporal_tensor_metadata_mode"] = temporal_mode
        for key in (
            "selected_times",
            "selected_positions",
            "centers",
            "widths",
            "gates",
            "allocation",
            "acquisition_matrix",
            "valid_mask",
            "time_coords",
            "process_logits",
            "start_logits",
            "end_logits",
            "boundary_logits",
            "uncertainty_logits",
            "redundancy_logits",
            "pair_logits",
            "pair_prob",
            "pair_valid_mask",
        ):
            value = reader_outputs.get(key)
            if torch.is_tensor(value):
                if key in _SLOT_ALIGNED_AUX_KEYS and value.shape[:2] == selected_mask.shape:
                    expand_dims = (1,) * (value.ndim - selected_mask.ndim)
                    row_mask = selected_mask.reshape(*selected_mask.shape, *expand_dims)
                    meta[key] = value.masked_fill(~row_mask, 0)
                else:
                    meta[key] = value
        return meta

    @staticmethod
    def _downsample_prefix_level(
        features: torch.Tensor,
        mask: torch.Tensor,
        *,
        level_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"level{level_idx - 1} features must be [B,C,T], got {tuple(features.shape)}")
        valid = _prefix_binary_mask(mask, name=f"level{level_idx - 1}_mask")
        if valid.shape != (features.shape[0], features.shape[2]):
            raise ValueError(
                f"level{level_idx - 1} feature/mask temporal shape mismatch: "
                f"{tuple(features.shape)} vs {tuple(valid.shape)}"
            )

        even_feat = features[:, :, 0::2]
        odd_feat = features[:, :, 1::2]
        even_valid = valid[:, 0::2]
        odd_valid = valid[:, 1::2]
        if odd_feat.shape[-1] < even_feat.shape[-1]:
            pad_len = even_feat.shape[-1] - odd_feat.shape[-1]
            odd_feat = torch.cat(
                (
                    odd_feat,
                    features.new_zeros((features.shape[0], features.shape[1], pad_len)),
                ),
                dim=-1,
            )
            odd_valid = torch.cat(
                (
                    odd_valid,
                    valid.new_zeros((valid.shape[0], pad_len)),
                ),
                dim=1,
            )

        even_weight = even_valid[:, None].to(dtype=features.dtype)
        odd_weight = odd_valid[:, None].to(dtype=features.dtype)
        pooled_mask = even_valid | odd_valid
        denom = (even_weight + odd_weight).clamp_min(1.0)
        pooled = (even_feat * even_weight + odd_feat * odd_weight) / denom
        pooled = pooled.masked_fill(~pooled_mask[:, None], 0.0).contiguous()
        pooled_mask = _prefix_binary_mask(pooled_mask, name=f"level{level_idx}_mask")
        return pooled, pooled_mask

    def _make_output_pyramid(
        self,
        level0_features: torch.Tensor,
        level0_mask: torch.Tensor,
    ) -> tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        feats = [level0_features]
        masks = [level0_mask]
        for level_idx in range(1, len(self.output_strides)):
            next_feat, next_mask = self._downsample_prefix_level(
                feats[-1],
                masks[-1],
                level_idx=level_idx,
            )
            feats.append(next_feat)
            masks.append(next_mask)
        return tuple(feats), tuple(masks)

    def _forward_reader_outputs(
        self,
        source_tokens: Optional[torch.Tensor],
        reader_outputs: Mapping[str, object],
        *,
        return_aux: bool,
        source_feature_level: Optional[int] = None,
        source_feature_num_levels: Optional[int] = None,
        source_feature_level_explicit: Optional[bool] = None,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]] | Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Dict[str, object]]:
        selected_tokens, selected_tokens_source, selected_tokens_source_verified = self._continuous_tokens(source_tokens, reader_outputs)
        selected_mask = self._selected_mask(reader_outputs, selected_tokens)
        reader_outputs, selected_tokens, selected_mask = self._repair_slot_order_for_temporal_metadata(
            reader_outputs,
            selected_tokens,
            selected_mask,
        )
        out = self.token_proj(selected_tokens)
        if self.time_proj is not None:
            out = out + self.time_proj(self._time_features(reader_outputs, selected_tokens))
        selected_tokens_for_aux = selected_tokens.masked_fill(~selected_mask.unsqueeze(-1), 0.0)
        out = self.norm(out).masked_fill(~selected_mask.unsqueeze(-1), 0.0)
        feats, masks = self._make_output_pyramid(out.transpose(1, 2).contiguous(), selected_mask)
        if not bool(return_aux):
            return feats, masks
        return feats, masks, self._metadata(
            reader_outputs,
            selected_tokens_for_aux,
            selected_mask,
            selected_tokens_source=selected_tokens_source,
            selected_tokens_source_verified=selected_tokens_source_verified,
            source_feature_level=source_feature_level,
            source_feature_num_levels=source_feature_num_levels,
            source_feature_level_explicit=source_feature_level_explicit,
        )

    def _forward_standard_neck(
        self,
        feat_list: object,
        mask_list: object,
        metas: object,
    ) -> tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], object]:
        features, input_mask, source_level, source_level_explicit, num_levels = self._validate_neck_inputs(
            feat_list,
            mask_list,
        )
        self._validate_standard_metas_batch_size(metas, batch_size=int(features.shape[0]))
        reader_outputs = self._reader_outputs_from_metas(metas)
        self._validate_reader_tensor_tree(
            reader_outputs,
            expected_device=features.device,
            location=_READER_OUTPUTS_META_KEY,
        )
        reader_valid = _require_tensor(reader_outputs, "valid_mask")
        if not torch.equal(_prefix_binary_mask(reader_valid, name="valid_mask"), input_mask):
            raise ValueError(f"valid_mask must match masks[{source_level}] in the standard neck path")
        source_tokens = features.transpose(1, 2).contiguous()
        feats, masks, aux = self._forward_reader_outputs(
            source_tokens,
            reader_outputs,
            return_aux=True,
            source_feature_level=source_level,
            source_feature_num_levels=num_levels,
            source_feature_level_explicit=source_level_explicit,
        )
        return feats, masks, self._write_bridge_metadata(metas, aux)

    def forward(
        self,
        source_tokens: Optional[torch.Tensor] = None,
        reader_outputs: Optional[Mapping[str, object]] = None,
        return_aux: bool = False,
        metas: object = None,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]] | Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Dict[str, object]] | tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], object]:
        if self._is_neck_call(source_tokens, reader_outputs):
            return self._forward_standard_neck(source_tokens, reader_outputs, metas)
        if reader_outputs is None and isinstance(source_tokens, Mapping):
            reader_outputs = source_tokens
            source_tokens = None
        if reader_outputs is None:
            raise ValueError("reader_outputs must be provided")
        if not isinstance(reader_outputs, Mapping):
            raise ValueError("reader_outputs must be a mapping")

        return self._forward_reader_outputs(source_tokens, reader_outputs, return_aux=return_aux)


ProcessConditionedOrderedTransportMRASDetectorBridge = PCOTMRASDetectorBridge

__all__ = ["PCOTMRASDetectorBridge", "ProcessConditionedOrderedTransportMRASDetectorBridge"]
