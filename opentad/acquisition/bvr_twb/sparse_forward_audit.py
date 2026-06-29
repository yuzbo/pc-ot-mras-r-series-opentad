from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .raw_handoff import build_sparse_raw_handoff
from .types import ROUTE_LABEL, sorted_unique_positions
from .validators import build_original_time_metadata


SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS = "sparse_forward_precheck_shape_only_no_metric_claim"
SPARSE_FORWARD_REAL_MODULE_CLAIM_STATUS = "sparse_forward_precheck_real_module_no_metric_claim"


@dataclass
class SparseRawHandoffLedger:
    video_name: str
    window_id: str
    dense_T: int
    selected_positions: Sequence[int]
    selector_method: str = "bvr_twb"
    selection_unit: str = "frame"
    route_label: str = ROUTE_LABEL
    claim_status: str = SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS
    audit_mode: str = "fake_raw"
    sparse_compute_claim: bool = False
    raw_frame_inds_in: Optional[Sequence[int]] = None
    decoded_frame_count: Optional[int] = None
    decoded_unique_count: Optional[int] = None
    padded_duplicate_count: int = 0
    backbone_input_shape_before_preprocess: Optional[Sequence[int]] = None
    backbone_input_shape_after_preprocess: Optional[Sequence[int]] = None
    dense_backbone_chunk_count: Optional[int] = None
    backbone_forward_chunk_count: Optional[int] = None
    time_embed_valid_count: Optional[int] = None
    time_embed_total_count: Optional[int] = None
    post_backbone_feature_len: Optional[int] = None
    post_backbone_interpolated_to_dense: bool = False
    detector_prepad_feature_len: Optional[int] = None
    detector_pad_len: Optional[int] = None
    detector_mask_true_count: Optional[int] = None
    rpn_valid_temporal_len: Optional[int] = None
    temporal_decode_uses_original_time: bool = True
    original_time_metadata: Optional[Dict[str, object]] = None
    dense_raw_backbone_handoff: bool = False
    selected_inputs_is_gathered: bool = True
    forbidden_deploy_fields_absent: bool = True
    module_forward_evidence: bool = False
    provenance: Dict[str, object] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        dense_T = int(self.dense_T)
        positions = sorted_unique_positions(self.selected_positions, dense_T)
        handoff = build_sparse_raw_handoff(positions, dense_T)
        original_time = self.original_time_metadata or build_original_time_metadata(
            dense_T,
            positions,
            fps=30.0,
        )
        valid_k = int(len(positions))
        row = {
            "route_label": self.route_label,
            "claim_status": self.claim_status,
            "audit_mode": self.audit_mode,
            "video_name": self.video_name,
            "window_id": self.window_id,
            "selector_method": self.selector_method,
            "selection_unit": self.selection_unit,
            "selected_positions_unit": self.selection_unit,
            "dense_T": dense_T,
            "selected_positions": positions,
            "valid_k": valid_k,
            "raw_frame_inds_in": list(self.raw_frame_inds_in) if self.raw_frame_inds_in is not None else handoff["raw_frame_inds_in"],
            "decoded_frame_count": (
                int(self.decoded_frame_count)
                if self.decoded_frame_count is not None
                else handoff["decoded_frame_count"]
            ),
            "decoded_unique_count": (
                int(self.decoded_unique_count)
                if self.decoded_unique_count is not None
                else handoff["decoded_unique_count"]
            ),
            "padded_duplicate_count": int(self.padded_duplicate_count),
            "backbone_input_shape_before_preprocess": _shape_or_default(
                self.backbone_input_shape_before_preprocess,
                [1, 1, valid_k, 3, 224, 224],
            ),
            "backbone_input_shape_after_preprocess": _shape_or_default(
                self.backbone_input_shape_after_preprocess,
                [valid_k, 1, 3, 1, 224, 224],
            ),
            "dense_backbone_chunk_count": int(self.dense_backbone_chunk_count or dense_T),
            "backbone_forward_chunk_count": int(self.backbone_forward_chunk_count or valid_k),
            "time_embed_valid_count": int(self.time_embed_valid_count or valid_k),
            "time_embed_total_count": int(self.time_embed_total_count or valid_k),
            "post_backbone_feature_len": int(self.post_backbone_feature_len or valid_k),
            "post_backbone_interpolated_to_dense": bool(self.post_backbone_interpolated_to_dense),
            "detector_prepad_feature_len": int(self.detector_prepad_feature_len or valid_k),
            "detector_pad_len": int(self.detector_pad_len or valid_k),
            "detector_mask_true_count": int(self.detector_mask_true_count or valid_k),
            "rpn_valid_temporal_len": int(self.rpn_valid_temporal_len or valid_k),
            "temporal_decode_uses_original_time": bool(self.temporal_decode_uses_original_time),
            "original_time_metadata": original_time,
            "dense_raw_backbone_handoff": bool(self.dense_raw_backbone_handoff),
            "selected_inputs_is_gathered": bool(self.selected_inputs_is_gathered),
            "sparse_compute_claim": bool(self.sparse_compute_claim),
            "forbidden_deploy_fields_absent": bool(self.forbidden_deploy_fields_absent),
            "module_forward_evidence": bool(self.module_forward_evidence),
            "provenance": _default_provenance(self.provenance),
        }
        row.update(self.extra)
        return row


class SparseForwardAuditContext:
    def __init__(
        self,
        video_name,
        window_id,
        dense_T,
        selected_positions,
        selection_unit="frame",
        audit_mode="fake_raw",
        claim_status=SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS,
    ):
        self.video_name = video_name
        self.window_id = window_id
        self.dense_T = int(dense_T)
        self.selected_positions = sorted_unique_positions(selected_positions, self.dense_T)
        self.selection_unit = selection_unit
        self.audit_mode = audit_mode
        self.claim_status = claim_status
        self.records = {}

    def record_raw_decode(self, raw_frame_inds_in, padded_duplicate_count=0):
        raw = [int(pos) for pos in raw_frame_inds_in]
        self.records.update(
            {
                "raw_frame_inds_in": raw,
                "decoded_frame_count": int(len(raw)),
                "decoded_unique_count": int(len(set(raw))),
                "padded_duplicate_count": int(padded_duplicate_count),
            }
        )

    def record_backbone(
        self,
        input_shape_before_preprocess,
        input_shape_after_preprocess,
        forward_chunk_count,
        dense_backbone_chunk_count=None,
        time_embed_valid_count=None,
        time_embed_total_count=None,
        post_backbone_feature_len=None,
        post_backbone_interpolated_to_dense=False,
    ):
        valid_k = len(self.selected_positions)
        self.records.update(
            {
                "backbone_input_shape_before_preprocess": list(input_shape_before_preprocess),
                "backbone_input_shape_after_preprocess": list(input_shape_after_preprocess),
                "backbone_forward_chunk_count": int(forward_chunk_count),
                "dense_backbone_chunk_count": int(dense_backbone_chunk_count or self.dense_T),
                "time_embed_valid_count": int(time_embed_valid_count or valid_k),
                "time_embed_total_count": int(time_embed_total_count or time_embed_valid_count or valid_k),
                "post_backbone_feature_len": int(post_backbone_feature_len or forward_chunk_count),
                "post_backbone_interpolated_to_dense": bool(post_backbone_interpolated_to_dense),
            }
        )

    def record_detector(self, prepad_feature_len, pad_len, mask_true_count, rpn_valid_temporal_len=None):
        self.records.update(
            {
                "detector_prepad_feature_len": int(prepad_feature_len),
                "detector_pad_len": int(pad_len),
                "detector_mask_true_count": int(mask_true_count),
                "rpn_valid_temporal_len": int(rpn_valid_temporal_len or mask_true_count),
            }
        )

    def record_temporal_decode(self, uses_original_time=True, original_time_metadata=None):
        self.records.update(
            {
                "temporal_decode_uses_original_time": bool(uses_original_time),
                "original_time_metadata": original_time_metadata,
            }
        )

    def build_ledger(self, sparse_compute_claim=False, module_forward_evidence=False):
        return SparseRawHandoffLedger(
            video_name=self.video_name,
            window_id=self.window_id,
            dense_T=self.dense_T,
            selected_positions=self.selected_positions,
            selection_unit=self.selection_unit,
            audit_mode=self.audit_mode,
            claim_status=self.claim_status,
            sparse_compute_claim=bool(sparse_compute_claim),
            module_forward_evidence=bool(module_forward_evidence),
            **self.records,
        ).to_dict()


def build_fake_sparse_forward_ledger(
    video_name="fake_video",
    window_id="fake_video_0000",
    dense_T=96,
    selected_positions=None,
    detector_pad_len=128,
    module_forward=False,
):
    positions = selected_positions or [0, 7, 18, 31, 47, 63, 79, 95]
    ctx = SparseForwardAuditContext(
        video_name=video_name,
        window_id=window_id,
        dense_T=dense_T,
        selected_positions=positions,
        audit_mode="module_fake_forward" if module_forward else "fake_raw",
    )
    handoff = build_sparse_raw_handoff(positions, dense_T)
    valid_k = len(positions)
    ctx.record_raw_decode(handoff["raw_frame_inds_in"], handoff["padded_duplicate_count"])
    ctx.record_backbone(
        input_shape_before_preprocess=[1, 1, valid_k, 3, 224, 224],
        input_shape_after_preprocess=[valid_k, 1, 3, 1, 224, 224],
        forward_chunk_count=valid_k,
        dense_backbone_chunk_count=dense_T,
        time_embed_valid_count=valid_k,
        time_embed_total_count=valid_k,
        post_backbone_feature_len=valid_k,
    )
    ctx.record_detector(
        prepad_feature_len=valid_k,
        pad_len=max(int(detector_pad_len), valid_k),
        mask_true_count=valid_k,
        rpn_valid_temporal_len=valid_k,
    )
    ctx.record_temporal_decode(
        uses_original_time=True,
        original_time_metadata=build_original_time_metadata(dense_T, positions, fps=30.0),
    )
    return ctx.build_ledger(sparse_compute_claim=False, module_forward_evidence=module_forward)


def _shape_or_default(shape, default):
    return [int(value) for value in (default if shape is None else shape)]


def _default_provenance(provenance):
    result = {
        "selection_uses_gt": False,
        "selection_uses_teacher": False,
        "selection_uses_prediction_cache": False,
        "selection_uses_raw_detector_prediction": False,
        "selection_uses_oracle_boundary": False,
        "selection_uses_oracle_residual": False,
    }
    result.update(dict(provenance or {}))
    return result
