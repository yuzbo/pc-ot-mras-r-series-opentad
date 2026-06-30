import torch
from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck
from .base import BaseDetector
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """
    Base class for single-stage detectors which should not have roi_extractors.
    """

    def __init__(self, backbone=None, projection=None, neck=None, rpn_head=None):
        super(SingleStageDetector, self).__init__()

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_kwargs = dict(kwargs)
            rpn_kwargs.setdefault("metas", metas)
            rpn_losses = self.rpn_head.forward_train(
                x,
                masks,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **rpn_kwargs,
            )
            losses.update(rpn_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            predictions = self.rpn_head.forward_test(x, masks, metas=metas)
        else:
            rpn_proposals = rpn_scores = None
            predictions = rpn_proposals, rpn_scores

        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        if len(predictions) == 3:
            rpn_proposals, rpn_scores, rpn_diagnostics = predictions
        else:
            rpn_proposals, rpn_scores = predictions
            rpn_diagnostics = None
        # rpn_proposals,  # [B,K,2]
        # rpn_scores,  # [B,K,num_classes] after sigmoid

        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = rpn_proposals[i].detach().cpu()  # [N,2]
            scores = rpn_scores[i].detach().cpu()  # [N,class]
            diagnostics = None if rpn_diagnostics is None else rpn_diagnostics[i]

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
                diagnostic_records = self._build_qc_v2_diagnostic_records(
                    diagnostics,
                    torch.arange(segments.shape[0]),
                    labels,
                )
                self._mark_qc_v2_pre_nms_state(
                    diagnostic_records,
                    scores,
                    rank_semantics="traversal_proposal_order_single_class_no_topk",
                )
            else:
                pred_prob = scores.flatten()  # [N*class]

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # 3. gather predicted proposals
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)

                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs
                diagnostic_records = self._build_qc_v2_diagnostic_records(diagnostics, pt_idxs, cls_idxs)
                self._mark_qc_v2_pre_nms_state(
                    diagnostic_records,
                    scores,
                    rank_semantics="score_sorted_after_threshold_topk",
                )

            # if not sliding window, do nms
            pre_nms_segments_for_diagnostic = segments.clone()
            pre_nms_candidate_records = diagnostic_records
            if post_cfg.sliding_window == False and post_cfg.nms is not None:
                pre_nms_segments = segments.clone()
                pre_nms_labels = labels.clone()
                pre_nms_diagnostic_records = diagnostic_records
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
                diagnostic_records = self._match_qc_v2_diagnostics_after_nms(
                    pre_nms_segments,
                    pre_nms_labels,
                    pre_nms_diagnostic_records,
                    segments,
                    labels,
                    scores,
                )
            else:
                self._mark_qc_v2_post_nms_state_without_nms(diagnostic_records, scores)

            video_id = metas[i]["video_name"]
            pre_nms_physical_segments_for_diagnostic = convert_to_seconds(
                pre_nms_segments_for_diagnostic.clone(),
                metas[i],
            )
            pre_nms_candidates = self._serialize_qc_v2_pre_nms_candidates(
                pre_nms_candidate_records,
                pre_nms_physical_segments_for_diagnostic,
                ext_cls,
            )

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for det_idx, (segment, label, score) in enumerate(zip(segments, labels, scores)):
                # convert to python scalars
                record = dict(
                    segment=[round(seg.item(), 2) for seg in segment],
                    label=label,
                    score=round(score.item(), 4),
                )
                if diagnostic_records is not None and det_idx < len(diagnostic_records):
                    self._attach_qc_v2_diagnostic_record(record, diagnostic_records[det_idx], segment)
                results_per_video.append(record)
            if pre_nms_candidates and results_per_video:
                results_per_video[0]["qc_v2_pre_nms_candidates"] = pre_nms_candidates

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

    @staticmethod
    def _tensor_row_to_list(tensor):
        return [round(value.item(), 4) for value in tensor]

    @staticmethod
    def _tensor_scalar_or_none(tensor):
        if tensor is None:
            return None
        return round(tensor.item(), 4)

    @staticmethod
    def _build_qc_v2_diagnostic_records(diagnostics, point_indices, class_indices):
        if diagnostics is None or not diagnostics.get("diagnostic_available", False):
            return None

        records = []
        quality_scores = diagnostics.get("quality_scores", None)
        fused_scores = diagnostics.get("fused_scores", None)
        proposal_widths = diagnostics.get("proposal_widths", None)
        level_ids = diagnostics.get("level_ids", None)
        diagnostic_point_indices = diagnostics.get("point_indices", None)
        for point_idx, class_idx in zip(point_indices, class_indices):
            point_idx = int(point_idx.item())
            class_idx = int(class_idx.item())
            cls_scores = diagnostics["cls_scores"][point_idx]
            record = {
                "coverage_available": bool(diagnostics.get("coverage_available", False)),
                "class_index": class_idx,
                "cls_score": cls_scores[class_idx],
                "fused_score": None if fused_scores is None else fused_scores[point_idx][class_idx],
                "quality_score": None if quality_scores is None else quality_scores[point_idx],
                "selected_segment": diagnostics["selected_segments"][point_idx],
                "selected_length": diagnostics["selected_lengths"][point_idx],
                "physical_length": diagnostics["physical_lengths"][point_idx],
                "proposal_width": None if proposal_widths is None else proposal_widths[point_idx],
                "gap_mean": diagnostics["gap_mean"][point_idx],
                "visibility_support": diagnostics["visibility_support"][point_idx],
                "coverage": diagnostics["coverage"][point_idx],
                "endpoint_support": diagnostics["endpoint_support"][point_idx],
                "level_id": None if level_ids is None else level_ids[point_idx],
                "point_index": None if diagnostic_point_indices is None else diagnostic_point_indices[point_idx],
            }
            records.append(record)
        return records

    @staticmethod
    def _mark_qc_v2_pre_nms_state(records, scores, rank_semantics):
        if records is None:
            return
        for rank, (record, score) in enumerate(zip(records, scores), start=1):
            record["pre_nms_rank"] = rank
            record["pre_nms_score"] = score
            record["rank_semantics"] = rank_semantics
            record["survived_after_topk"] = True
            record["post_topk_rank"] = rank
            record["post_topk_score"] = score
            record["survived_after_nms"] = False
            record["post_nms_rank"] = None
            record["post_nms_score"] = None

    @staticmethod
    def _mark_qc_v2_post_nms_state_without_nms(records, scores):
        if records is None:
            return
        for rank, (record, score) in enumerate(zip(records, scores), start=1):
            record["survived_after_nms"] = True
            record["post_nms_rank"] = rank
            record["post_nms_score"] = score

    @staticmethod
    def _match_qc_v2_diagnostics_after_nms(old_segments, old_labels, old_records, new_segments, new_labels, new_scores=None):
        if old_records is None:
            return None
        matched = []
        used = set()
        for post_rank, (segment, label) in enumerate(zip(new_segments, new_labels), start=1):
            found = None
            for idx, (old_segment, old_label) in enumerate(zip(old_segments, old_labels)):
                if idx in used or int(old_label.item()) != int(label.item()):
                    continue
                if torch.allclose(old_segment, segment, atol=1e-4, rtol=0.0):
                    found = old_records[idx]
                    found["survived_after_nms"] = True
                    found["post_nms_rank"] = post_rank
                    found["post_nms_score"] = None if new_scores is None else new_scores[post_rank - 1]
                    used.add(idx)
                    break
            matched.append(found or {"coverage_available": False})
        return matched

    def _serialize_qc_v2_pre_nms_candidates(self, diagnostic_records, physical_segments, ext_cls):
        if diagnostic_records is None:
            return []

        candidates = []
        for diagnostic_record, physical_segment in zip(diagnostic_records, physical_segments):
            if not diagnostic_record:
                continue
            class_index = diagnostic_record.get("class_index")
            if isinstance(ext_cls, list) and class_index is not None:
                label = ext_cls[int(class_index)]
            else:
                label = None if class_index is None else int(class_index)
            candidates.append(
                {
                    "label": label,
                    "class_index": None if class_index is None else int(class_index),
                    "cls_score": self._tensor_scalar_or_none(diagnostic_record.get("cls_score")),
                    "fused_score": self._tensor_scalar_or_none(diagnostic_record.get("fused_score")),
                    "quality_score": self._tensor_scalar_or_none(diagnostic_record.get("quality_score")),
                    "score": self._tensor_scalar_or_none(diagnostic_record.get("pre_nms_score")),
                    "pre_nms_score": self._tensor_scalar_or_none(diagnostic_record.get("pre_nms_score")),
                    "pre_nms_rank": diagnostic_record.get("pre_nms_rank"),
                    "rank_semantics": diagnostic_record.get("rank_semantics"),
                    "survived_after_topk": bool(diagnostic_record.get("survived_after_topk", False)),
                    "post_topk_rank": diagnostic_record.get("post_topk_rank"),
                    "post_topk_score": self._tensor_scalar_or_none(diagnostic_record.get("post_topk_score")),
                    "survived_after_nms": bool(diagnostic_record.get("survived_after_nms", False)),
                    "post_nms_rank": diagnostic_record.get("post_nms_rank"),
                    "post_nms_score": self._tensor_scalar_or_none(diagnostic_record.get("post_nms_score")),
                    "selected_segment": self._tensor_row_to_list(diagnostic_record["selected_segment"]),
                    "physical_segment": self._tensor_row_to_list(physical_segment),
                    "selected_length": self._tensor_scalar_or_none(diagnostic_record.get("selected_length")),
                    "physical_length": round((physical_segment[1] - physical_segment[0]).item(), 4),
                    "proposal_width": self._tensor_scalar_or_none(diagnostic_record.get("proposal_width")),
                    "gap_mean": self._tensor_scalar_or_none(diagnostic_record.get("gap_mean")),
                    "visibility_support": self._tensor_scalar_or_none(diagnostic_record.get("visibility_support")),
                    "coverage": self._tensor_scalar_or_none(diagnostic_record.get("coverage")),
                    "endpoint_support": self._tensor_scalar_or_none(diagnostic_record.get("endpoint_support")),
                    "level_id": None
                    if diagnostic_record.get("level_id") is None
                    else int(diagnostic_record["level_id"].item()),
                    "point_index": None
                    if diagnostic_record.get("point_index") is None
                    else int(diagnostic_record["point_index"].item()),
                    "coverage_available": bool(diagnostic_record.get("coverage_available", False)),
                }
            )
        return candidates

    def _attach_qc_v2_diagnostic_record(self, output_record, diagnostic_record, physical_segment):
        if not diagnostic_record:
            output_record["coverage_available"] = False
            return

        output_record.update(
            {
                "cls_score": self._tensor_scalar_or_none(diagnostic_record.get("cls_score")),
                "fused_score": self._tensor_scalar_or_none(diagnostic_record.get("fused_score")),
                "quality_score": self._tensor_scalar_or_none(diagnostic_record.get("quality_score")),
                "selected_segment": self._tensor_row_to_list(diagnostic_record["selected_segment"]),
                "physical_segment": self._tensor_row_to_list(physical_segment),
                "selected_length": self._tensor_scalar_or_none(diagnostic_record["selected_length"]),
                "physical_length": round((physical_segment[1] - physical_segment[0]).item(), 4),
                "proposal_width": self._tensor_scalar_or_none(diagnostic_record.get("proposal_width")),
                "gap_mean": self._tensor_scalar_or_none(diagnostic_record.get("gap_mean")),
                "visibility_support": self._tensor_scalar_or_none(diagnostic_record.get("visibility_support")),
                "coverage": self._tensor_scalar_or_none(diagnostic_record.get("coverage")),
                "endpoint_support": self._tensor_scalar_or_none(diagnostic_record.get("endpoint_support")),
                "pre_nms_rank": diagnostic_record.get("pre_nms_rank"),
                "pre_nms_score": self._tensor_scalar_or_none(diagnostic_record.get("pre_nms_score")),
                "rank_semantics": diagnostic_record.get("rank_semantics"),
                "survived_after_topk": bool(diagnostic_record.get("survived_after_topk", False)),
                "post_topk_rank": diagnostic_record.get("post_topk_rank"),
                "post_topk_score": self._tensor_scalar_or_none(diagnostic_record.get("post_topk_score")),
                "survived_after_nms": bool(diagnostic_record.get("survived_after_nms", False)),
                "post_nms_rank": diagnostic_record.get("post_nms_rank"),
                "post_nms_score": self._tensor_scalar_or_none(diagnostic_record.get("post_nms_score")),
                "level_id": None
                if diagnostic_record.get("level_id") is None
                else int(diagnostic_record["level_id"].item()),
                "point_index": None
                if diagnostic_record.get("point_index") is None
                else int(diagnostic_record["point_index"].item()),
                "coverage_available": bool(diagnostic_record.get("coverage_available", False)),
            }
        )
