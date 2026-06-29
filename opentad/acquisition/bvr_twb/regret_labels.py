import numpy as np

from .types import ROUTE_LABEL
from .validators import validate_no_leakage, validate_route_identity


REGRET_COMPONENT_KEYS = (
    "boundary_coverage_loss",
    "short_action_miss_risk",
    "bracket_width_uncertainty",
    "gap_risk",
    "entropy_reduction",
    "width_reduction",
    "boundary_risk_reduction",
    "omission_regret",
)


def _as_segments(gt_segments):
    if gt_segments is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(gt_segments, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    arr = arr.reshape(-1, 2)
    return arr[np.asarray(arr[:, 1] > arr[:, 0]).reshape(-1)]


def _packet_positions(packet):
    return np.asarray(getattr(packet, "positions", []), dtype=np.float32).reshape(-1)


def _distance_score(positions, targets, radius):
    if positions.size == 0 or targets.size == 0:
        return 0.0
    distances = np.abs(positions[:, None] - targets[None, :])
    best = float(np.min(distances))
    return float(np.clip(1.0 - best / float(max(radius, 1e-6)), 0.0, 1.0))


def _short_action_score(positions, gt_segments, max_short_len, radius):
    if positions.size == 0 or gt_segments.shape[0] == 0:
        return 0.0
    lengths = gt_segments[:, 1] - gt_segments[:, 0]
    short = gt_segments[lengths <= float(max_short_len)]
    if short.shape[0] == 0:
        return 0.0
    centers = 0.5 * (short[:, 0] + short[:, 1])
    return _distance_score(positions, centers, radius)


def validate_regret_label_context(split, metadata=None):
    split = str(split)
    metadata = {} if metadata is None else dict(metadata)
    validate_route_identity({"route_label": metadata.get("route_label", ROUTE_LABEL)})
    validate_no_leakage(metadata.get("selector_metadata", {}))
    if split != "train":
        for key in ("gt_segments", "gt_labels", "teacher_logits", "teacher_scores", "prediction_cache"):
            value = metadata.get(key)
            if value is not None and value is not False:
                if isinstance(value, (list, tuple, dict)) and len(value) == 0:
                    continue
                raise ValueError(f"BVR-TWB regret labels are train-only; forbidden {key} present for split={split}")
    return True


def build_packet_regret_labels(
    packets,
    gt_segments,
    dense_T,
    split,
    gt_labels=None,
    boundary_radius=4.0,
    short_action_frames=12.0,
    metadata=None,
):
    """Build train-only geometry proxy labels for packet omission regret.

    The labels are supervision targets only. They are forbidden for val/test/deploy
    selector decisions and are not used by the fallback heuristic at test time.
    """

    dense_T = int(dense_T)
    if split != "train":
        validate_regret_label_context(
            split,
            {
                "route_label": ROUTE_LABEL,
                "selector_metadata": {},
                "gt_segments": gt_segments,
                "gt_labels": gt_labels,
            },
        )
        raise ValueError("BVR-TWB regret labels may only be built for split='train'")

    validate_regret_label_context(split, {"route_label": ROUTE_LABEL, "selector_metadata": metadata or {}})
    segments = _as_segments(gt_segments)
    boundaries = np.concatenate([segments[:, 0], segments[:, 1]]) if segments.shape[0] else np.zeros((0,), dtype=np.float32)
    labels = []
    for packet in packets:
        positions = _packet_positions(packet)
        features = getattr(packet, "feature_summary", {}) or {}
        boundary_loss = _distance_score(positions, boundaries, boundary_radius)
        short_risk = _short_action_score(positions, segments, short_action_frames, boundary_radius)
        width_uncertainty = float(
            np.clip(
                0.5 * float(features.get("mean_uncertainty", 0.0))
                + 0.5 * float(features.get("bracket_width_frames", 0.0)) / float(max(dense_T, 1)),
                0.0,
                1.0,
            )
        )
        gap_risk = float(
            np.clip(
                max(
                    float(features.get("gap_risk", 0.0)),
                    float(features.get("gap_if_omitted_frames", 0.0)) / float(max(dense_T, 1)),
                ),
                0.0,
                1.0,
            )
        )
        entropy_reduction = float(
            np.clip(
                0.45 * float(features.get("belief_entropy", features.get("mean_uncertainty", 0.0)))
                + 0.25 * boundary_loss
                + 0.15 * float(features.get("two_sided_state_contrast", 0.0))
                + 0.15 * short_risk,
                0.0,
                1.0,
            )
        )
        width_reduction = float(
            np.clip(
                0.55 * float(features.get("bracket_width_frames", 0.0)) / float(max(dense_T, 1))
                + 0.30 * boundary_loss
                + 0.15 * float(features.get("mean_uncertainty", 0.0)),
                0.0,
                1.0,
            )
        )
        boundary_risk_reduction = float(
            np.clip(0.50 * boundary_loss + 0.20 * width_reduction + 0.15 * entropy_reduction + 0.15 * gap_risk, 0.0, 1.0)
        )
        regret = float(
            np.clip(
                0.34 * boundary_loss
                + 0.20 * short_risk
                + 0.16 * width_uncertainty
                + 0.12 * gap_risk
                + 0.10 * entropy_reduction
                + 0.08 * boundary_risk_reduction,
                0.0,
                1.0,
            )
        )
        labels.append(
            {
                "route_label": ROUTE_LABEL,
                "packet_id": int(packet.packet_id),
                "packet_role": packet.role,
                "packet_positions": [int(pos) for pos in packet.positions],
                "target_regret": regret,
                "target_omission_regret": regret,
                "target_entropy_reduction": entropy_reduction,
                "target_width_reduction": width_reduction,
                "target_boundary_risk_reduction": boundary_risk_reduction,
                "regret_components": {
                    "boundary_coverage_loss": float(boundary_loss),
                    "short_action_miss_risk": float(short_risk),
                    "bracket_width_uncertainty": float(width_uncertainty),
                    "gap_risk": float(gap_risk),
                    "entropy_reduction": float(entropy_reduction),
                    "width_reduction": float(width_reduction),
                    "boundary_risk_reduction": float(boundary_risk_reduction),
                    "omission_regret": float(regret),
                },
                "voi_targets": {
                    "entropy_reduction": float(entropy_reduction),
                    "width_reduction": float(width_reduction),
                    "boundary_risk_reduction": float(boundary_risk_reduction),
                    "omission_regret": float(regret),
                },
                "training_only": True,
                "diagnostic_only": False,
                "forbidden_splits": ["val", "test", "deploy"],
                "uses_gt": True,
                "uses_teacher": False,
                "uses_prediction_cache": False,
                "label_source": "train_only_gt_geometry_proxy",
            }
        )
    return labels


def validate_regret_label_schema(row):
    required = {
        "route_label",
        "packet_id",
        "packet_role",
        "packet_positions",
        "target_regret",
        "target_omission_regret",
        "target_entropy_reduction",
        "target_width_reduction",
        "target_boundary_risk_reduction",
        "regret_components",
        "voi_targets",
        "training_only",
        "forbidden_splits",
        "uses_gt",
        "uses_teacher",
        "uses_prediction_cache",
        "label_source",
    }
    missing = sorted(required.difference(row.keys()))
    if missing:
        raise ValueError(f"regret label missing fields: {missing}")
    validate_route_identity(row)
    if row["training_only"] is not True or row["uses_gt"] is not True:
        raise ValueError("BVR-TWB regret labels must be explicit train-only GT proxy labels")
    if row["uses_teacher"] or row["uses_prediction_cache"]:
        raise ValueError("BVR-TWB regret labels must not use teacher/cache shortcuts")
    if not {"val", "test", "deploy"}.issubset(set(row["forbidden_splits"])):
        raise ValueError("BVR-TWB regret labels must forbid val/test/deploy")
    components = row["regret_components"]
    missing_components = sorted(set(REGRET_COMPONENT_KEYS).difference(components.keys()))
    if missing_components:
        raise ValueError(f"regret label components missing fields: {missing_components}")
    target = float(row["target_regret"])
    if not np.isfinite(target) or target < 0.0 or target > 1.0:
        raise ValueError("target_regret must be finite in [0, 1]")
    for key in ("target_omission_regret", "target_entropy_reduction", "target_width_reduction", "target_boundary_risk_reduction"):
        value = float(row[key])
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{key} must be finite in [0, 1]")
    voi_targets = row["voi_targets"]
    for key in ("entropy_reduction", "width_reduction", "boundary_risk_reduction", "omission_regret"):
        if key not in voi_targets:
            raise ValueError(f"VOI target missing field: {key}")
    return True
