import numpy as np


def _boundary_distance_regret(pos, gt_segments, dense_T):
    gt = np.asarray(gt_segments, dtype=np.float64).reshape(-1, 2)
    if gt.size == 0:
        return 0.0
    boundaries = np.concatenate([gt[:, 0], gt[:, 1]])
    distance = float(np.min(np.abs(boundaries - float(pos))))
    return float(max(0.0, 1.0 - distance / max(float(dense_T) * 0.12, 1.0)))


def build_regret_labels(probes, gt_segments, dense_T, split):
    split = str(split)
    if split != "train":
        raise ValueError("RBA-RBR regret/value labels are train-only and forbidden on val/test/deploy")
    if gt_segments is None:
        raise ValueError("train regret labels require train gt_segments")
    labels = []
    for probe in probes:
        boundary_regret = _boundary_distance_regret(probe.center_pos, gt_segments, dense_T)
        omission_regret = float(max(boundary_regret, probe.predicted_regret if probe.stage == "rescue" else 0.0))
        labels.append(
            {
                "route_label": probe.route_label,
                "probe_id": int(probe.probe_id),
                "center_pos": int(probe.center_pos),
                "stage": probe.stage,
                "target_boundary_regret": float(boundary_regret),
                "target_omission_regret": float(omission_regret),
                "target_regret": float(omission_regret),
                "train_only": True,
                "split": split,
            }
        )
    return labels
