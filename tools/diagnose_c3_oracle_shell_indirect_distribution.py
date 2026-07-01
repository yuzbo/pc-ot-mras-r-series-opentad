import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.datasets.transforms.end_to_end import LoadFrames
from opentad.datasets.transforms.pseudo_boundary import load_coarse_score_cache, select_coarse_oracle_shell_positions, slice_global_scores_for_window


def _as_segments(segments):
    clean = []
    for row in np.asarray(segments if segments is not None else [], dtype=np.float32):
        if len(row) >= 2 and float(row[1]) > float(row[0]):
            clean.append([float(row[0]), float(row[1])])
    return clean


def _nearest_distances(src, dst):
    src = np.asarray(sorted(set(int(x) for x in src)), dtype=np.int64)
    dst = np.asarray(sorted(set(int(x) for x in dst)), dtype=np.int64)
    if src.size == 0:
        return []
    if dst.size == 0:
        return [None for _ in src]
    return [int(np.min(np.abs(dst - x))) for x in src]


def _max_gap(positions, valid_len):
    positions = sorted(set(int(x) for x in positions if 0 <= int(x) < valid_len))
    if not positions:
        return valid_len
    gaps = [positions[0]]
    gaps.extend(curr - prev for prev, curr in zip(positions, positions[1:]))
    gaps.append(max(0, valid_len - 1 - positions[-1]))
    return int(max(gaps)) if gaps else 0


def _boundary_stats(positions, gt_segments, valid_len, radius):
    positions = sorted(set(int(x) for x in positions if 0 <= int(x) < valid_len))
    boundaries = []
    for start, end in _as_segments(gt_segments):
        boundaries.extend([start, end])
    if not boundaries:
        return 0.0, 0.0
    boundary_hits = sum(1 for b in boundaries if any(abs(float(p) - b) <= radius for p in positions))
    near_selected = sum(1 for p in positions if any(abs(float(p) - b) <= radius for b in boundaries))
    return boundary_hits / float(len(boundaries)), near_selected / float(len(positions) or 1)


def _action_stats(positions, gt_segments, valid_len):
    positions = sorted(set(int(x) for x in positions if 0 <= int(x) < valid_len))
    action_positions = set()
    for start, end in _as_segments(gt_segments):
        first = max(0, int(math.ceil(start)))
        stop = min(valid_len, int(math.ceil(end)))
        action_positions.update(range(first, stop))
    if not action_positions:
        return 0.0, 0.0
    selected_action = sum(1 for p in positions if p in action_positions)
    return selected_action / float(len(positions) or 1), selected_action / float(len(action_positions))


def _histogram_positions(positions, valid_len, bins=8):
    positions = np.asarray([int(x) for x in positions if 0 <= int(x) < valid_len], dtype=np.float32)
    if positions.size == 0 or valid_len <= 0:
        return [0 for _ in range(bins)]
    hist, _ = np.histogram(positions / float(valid_len), bins=bins, range=(0.0, 1.0))
    return hist.astype(int).tolist()


def compare_oracle_and_indirect_window(
    valid_len,
    target_frame_num,
    gt_segments,
    indirect_positions=None,
    action_score=None,
    action_logit=None,
    boundary_radius=2,
    sample_key="oracle-vs-indirect",
):
    valid_len = int(valid_len)
    oracle_loader = LoadFrames(
        method="oracle_action_boundary_subsample",
        target_len=int(target_frame_num),
        scale_factor=1,
        oracle_boundary_radius=int(boundary_radius),
    )
    oracle_positions = oracle_loader._select_oracle_positions(
        valid_len=valid_len,
        gt_segments=np.asarray(gt_segments, dtype=np.float32),
        target_frame_num=int(target_frame_num),
        profile="action_boundary_dense",
    )
    if indirect_positions is None:
        indirect_positions = select_coarse_oracle_shell_positions(
            valid_len=valid_len,
            target_frame_num=target_frame_num,
            action_score=action_score,
            action_logit=action_logit,
            sample_key=sample_key,
        )
    indirect_positions = np.asarray(indirect_positions, dtype=np.int64).copy()
    oracle_set = set(int(x) for x in oracle_positions)
    indirect_set = set(int(x) for x in indirect_positions)
    union = oracle_set | indirect_set
    inter = oracle_set & indirect_set
    oracle_only = sorted(oracle_set - indirect_set)
    indirect_only = sorted(indirect_set - oracle_set)
    oracle_boundary_recall, oracle_boundary_rate = _boundary_stats(oracle_set, gt_segments, valid_len, boundary_radius)
    indirect_boundary_recall, indirect_boundary_rate = _boundary_stats(indirect_set, gt_segments, valid_len, boundary_radius)
    oracle_action_precision, oracle_action_coverage = _action_stats(oracle_set, gt_segments, valid_len)
    indirect_action_precision, indirect_action_coverage = _action_stats(indirect_set, gt_segments, valid_len)
    return {
        "diagnostic_only": True,
        "official_map_claim": False,
        "selection_mutated": False,
        "valid_len": valid_len,
        "target_frame_num": int(target_frame_num),
        "overlap_count": len(inter),
        "jaccard": len(inter) / float(len(union)) if union else 1.0,
        "oracle_boundary_recall": oracle_boundary_recall,
        "indirect_boundary_recall": indirect_boundary_recall,
        "oracle_boundary_near_selected_rate": oracle_boundary_rate,
        "indirect_boundary_near_selected_rate": indirect_boundary_rate,
        "oracle_action_selected_precision": oracle_action_precision,
        "indirect_action_selected_precision": indirect_action_precision,
        "oracle_action_coverage": oracle_action_coverage,
        "indirect_action_coverage": indirect_action_coverage,
        "oracle_max_gap": _max_gap(oracle_set, valid_len),
        "indirect_max_gap": _max_gap(indirect_set, valid_len),
        "oracle_position_histogram_8": _histogram_positions(oracle_set, valid_len),
        "indirect_position_histogram_8": _histogram_positions(indirect_set, valid_len),
        "oracle_only_count": len(oracle_only),
        "indirect_only_count": len(indirect_only),
        "oracle_only_distance_to_indirect": _nearest_distances(oracle_only, indirect_set),
        "indirect_only_distance_to_oracle": _nearest_distances(indirect_only, oracle_set),
    }


def _annotation_gt_segments(video_info):
    gt_segments = []
    for anno in video_info.get("annotations", []):
        if anno.get("label") == "Ambiguous":
            continue
        start = int(float(anno["segment"][0]) / float(video_info["duration"]) * int(video_info["frame"]))
        end = int(float(anno["segment"][1]) / float(video_info["duration"]) * int(video_info["frame"]))
        if end > start:
            gt_segments.append([float(start), float(end)])
    return np.asarray(gt_segments, dtype=np.float32)


def _window_starts(snippet_num, window_size, window_overlap_ratio):
    window_stride = int(window_size * (1.0 - float(window_overlap_ratio)))
    if window_stride <= 0:
        raise ValueError("window_overlap_ratio must leave a positive window stride")
    last_window = False
    starts = []
    for idx in range(max(1, snippet_num // window_stride)):
        window_start = idx * window_stride
        window_end = window_start + window_size
        if window_end > snippet_num:
            window_end = snippet_num
            window_start = max(0, window_end - window_size)
            last_window = True
        starts.append((int(window_start), int(window_end)))
        if last_window:
            break
    return starts


def _local_gt_for_window(gt_segments_frame, window_start_frame, window_end_frame, snippet_stride):
    local = []
    for start, end in np.asarray(gt_segments_frame, dtype=np.float32):
        clipped_start = max(float(start), float(window_start_frame))
        clipped_end = min(float(end), float(window_end_frame))
        if clipped_end > clipped_start:
            local.append(
                [
                    (clipped_start - float(window_start_frame)) / float(snippet_stride),
                    (clipped_end - float(window_start_frame)) / float(snippet_stride),
                ]
            )
    return np.asarray(local, dtype=np.float32)


def iter_annotation_cache_comparisons(
    ann_file,
    cache_dir,
    subset_name,
    target_frame_num=384,
    window_size=768,
    window_overlap_ratio=0.25,
    feature_stride=16,
    sample_stride=1,
    boundary_radius=2,
    limit_windows=None,
):
    database = json.loads(Path(ann_file).read_text(encoding="utf-8"))["database"]
    snippet_stride = int(feature_stride) * int(sample_stride)
    records = []
    for video_name, video_info in database.items():
        if video_info.get("subset") not in subset_name:
            continue
        raw_scores, _manifest = load_coarse_score_cache(cache_dir, video_name)
        num_frames = int(video_info["frame"])
        snippet_centers = np.arange(0, num_frames, snippet_stride)
        gt_segments_frame = _annotation_gt_segments(video_info)
        for window_start, window_end in _window_starts(len(snippet_centers), window_size, window_overlap_ratio):
            window_centers = snippet_centers[window_start:window_end]
            if window_centers.size == 0:
                continue
            global_indices = window_start + np.arange(window_centers.size, dtype=np.int64)
            fields = slice_global_scores_for_window(raw_scores, global_indices)
            window_start_frame = float(window_centers[0])
            window_end_frame = float(window_centers[-1])
            local_gt = _local_gt_for_window(gt_segments_frame, window_start_frame, window_end_frame, snippet_stride)
            row = compare_oracle_and_indirect_window(
                valid_len=int(window_centers.size),
                target_frame_num=target_frame_num,
                gt_segments=local_gt,
                action_score=fields.get("action_score"),
                action_logit=fields.get("action_logit"),
                boundary_radius=boundary_radius,
                sample_key=f"{video_name}|{window_start}|{window_end}",
            )
            row.update(
                {
                    "video_name": video_name,
                    "window_start": int(window_start),
                    "window_end": int(window_end),
                    "subset": video_info.get("subset"),
                }
            )
            records.append(row)
            if limit_windows is not None and len(records) >= int(limit_windows):
                return records
    return records


def _aggregate(records):
    numeric_keys = [
        "jaccard",
        "oracle_boundary_recall",
        "indirect_boundary_recall",
        "oracle_boundary_near_selected_rate",
        "indirect_boundary_near_selected_rate",
        "oracle_action_selected_precision",
        "indirect_action_selected_precision",
        "oracle_action_coverage",
        "indirect_action_coverage",
        "oracle_max_gap",
        "indirect_max_gap",
    ]
    summary = {
        "diagnostic_only": True,
        "official_map_claim": False,
        "windows": len(records),
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in records if key in row and row[key] is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    return summary


def _write_records(records, json_path, csv_path=None, summary_path=None):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_aggregate(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_path is None:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in records for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(row.get(k)) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in keys})


def main():
    parser = argparse.ArgumentParser(description="Diagnostic-only oracle-vs-coarse-indirect selection comparison.")
    parser.add_argument("--records-jsonl", default=None, help="JSONL rows with video_name, valid_len, gt_segments, action_score/logit")
    parser.add_argument("--ann-file", default=None, help="THUMOS annotation JSON for offline GT diagnostics")
    parser.add_argument("--cache-dir", default=None, help="coarse score cache dir with manifest.json")
    parser.add_argument("--subset-name", nargs="+", default=["validation"], help="subset names for annotation/cache diagnostics")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-summary-json", default=None)
    parser.add_argument("--target-frame-num", type=int, default=384)
    parser.add_argument("--window-size", type=int, default=768)
    parser.add_argument("--window-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--feature-stride", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--limit-windows", type=int, default=None)
    args = parser.parse_args()

    records = []
    if args.records_jsonl:
        with Path(args.records_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                records.append(
                    compare_oracle_and_indirect_window(
                        valid_len=row["valid_len"],
                        target_frame_num=args.target_frame_num,
                        gt_segments=row.get("gt_segments", []),
                        action_score=row.get("action_score"),
                        action_logit=row.get("action_logit"),
                        boundary_radius=args.boundary_radius,
                        sample_key=row.get("video_name", "unknown"),
                    )
                )
    elif args.ann_file and args.cache_dir:
        records = iter_annotation_cache_comparisons(
            ann_file=args.ann_file,
            cache_dir=args.cache_dir,
            subset_name=args.subset_name,
            target_frame_num=args.target_frame_num,
            window_size=args.window_size,
            window_overlap_ratio=args.window_overlap_ratio,
            feature_stride=args.feature_stride,
            sample_stride=args.sample_stride,
            boundary_radius=args.boundary_radius,
            limit_windows=args.limit_windows,
        )
    else:
        raise ValueError("Provide either --records-jsonl or both --ann-file and --cache-dir")
    _write_records(records, args.out_json, args.out_csv, args.out_summary_json)


if __name__ == "__main__":
    main()
