from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.acquisition.abr import ABR_ROUTE_LABEL


SCOUT_SOURCE = "deploy_visible_raw_video_graydiff_v1"

FORBIDDEN_OUTPUT_KEYS = {
    "gt_segments",
    "gt_labels",
    "annotations",
    "teacher_logits",
    "teacher_features",
    "teacher_scout_curve",
    "prediction_cache",
    "prediction_cache_path",
    "prediction_scout_curve",
    "raw_predictions",
    "raw_detector_outputs",
    "detector_outputs",
    "detector_predictions",
    "detector_feedback",
    "detector_scout_curve",
    "dense_backbone_features",
    "dense_backbone_handoff",
    "dense_raw_backbone_handoff",
}

FORBIDDEN_OUTPUT_PREFIXES = ("teacher", "prediction", "detector", "dense_backbone")

COMMON_VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".webm", ".mov")


class DeployVisibleScoutExportError(RuntimeError):
    pass


def export_deploy_visible_scout(
    annotation_json: Path | str,
    video_roots: Sequence[Path | str],
    subset: str,
    out_json: Path | str,
    curve_len: int = 384,
    resize: int = 96,
    max_videos: int | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    annotation_path = Path(annotation_json)
    roots = [Path(root) for root in video_roots]
    output_path = Path(out_json)
    curve_len = _positive_int(curve_len, "curve_len")
    resize = _positive_int(resize, "resize")
    if not roots:
        raise DeployVisibleScoutExportError("at least one --video-root is required")

    videos = _load_annotation_videos(annotation_path, subset=subset, max_videos=max_videos)
    payload: dict[str, Any] = {
        "route_label": ABR_ROUTE_LABEL,
        "method": "abr_deploy_visible_scout_export",
        "scout_source": SCOUT_SOURCE,
        "deploy_visible_only": True,
        "no_gt_in_output": True,
        "no_teacher": True,
        "no_prediction_cache": True,
        "no_detector": True,
        "no_dense_backbone": True,
        "subset": subset,
        "curve_len": curve_len,
        "resize": resize,
        "video_roots": [str(root) for root in roots],
        "videos": {},
        "skipped_missing": [],
        "success_count": 0,
        "missing_count": 0,
    }

    for video_id, info in videos.items():
        video_path = _find_video(video_id, roots)
        if video_path is None:
            payload["missing_count"] += 1
            payload["skipped_missing"].append({"video_id": video_id, "reason": "missing video"})
            if not allow_missing:
                _write_json(output_path, payload)
                raise DeployVisibleScoutExportError(f"missing video for {video_id}")
            continue
        curve, capture_meta = _extract_graydiff_curve(video_path, curve_len=curve_len, resize=resize)
        record = {
            "video_id": video_id,
            "deploy_visible_scout_curve": curve,
            "scout_source": SCOUT_SOURCE,
            "duration": info["duration"],
            "total_frames": int(capture_meta.get("total_frames") or info.get("frame") or 0),
            "avg_fps": float(capture_meta.get("avg_fps") or info.get("fps") or _fps_from_metadata(info)),
            "curve_len": len(curve),
            "source_video_name": video_path.name,
        }
        _assert_no_forbidden_output_keys(record)
        payload["videos"][video_id] = record
        payload["success_count"] += 1

    _assert_no_forbidden_output_keys(payload)
    _write_json(output_path, payload)
    if payload["success_count"] <= 0:
        raise DeployVisibleScoutExportError("no videos were successfully exported")
    return payload


def _load_annotation_videos(path: Path, subset: str, max_videos: int | None) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise DeployVisibleScoutExportError(f"missing annotation JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DeployVisibleScoutExportError(f"invalid annotation JSON: {exc}") from exc

    database = payload.get("database", payload) if isinstance(payload, Mapping) else None
    if not isinstance(database, Mapping):
        raise DeployVisibleScoutExportError("annotation JSON must contain a database mapping")

    selected: dict[str, dict[str, Any]] = {}
    for video_id, info in database.items():
        if not isinstance(info, Mapping):
            continue
        if subset and str(info.get("subset", "")).lower() != subset.lower():
            continue
        duration = float(info.get("duration", 0.0))
        frame = int(info.get("frame", info.get("num_frames", info.get("total_frames", 0))))
        fps = float(info.get("fps", info.get("avg_fps", frame / duration if duration > 0.0 and frame > 0 else 0.0)))
        if not math.isfinite(duration) or duration <= 0.0:
            raise DeployVisibleScoutExportError(f"video {video_id} has missing/invalid duration")
        selected[str(video_id)] = {"duration": duration, "frame": frame, "fps": fps}
        if max_videos is not None and len(selected) >= int(max_videos):
            break
    if not selected:
        raise DeployVisibleScoutExportError(f"no videos found for subset={subset!r}")
    return selected


def _find_video(video_id: str, roots: Sequence[Path]) -> Path | None:
    name = str(video_id)
    candidates = [name]
    if Path(name).suffix:
        candidates.append(Path(name).stem)
    for root in roots:
        for candidate in candidates:
            direct = root / candidate
            if direct.is_file():
                return direct
            for suffix in COMMON_VIDEO_SUFFIXES:
                with_suffix = root / f"{candidate}{suffix}"
                if with_suffix.is_file():
                    return with_suffix
    return None


def _extract_graydiff_curve(video_path: Path, curve_len: int, resize: int) -> tuple[list[float], dict[str, Any]]:
    cv2 = _load_cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise DeployVisibleScoutExportError(f"could not open video: {video_path}")
        total_frames = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)))
        if total_frames <= 0:
            raise DeployVisibleScoutExportError(f"video has no readable frames: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_indices = np.linspace(0, max(total_frames - 1, 0), num=curve_len)
        grays: list[np.ndarray] = []
        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(index))))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise DeployVisibleScoutExportError(f"failed reading sampled frame {int(round(float(index)))} from {video_path}")
            grays.append(_low_res_gray(frame, cv2, resize))
    finally:
        cap.release()

    brightness = np.asarray([float(gray.mean()) for gray in grays], dtype=np.float32)
    frame_diff = np.zeros(len(grays), dtype=np.float32)
    for idx in range(1, len(grays)):
        frame_diff[idx] = float(np.mean(np.abs(grays[idx] - grays[idx - 1])))

    curve = 0.35 * _normalize01(brightness) + 0.65 * _normalize01(frame_diff)
    curve = np.clip(curve, 0.0, 1.0)
    return [round(float(value), 8) for value in curve.tolist()], {
        "total_frames": total_frames,
        "avg_fps": fps if math.isfinite(fps) and fps > 0.0 else 0.0,
    }


def _low_res_gray(frame: np.ndarray, cv2: Any, resize: int) -> np.ndarray:
    if resize > 0:
        frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return np.asarray(gray, dtype=np.float32)


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _fps_from_metadata(info: Mapping[str, Any]) -> float:
    duration = float(info.get("duration", 0.0))
    frame = int(info.get("frame", 0))
    if duration > 0.0 and frame > 0:
        return float(frame) / duration
    return 0.0


def _assert_no_forbidden_output_keys(payload: Any, prefix: str = "") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            child = key_text if not prefix else f"{prefix}.{key_text}"
            if key_text in FORBIDDEN_OUTPUT_KEYS or key_text.startswith(FORBIDDEN_OUTPUT_PREFIXES):
                raise DeployVisibleScoutExportError(f"forbidden scout output key: {child}")
            _assert_no_forbidden_output_keys(value, child)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for idx, value in enumerate(payload):
            _assert_no_forbidden_output_keys(value, f"{prefix}[{idx}]")


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise DeployVisibleScoutExportError(f"{name} must be positive")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise DeployVisibleScoutExportError("OpenCV/cv2 is required to export deploy-visible scout curves") from exc
    return cv2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export ABR deploy-visible scout curves from raw low-resolution video pixels."
    )
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--video-root", action="append", required=True)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--curve-len", type=int, default=384)
    parser.add_argument("--resize", type=int, default=96)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = export_deploy_visible_scout(
            annotation_json=Path(args.annotation_json),
            video_roots=[Path(root) for root in args.video_root],
            subset=str(args.subset),
            out_json=Path(args.out_json),
            curve_len=int(args.curve_len),
            resize=int(args.resize),
            max_videos=args.max_videos,
            allow_missing=bool(args.allow_missing),
        )
    except Exception as exc:
        error_payload = {
            "route_label": ABR_ROUTE_LABEL,
            "method": "abr_deploy_visible_scout_export",
            "status": "LOCKED",
            "error": str(exc),
            "scout_source": SCOUT_SOURCE,
            "deploy_visible_only": True,
            "no_gt_in_output": True,
            "no_teacher": True,
            "no_prediction_cache": True,
            "no_detector": True,
            "no_dense_backbone": True,
        }
        if "args" in locals() and getattr(args, "out_json", None):
            _write_json(Path(args.out_json), error_payload)
        print(json.dumps(error_payload, indent=2, sort_keys=True))
        return 1

    payload["status"] = "PASS_DEPLOY_VISIBLE_SCOUT_EXPORT"
    _write_json(Path(args.out_json), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
