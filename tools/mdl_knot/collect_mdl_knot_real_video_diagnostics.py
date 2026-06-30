from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.mdl_knot import (  # noqa: E402
    MDL_KNOT_ROUTE_LABEL,
    MDLKnotConfig,
    apply_mdl_knot_to_dense_window,
    build_frame_metadata_scout_curve,
    build_pipeline_diagnostic,
    build_raw_frame_motion_scout_curve,
    summarize_pipeline_diagnostics,
)
from opentad.acquisition.mdl_knot.diagnostics import validate_shortdiag_train_log_content  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "input_mdl_knot_dynamic_adapter_irregular_headv3.py"


class FixtureVideoReader:
    """Small in-memory reader that exercises the raw-frame LoadFrames branch."""

    def __init__(self, frame_count: int, pattern_id: int) -> None:
        self.frame_count = int(frame_count)
        self.pattern_id = int(pattern_id)

    def __len__(self) -> int:
        return self.frame_count

    def _frame(self, index: int) -> np.ndarray:
        idx = int(index)
        yy, xx = np.mgrid[0:48, 0:64]
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = (idx * (7 + self.pattern_id * 3) + xx) % 255
        frame[:, :, 1] = ((idx // (2 + self.pattern_id)) * 17 + yy * (self.pattern_id + 1)) % 255
        frame[:, :, 2] = ((xx + yy + idx * (3 + self.pattern_id)) // 2) % 255
        if self.pattern_id % 2 == 1:
            start = (idx * 3) % 40
            frame[8:24, start : start + 16, 1] = 255
        return frame

    def __getitem__(self, index: int) -> np.ndarray:
        return self._frame(index)

    def get_batch(self, indices: Iterable[int]) -> np.ndarray:
        return np.stack([self._frame(int(index)) for index in indices], axis=0)


class EquivalentMDLKnotLoadFrames:
    """Torch-free equivalent of the MDL-Knot LoadFrames branch for diagnostics."""

    def __init__(self, **kwargs) -> None:
        self.target_len = int(kwargs.get("target_len", kwargs.get("mdl_knot_max_k", 384)))
        self.mdl_knot_bridge = str(kwargs.get("mdl_knot_bridge", "fixed_pad"))
        self.mdl_knot_deploy_scout_source = str(
            kwargs.get("mdl_knot_deploy_scout_source", "raw_frame_motion_scout_with_metadata_fallback")
        )
        self.mdl_knot_scout_stride = int(kwargs.get("mdl_knot_scout_stride", 8))
        self.mdl_knot_scout_max_frames = int(kwargs.get("mdl_knot_scout_max_frames", 96))
        self.mdl_knot_allow_synthetic_fallback = bool(kwargs.get("mdl_knot_allow_synthetic_fallback", False))
        if self.mdl_knot_bridge != "fixed_pad":
            raise ValueError("MDL-Knot real diagnostic collector requires fixed_pad bridge")
        if self.mdl_knot_allow_synthetic_fallback:
            raise ValueError("MDL-Knot real diagnostic collector requires synthetic fallback disabled")
        self.config = MDLKnotConfig(
            route_label=MDL_KNOT_ROUTE_LABEL,
            min_k=int(kwargs.get("mdl_knot_min_k", 4)),
            max_k=int(kwargs.get("mdl_knot_max_k", self.target_len)),
            target_weighted_error=float(kwargs.get("mdl_knot_target_weighted_error", 0.02)),
            max_gap=int(kwargs.get("mdl_knot_max_gap", 32)),
        )

    @staticmethod
    def _to_numpy_frame(frame) -> np.ndarray:
        if hasattr(frame, "asnumpy"):
            return frame.asnumpy()
        if hasattr(frame, "detach"):
            return frame.detach().cpu().numpy()
        return np.asarray(frame)

    def _read_probe_frames(self, reader, frame_indices: list[int]) -> list[np.ndarray]:
        if hasattr(reader, "get_batch"):
            batch = self._to_numpy_frame(reader.get_batch(frame_indices))
            return [batch[idx] for idx in range(batch.shape[0])]
        return [self._to_numpy_frame(reader[int(index)]) for index in frame_indices]

    def _dense_window(self, results: Mapping[str, object]) -> list[int]:
        start = int(results.get("feature_start_idx", 0))
        end = int(results.get("feature_end_idx", start + int(results.get("window_size", 128)) - 1))
        if end < start:
            end = start
        return list(range(start, end + 1))

    def _metadata_scout(self, results: Mapping[str, object], dense_window: list[int]):
        return build_frame_metadata_scout_curve(
            dense_t=len(dense_window),
            time_index=dense_window,
            total_frames=int(results.get("total_frames", len(dense_window))),
            duration=results.get("duration"),
            fps=results.get("avg_fps", results.get("fps")),
            source="frame_metadata_scout",
            provenance={
                "video_name": results.get("video_name", "unknown"),
                "scout_policy": self.mdl_knot_deploy_scout_source,
            },
        )

    def _raw_scout(self, results: Mapping[str, object], dense_window: list[int]):
        reader = results.get("video_reader") or results.get("decord_reader")
        if reader is None or len(dense_window) < 2:
            return None
        stride = max(self.mdl_knot_scout_stride, 1)
        probe_positions = np.arange(0, len(dense_window), stride, dtype=np.int64)
        if probe_positions[-1] != len(dense_window) - 1:
            probe_positions = np.concatenate([probe_positions, np.asarray([len(dense_window) - 1], dtype=np.int64)])
        max_frames = max(self.mdl_knot_scout_max_frames, 2)
        if probe_positions.size > max_frames:
            probe_positions = np.unique(np.rint(np.linspace(0, len(dense_window) - 1, num=max_frames)).astype(np.int64))
        frame_indices = np.asarray(dense_window, dtype=np.int64)[probe_positions]
        try:
            probe_frames = self._read_probe_frames(reader, frame_indices.astype(int).tolist())
        except Exception:
            return None
        return build_raw_frame_motion_scout_curve(
            probe_frames=probe_frames,
            probe_positions=probe_positions.astype(int).tolist(),
            dense_t=len(dense_window),
            source="raw_frame_motion_scout",
            provenance={
                "video_name": results.get("video_name", "unknown"),
                "scout_policy": self.mdl_knot_deploy_scout_source,
                "raw_probe_stride": stride,
                "raw_probe_max_frames": max_frames,
                "raw_probe_frame_indices": frame_indices.astype(int).tolist(),
            },
        )

    def _scout_curve(self, results: Mapping[str, object], dense_window: list[int]):
        policy = self.mdl_knot_deploy_scout_source
        if policy in ("raw_frame_motion_scout", "raw_frame_motion_scout_with_metadata_fallback"):
            curve = self._raw_scout(results, dense_window)
            if curve is not None:
                return curve
            if policy == "raw_frame_motion_scout_with_metadata_fallback":
                return self._metadata_scout(results, dense_window)
            raise ValueError("raw_frame_motion_scout unavailable and synthetic fallback is disabled")
        if policy == "frame_metadata_scout":
            return self._metadata_scout(results, dense_window)
        raise ValueError(f"unsupported MDL-Knot deploy scout source: {policy}")

    def __call__(self, results: dict) -> dict:
        dense_window = self._dense_window(results)
        scout_curve = self._scout_curve(results, dense_window)
        apply_mdl_knot_to_dense_window(
            results=results,
            dense_window=dense_window,
            scout_curve=scout_curve,
            config=self.config,
            adapter_target_len=self.target_len,
        )
        results["mdl_knot_selector_used_gt"] = False
        results["mdl_knot_route_label"] = MDL_KNOT_ROUTE_LABEL
        results["mdl_knot_deploy_scout_source"] = scout_curve.source
        results["mdl_knot_deploy_scout_provenance"] = dict(scout_curve.provenance)
        results["mdl_knot_pipeline_diagnostic"] = build_pipeline_diagnostic(
            ledger=results["mdl_knot_ledger"],
            sparse_meta=results["mdl_knot_sparse_meta"],
            masks=results["masks"],
            scout_source=scout_curve.source,
            scout_provenance=scout_curve.provenance,
            bridge=self.mdl_knot_bridge,
            adapter_target_len=self.target_len,
        )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MDL-Knot real-video pipeline diagnostics only.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="MDL-Knot config with LoadFrames pipeline.")
    parser.add_argument("--out", required=True, help="Output formal-readiness summary JSON.")
    parser.add_argument("--annotation", default=None, help="Optional THUMOS annotation JSON for real video windows.")
    parser.add_argument("--video-root", action="append", default=[], help="Root directory containing THUMOS mp4 files.")
    parser.add_argument("--split", default=None, help="Optional annotation subset/split filter.")
    parser.add_argument("--window-count", type=int, default=8, help="Maximum windows to inspect.")
    parser.add_argument("--window-size", type=int, default=128, help="Dense frame window size for annotation sampling.")
    parser.add_argument("--shortdiag-log", default=None, help="Validated one-epoch shortdiag train log path.")
    parser.add_argument(
        "--dry-run-fixture",
        action="store_true",
        help="Use deterministic in-memory video readers for local tests; no training/evaluation is run.",
    )
    return parser.parse_args()


def _load_config(config_arg: str) -> tuple[Path, dict]:
    path = Path(config_arg)
    if not path.is_absolute():
        path = ROOT / path
    return path, runpy.run_path(str(path))


def _loadframes_kwargs(cfg: Mapping[str, object]) -> dict:
    dataset = dict(cfg.get("dataset", {}))
    pipeline = list(dict(dataset.get("train", {})).get("pipeline", []))
    load_steps = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    if len(load_steps) != 1:
        raise ValueError(f"expected exactly one train LoadFrames step, got {len(load_steps)}")
    kwargs = dict(load_steps[0])
    kwargs.pop("type", None)
    return kwargs


def _decord_available() -> bool:
    try:
        import decord  # noqa: F401

        return True
    except Exception:
        return False


def _build_loader(cfg: Mapping[str, object]):
    kwargs = _loadframes_kwargs(cfg)
    try:
        from opentad.datasets.transforms.end_to_end import LoadFrames

        return LoadFrames(**kwargs), {
            "loader": "opentad.datasets.transforms.end_to_end.LoadFrames",
            "equivalent_loader_fallback": False,
        }
    except Exception as exc:
        print(f"COLLECTOR_NOTICE: using torch-free equivalent MDL LoadFrames branch because import failed: {exc}")
        return EquivalentMDLKnotLoadFrames(**kwargs), {
            "loader": "EquivalentMDLKnotLoadFrames",
            "equivalent_loader_fallback": True,
            "opentad_loadframes_import_error": str(exc),
        }


def _fixture_windows(window_count: int) -> Iterator[dict]:
    count = max(int(window_count), 1)
    sizes = [96, 128, 160, 192, 112, 144]
    for idx in range(count):
        size = sizes[idx % len(sizes)]
        use_reader = idx % 3 != 2
        results = {
            "video_name": f"mdl_knot_realdiag_fixture_{idx:03d}",
            "total_frames": size,
            "avg_fps": 30.0,
            "duration": size / 30.0,
            "snippet_stride": 1,
            "window_size": size,
            "feature_start_idx": 0,
            "feature_end_idx": size - 1,
            "window_id": idx,
        }
        if use_reader:
            results["video_reader"] = FixtureVideoReader(size, pattern_id=idx + 1)
        yield results


def _annotation_items(annotation: Mapping[str, object]) -> Iterator[tuple[str, Mapping[str, object]]]:
    database = annotation.get("database") if isinstance(annotation.get("database"), Mapping) else annotation
    for key, value in dict(database).items():
        if isinstance(value, Mapping):
            yield str(value.get("video_name", value.get("id", key))), value


def _find_video(video_id: str, video_roots: Iterable[str]) -> Path | None:
    suffixes = (".mp4", ".avi", ".mkv")
    for root_arg in video_roots:
        root = Path(root_arg)
        if not root.exists():
            continue
        for suffix in suffixes:
            for candidate in (root / f"{video_id}{suffix}", root / video_id / f"{video_id}{suffix}"):
                if candidate.exists():
                    return candidate
    return None


def _open_decord_reader(path: Path):
    try:
        from decord import VideoReader
    except Exception:
        return None
    try:
        return VideoReader(str(path))
    except Exception:
        return None


def _annotation_windows(args: argparse.Namespace) -> Iterator[dict]:
    if not args.annotation:
        raise ValueError("--annotation is required unless --dry-run-fixture is set")
    ann_path = Path(args.annotation)
    if not ann_path.is_absolute():
        ann_path = ROOT / ann_path
    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    emitted = 0
    for video_id, item in _annotation_items(annotation):
        subset = str(item.get("subset", item.get("split", "")))
        if args.split and subset and subset.lower() != str(args.split).lower():
            continue
        total_frames = int(item.get("total_frames", item.get("num_frames", item.get("frame_count", 0))) or 0)
        fps = float(item.get("fps", item.get("avg_fps", 30.0)) or 30.0)
        duration = float(item.get("duration", 0.0) or 0.0)
        if total_frames <= 0 and duration > 0:
            total_frames = max(int(round(duration * fps)), 2)
        total_frames = max(total_frames, int(args.window_size), 2)
        window_size = min(max(int(args.window_size), 2), total_frames)
        video_path = _find_video(video_id, args.video_root)
        reader = _open_decord_reader(video_path) if video_path is not None else None
        starts = np.linspace(0, max(total_frames - window_size, 0), num=2, dtype=np.int64)
        for start in starts.tolist():
            end = int(start) + window_size - 1
            results = {
                "video_name": video_id,
                "total_frames": total_frames,
                "avg_fps": fps,
                "duration": duration if duration > 0 else total_frames / max(fps, 1.0),
                "snippet_stride": 1,
                "window_size": window_size,
                "feature_start_idx": int(start),
                "feature_end_idx": int(end),
                "window_id": emitted,
            }
            if reader is not None:
                results["video_reader"] = reader
            yield results
            emitted += 1
            if emitted >= int(args.window_count):
                return


def _shortdiag_evidence(shortdiag_log: str | None, out_path: Path) -> dict:
    if not shortdiag_log:
        return {
            "validated": False,
            "formal_train_unlocked": False,
            "no_sparse_compute_claim": True,
            "evidence_scope": "missing_shortdiag_log",
            "log_evidence": None,
        }
    log_evidence = validate_shortdiag_train_log_content(shortdiag_log, evidence_roots=[out_path.parent, ROOT])
    return {
        "validated": True,
        "formal_train_unlocked": False,
        "no_sparse_compute_claim": True,
        "metric_claim": False,
        "sparse_compute_claim": False,
        "runtime_claim": False,
        "deploy_claim": False,
        "paper_claim": False,
        "evidence_scope": "one_epoch_train_log",
        "log_evidence": log_evidence,
    }


def _collect_diagnostics(loader, windows: Iterable[dict]) -> list[dict]:
    diagnostics = []
    for results in windows:
        out = loader(dict(results))
        diagnostic = dict(out.get("mdl_knot_pipeline_diagnostic", {}))
        if not diagnostic:
            raise ValueError(f"LoadFrames produced no MDL-Knot diagnostic for {results.get('video_name')}")
        diagnostics.append(diagnostic)
    return diagnostics


def _formal_summary(
    *,
    config_path: Path,
    diagnostics: list[dict],
    shortdiag: dict,
    args: argparse.Namespace,
    reader_backend: Mapping[str, object],
) -> dict:
    real_diag = summarize_pipeline_diagnostics(diagnostics)
    dry_run = bool(args.dry_run_fixture)
    annotation_path = None
    if args.annotation:
        annotation_path = str(Path(args.annotation).resolve())
    raw_count = int(real_diag.get("raw_frame_scout_count", real_diag.get("raw_frame_scout_windows", 0)))
    fixture_count = int(real_diag.get("window_count", 0)) if dry_run else 0
    real_reader_count = 0 if dry_run else raw_count
    return {
        "route_label": MDL_KNOT_ROUTE_LABEL,
        "validated": True,
        "validation_scope": "real_video_pipeline_diagnostics_for_formal_readiness_packet",
        "config_path": str(config_path),
        "source_mode": "fixture_schema_only" if dry_run else "real_video_pipeline",
        "dry_run_fixture": dry_run,
        "annotation_path": annotation_path,
        "video_root": [str(Path(root).resolve()) for root in args.video_root],
        "reader_backend": {
            "mode": "fixture" if dry_run else "annotation_video_root",
            "decord_available": _decord_available(),
            "real_video_reader_required_for_formal_readiness": True,
            **dict(reader_backend),
        },
        "real_video_reader_window_count": real_reader_count,
        "real_video_raw_scout_count": real_reader_count,
        "fixture_window_count": fixture_count,
        "synthetic": False,
        "formal_train_unlocked": False,
        "full_train_unlocked": False,
        "no_mAP": True,
        "no_tools_test": True,
        "no_train": True,
        "no_eval": True,
        "locked_actions": {
            "remote_sync": True,
            "slurm": True,
            "training": True,
            "evaluation": True,
            "tools_test_py": True,
        },
        "no_claims": {
            "mAP": True,
            "runtime": True,
            "FLOPs": True,
            "deploy": True,
            "paper": True,
            "sparse_compute": True,
        },
        "real_video_pipeline_diagnostics": real_diag,
        "shortdiag_evidence": shortdiag,
    }


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config_path, cfg = _load_config(args.config)
    if cfg.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        print(f"LOCKED: route label mismatch in config: {cfg.get('route_label')}")
        return 2
    loader, reader_backend = _build_loader(cfg)
    windows = _fixture_windows(args.window_count) if args.dry_run_fixture else _annotation_windows(args)
    diagnostics = _collect_diagnostics(loader, windows)
    if not diagnostics:
        print("LOCKED: no diagnostic windows collected")
        return 2
    summary = _formal_summary(
        config_path=config_path,
        diagnostics=diagnostics,
        shortdiag=_shortdiag_evidence(args.shortdiag_log, out_path),
        args=args,
        reader_backend=reader_backend,
    )
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"MDL_KNOT_REAL_VIDEO_DIAGNOSTIC_SUMMARY={out_path}")
    print("No training, no evaluation, no tools/test.py, no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
