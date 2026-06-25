from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class FrameTokenHybridPreviewProbe:
    """Write deploy-visible preview metadata for the Frame/Token Hybrid route.

    This hook runs after ``LoadFrames`` and before decode/augmentation consumers
    need ``metas``. It only uses planned frame indices and prefix masks, so it
    is a low-cost controller input rather than a GT/teacher/cache/detector
    shortcut. It does not prove raw decode saving by itself; it only closes the
    normal metadata path required by the selector.
    """

    def __init__(
        self,
        signal_meta_key: str = "frame_token_hybrid_preview_signal",
        positions_meta_key: str = "frame_token_hybrid_preview_positions",
        source_meta_key: str = "frame_token_hybrid_preview_source",
    ) -> None:
        self.signal_meta_key = str(signal_meta_key)
        self.positions_meta_key = str(positions_meta_key)
        self.source_meta_key = str(source_meta_key)

    def __call__(self, results):
        if "frame_inds" not in results:
            raise ValueError("FrameTokenHybridPreviewProbe requires frame_inds from LoadFrames")
        frame_inds = np.asarray(results["frame_inds"], dtype=np.float32)
        if frame_inds.ndim == 0:
            raise ValueError("FrameTokenHybridPreviewProbe received scalar frame_inds")
        if frame_inds.ndim == 1:
            centers = frame_inds
        else:
            centers = frame_inds.reshape(frame_inds.shape[0], -1).mean(axis=1)
        valid_len = self._valid_len(results.get("masks"), len(centers))
        if valid_len <= 0:
            raise ValueError("FrameTokenHybridPreviewProbe requires at least one valid position")

        valid_centers = centers[:valid_len]
        signal = self._temporal_gap_signal(valid_centers)
        results[self.signal_meta_key] = [float(value) for value in signal]
        results[self.positions_meta_key] = [int(idx) for idx in range(valid_len)]
        results[self.source_meta_key] = "frame_inds_temporal_gap_preview_probe"
        return results

    @staticmethod
    def _valid_len(masks: Any, fallback_len: int) -> int:
        if masks is None:
            return int(fallback_len)
        if hasattr(masks, "detach"):
            values = masks.detach().cpu().numpy()
        else:
            values = np.asarray(masks)
        values = values.astype(bool).reshape(-1)
        if values.size == 0:
            return 0
        return int(values.sum())

    @staticmethod
    def _temporal_gap_signal(centers: Sequence[float]) -> np.ndarray:
        centers = np.asarray(centers, dtype=np.float32).reshape(-1)
        if centers.size == 1:
            return np.zeros((1,), dtype=np.float32)
        gaps = np.abs(np.diff(centers))
        reference_gap = float(np.median(gaps)) if gaps.size else 0.0
        denom = max(reference_gap, 1.0)
        signal = np.zeros((centers.size,), dtype=np.float32)
        signal[1:] = np.abs(gaps - reference_gap) / denom
        return signal
