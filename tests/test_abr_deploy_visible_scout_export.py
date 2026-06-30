import json

import numpy as np
import pytest

from opentad.acquisition.abr import ABRConfig
from tools.abr.audit_abr_first_round_bracket_recall import run_audit

from tools.abr.export_abr_deploy_visible_scout import (
    FORBIDDEN_OUTPUT_KEYS,
    DeployVisibleScoutExportError,
    export_deploy_visible_scout,
    main,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _annotation_payload(video_id="video_0001", subset="validation", annotations=None):
    return {
        "database": {
            video_id: {
                "subset": subset,
                "duration": 12.0,
                "frame": 12,
                "fps": 1.0,
                "annotations": annotations
                if annotations is not None
                else [{"segment": [2.0, 4.0], "label": "BaseballPitch"}],
            }
        }
    }


class FakeCapture:
    def __init__(self, path):
        self.path = str(path)
        self.frames = FAKE_VIDEOS.get(self.path)
        self.index = 0

    def isOpened(self):
        return self.frames is not None

    def get(self, prop):
        if prop == FakeCV2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames or []))
        if prop == FakeCV2.CAP_PROP_FPS:
            return 1.0
        return 0.0

    def set(self, prop, value):
        if prop == FakeCV2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
        return True

    def read(self):
        if self.frames is None or self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        return True, frame.copy()

    def release(self):
        return None


class FakeCV2:
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FPS = 5
    CAP_PROP_POS_FRAMES = 1
    COLOR_BGR2GRAY = 6
    INTER_AREA = 3

    @staticmethod
    def VideoCapture(path):
        return FakeCapture(path)

    @staticmethod
    def resize(frame, size, interpolation=None):
        del interpolation
        width, height = size
        return np.resize(frame, (height, width, frame.shape[2])).astype(frame.dtype)

    @staticmethod
    def cvtColor(frame, code):
        assert code == FakeCV2.COLOR_BGR2GRAY
        return frame.mean(axis=2).astype(np.float32)


FAKE_VIDEOS = {}


@pytest.fixture(autouse=True)
def fake_cv2(monkeypatch):
    import tools.abr.export_abr_deploy_visible_scout as exporter

    FAKE_VIDEOS.clear()
    monkeypatch.setattr(exporter, "_load_cv2", lambda: FakeCV2)


def _make_fake_video(path):
    path.write_bytes(b"fake video placeholder")
    intensities = [0, 0, 25, 25, 220, 220, 40, 40, 180, 180, 10, 10]
    FAKE_VIDEOS[str(path)] = [
        np.full((8, 8, 3), value, dtype=np.uint8) for value in intensities
    ]
    return path


def test_export_schema_has_deploy_visible_curve_source_and_no_forbidden_fields(tmp_path):
    ann = _write_json(tmp_path / "ann.json", _annotation_payload(annotations=[]))
    video_root = tmp_path / "videos"
    video_root.mkdir()
    _make_fake_video(video_root / "video_0001.mp4")

    payload = export_deploy_visible_scout(
        annotation_json=ann,
        video_roots=[video_root],
        subset="validation",
        out_json=tmp_path / "scout.json",
        curve_len=6,
        resize=4,
    )

    assert payload["scout_source"] == "deploy_visible_raw_video_graydiff_v1"
    assert payload["success_count"] == 1
    record = payload["videos"]["video_0001"]
    assert record["scout_source"] == "deploy_visible_raw_video_graydiff_v1"
    assert len(record["deploy_visible_scout_curve"]) == 6
    assert all(0.0 <= value <= 1.0 for value in record["deploy_visible_scout_curve"])
    assert len(set(round(value, 6) for value in record["deploy_visible_scout_curve"])) > 1

    serialized = json.loads((tmp_path / "scout.json").read_text(encoding="utf-8"))
    per_video_text = json.dumps(serialized["videos"]["video_0001"])
    for key in FORBIDDEN_OUTPUT_KEYS:
        assert key not in serialized["videos"]["video_0001"]
        assert key not in per_video_text


def test_annotation_gt_does_not_change_exported_curve(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    _make_fake_video(video_root / "video_0001.mp4")
    ann_a = _write_json(
        tmp_path / "ann_a.json",
        _annotation_payload(annotations=[{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    ann_b = _write_json(
        tmp_path / "ann_b.json",
        _annotation_payload(annotations=[{"segment": [9.0, 11.0], "label": "GolfSwing"}]),
    )

    first = export_deploy_visible_scout(ann_a, [video_root], "validation", tmp_path / "a.json", 6, 4)
    second = export_deploy_visible_scout(ann_b, [video_root], "validation", tmp_path / "b.json", 6, 4)

    assert first["videos"]["video_0001"]["deploy_visible_scout_curve"] == second["videos"]["video_0001"][
        "deploy_visible_scout_curve"
    ]


def test_missing_video_fails_closed_by_default_and_allow_missing_records_skip(tmp_path):
    ann = _write_json(tmp_path / "ann.json", _annotation_payload())
    video_root = tmp_path / "videos"
    video_root.mkdir()

    with pytest.raises(DeployVisibleScoutExportError, match="missing video"):
        export_deploy_visible_scout(ann, [video_root], "validation", tmp_path / "scout.json", 6, 4)

    with pytest.raises(DeployVisibleScoutExportError, match="no videos were successfully exported"):
        export_deploy_visible_scout(
            ann,
            [video_root],
            "validation",
            tmp_path / "scout_missing.json",
            6,
            4,
            allow_missing=True,
        )

    out = json.loads((tmp_path / "scout_missing.json").read_text(encoding="utf-8"))
    assert out["success_count"] == 0
    assert out["missing_count"] == 1
    assert out["skipped_missing"][0]["video_id"] == "video_0001"


def test_exported_scout_connects_to_run_audit_minimal_case(tmp_path):
    ann = _write_json(
        tmp_path / "ann.json",
        _annotation_payload(annotations=[{"segment": [2.0, 4.0], "label": "BaseballPitch"}]),
    )
    video_root = tmp_path / "videos"
    video_root.mkdir()
    _make_fake_video(video_root / "video_0001.mp4")
    scout_path = tmp_path / "scout.json"

    export_deploy_visible_scout(ann, [video_root], "validation", scout_path, 12, 4)
    audit = run_audit(
        ann,
        scout_path,
        abr_config=ABRConfig(k0=12, k1_cap=0, k2_cap=0, max_total_k=12, max_gap=1, round2_enabled=False),
    )

    assert audit["real_deploy_visible_recall_evidence"] is True
    assert audit["diagnostic_fallback_used"] is False
    assert audit["selector_gt_visible"] is False
    assert audit["scout_sources"] == ["deploy_visible_raw_video_graydiff_v1"]


def test_cli_exports_with_repeated_video_root_and_max_videos(tmp_path):
    ann = _write_json(tmp_path / "ann.json", _annotation_payload(annotations=[]))
    empty_root = tmp_path / "empty"
    video_root = tmp_path / "videos"
    empty_root.mkdir()
    video_root.mkdir()
    _make_fake_video(video_root / "video_0001.mp4")
    out = tmp_path / "scout_cli.json"

    rc = main(
        [
            "--annotation-json",
            str(ann),
            "--video-root",
            str(empty_root),
            "--video-root",
            str(video_root),
            "--subset",
            "validation",
            "--out-json",
            str(out),
            "--curve-len",
            "6",
            "--resize",
            "4",
            "--max-videos",
            "1",
        ],
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert list(payload["videos"]) == ["video_0001"]
    assert payload["success_count"] == 1
