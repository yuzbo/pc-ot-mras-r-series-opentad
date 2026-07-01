from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.bata.c3_coarse_classifier_model_matrix import MODEL_MATRIX, iter_matrix


def test_first_wave_matrix_has_broad_model_families() -> None:
    entries = list(iter_matrix(tier="first_wave"))
    families = {entry["family"] for entry in entries}
    assert "image_backbone_temporal_head" in families
    assert "native_video_classifier" in families
    assert "video_transformer_teacher" in families
    assert len(entries) >= 10


def test_matrix_entries_are_c3_diagnostic_only_and_have_selection_rationale() -> None:
    ids = set()
    for entry in MODEL_MATRIX:
        assert entry["id"] not in ids
        ids.add(entry["id"])
        assert entry["tier"] in {"first_wave", "second_wave"}
        assert entry["backend"] in {"timm", "torchvision_video", "pytorchvideo_hub", "hf_snapshot"}
        assert isinstance(entry.get("why"), str) and len(entry["why"]) >= 32
        assert isinstance(entry.get("intended_head"), str) and len(entry["intended_head"]) >= 8
        assert isinstance(entry.get("compute_class"), str) and entry["compute_class"]
        assert "DIVERGENT_INNOVATION" not in json.dumps(entry)


def test_second_wave_is_not_default_downloaded() -> None:
    second_wave = [entry for entry in MODEL_MATRIX if entry["tier"] == "second_wave"]
    assert second_wave
    assert all(not entry["default_download"] for entry in second_wave)


def test_cli_dry_run_writes_download_status(tmp_path: Path) -> None:
    output = tmp_path / "matrix.json"
    script = Path("tools/bata/c3_coarse_classifier_model_matrix.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--download",
            "--dry-run",
            "--tier",
            "first_wave",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["route_label"] == "C3_MAINLINE_OPTIMIZATION"
    assert payload["diagnostic_only"] is True
    assert payload["no_detector_training"] is True
    assert payload["no_detector_eval"] is True
    assert payload["download_results"]
    assert {item["status"] for item in payload["download_results"]} == {"dry_run"}
    assert payload["schema_version"] == "c3_coarse_classifier_model_matrix_v1"
    assert proc.stderr == ""
