import pytest
import subprocess
import sys

torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
if torch_probe.returncode != 0:
    pytest.skip(
        "torch unavailable in this process: "
        + (torch_probe.stderr.strip().splitlines()[-1] if torch_probe.stderr.strip() else f"exit {torch_probe.returncode}"),
        allow_module_level=True,
    )

import torch

from opentad.models.utils.post_processing.utils import convert_to_seconds


def _base_meta():
    return {
        "fps": 10.0,
        "duration": 100.0,
        "snippet_stride": 4.0,
        "window_start_frame": 100.0,
        "offset_frames": 2.0,
    }


def test_convert_to_seconds_maps_selected_axis_back_to_physical_dense_time():
    segments = torch.tensor([[0.0, 2.0], [1.5, 3.0]], dtype=torch.float32)
    meta = {
        **_base_meta(),
        "irregular_selected_positions": [0.0, 10.0, 20.0],
        "irregular_selected_valid_len": 30.0,
        "irregular_native_axis": False,
    }

    out = convert_to_seconds(segments.clone(), meta)

    expected = torch.tensor([[10.2, 18.2], [16.2, 22.2]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1.0e-5)


def test_convert_to_seconds_keeps_regular_axis_when_irregular_native_axis_is_true():
    segments = torch.tensor([[0.0, 2.0], [1.5, 3.0]], dtype=torch.float32)
    meta = {
        **_base_meta(),
        "irregular_selected_positions": [0.0, 10.0, 20.0],
        "irregular_selected_valid_len": 30.0,
        "irregular_native_axis": True,
    }

    out = convert_to_seconds(segments.clone(), meta)

    expected = torch.tensor([[10.2, 11.0], [10.8, 11.4]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1.0e-5)


def test_convert_to_seconds_rejects_unsorted_selected_axis_positions():
    segments = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    meta = {
        **_base_meta(),
        "irregular_selected_positions": [0.0, 20.0, 10.0],
        "irregular_selected_valid_len": 30.0,
        "irregular_native_axis": False,
    }

    with pytest.raises(ValueError, match="irregular_selected_positions must be sorted"):
        convert_to_seconds(segments.clone(), meta)
