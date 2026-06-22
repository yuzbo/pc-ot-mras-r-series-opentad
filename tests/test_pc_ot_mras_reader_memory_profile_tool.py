import json
import subprocess
import sys
from pathlib import Path

import pytest


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


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "profile_pc_ot_mras_reader_memory.py"


def _run_profile(*args):
    proc = subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return json.loads(proc.stdout)


def test_profile_tool_skips_pair_outputs_when_disabled(tmp_path):
    json_path = tmp_path / "reader_profile.json"

    payload = _run_profile(
        "--batch-size",
        "1",
        "--time",
        "6",
        "--hidden-dim",
        "8",
        "--num-slots",
        "3",
        "--emit-pair-distribution",
        "0",
        "--device",
        "cpu",
        "--json-output",
        str(json_path),
    )
    file_payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload == file_payload
    assert payload["success"] is True
    assert payload["emit_pair_distribution"] is False
    assert payload["regularizers_finite"] is True
    assert payload["pair_shape"] is None
    assert "pair_logits" not in payload["output_keys"]
    assert "pair_prob" not in payload["output_keys"]
    assert "pair_valid_mask" not in payload["output_keys"]


def test_profile_tool_records_pair_shape_when_enabled():
    payload = _run_profile(
        "--batch-size",
        "1",
        "--time",
        "5",
        "--hidden-dim",
        "8",
        "--num-slots",
        "3",
        "--emit-pair-distribution",
        "1",
        "--device",
        "cpu",
    )

    assert payload["success"] is True
    assert payload["emit_pair_distribution"] is True
    assert payload["regularizers_finite"] is True
    assert payload["pair_shape"] == [1, 5, 5]
    assert "pair_logits" in payload["output_keys"]
    assert "pair_prob" in payload["output_keys"]
    assert "pair_valid_mask" in payload["output_keys"]
