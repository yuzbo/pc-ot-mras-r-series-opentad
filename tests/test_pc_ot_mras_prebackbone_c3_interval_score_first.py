from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
CONFIG_PATH = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_interval_score_first_full_train_candidate_n16r4.py"
)
PRECHECK_CONFIG_PATH = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_interval_score_first_precheck_n16r4.py"
)
LAUNCHER_PATH = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_c3_interval_score_first_precheck_n16r4.sbatch"


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _import_torch_or_skip():
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else f"exit {probe.returncode}"
        pytest.skip(f"torch unavailable in this process: {detail}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local DLL state.
        pytest.skip(f"torch unavailable in this process: {exc}")
    return torch


def _load_prebackbone_selector_module(fake_reader):
    for name in (
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        "opentad.models.builder",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    builder.build_selector = lambda _cfg: fake_reader
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        SELECTOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_time_index_inputs(torch, *, dense_len: int = 8):
    values = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len, 1, 1)
    inputs = values.expand(1, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((1, dense_len), dtype=torch.bool)
    metas = [{"sample_id": "c3-interval-score-first"}]
    gt_segments = [torch.tensor([[2.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]
    return inputs, masks, metas, gt_segments, gt_labels


class _ScoreFirstIntervalReader:
    def __init__(self, torch):
        self.action_logits = torch.nn.Parameter(
            torch.tensor([-6.0, -5.0, 2.0, 8.0, 8.0, 2.0, -5.0, -6.0], dtype=torch.float32)
        )
        self.start_logits = torch.nn.Parameter(
            torch.tensor([-8.0, -7.0, 10.0, -2.0, -3.0, -6.0, 4.0, -5.0], dtype=torch.float32)
        )
        self.end_logits = torch.nn.Parameter(
            torch.tensor([-8.0, -7.0, -6.0, -3.0, -2.0, 10.0, -5.0, 4.0], dtype=torch.float32)
        )
        self.frame_logits = torch.nn.Parameter(
            torch.tensor([12.0, 11.0, -8.0, -8.0, -8.0, -8.0, 10.0, 9.0], dtype=torch.float32)
        )

    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slot_logits = lowcost_features.new_zeros((batch, 4, time))
        action = self.action_logits[:time].to(device=lowcost_features.device).unsqueeze(0).expand(batch, -1)
        start = self.start_logits[:time].to(device=lowcost_features.device).unsqueeze(0).expand(batch, -1)
        end = self.end_logits[:time].to(device=lowcost_features.device).unsqueeze(0).expand(batch, -1)
        frame = self.frame_logits[:time].to(device=lowcost_features.device).unsqueeze(0).expand(batch, -1)
        boundary = torch.maximum(start, end)
        zeros = lowcost_features.new_zeros((batch, time))
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame,
            "actionness_logits": action,
            "action_logits": action,
            "value_logits": action,
            "start_logits": start,
            "end_logits": end,
            "boundary_logits": boundary,
            "uncertainty_logits": zeros,
            "redundancy_logits": zeros,
            "regularizers": {"total_regularizer": slot_logits.sum() * 0.0},
        }


def _make_score_first_selector(module):
    return module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ScoreFirstIntervalReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="interval_score_first_packet",
        protected_uniform_count=0,
        coverage_guard_count=0,
        interval_boundary_budget_ratio=0.5,
        interval_candidate_topk=4,
        frame_score_st_surrogate="global_softmax",
        frame_score_st_temperature=1.0,
    )


def test_interval_score_first_hard_selection_uses_interval_score_before_packet_allocation():
    torch = _import_torch_or_skip()
    reader = _ScoreFirstIntervalReader(torch)
    module = _load_prebackbone_selector_module(reader)
    selector = _make_score_first_selector(module)
    inputs, masks, metas, _gt_segments, _gt_labels = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_test(inputs, masks, metas)
    meta = outputs["metas"][0]

    selected = meta["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [2, 3, 4, 5]
    assert selected != [0, 1, 6, 7]
    assert meta["pc_ot_mras_prebackbone_selection_strategy"] == "interval_score_first_packet"
    assert meta["pc_ot_mras_prebackbone_hard_selection_source"] == "interval_score_first_packet"
    assert "boundary_packet" in meta["pc_ot_mras_prebackbone_selected_roles"]
    assert "interior_action_packet" in meta["pc_ot_mras_prebackbone_selected_roles"]
    packet_meta = meta["pc_ot_mras_prebackbone_interval_packet_metadata"]
    assert packet_meta["status"] == "interval_score_first_packet"
    assert packet_meta["score_first"] is True
    assert packet_meta["boundary_positions"] == [2, 5]
    assert packet_meta["interior_positions"] == [3, 4]
    assert packet_meta["top_intervals"][0]["start"] == 2
    assert packet_meta["top_intervals"][0]["end"] == 5
    assert "interval_score" in packet_meta["source_heads"]
    assert packet_meta["uses_gt"] is False
    assert packet_meta["uses_teacher"] is False
    assert packet_meta["uses_raw_prediction_cache"] is False


def test_interval_score_first_train_test_match_and_backpropagates_interval_heads():
    torch = _import_torch_or_skip()
    reader = _ScoreFirstIntervalReader(torch)
    module = _load_prebackbone_selector_module(reader)
    selector = _make_score_first_selector(module)
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs(torch, dense_len=8)

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, [{"split": "test"}])

    assert train_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"] == [2, 3, 4, 5]
    assert test_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"] == [2, 3, 4, 5]
    assert train_outputs["masks"].tolist() == [[True, True, True, True]]
    assert test_outputs["masks"].tolist() == [[True, True, True, True]]

    detector_like_loss = train_outputs["inputs"].sum()
    detector_like_loss.backward()

    assert reader.action_logits.grad is not None
    assert reader.start_logits.grad is not None
    assert reader.end_logits.grad is not None
    assert torch.isfinite(reader.action_logits.grad).all()
    assert torch.isfinite(reader.start_logits.grad).all()
    assert torch.isfinite(reader.end_logits.grad).all()
    assert reader.action_logits.grad.abs().sum().item() > 0.0
    assert reader.start_logits.grad.abs().sum().item() > 0.0
    assert reader.end_logits.grad.abs().sum().item() > 0.0


def test_interval_score_first_configs_and_launcher_are_fail_closed():
    cfg_globals = runpy.run_path(str(CONFIG_PATH))
    precheck_globals = runpy.run_path(str(PRECHECK_CONFIG_PATH))
    selector = cfg_globals["model"]["frame_selector"]
    gate = cfg_globals["pc_ot_mras_prebackbone_e2e_acquisition_gate"]

    assert cfg_globals["variant_id"] == "C3-IntervalScoreFirst-BoundaryPacket-OriginalAdaTAD"
    assert cfg_globals["route_id"] == "pc_ot_mras_prebackbone_c3_interval_score_first"
    assert selector["selection_strategy"] == "interval_score_first_packet"
    assert selector["frame_score_st_surrogate"] == "global_softmax"
    assert selector["max_dense_gap"] == 0
    assert selector["max_gap_guard_count"] == 0
    assert cfg_globals["experiment_scope"]["changes_input_sampling"] is True
    assert cfg_globals["experiment_scope"]["changes_detector_head"] is False
    assert cfg_globals["experiment_scope"]["uses_teacher"] is False
    assert cfg_globals["experiment_scope"]["uses_test_gt"] is False
    assert cfg_globals["experiment_scope"]["uses_raw_prediction_cache"] is False
    assert gate["formal_train_candidate"] is False
    assert gate["allow_tools_train"] is False
    assert gate["allow_tools_test"] is False
    assert gate["allow_detector_map"] is False
    assert gate["allow_long_training"] is False
    assert gate["allowed_entrypoints"] == ()
    assert precheck_globals["stage_id"] == "c3_interval_score_first_precheck_n16r4"
    assert precheck_globals["pc_ot_mras_prebackbone_e2e_acquisition_gate"]["remote_precheck_only_candidate"] is True

    launcher_text = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "PRECHECK_ONLY=\"${PRECHECK_ONLY:-1}\"" in launcher_text
    assert "ALLOW_C3_INTERVAL_SCORE_FIRST_FULL_TRAIN" in launcher_text
    assert "tools/train.py" not in launcher_text.split("if [ \"$PRECHECK_ONLY\" = \"1\" ]", maxsplit=1)[0]
    assert "tools/test.py" not in launcher_text
    assert "raw_prediction" in launcher_text
