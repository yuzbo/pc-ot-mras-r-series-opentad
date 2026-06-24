from __future__ import annotations

import importlib.util
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
    / "pc_ot_mras_prebackbone_c3_global_rank_st_full_train_candidate_n16r4.py"
)
LAUNCHER_PATH = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_c3_global_rank_st_precheck_n16r4.sbatch"


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


class _FrameScoreFirstReader:
    def __init__(self, torch, frame_logits):
        self.frame_logits = torch.nn.Parameter(torch.tensor(frame_logits, dtype=torch.float32))

    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        device = lowcost_features.device
        slot_logits = lowcost_features.new_zeros((batch, 4, time))
        slot_logits[:, 0, -1] = 100.0
        slot_logits[:, 1, 0] = 90.0
        slot_logits[:, 2, 2] = 80.0
        slot_logits[:, 3, 5] = 70.0
        frame_scores = self.frame_logits[:time].to(device=device).unsqueeze(0).expand(batch, -1)
        zeros = lowcost_features.new_zeros((batch, time))
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "start_logits": frame_scores,
            "end_logits": frame_scores,
            "uncertainty_logits": zeros,
            "redundancy_logits": zeros,
            "regularizers": {"total_regularizer": frame_scores.sum() * 0.0},
        }


def _make_time_index_inputs(torch, *, dense_len: int = 8):
    values = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len, 1, 1)
    inputs = values.expand(1, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((1, dense_len), dtype=torch.bool)
    metas = [{"sample_id": "c3-global-rank-st"}]
    return inputs, masks, metas


def _global_rank_selector(module):
    return module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "FrameScoreFirstReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_global_rank_st",
        frame_score_st_surrogate="global_rank_topk",
        frame_score_st_temperature=1.0,
        frame_score_st_logit_clamp=12.0,
        frame_score_st_gradient_scale=0.25,
        global_rank_st_temperature=0.75,
        global_rank_st_topk=4,
        global_rank_st_rank_width=1.25,
        protected_uniform_count=0,
        coverage_guard_count=0,
        max_gap=0,
        st_surrogate_mode="full_flat",
        aux_gt_acquisition_loss_weight=0.0,
        reader_regularizer_loss_weight=0.0,
    )


def test_c3_global_rank_st_uses_frame_scores_for_hard_real_frame_selection_in_train_and_test():
    torch = _import_torch_or_skip()
    reader = _FrameScoreFirstReader(torch, [0.0, 10.0, 2.0, 8.0, 7.0, 1.0, 9.0, -5.0])
    module = _load_prebackbone_selector_module(reader)
    selector = _global_rank_selector(module)
    inputs, masks, metas = _make_time_index_inputs(torch)

    train_outputs = selector.forward_train(
        inputs.clone(),
        masks,
        [dict(meta) for meta in metas],
        gt_segments=[torch.tensor([[0.0, 4.0]], dtype=torch.float32)],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    test_outputs = selector.forward_test(inputs.clone(), masks, [dict(meta) for meta in metas])

    selected_train = train_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    selected_test = test_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected_train == [1, 3, 4, 6]
    assert selected_test == selected_train
    assert selected_train == sorted(set(selected_train))
    assert torch.equal(train_outputs["masks"], test_outputs["masks"])
    assert torch.allclose(train_outputs["inputs"][0, 0, :, 0, 0], torch.tensor([1.0, 3.0, 4.0, 6.0]))

    meta = train_outputs["metas"][0]
    assert meta["pc_ot_mras_prebackbone_selection_strategy"] == "frame_score_global_rank_st"
    assert meta["pc_ot_mras_prebackbone_hard_selection_source"] == "frame_selection_logits"
    assert meta["pc_ot_mras_prebackbone_slot_not_hard_source"] is True
    assert meta["pc_ot_mras_prebackbone_frame_score_st_surrogate"] == "global_rank_topk"
    assert meta["pc_ot_mras_prebackbone_global_rank_st_topk"] == 4


def test_c3_global_rank_st_surrogate_backpropagates_finite_gradient_to_frame_logits():
    torch = _import_torch_or_skip()
    reader = _FrameScoreFirstReader(torch, [0.0, 10.0, 2.0, 8.0, 7.0, 1.0, 9.0, -5.0])
    module = _load_prebackbone_selector_module(reader)
    selector = _global_rank_selector(module)
    inputs, masks, metas = _make_time_index_inputs(torch)
    inputs = inputs.requires_grad_(True)

    outputs = selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=[torch.tensor([[0.0, 4.0]], dtype=torch.float32)],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    outputs["inputs"].square().mean().backward()

    assert reader.frame_logits.grad is not None
    assert torch.isfinite(reader.frame_logits.grad).all()
    assert reader.frame_logits.grad.abs().sum().item() > 0.0
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_c3_global_rank_st_config_identity_and_gate_remain_locked():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG_PATH))
    frame_selector = cfg.model.frame_selector
    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate

    assert cfg.variant_id == "C3-GlobalRankST-BoundaryDifficulty-OriginalAdaTAD"
    assert cfg.experiment_scope.route_family == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.experiment_scope.route == "pc_ot_mras_prebackbone_c3_global_rank_st"
    assert cfg.experiment_scope.selection_strategy == "frame_score_global_rank_st"
    assert cfg.experiment_scope.rank_transport_surrogate == "global_rank_topk"
    assert cfg.experiment_scope.uses_p2 is False
    assert cfg.experiment_scope.uses_teacher is False
    assert cfg.experiment_scope.uses_test_gt is False
    assert cfg.experiment_scope.uses_raw_prediction_cache is False
    assert "DIVERGENT_INNOVATION_BH_SDC" not in repr(cfg)
    assert "PCOTMRASDetectorBridge" not in repr(cfg.model)

    assert frame_selector.selection_strategy == "frame_score_global_rank_st"
    assert frame_selector.frame_score_st_surrogate == "global_rank_topk"
    assert frame_selector.global_rank_st_topk == 384
    assert float(frame_selector.global_rank_st_temperature) > 0.0
    assert frame_selector.reader.type == "PCOTMRASBoundaryDifficultyTemporalFrameScout"
    assert int(frame_selector.target_len) == 384
    assert int(frame_selector.dense_window_size) == 768
    assert int(frame_selector.max_dense_gap) == 0
    assert int(frame_selector.max_gap_guard_count) == 0

    assert gate.formal_train_candidate is False
    assert gate.launch_gate_passed is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_long_training is False
    assert tuple(gate.allowed_entrypoints) == ()


def test_c3_global_rank_st_precheck_launcher_is_fail_closed_and_not_a_training_launcher():
    assert LAUNCHER_PATH.is_file()
    text = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "C3_ORIGINAL_OPTIMIZATION_ROUTE" in text
    assert "pc_ot_mras_prebackbone_c3_global_rank_st" in text
    assert "pc_ot_mras_prebackbone_c3_global_rank_st_full_train_candidate_n16r4.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert "ALLOW_C3_GLOBAL_RANK_ST_TRAIN" in text
    assert "tools/test.py" not in text
    assert "torchrun" not in text
    assert "tools/train.py \"$CONFIG\"" not in text
    assert "raw prediction/cache is forbidden" in text
    assert "PRECHECK_ONLY=0 remains locked" in text
