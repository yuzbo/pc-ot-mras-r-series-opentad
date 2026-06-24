from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
FIXED_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py"
)
DYNAMIC_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_pro_dynamic_marginal_budget_guard_candidate_n16r4.py"
)
LEARNED_DYNAMIC_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_learned_dynamic_budget_candidate_n16r4.py"
)
LEARNED_DYNAMIC_PRECHECK_LAUNCHER = (
    ROOT
    / "scripts"
    / "run_pc_ot_mras_prebackbone_c3_learned_dynamic_budget_precheck_n16r4.sbatch"
)


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


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


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


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


def _make_time_index_inputs(torch, *, batch: int, dense_len: int):
    values = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len, 1, 1)
    inputs = values.expand(batch, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"c3-pro-dynamic-{idx}"} for idx in range(batch)]
    return inputs, masks, metas


class _ThreeLevelMarginalUtilityReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        torch = _import_torch_or_skip()
        batch, time, _dim = lowcost_features.shape
        slot_logits = lowcost_features.new_zeros((batch, 8, time))
        frame_scores = lowcost_features.new_full((batch, time), -8.0)
        frame_scores[0, :time] = 8.0
        frame_scores[1, :time] = 0.0
        frame_scores[2, :time] = -8.0
        boundary_logits = torch.stack(
            [
                lowcost_features.new_full((time,), 6.0),
                lowcost_features.new_zeros((time,)),
                lowcost_features.new_full((time,), -6.0),
            ],
            dim=0,
        )
        uncertainty_logits = torch.stack(
            [
                lowcost_features.new_full((time,), 3.0),
                lowcost_features.new_zeros((time,)),
                lowcost_features.new_full((time,), -3.0),
            ],
            dim=0,
        )
        redundancy_logits = torch.stack(
            [
                lowcost_features.new_full((time,), -6.0),
                lowcost_features.new_zeros((time,)),
                lowcost_features.new_full((time,), 6.0),
            ],
            dim=0,
        )
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": torch.softmax(slot_logits, dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "boundary_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
            "redundancy_logits": redundancy_logits,
            "regularizers": {"total_regularizer": lowcost_features.sum() * 0.0},
        }


class _SparseTailFrameScoreReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        torch = _import_torch_or_skip()
        batch, time, _dim = lowcost_features.shape
        slot_logits = lowcost_features.new_zeros((batch, 4, time))
        frame_scores = lowcost_features.new_full((batch, time), -20.0)
        for score, pos in zip((40.0, 39.0, 38.0, 37.0), (31, 30, 1, 0)):
            frame_scores[:, min(pos, time - 1)] = score
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": torch.softmax(slot_logits, dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "regularizers": {"total_regularizer": lowcost_features.sum() * 0.0},
        }


class _LearnedBudgetReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        torch = _import_torch_or_skip()
        batch, time, _dim = lowcost_features.shape
        slot_logits = lowcost_features.new_zeros((batch, 8, time))
        frame_scores = lowcost_features.new_full((batch, time), -40.0)
        preferred_positions = torch.tensor([15, 0, 14, 1, 13, 2, 12, 3], device=frame_scores.device)
        preferred_positions = preferred_positions.clamp(max=time - 1)
        for rank, pos in enumerate(preferred_positions.tolist()):
            frame_scores[:, pos] = 100.0 - rank
        budget_logits = lowcost_features.new_tensor(
            [
                [-8.0, -4.0, 8.0],
                [-8.0, 8.0, -4.0],
                [8.0, -4.0, -8.0],
            ]
        )
        if batch != 3:
            budget_logits = budget_logits[:batch]
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": torch.softmax(slot_logits, dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "budget_logits": budget_logits,
            "boundary_logits": frame_scores * 0.0,
            "uncertainty_logits": frame_scores * 0.0,
            "redundancy_logits": frame_scores * 0.0,
            "regularizers": {"total_regularizer": lowcost_features.sum() * 0.0},
        }


def _build_learned_budget_selector(module):
    return module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "LearnedBudgetReader"},
        target_len=8,
        dense_window_size=16,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        dynamic_budget=dict(
            enabled=True,
            protocol="learned_dynamic_budget",
            budget_bins=(4, 6, 8),
            budget_logit_key="budget_logits",
            hard_budget=True,
            train_test_same_hard_budget=True,
            budget_average_loss_weight=0.05,
            budget_entropy_loss_weight=0.005,
            budget_difficulty_target_loss_weight=0.05,
            budget_target_temperature=2.0,
        ),
    )


def test_learned_dynamic_budget_hard_forward_uses_true_frames_and_invalid_tail_mask():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_LearnedBudgetReader())
    selector = _build_learned_budget_selector(module)
    inputs, masks, metas = _make_time_index_inputs(torch, batch=3, dense_len=16)

    outputs = selector.forward_test(inputs, masks, metas)

    assert outputs["masks"].long().sum(dim=1).tolist() == [8, 6, 4]
    expected_prefixes = ([0, 1, 2, 3, 12, 13, 14, 15], [0, 1, 2, 13, 14, 15], [0, 1, 14, 15])
    for batch_idx, expected_prefix in enumerate(expected_prefixes):
        meta = outputs["metas"][batch_idx]
        selected = meta["pc_ot_mras_prebackbone_selected_dense_indices"]
        valid_len = int(meta["irregular_selected_output_valid_len"])
        assert selected[:valid_len] == list(expected_prefix)
        assert outputs["inputs"][batch_idx, 0, :valid_len, 0, 0].tolist() == pytest.approx(expected_prefix)
        assert outputs["masks"][batch_idx, :valid_len].all()
        assert not outputs["masks"][batch_idx, valid_len:].any()
        if valid_len < outputs["inputs"].shape[2]:
            assert float(outputs["inputs"][batch_idx, :, valid_len:, :, :].abs().sum().item()) == 0.0
        assert all(float(value).is_integer() for value in outputs["inputs"][batch_idx, 0, :valid_len, 0, 0])
        budget_meta = meta["pc_ot_mras_prebackbone_dynamic_budget"]
        assert budget_meta["protocol"] == "learned_dynamic_budget"
        assert budget_meta["predicted_budget"] == valid_len
        assert budget_meta["budget"] == valid_len
        assert budget_meta["budget_bins"] == [4, 6, 8]
        assert len(budget_meta["budget_probs"]) == 3
        assert budget_meta["budget_entropy"] >= 0.0
        assert budget_meta["uses_gt"] is False
        assert budget_meta["uses_teacher"] is False
        assert budget_meta["uses_raw_prediction_cache"] is False
        assert budget_meta["uses_p2"] is False


def test_learned_dynamic_budget_train_and_test_use_same_hard_indices():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_LearnedBudgetReader())
    selector = _build_learned_budget_selector(module)
    inputs, masks, test_metas = _make_time_index_inputs(torch, batch=3, dense_len=16)
    train_metas = [{"sample_id": f"train-{idx}"} for idx in range(3)]
    gt_segments = [torch.empty((0, 2), dtype=torch.float32) for _ in range(3)]
    gt_labels = [torch.empty((0,), dtype=torch.long) for _ in range(3)]

    test_outputs = selector.forward_test(inputs, masks, test_metas)
    train_outputs = selector.forward_train(inputs, masks, train_metas, gt_segments, gt_labels)

    assert train_outputs["masks"].long().sum(dim=1).tolist() == test_outputs["masks"].long().sum(dim=1).tolist()
    for train_meta, test_meta in zip(train_outputs["metas"], test_outputs["metas"]):
        train_valid = int(train_meta["irregular_selected_output_valid_len"])
        test_valid = int(test_meta["irregular_selected_output_valid_len"])
        assert train_valid == test_valid
        assert train_meta["pc_ot_mras_prebackbone_selected_dense_indices"][:train_valid] == (
            test_meta["pc_ot_mras_prebackbone_selected_dense_indices"][:test_valid]
        )
        assert train_meta["pc_ot_mras_prebackbone_dynamic_budget"]["predicted_budget"] == (
            test_meta["pc_ot_mras_prebackbone_dynamic_budget"]["predicted_budget"]
        )


def test_real_boundary_difficulty_learned_budget_head_receives_nonzero_aux_gradient():
    torch = _import_torch_or_skip()
    torch.manual_seed(7)
    module = _load_prebackbone_selector_module(None)
    reader = module.PCOTMRASBoundaryDifficultyTemporalFrameScout(
        in_dim=12,
        hidden_dim=16,
        num_slots=8,
        temporal_layers=1,
        temporal_kernel_size=3,
        dropout=0.0,
        budget_bin_count=3,
    )
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASBoundaryDifficultyTemporalFrameScout"},
        target_len=8,
        dense_window_size=16,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        aux_gt_acquisition_loss_weight=0.0,
        aux_duplicate_cap_loss_weight=0.0,
        reader_regularizer_loss_weight=0.0,
        dynamic_budget=dict(
            enabled=True,
            protocol="learned_dynamic_budget",
            budget_bins=(4, 6, 8),
            target_budget=6,
            average_budget=6,
            budget_logit_key="budget_logits",
            hard_budget=True,
            train_test_same_hard_budget=True,
            budget_average_loss_weight=0.05,
            budget_entropy_loss_weight=0.005,
            budget_difficulty_target_loss_weight=0.10,
            budget_target_temperature=2.0,
        ),
    )
    selector.reader = reader
    inputs = torch.randn(2, 3, 16, 2, 2)
    masks = torch.ones((2, 16), dtype=torch.bool)
    metas = [{"sample_id": "budget-grad-0"}, {"sample_id": "budget-grad-1"}]
    gt_segments = [torch.empty((0, 2), dtype=torch.float32) for _ in range(2)]
    gt_labels = [torch.empty((0,), dtype=torch.long) for _ in range(2)]

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    assert "selector_budget_average_loss" in outputs["losses"]
    assert "selector_budget_entropy_loss" in outputs["losses"]
    assert "selector_budget_difficulty_target_loss" in outputs["losses"]

    total_loss = sum(outputs["losses"].values())
    total_loss.backward()

    grad = selector.reader.budget_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0


def test_c3_pro_dynamic_marginal_budget_selector_outputs_variable_budget_metadata():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_ThreeLevelMarginalUtilityReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ThreeLevelMarginalUtilityReader"},
        target_len=8,
        dense_window_size=16,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        dynamic_budget=dict(
            enabled=True,
            protocol="marginal_utility_v0",
            min_budget=4,
            target_budget=6,
            max_budget=8,
            average_budget=6,
            budget_step=2,
            actionness_weight=1.0,
            boundary_weight=0.35,
            uncertainty_weight=0.20,
            redundancy_weight=0.35,
        ),
    )
    inputs, masks, metas = _make_time_index_inputs(torch, batch=3, dense_len=16)

    outputs = selector.forward_test(inputs, masks, metas)

    assert outputs["masks"].long().sum(dim=1).tolist() == [8, 6, 4]
    for meta, expected_budget in zip(outputs["metas"], (8, 6, 4)):
        budget_meta = meta["pc_ot_mras_prebackbone_dynamic_budget"]
        assert budget_meta["enabled"] is True
        assert budget_meta["protocol"] == "marginal_utility_v0"
        assert budget_meta["budget"] == expected_budget
        assert budget_meta["min_budget"] == 4
        assert budget_meta["target_budget"] == 6
        assert budget_meta["max_budget"] == 8
        assert budget_meta["average_budget"] == 6
        assert budget_meta["uses_gt"] is False
        assert budget_meta["uses_teacher"] is False
        assert budget_meta["uses_raw_prediction_cache"] is False
        assert budget_meta["uses_p2"] is False
        assert budget_meta["deploy_time_signals"] == [
            "frame_selection_logits",
            "actionness_logits",
            "boundary_logits",
            "uncertainty_logits",
            "redundancy_logits",
            "valid_len",
        ]


def test_frame_score_topk_max_gap_guard_is_applied_before_score_fill():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_SparseTailFrameScoreReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "SparseTailFrameScoreReader"},
        target_len=4,
        dense_window_size=32,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        max_dense_gap=10,
        max_gap_guard_count=4,
    )
    inputs, masks, metas = _make_time_index_inputs(torch, batch=1, dense_len=32)

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    roles = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_roles"]
    assert selected == sorted(selected)
    assert len(selected) == len(set(selected)) == 4
    assert max(right - left for left, right in zip(selected[:-1], selected[1:])) <= 10
    assert "max_gap_guard" in roles
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_max_gap_guard"]["enabled"] is True
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_max_gap_guard"]["safety_gate_only"] is True


def test_c3_pro_dynamic_budget_config_parse_gate_and_fixed_control_survives():
    mmengine_config = pytest.importorskip("mmengine.config")

    fixed_cfg = mmengine_config.Config.fromfile(str(FIXED_CONFIG))
    dynamic_cfg = mmengine_config.Config.fromfile(str(DYNAMIC_CONFIG))

    assert fixed_cfg.experiment_scope.budget_protocol == (
        "fixed384_over_dense768_frame_score_first_topk_no_hard_uniform_guard"
    )
    assert fixed_cfg.model.frame_selector.target_len == 384
    assert fixed_cfg.model.frame_selector.max_dense_gap == 0
    assert fixed_cfg.model.frame_selector.max_gap_guard_count == 0

    assert dynamic_cfg.experiment_scope.budget_protocol == (
        "dynamic_marginal_utility_min320_target384_avg384_max448_over_dense768_frame_score_first_topk"
    )
    assert dynamic_cfg.experiment_scope.dynamic_budget_protocol_candidate is True
    assert dynamic_cfg.experiment_scope.max_gap_guard_safety_gate is True
    assert dynamic_cfg.experiment_scope.uses_p2 is False
    assert dynamic_cfg.experiment_scope.uses_teacher is False
    assert dynamic_cfg.experiment_scope.uses_test_gt is False
    assert dynamic_cfg.experiment_scope.uses_raw_prediction_cache is False
    assert dynamic_cfg.inference.load_from_raw_predictions is False
    assert dynamic_cfg.inference.save_raw_prediction is False

    selector = dynamic_cfg.model.frame_selector
    assert selector.selection_strategy == "frame_score_topk"
    assert selector.max_dense_gap > 0
    assert selector.max_gap_guard_count > 0
    assert selector.dynamic_budget.enabled is True
    assert selector.dynamic_budget.protocol == "marginal_utility_v0"
    assert selector.dynamic_budget.min_budget == 320
    assert selector.dynamic_budget.target_budget == 384
    assert selector.dynamic_budget.average_budget == 384
    assert selector.dynamic_budget.max_budget == 448
    assert selector.target_len >= selector.dynamic_budget.max_budget
    assert selector.reader.num_slots == selector.target_len
    assert dynamic_cfg.model.backbone.backbone.total_frames == selector.target_len
    assert dynamic_cfg.model.projection.max_seq_len == selector.target_len
    assert selector.dynamic_budget.deploy_visible_signals == (
        "frame_selection_logits",
        "actionness_logits",
        "boundary_logits",
        "uncertainty_logits",
        "redundancy_logits",
        "valid_len",
    )


def test_c3_learned_dynamic_budget_config_is_precheck_only_and_forbids_shortcuts():
    mmengine_config = pytest.importorskip("mmengine.config")

    cfg = mmengine_config.Config.fromfile(str(LEARNED_DYNAMIC_CONFIG))

    assert cfg.experiment_scope.route == "pc_ot_mras_prebackbone_c3_learned_dynamic_budget"
    assert cfg.experiment_scope.selection_strategy == "frame_score_topk"
    assert cfg.experiment_scope.budget_policy == "learned_dynamic_budget"
    assert cfg.experiment_scope.budget_bins == (256, 320, 384)
    assert cfg.experiment_scope.max_output_budget == 384
    assert cfg.experiment_scope.uses_p2 is False
    assert cfg.experiment_scope.uses_offline_ledger is False
    assert cfg.experiment_scope.uses_teacher is False
    assert cfg.experiment_scope.uses_test_gt is False
    assert cfg.experiment_scope.uses_raw_prediction_cache is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_precheck_only is True
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_tools_train is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_tools_test is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_slurm_precheck is True
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_gpu_precheck is True
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_slurm_long_training is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_gpu_long_training is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_detector_map is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    selector = cfg.model.frame_selector
    assert selector.target_len == 384
    assert selector.reader.num_slots == 384
    assert selector.selection_strategy == "frame_score_topk"
    assert selector.dynamic_budget.enabled is True
    assert selector.dynamic_budget.protocol == "learned_dynamic_budget"
    assert selector.dynamic_budget.budget_bins == (256, 320, 384)
    assert selector.dynamic_budget.max_budget == 384
    assert selector.dynamic_budget.budget_average_loss_weight > 0.0
    assert selector.dynamic_budget.budget_entropy_loss_weight > 0.0
    assert selector.dynamic_budget.budget_difficulty_target_loss_weight > 0.0
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384


def test_c3_learned_dynamic_budget_precheck_launcher_is_fail_closed():
    text = LEARNED_DYNAMIC_PRECHECK_LAUNCHER.read_text(encoding="utf-8")

    assert "PRECHECK_ONLY=\"${PRECHECK_ONLY:-1}\"" in text
    assert "PRECHECK_ONLY must stay 1" in text
    assert "allow_slurm_precheck is not True" in text
    assert "allow_slurm_long_training is not False" in text
    assert "codex/c3-dynamic-budget-20260624" in text
    assert "pc_ot_mras_prebackbone_c3_learned_dynamic_budget_candidate_n16r4.py" in text
    assert "py_compile" in text
    assert "Config.fromfile" in text
    assert "test_pc_ot_mras_prebackbone_c3_pro_dynamic_budget.py" in text
    assert "tools/train.py" not in text
    assert "tools/test.py" not in text
    assert "sbatch " not in text
    assert "srun " not in text
    assert "PREBACKBONE_C3_LEARNED_DYNAMIC_BUDGET_PRECHECK_PASS" in text
