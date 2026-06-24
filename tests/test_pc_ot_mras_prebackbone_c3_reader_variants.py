from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"


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


def _load_prebackbone_selector_module():
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
    builder.build_selector = lambda cfg: cfg
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        SELECTOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _init_reader(reader_cls, *, in_dim: int, hidden_dim: int, num_slots: int):
    kwargs = {
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "num_slots": num_slots,
        "num_layers": 2,
        "num_heads": 2,
        "kernel_size": 3,
        "dropout": 0.0,
    }
    signature = inspect.signature(reader_cls.__init__)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return reader_cls(**filtered)


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


def test_c3_masked_slot_transport_keeps_probability_path_float32_under_half_logits():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    logits = torch.tensor(
        [
            [
                [512.0, 500.0, -512.0, -640.0],
                [-640.0, 500.0, 512.0, -640.0],
            ]
        ],
        dtype=torch.float16,
    )
    valid = torch.tensor([[True, True, True, False]], dtype=torch.bool)

    masked_logits, acquisition_matrix = module._masked_slot_transport(logits, valid)

    assert masked_logits.dtype == torch.float32
    assert acquisition_matrix.dtype == torch.float32
    assert torch.isfinite(masked_logits[valid[:, None, :].expand_as(masked_logits)]).all()
    assert torch.isfinite(acquisition_matrix).all()
    assert torch.all(acquisition_matrix[..., 3] == 0.0)
    row_sums = acquisition_matrix.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-6)


def test_c3_sparse_transport_rejects_nonfinite_reader_outputs():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector_cls = module.PCOTMRASPreBackboneFrameSelector
    selector = selector_cls.__new__(selector_cls)
    selector.target_len = 2
    selector.residual_count = None
    selector.protected_uniform_count = 0
    selector.coverage_guard_count = 0
    selector.straight_through_detector_loss = True
    selector.residual_slot_role = "learned_residual"

    valid = torch.tensor([[True, True, True]], dtype=torch.bool)
    candidate_valid = valid.clone()
    candidate_dense_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 1] = float("nan")

    with pytest.raises(ValueError, match="slot_logits must be finite"):
        selector._sparse_transport_plan(
            {"slot_logits": logits},
            valid,
            candidate_valid=candidate_valid,
            candidate_dense_indices=candidate_dense_indices,
            training=True,
        )


def test_c3_acquisition_aux_loss_keeps_gradient_for_duplicate_positive_slots():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector_cls = module.PCOTMRASPreBackboneFrameSelector
    selector = selector_cls.__new__(selector_cls)
    selector.aux_gt_acquisition_loss_weight = 1.0
    selector.aux_value_loss_weight = 0.0
    selector.aux_risk_loss_weight = 0.0
    selector.aux_role_entropy_loss_weight = 0.0
    selector.reader_regularizer_loss_weight = 0.0

    matrix = torch.tensor(
        [
            [
                [0.90, 0.10, 0.00, 0.00],
                [0.90, 0.10, 0.00, 0.00],
                [0.90, 0.10, 0.00, 0.00],
                [0.90, 0.10, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    candidate_dense_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    gt_segments = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]

    losses = selector._losses(
        reader_outputs={"acquisition_matrix": matrix},
        valid_mask=valid,
        candidate_dense_indices=candidate_dense_indices,
        gt_segments=gt_segments,
    )
    losses["selector_gt_acquisition_loss"].backward()

    assert torch.isfinite(losses["selector_gt_acquisition_loss"])
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()
    assert matrix.grad[0, :, 0].abs().sum().item() > 0.0


@pytest.mark.parametrize(
    "reader_name",
    [
        "PCOTMRASCNNFrameScout",
        "PCOTMRASMotionTCNFrameScout",
        "PCOTMRASHybridFrameScout",
        "PCOTMRASRSeriesHybridFrameScout",
    ],
)
def test_c3_reader_variants_emit_masked_slot_transport_with_gradients(reader_name):
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    reader_cls = getattr(module, reader_name, None)
    if reader_cls is None:
        pytest.skip(f"{reader_name} is not implemented in pc_ot_mras_prebackbone_frame_selector.py yet")

    batch, time, channels, slots = 2, 9, 5, 4
    torch.manual_seed(20260624)
    reader = _init_reader(reader_cls, in_dim=channels, hidden_dim=16, num_slots=slots)
    reader.train()
    features = torch.randn(batch, time, channels, requires_grad=True)
    valid = torch.tensor(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    time_coords = torch.linspace(0.0, 1.0, steps=time).unsqueeze(0).expand(batch, -1).clone()

    outputs = reader(features, valid, time_coords=time_coords)

    assert set(("slot_logits", "acquisition_matrix")).issubset(outputs)
    slot_logits = outputs["slot_logits"]
    acquisition_matrix = outputs["acquisition_matrix"]
    assert slot_logits.shape == (batch, slots, time)
    assert acquisition_matrix.shape == (batch, slots, time)
    assert torch.isfinite(slot_logits[valid.unsqueeze(1).expand_as(slot_logits)]).all()
    assert torch.isfinite(acquisition_matrix).all()
    assert torch.all(acquisition_matrix >= 0.0)
    assert torch.all(acquisition_matrix[1, :, 5:] == 0.0)
    row_sums = acquisition_matrix.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-5)

    weighted_feature = (acquisition_matrix * features[:, None, :, 0]).sum()
    valid_logit_mean = slot_logits.masked_fill(~valid[:, None, :], 0.0).sum() / valid.sum().clamp_min(1)
    loss = weighted_feature + valid_logit_mean * 0.01
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0.0
    trainable_grads = [param.grad for param in reader.parameters() if param.requires_grad]
    assert trainable_grads
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0.0 for grad in trainable_grads)


_C3_CONFIG_VARIANTS = (
    ("C3-CNN-Lite", "PCOTMRASCNNFrameScout", ("c3", "cnn", "lite")),
    ("C3-Motion-TCN", "PCOTMRASMotionTCNFrameScout", ("c3", "motion", "tcn")),
    ("C3-Hybrid", "PCOTMRASHybridFrameScout", ("c3", "hybrid")),
    ("C3-Pro-BoundaryDifficulty", "PCOTMRASBoundaryDifficultyTemporalFrameScout", ("c3", "pro", "boundary")),
)


def _normalized_stem(path: Path) -> str:
    return path.stem.lower().replace("-", "_")


def _candidate_config_paths(tokens: tuple[str, ...]) -> list[Path]:
    paths = []
    for path in CONFIG_DIR.glob("*.py"):
        stem = _normalized_stem(path)
        if all(token in stem for token in tokens):
            paths.append(path)
    return sorted(paths)


def _mapping_get(mapping, key, default=None):
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        value = mapping.get(key, default)
        if value is not default:
            return value
    return getattr(mapping, key, default)


def _first_existing_value(owners, names: tuple[str, ...]):
    sentinel = object()
    for owner in owners:
        for name in names:
            value = _mapping_get(owner, name, sentinel)
            if value is not sentinel:
                return value
    raise AssertionError(f"none of {names} found")


def _pipeline_step(pipeline, step_type: str):
    for step in pipeline:
        if _mapping_get(step, "type") == step_type:
            return step
    raise AssertionError(f"{step_type} step not found")


def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _iter_new_c3_configs(label: str, reader_type: str, tokens: tuple[str, ...]):
    paths = _candidate_config_paths(tokens)
    if not paths:
        pytest.skip(f"{label} config is not present yet; searched tokens={tokens}")
    mmengine_config = pytest.importorskip("mmengine.config")
    for path in paths:
        yield path, mmengine_config.Config.fromfile(str(path)), reader_type


@pytest.mark.parametrize("label,reader_type,tokens", _C3_CONFIG_VARIANTS)
def test_c3_reader_variant_configs_load_with_fixed_768_to_384_prebackbone_contract(label, reader_type, tokens):
    for path, cfg, expected_reader in _iter_new_c3_configs(label, reader_type, tokens):
        torch = _import_torch_or_skip()
        selector_module = _load_prebackbone_selector_module()
        frame_selector = cfg.model.frame_selector
        assert frame_selector.type == "PCOTMRASPreBackboneFrameSelector", path.name
        assert frame_selector.reader.type == expected_reader, path.name

        source_frames = _first_existing_value((cfg, frame_selector), ("source_frames", "dense_window_size"))
        target_frames = _first_existing_value((cfg, frame_selector), ("target_frames", "target_len", "window_size"))
        assert int(source_frames) == 768, path.name
        assert int(target_frames) == 384, path.name
        assert int(frame_selector.selection_unit) == 1, path.name
        if expected_reader == "PCOTMRASBoundaryDifficultyTemporalFrameScout":
            assert frame_selector.selection_strategy == "frame_score_topk", path.name
            assert int(frame_selector.max_dense_gap) == 0, path.name
            assert int(frame_selector.max_gap_guard_count) == 0, path.name
            assert float(frame_selector.frame_score_st_temperature) > 0.0, path.name
            assert float(frame_selector.frame_score_st_local_width) > 0.0, path.name
            assert float(frame_selector.reader_regularizer_loss_weight) == 0.0, path.name
            assert float(frame_selector.reader.soft_order_regularizer_weight) > 0.0, path.name
            assert float(frame_selector.reader.duplicate_mass_regularizer_weight) > 0.0, path.name
            assert float(frame_selector.reader.duplicate_mass_cap_factor) >= 1.0, path.name
        assert int(_pipeline_step(cfg.dataset.train.pipeline, "LoadFrames").trunc_len) == 768, path.name
        assert int(cfg.dataset.val.window_size) == 768, path.name
        assert int(cfg.dataset.test.window_size) == 768, path.name
        assert cfg.model.get("neck", {}).get("type") != "PCOTMRASDetectorBridge", path.name
        assert "PCOTMRASDetectorBridge" not in repr(cfg.model), path.name

        reader_cls = getattr(selector_module, expected_reader)
        reader_kwargs = {
            key: value for key, value in frame_selector.reader.items() if key not in {"type", "_delete_"}
        }
        reader = reader_cls(**reader_kwargs)
        features = torch.randn(1, 6, int(frame_selector.reader.in_dim))
        valid = torch.tensor([[True, True, True, True, False, False]], dtype=torch.bool)
        time_coords = torch.linspace(0.0, 1.0, steps=6).unsqueeze(0)
        outputs = reader(features, valid, time_coords=time_coords)
        assert outputs["slot_logits"].shape == (1, 384, 6), path.name
        assert outputs["acquisition_matrix"].shape == (1, 384, 6), path.name
        if expected_reader == "PCOTMRASBoundaryDifficultyTemporalFrameScout":
            for head_name in (
                "actionness_logits",
                "start_logits",
                "end_logits",
                "uncertainty_logits",
                "redundancy_logits",
                "frame_selection_logits",
            ):
                assert outputs[head_name].shape == (1, 6), f"{path.name} missing {head_name}"
        assert torch.all(outputs["acquisition_matrix"][..., 4:] == 0.0), path.name
        row_sums = outputs["acquisition_matrix"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-5), path.name


@pytest.mark.parametrize("label,reader_type,tokens", _C3_CONFIG_VARIANTS)
def test_c3_reader_variant_configs_forbid_eval_cache_claims_and_keep_original_actionformer(label, reader_type, tokens):
    for path, cfg, _expected_reader in _iter_new_c3_configs(label, reader_type, tokens):
        scope = cfg.experiment_scope
        gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
        context = gate.entrypoint_gate_context
        forbidden = set(_as_tuple(_mapping_get(context, "forbidden_true_keys")))
        allowed_entrypoints = set(_as_tuple(_mapping_get(gate, "allowed_entrypoints")))

        assert scope.detector_stack == "original_adatad_actionformer_adapter", path.name
        assert "ActionFormer" in repr(cfg.model) or scope.detector_stack == "original_adatad_actionformer_adapter", path.name
        assert "PCOTMRASDetectorBridge" not in repr(cfg.model), path.name
        assert "tools/test.py" not in allowed_entrypoints, path.name
        assert gate.allow_tools_test is False, path.name
        assert gate.allow_detector_map is False, path.name
        assert cfg.inference.load_from_raw_predictions is False, path.name
        assert cfg.inference.save_raw_prediction is False, path.name

        for key in (
            "tools_test",
            "allow_tools_test",
            "raw_prediction_cache",
            "load_from_raw_predictions",
            "save_raw_prediction",
            "paper_claim",
            "paper_claim_allowed",
            "deploy_claim",
            "deploy_claim_allowed",
            "runtime_claim",
            "runtime_flops_claim",
            "runtime_flops_claim_allowed",
        ):
            assert key in forbidden, f"{path.name} does not forbid {key}"

        assert scope.uses_teacher is False, path.name
        assert scope.uses_test_gt is False, path.name
        assert scope.uses_raw_prediction_cache is False, path.name
        assert scope.paper_claim_allowed is False, path.name
        assert scope.deploy_claim_allowed is False, path.name
        assert scope.runtime_flops_claim_allowed is False, path.name
        assert gate.paper_claim_allowed is False, path.name
        assert gate.deploy_claim_allowed is False, path.name
        assert gate.runtime_flops_claim_allowed is False, path.name


def test_c3_pro_boundary_full_train_config_saves_current_result_detection_json():
    mmengine_config = pytest.importorskip("mmengine.config")
    path = CONFIG_DIR / "pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py"
    cfg = mmengine_config.Config.fromfile(str(path))

    assert cfg.post_processing.save_dict is True
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.formal_train_candidate is True
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_tools_test is False


def test_c3_selector_metadata_env_hook_writes_analyzer_ready_current_batch_jsonl(tmp_path, monkeypatch):
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector_cls = module.PCOTMRASPreBackboneFrameSelector
    selector = selector_cls.__new__(selector_cls)
    selector.target_len = 4
    selector.selection_unit = 1
    selector.remap_gt_to_selected_axis = True
    selector.residual_count = None
    selector.residual_slot_role = "learned_residual"
    selector.selector_support_status = "supported"
    selector.selection_strategy = "frame_score_topk"
    selector.frame_score_st_surrogate = "local_softmax"
    selector.frame_score_st_logit_clamp = 12.0
    selector.frame_score_st_gradient_scale = 0.05
    selector.frame_score_aux_logit_clamp = 12.0
    selector.dynamic_budget = None
    selector.max_dense_gap = 0
    selector.max_gap_guard_count = 0
    selector.meta_source = "pc_ot_mras_prebackbone_e2e_frame_selector"
    selector.scout_feature_source = "compressed_pixels"
    selector.scout_spatial_size = (32, 32)
    selector._metadata_dump_count = 0

    dump_path = tmp_path / "selector_metadata.jsonl"
    monkeypatch.setenv("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL", str(dump_path))
    monkeypatch.setenv("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_MAX_ROWS", "8")

    selected_positions = torch.tensor([[0.0, 2.0, 4.0, 6.0]], dtype=torch.float32)
    valid_lengths = torch.tensor([8], dtype=torch.long)
    selected_output_valid_lengths = torch.tensor([4], dtype=torch.long)
    reader_outputs = {
        "frame_selection_logits": torch.tensor([[0.1, 0.5, 0.2, 0.7, 0.3, 0.9, 0.4, 0.8]], dtype=torch.float32),
        "boundary_logits": torch.tensor([[0.1, 0.2, 0.7, 0.6, 0.3, 0.2, 0.9, 0.4]], dtype=torch.float32),
    }
    candidate_valid = torch.tensor([[True, True, True, True, True, True, True, True]], dtype=torch.bool)
    candidate_dense_indices = torch.arange(8, dtype=torch.long).unsqueeze(0)
    metas = [{"video_name": "video_a", "window_start_frame": 0}]
    gt_segments = [torch.tensor([[1.0, 5.0]], dtype=torch.float32)]

    selector._write_selected_axis_meta(
        metas=metas,
        selected_positions=selected_positions,
        valid_lengths=valid_lengths,
        selected_output_valid_lengths=selected_output_valid_lengths,
        selected_roles=[["coverage", "boundary", "interior", "boundary"]],
        raw_slot_dense_indices=[[0, 2, 2, 4]],
        raw_slot_duplicate_rates=[0.25],
        raw_slot_unique_counts=[3],
        reader_fill_counts=[1],
        st_active_row_counts=[4],
        reader_outputs=reader_outputs,
        candidate_valid=candidate_valid,
        candidate_dense_indices=candidate_dense_indices,
        gt_segments=gt_segments,
        training=True,
    )

    rows = [json.loads(line) for line in dump_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "video_a|window_start_frame=0"
    assert row["phase"] == "train"
    assert row["selected_dense_indices"] == [0, 2, 4, 6]
    assert row["valid_len"] == 8
    assert row["gt_segments"] == [[1.0, 5.0]]
    assert row["selector_scores"] == pytest.approx([0.1, 0.5, 0.2, 0.7, 0.3, 0.9, 0.4, 0.8])
    assert row["pc_ot_mras_prebackbone_raw_slot_dense_indices"] == [0, 2, 2, 4]
    assert row["pc_ot_mras_prebackbone_reader_fill_count"] == 1
    assert row["pc_ot_mras_prebackbone_st_active_row_count"] == 4
    assert row["irregular_selected_positions"] == [0.0, 2.0, 4.0, 6.0]
    assert row["irregular_dense_valid_len"] == 8
    assert row["irregular_selected_valid_len"] == 8

    monkeypatch.delenv("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL", raising=False)
    monkeypatch.delenv("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_MAX_ROWS", raising=False)


def test_c3_reader_full_train_launcher_prechecks_formal_selector_and_ranking_diagnostics():
    launcher = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch"
    text = launcher.read_text(encoding="utf-8")

    assert "codex/c3-pro-boundary-stability-20260624" in text
    assert "validate_pc_ot_mras_c3_diagnostic_gate.py" in text
    assert "analyze_pc_ot_mras_selector_posttrain_diagnostics.py" in text
    assert "analyze_p2_proposal_localization.py" in text
    assert "analyze_actionformer_post_nms_overload.py" in text
    assert "test_pc_ot_mras_prebackbone_nan_guards.py" in text
    assert "test_pc_ot_mras_prebackbone_c3_pro_stability.py" in text
    assert "test_pc_ot_mras_selector_posttrain_diagnostics.py" in text
    assert "test_pc_ot_mras_c3_diagnostic_gate.py" in text
    assert "tools/bata/validate_pc_ot_mras_c3_diagnostic_gate.py" in text
    assert "tools/bata/analyze_pc_ot_mras_selector_posttrain_diagnostics.py" in text
    assert "tools/bata/analyze_p2_proposal_localization.py" in text
    assert "tools/bata/analyze_actionformer_post_nms_overload.py" in text
    assert "PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL" in text
    assert "C3_SELECTOR_METADATA_JSONL" in text
    assert "converted_result_detection_proposals.jsonl" in text


def test_c3_reader_full_train_launcher_requires_slurm_for_full_train_and_postrun_gate():
    launcher = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch"
    text = launcher.read_text(encoding="utf-8")

    assert "ALLOW_LOGIN_NODE_DEBUG=1 only permits PRECHECK_ONLY=1" in text
    assert 'if [ "$PRECHECK_ONLY" = "0" ] && [ -z "${SLURM_JOB_ID:-}" ]; then' in text
    assert "POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED" in text
    assert "C3_SELECTOR_SUMMARY_JSON" in text
    assert "C3_PROPOSAL_SUMMARY_JSON" in text
    assert "C3_OVERLOAD_SUMMARY_JSON" in text
    assert "C3_DIAGNOSTIC_GATE_JSON" in text
    assert "tools/bata/validate_pc_ot_mras_c3_diagnostic_gate.py" in text
    assert "PREBACKBONE_C3_READER_FULL_TRAIN_PASS_WITH_DIAGNOSTIC_GATE" in text
    assert "running C3 selector diagnostic producer" in text
    assert "running C3 result_detection-to-proposal converter" in text
    assert "running C3 proposal localization diagnostic producer" in text
    assert "running C3 post-NMS overload diagnostic producer" in text
    assert "POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED=1 but C3_SELECTOR_METADATA_JSONL is missing" in text
    assert "full training requires POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED=1" in text
    assert "POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED=0; run is not attribution-ready" not in text
    assert 'C3_SELECTOR_METADATA_JSONL="$RUN_ROOT/selector_metadata.jsonl"' in text
    assert "${C3_SELECTOR_METADATA_JSONL:-" not in text
    assert 'C3_RESULT_DETECTION_PROPOSALS_JSONL="$C3_RESULT_DETECTION_GEOMETRY_DIR/converted_result_detection_proposals.jsonl"' in text
    assert "${C3_RESULT_DETECTION_PROPOSALS_JSONL:-" not in text
    assert 'C3_RESULT_DETECTION_JSON="$FINAL_WORK_DIR/result_detection.json"' in text
    assert "${C3_RESULT_DETECTION_JSON:-" not in text
    assert 'rm -f "$C3_SELECTOR_METADATA_JSONL"' in text
    assert 'grep -Eiq "nan|\\\\binf\\\\b' not in text
    assert "train stdout contains raw prediction/cache marker" in text
    assert "C3_RESULT_DETECTION_JSON" in text
    assert "--provenance-run-root \"$RUN_ROOT\"" in text
    assert "--provenance-work-dir \"$FINAL_WORK_DIR\"" in text
    assert "--provenance-train-stdout \"$TRAIN_STDOUT\"" in text
    assert "--provenance-result-detection-json \"$C3_RESULT_DETECTION_JSON\"" in text
    assert "--expected-run-root \"$RUN_ROOT\"" in text
    assert "--expected-work-dir \"$FINAL_WORK_DIR\"" in text
    assert "--expected-train-stdout \"$TRAIN_STDOUT\"" in text
    assert "--expected-result-detection-json \"$C3_RESULT_DETECTION_JSON\"" in text
