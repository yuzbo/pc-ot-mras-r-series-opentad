import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_e2e_frame_acquisition_actionformer_adapter_fixed50_candidate_n16r4.py"
)
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
READER_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py"
ACTIONFORMER_PATH = ROOT / "opentad" / "models" / "detectors" / "actionformer.py"


def _load_cfg():
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(CONFIG))


def _source(path):
    return path.read_text(encoding="utf-8")


def _tree(path):
    return ast.parse(_source(path), filename=str(path))


def _find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _find_method(class_node, name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found")


def _call_lines(method_node, predicate):
    lines = []
    for node in ast.walk(method_node):
        if isinstance(node, ast.Call) and predicate(node):
            lines.append(node.lineno)
    return sorted(lines)


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _contains_attr_call(call, attr):
    return isinstance(call.func, ast.Attribute) and call.func.attr == attr


def _first_line(method_node, predicate, label):
    lines = _call_lines(method_node, predicate)
    assert lines, f"{label} call not found in {method_node.name}"
    return lines[0]


def test_config_main_chain_is_prebackbone_frame_selector_not_token_bridge():
    cfg = _load_cfg()
    model_text = repr(cfg.model)
    pipeline_text = repr(cfg.dataset).lower()

    assert cfg.experiment_scope.selection_surface == "pre_backbone_raw_frame"
    assert cfg.experiment_scope.selection_timing == "online_before_backbone"
    assert cfg.experiment_scope.uses_offline_ledger is False
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"

    assert "frame_selector" in cfg.model
    assert cfg.model.frame_selector.type == "PCOTMRASPreBackboneFrameSelector"
    assert cfg.model.frame_selector.target_len == cfg.window_size == 384
    assert cfg.model.frame_selector.dense_window_size == cfg.dense_window_size == 768
    assert cfg.experiment_scope.budget_protocol == "fixed384_frame_slot_candidate"
    assert cfg.experiment_scope.s80r16_cell96x4_enabled is False
    assert "future route" in cfg.experiment_scope.s80r16_cell96x4_note.lower()
    assert "not" in cfg.experiment_scope.s80r16_cell96x4_note.lower()
    assert cfg.model.backbone.backbone.total_frames == cfg.window_size
    assert cfg.model.projection.max_seq_len == cfg.window_size

    assert "pc_ot_mras_reader" not in cfg.model
    assert "token_compressor" not in cfg.model
    assert cfg.model.get("neck", {}).get("type") != "PCOTMRASDetectorBridge"
    assert "PCOTMRASDetectorBridge" not in model_text
    assert "post_projection" not in model_text
    assert "offline_ledger" not in model_text
    assert "value_transport_ledger" not in model_text
    assert "bata_value_transport_ledger_subsample" not in pipeline_text
    assert "hard_positions" not in pipeline_text
    assert "teacher" not in pipeline_text
    assert "oracle" not in pipeline_text


def test_prebackbone_transport_protocol_is_hard_top1_for_train_and_eval():
    cfg = _load_cfg()
    frame_selector = cfg.model.frame_selector
    selector_source = _source(SELECTOR_PATH)

    assert cfg.experiment_scope.first_version_forward_contract == "hard_top1_train_eval"
    assert cfg.experiment_scope.selector_gradient == "hard_top1_st_detector_loss_and_train_gt_acquisition_aux"
    assert frame_selector.transport_topk == 1
    assert frame_selector.eval_transport_topk == 1
    assert frame_selector.transport_topk == frame_selector.eval_transport_topk
    assert frame_selector.straight_through_detector_loss is True
    assert "hard + soft_surrogate" in selector_source
    assert "soft_surrogate[" in selector_source
    assert ".detach()" in selector_source
    assert "frame_proxy = flat.detach().mean" in selector_source
    assert "torch.bmm(transport_weights, frame_proxy)" in selector_source
    assert "torch.bmm(transport_weights, flat)" not in selector_source


def test_actionformer_calls_frame_selector_before_backbone_projection_reader_and_head():
    actionformer = _find_class(_tree(ACTIONFORMER_PATH), "ActionFormer")

    for method_name, selector_call in (
        ("forward_train", "forward_train"),
        ("forward_test", "forward_test"),
    ):
        method = _find_method(actionformer, method_name)
        frame_selector_line = _first_line(
            method,
            lambda call: (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == selector_call
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "frame_selector"
            ),
            "frame_selector",
        )
        backbone_line = _first_line(
            method,
            lambda call: (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "backbone"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
            ),
            "backbone",
        )
        projection_line = _first_line(
            method,
            lambda call: (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "projection"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
            ),
            "projection",
        )
        reader_injection_line = _first_line(
            method,
            lambda call: _contains_attr_call(call, "_inject_pc_ot_mras_reader_outputs"),
            "reader injection",
        )
        head_line = _first_line(
            method,
            lambda call: _call_name(call).startswith("_call_rpn_head_forward"),
            "rpn head",
        )

        assert frame_selector_line < backbone_line
        assert frame_selector_line < projection_line
        assert frame_selector_line < reader_injection_line
        assert frame_selector_line < head_line


def test_selector_source_consumes_raw_frames_and_does_not_read_projection_tokens():
    tree = _tree(SELECTOR_PATH)
    selector = _find_class(tree, "PCOTMRASPreBackboneFrameSelector")
    source = _source(SELECTOR_PATH)

    assert "Online PC-OT-MRAS frame acquisition before the video backbone" in source
    assert "inputs: torch.Tensor" in source
    assert "def _scout_frame_features" in source
    assert "def _compressed_pixel_features" in source
    assert "def _raw_frame_descriptors" in source
    assert "_video_tensor_for_descriptors(inputs)" in source
    assert 'scout_feature_source == "compressed_pixels"' in source
    assert "F.interpolate" in source
    assert "video.mean" in source
    assert "video.std" in source
    assert "motion[:, 1:]" in source
    assert "edge_delta[:, 1:]" in source
    assert "motion_delta[:, 1:]" in source
    assert "coords.unsqueeze(-1)" in source
    assert "descriptors = torch.cat" in source
    assert "_fit_descriptor_width(descriptors, width=self.descriptor_dim)" in source

    select = _find_method(selector, "_select")
    scout_feature_line = _first_line(
        select,
        lambda call: _contains_attr_call(call, "_scout_frame_features"),
        "scout-frame feature",
    )
    reader_line = _first_line(
        select,
        lambda call: isinstance(call.func, ast.Attribute)
        and call.func.attr == "reader"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self",
        "reader",
    )
    transport_line = _first_line(
        select,
        lambda call: _contains_attr_call(call, "_apply_sparse_transport"),
        "sparse transport",
    )
    assert scout_feature_line < reader_line < transport_line

    forbidden_identifier_fragments = ("projection", "post_projection", "backbone_feature", "token_feature")
    identifiers = []
    for node in ast.walk(selector):
        if isinstance(node, ast.Name):
            identifiers.append(node.id)
        elif isinstance(node, ast.arg):
            identifiers.append(node.arg)
        elif isinstance(node, ast.Attribute):
            identifiers.append(node.attr)
    offenders = [
        item
        for item in identifiers
        if any(fragment in item.lower() for fragment in forbidden_identifier_fragments)
    ]
    assert offenders == []


def test_selector_outputs_frame_tensor_masks_and_dense_index_metadata():
    source = _source(SELECTOR_PATH)

    assert '"inputs": outputs["inputs"]' in source
    assert '"masks": outputs["masks"]' in source
    assert '"metas": outputs["metas"]' in source
    assert '"inputs": selected_inputs' in source
    assert "selected_masks = output_axis < plan" in source
    assert '"masks": selected_masks' in source
    assert '"selected_positions": plan["selected_positions"]' in source
    assert 'meta["irregular_selected_positions"]' in source
    assert 'meta["irregular_selected_valid_len"]' in source
    assert 'meta["irregular_native_axis"] = False' in source
    assert 'meta["pc_ot_mras_prebackbone_selected_dense_indices"]' in source
    assert 'meta["pc_ot_mras_prebackbone_selector_source"]' in source


def test_gate_forbids_test_time_shortcuts_but_keeps_train_only_gt_remap_allowed():
    cfg = _load_cfg()
    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    context = gate.entrypoint_gate_context
    forbidden = set(context.forbidden_true_keys)

    for key in (
        "tools_test",
        "allow_tools_test",
        "detector_map",
        "allow_detector_map",
        "offline_ledger",
        "frozen_reader_ledger",
        "post_projection_bridge",
        "raw_prediction_cache",
        "prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "uses_teacher",
        "uses_oracle",
        "uses_raw_prediction",
        "metric_claim",
        "paper_claim",
        "runtime_flops_claim",
        "deploy_claim",
    ):
        assert key in forbidden

    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    frame_selector = cfg.model.frame_selector
    assert frame_selector.remap_gt_to_selected_axis is True
    assert frame_selector.aux_gt_acquisition_loss_weight > 0.0
    assert "gt_segments" in _source(SELECTOR_PATH)
    assert "gt_labels" in _source(SELECTOR_PATH)
    assert "def forward_test(self, inputs, masks, metas=None)" in _source(SELECTOR_PATH)


def test_irregular_temporal_metadata_contract_is_not_uniform_axis_only():
    selector_source = _source(SELECTOR_PATH)
    actionformer_source = _source(ACTIONFORMER_PATH)

    required_selector_keys = (
        "irregular_selected_positions",
        "irregular_selected_valid_len",
        "irregular_native_axis",
        "pc_ot_mras_prebackbone_selected_dense_indices",
    )
    for key in required_selector_keys:
        assert key in selector_source

    assert "metas = selector_outputs.get(\"metas\", metas)" in actionformer_source
    assert "_call_rpn_head_forward_train" in actionformer_source
    assert "_call_rpn_head_forward_test" in actionformer_source
    assert "metas=metas" in actionformer_source


def test_selector_reader_guard_and_scout_knobs_are_explicit_in_config():
    cfg = _load_cfg()
    frame_selector = cfg.model.frame_selector
    reader = frame_selector.reader
    reader_source = _source(READER_PATH)

    assert int(frame_selector.descriptor_dim) > 0
    assert int(reader.in_dim) == int(frame_selector.descriptor_dim)
    assert frame_selector.transport_topk == 1
    assert frame_selector.eval_transport_topk == 1
    assert frame_selector.transport_topk == frame_selector.eval_transport_topk
    assert frame_selector.straight_through_detector_loss is True
    assert frame_selector.reader_regularizer_loss_weight > 0.0

    contract_gaps = {}
    if "if int(descriptor_dim) != 4" in _source(SELECTOR_PATH) and int(frame_selector.descriptor_dim) != 4:
        contract_gaps["descriptor_dim_source_config_mismatch"] = (
            "config requests descriptor_dim="
            f"{int(frame_selector.descriptor_dim)} but selector source still rejects non-4 descriptors"
        )
    if "coverage_guard_count" in _source(SELECTOR_PATH) and "coverage_guard_count" not in frame_selector:
        contract_gaps["coverage_guard_count"] = "selector coverage guard count must be explicit in config"

    assert "column_cap" in reader_source
    assert "enable_value_heads" in reader_source
    required_explicit_reader_knobs = {
        "column_cap": "coverage/column-mass guard must not rely on an implicit reader default",
        "enable_value_heads": "reader value-head opt-in/out must be explicit for leakage attribution",
    }
    contract_gaps.update({
        key: reason
        for key, reason in required_explicit_reader_knobs.items()
        if key not in reader
    })
    assert contract_gaps == {}
    assert isinstance(reader.enable_value_heads, bool)
