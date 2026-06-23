import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path


_MISSING = object()


def _get_value(node, key, default=_MISSING):
    if isinstance(node, Mapping):
        return node.get(key, default)

    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                pass

    try:
        return node[key]
    except Exception:
        pass

    cfg_dict = getattr(node, "_cfg_dict", None)
    if cfg_dict is not None and cfg_dict is not node:
        return _get_value(cfg_dict, key, default)

    return getattr(node, key, default)


def _iter_items(node):
    if isinstance(node, Mapping):
        return tuple(node.items())

    items = getattr(node, "items", None)
    if callable(items):
        try:
            return tuple(items())
        except Exception:
            pass

    cfg_dict = getattr(node, "_cfg_dict", None)
    if cfg_dict is not None and cfg_dict is not node:
        return _iter_items(cfg_dict)

    return tuple()


def _iter_option_paths(node, prefix=""):
    if node is None:
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(value, Mapping):
                yield from _iter_option_paths(value, path)
            else:
                yield path
        return
    yield prefix or str(node)


def _is_false(value):
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return False


def _is_true(value):
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _is_gate_name(name):
    return name in {"training_guard", "local_only_gate", "local_only_training_gate"} or name.endswith("_gate")


def _is_pc_ot_mras_gate(gate):
    route = _lower_text(_get_value(gate, "route", _MISSING))
    stage = _lower_text(_get_value(gate, "stage", _MISSING))
    return "pc-ot-mras" in route or "pc_ot_mras" in stage or "pc-ot-mras" in stage


def _as_int(value, default=None):
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _lower_text(value):
    if value in (_MISSING, None):
        return ""
    return str(value).strip().lower()


def _is_smoke_gate(gate_name, gate):
    if _get_value(gate, "smoke_only", _MISSING) is not _MISSING:
        return True
    stage = _lower_text(_get_value(gate, "stage", _MISSING))
    return "smoke" in _lower_text(gate_name) or "smoke" in stage


def _training_block_reason(gate):
    if _is_false(_get_value(gate, "allow_detector_training", _MISSING)):
        return "allow_detector_training=False"
    if _is_true(_get_value(gate, "requires_launch_gate", _MISSING)) and not _is_true(
        _get_value(gate, "launch_gate_passed", _MISSING)
    ):
        return "requires_launch_gate=True and launch_gate_passed!=True"
    for key in ("local_only_training", "local_synthetic_gate_only", "precheck_only"):
        if _is_true(_get_value(gate, key, _MISSING)):
            return f"{key}=True"
    return None


def _smoke_scope_block_reason(cfg, gate_name, gate, entrypoint):
    if not _is_smoke_gate(gate_name, gate):
        return None
    if not _is_true(_get_value(gate, "smoke_only", _MISSING)):
        return "smoke gate requires smoke_only=True"

    entrypoint_text = str(entrypoint)
    if entrypoint_text == "tools/train.py" and not _is_true(_get_value(gate, "allow_tools_train", _MISSING)):
        return "smoke_only=True requires allow_tools_train=True for tools/train.py"
    if entrypoint_text == "tools/test.py":
        if not _is_true(_get_value(gate, "allow_tools_test", _MISSING)):
            return "smoke_only=True forbids tools/test.py when allow_tools_test!=True"
        if not _is_true(_get_value(gate, "allow_detector_map", _MISSING)):
            return "smoke_only=True forbids tools/test.py when allow_detector_map!=True"
    if _is_true(_get_value(gate, "allow_detector_map", _MISSING)):
        return "smoke_only=True forbids allow_detector_map=True"

    allowed_entrypoints = _get_value(gate, "allowed_entrypoints", _MISSING)
    if allowed_entrypoints is not _MISSING:
        if isinstance(allowed_entrypoints, str):
            allowed = {allowed_entrypoints}
        else:
            try:
                allowed = {str(item) for item in allowed_entrypoints}
            except TypeError:
                allowed = set()
        if str(entrypoint) not in allowed:
            return f"smoke_only=True but entrypoint {entrypoint} is not in allowed_entrypoints"

    if _is_true(_get_value(gate, "allow_long_training", _MISSING)):
        return "smoke_only=True requires allow_long_training!=True"

    workflow = _get_value(cfg, "workflow", {})
    max_epochs = _as_int(_get_value(gate, "max_epochs", _MISSING), default=1)
    end_epoch = _as_int(_get_value(workflow, "end_epoch", _MISSING), default=None)
    if end_epoch is None:
        return "smoke_only=True requires workflow.end_epoch"
    if end_epoch > max_epochs:
        return f"workflow.end_epoch={end_epoch} exceeds smoke max_epochs={max_epochs}"

    max_train_iters_limit = _as_int(_get_value(gate, "max_train_iters", _MISSING), default=None)
    max_train_iters = _as_int(_get_value(workflow, "max_train_iters", _MISSING), default=None)
    if max_train_iters_limit is not None:
        if max_train_iters is None:
            return "smoke_only=True requires workflow.max_train_iters"
        if max_train_iters > max_train_iters_limit:
            return f"workflow.max_train_iters={max_train_iters} exceeds smoke max_train_iters={max_train_iters_limit}"

    if _as_int(_get_value(workflow, "val_eval_interval", _MISSING), default=-1) > 0:
        return "smoke_only=True requires workflow.val_eval_interval<=0"
    if _as_int(_get_value(workflow, "val_loss_interval", _MISSING), default=-1) > 0:
        return "smoke_only=True requires workflow.val_loss_interval<=0"
    if _is_true(_get_value(gate, "disable_checkpoint", _MISSING)) and not _is_true(
        _get_value(workflow, "disable_checkpoint", _MISSING)
    ):
        return "smoke_only=True requires workflow.disable_checkpoint=True when gate.disable_checkpoint=True"

    inference = _get_value(cfg, "inference", {})
    if _is_true(_get_value(inference, "load_from_raw_predictions", _MISSING)):
        return "smoke_only=True forbids inference.load_from_raw_predictions=True"
    if _is_true(_get_value(inference, "save_raw_prediction", _MISSING)):
        return "smoke_only=True forbids inference.save_raw_prediction=True"

    return None


def _entrypoint_scope_block_reason(gate, entrypoint):
    entrypoint_text = str(entrypoint)

    if entrypoint_text == "tools/train.py" and _is_false(_get_value(gate, "allow_tools_train", _MISSING)):
        return "allow_tools_train=False forbids tools/train.py"
    if entrypoint_text == "tools/test.py":
        if _is_false(_get_value(gate, "allow_tools_test", _MISSING)):
            return "allow_tools_test=False forbids tools/test.py"
        if _is_false(_get_value(gate, "allow_detector_map", _MISSING)):
            return "allow_detector_map=False forbids tools/test.py"

    allowed_entrypoints = _get_value(gate, "allowed_entrypoints", _MISSING)
    if allowed_entrypoints is not _MISSING:
        if isinstance(allowed_entrypoints, str):
            allowed = {allowed_entrypoints}
        else:
            try:
                allowed = {str(item) for item in allowed_entrypoints}
            except TypeError:
                allowed = set()
        if entrypoint_text not in allowed:
            return f"entrypoint {entrypoint} is not in allowed_entrypoints"

    return None


def _format_detail(value):
    if value in (_MISSING, None, ""):
        return None
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_training_block_error(gate_name, gate, reason, entrypoint):
    details = []
    for key in ("route", "stage", "reviewed_predecessor"):
        detail = _format_detail(_get_value(gate, key, _MISSING))
        if detail is not None:
            details.append(f"{key}={detail}")
    detail_text = f" ({'; '.join(details)})" if details else ""

    allowed = _format_detail(_get_value(gate, "allowed_checks", _MISSING))
    allowed_text = f" Allowed checks: {allowed}." if allowed else ""

    return (
        f"{entrypoint} is blocked by local-only config gate '{gate_name}'{detail_text}: {reason}. "
        "This guard runs after config loading and before DDP, dataset, model, or runner construction. "
        "Use the local/static/synthetic checks recorded by the gate, or create a separately reviewed "
        "training config without the local-only detector-training block."
        f"{allowed_text}"
    )


def _has_pc_ot_mras_gate(cfg):
    return any(_is_pc_ot_mras_gate(gate) for _, gate in _iter_candidate_gates(cfg))


def assert_safe_cfg_options_for_gated_config(cfg, cfg_options, entrypoint="tools/train.py"):
    """Reject CLI config overrides that can mutate PC-OT-MRAS gate boundaries."""
    if not cfg_options or not _has_pc_ot_mras_gate(cfg):
        return

    safe_exact = {
        "work_dir",
        "model.projection.pretrained",
        "dataset.train.ann_file",
        "dataset.val.ann_file",
        "dataset.test.ann_file",
        "dataset.train.class_map",
        "dataset.val.class_map",
        "dataset.test.class_map",
        "dataset.train.data_path",
        "dataset.val.data_path",
        "dataset.test.data_path",
        "evaluation.ground_truth_filename",
    }
    unsafe_fragments = (
        "_gate",
        "training_guard",
        "local_only",
        "allow_",
        "allowed_checks",
        "forbidden_checks",
        "raw_prediction",
        "prediction_cache",
        "checkpoint",
        "resume",
        "load_from",
        "teacher",
        "oracle",
        "workflow.val_eval_interval",
        "workflow.val_start_epoch",
        "workflow.max_train_iters",
        "workflow.end_epoch",
        "tools_train",
        "tools_test",
        "detector_map",
        "metric_claim",
        "paper_claim",
        "runtime_flops",
        "deploy_claim",
        "scanner_quality",
        "dynamic_budget",
    )

    bad_paths = []
    for path in _iter_option_paths(cfg_options):
        path = str(path)
        path_lower = path.lower()
        if path in safe_exact:
            continue
        if any(fragment in path_lower for fragment in unsafe_fragments):
            bad_paths.append(path)
            continue
        bad_paths.append(path)

    if bad_paths:
        joined = ", ".join(sorted(bad_paths))
        raise RuntimeError(
            f"{entrypoint} rejected unsafe --cfg-options for PC-OT-MRAS gated config: {joined}. "
            "Use the reviewed launcher allowlist for runtime paths only; gate, workflow, "
            "checkpoint, raw-prediction, metric, and claim fields are immutable."
        )


def assert_safe_entrypoint_args_for_gated_config(cfg, args, entrypoint="tools/train.py"):
    """Reject CLI entrypoint arguments that bypass gated-config launchers."""
    if not _has_pc_ot_mras_gate(cfg):
        return

    if getattr(args, "resume", None) is not None:
        raise RuntimeError(
            f"{entrypoint} rejected --resume for PC-OT-MRAS gated config before DDP, dataset, "
            "model, or checkpoint access. Use a separately reviewed launcher/gate for any resume path."
        )


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entrypoint_gate_context_block_reason(gate):
    context = _get_value(gate, "entrypoint_gate_context", _MISSING)
    if context in (_MISSING, None) or not _is_true(_get_value(context, "required", _MISSING)):
        return None

    gate_json_env = str(_get_value(context, "gate_json_env", "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON"))
    gate_sha_env = str(_get_value(context, "gate_sha256_env", "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256"))
    manifest_env = str(
        _get_value(context, "active_manifest_sha256_env", "OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256")
    )
    resolved_env = str(
        _get_value(context, "resolved_config_sha256_env", "OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256")
    )

    gate_json_path = os.environ.get(gate_json_env)
    gate_sha256 = os.environ.get(gate_sha_env)
    active_manifest_sha256 = os.environ.get(manifest_env)
    resolved_config_sha256 = os.environ.get(resolved_env)

    if not gate_json_path:
        return f"missing required entrypoint gate env {gate_json_env}"
    if not gate_sha256:
        return f"missing required entrypoint gate env {gate_sha_env}"
    if not active_manifest_sha256:
        return f"missing required entrypoint gate env {manifest_env}"
    if _is_true(_get_value(context, "require_resolved_config_sha256", True)) and not resolved_config_sha256:
        return f"missing required entrypoint gate env {resolved_env}"

    gate_path = Path(gate_json_path)
    if not gate_path.is_file():
        return f"entrypoint gate JSON does not exist: {gate_json_path}"
    actual_sha256 = _sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        return f"entrypoint gate JSON sha256 mismatch: expected={gate_sha256} actual={actual_sha256}"

    try:
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"entrypoint gate JSON is not valid JSON: {exc}"

    allowed_decisions = _get_value(context, "allowed_decisions", _MISSING)
    if allowed_decisions is not _MISSING:
        if isinstance(allowed_decisions, str):
            allowed = {allowed_decisions}
        else:
            allowed = {str(item) for item in allowed_decisions}
        if str(gate_payload.get("decision")) not in allowed:
            return f"entrypoint gate decision is not allowed: {gate_payload.get('decision')}"

    expected_manifest = gate_payload.get("active_sha256_manifest_sha256") or gate_payload.get(
        "expected_active_sha256_manifest_sha256"
    )
    if expected_manifest != active_manifest_sha256:
        return (
            "entrypoint gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    require_resolved_config_sha256 = _is_true(_get_value(context, "require_resolved_config_sha256", True))
    expected_resolved = gate_payload.get("resolved_config_sha256") or gate_payload.get(
        "expected_resolved_config_sha256"
    )
    if require_resolved_config_sha256 and expected_resolved is None:
        return "entrypoint gate JSON missing resolved_config_sha256"
    if resolved_config_sha256 and expected_resolved != resolved_config_sha256:
        return (
            "entrypoint gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    forbidden_true = _get_value(context, "forbidden_true_keys", _MISSING)
    if forbidden_true is not _MISSING:
        for key in forbidden_true:
            if gate_payload.get(str(key)) is True:
                return f"entrypoint gate must not set {key}=true"

    if _is_true(_get_value(context, "strict_payload_validation", _MISSING)):
        exact_values = _get_value(context, "required_exact_values", {})
        for key, expected in _iter_items(exact_values):
            if gate_payload.get(str(key)) != expected:
                return f"entrypoint gate must set {key}={expected!r}"

        required_true = _get_value(context, "required_true_keys", _MISSING)
        if required_true is not _MISSING:
            for key in required_true:
                if gate_payload.get(str(key)) is not True:
                    return f"entrypoint gate must set {key}=true"

        if forbidden_true is not _MISSING:
            for key in forbidden_true:
                if str(key) in gate_payload and gate_payload[str(key)] is not False:
                    return f"entrypoint gate must keep {key}=false/absent; got {gate_payload[str(key)]!r}"

        if _get_value(context, "unknown_key_policy", _MISSING) == "reject_unknown_except_explicit_harmless_metadata":
            harmless = _get_value(context, "harmless_metadata_keys", ())
            allowed_keys = {
                "decision",
                "route",
                "active_sha256_manifest_sha256",
                "expected_active_sha256_manifest_sha256",
                "resolved_config_sha256",
                "expected_resolved_config_sha256",
            }
            for key, _ in _iter_items(exact_values):
                allowed_keys.add(str(key))
            if required_true is not _MISSING:
                allowed_keys.update(str(key) for key in required_true)
            if forbidden_true is not _MISSING:
                allowed_keys.update(str(key) for key in forbidden_true)
            allowed_keys.update(str(key) for key in harmless)
            for key in gate_payload:
                if str(key) not in allowed_keys:
                    return f"entrypoint gate contains unknown or unallowlisted key: {key}"

    return None


def _iter_candidate_gates(cfg):
    direct = _get_value(cfg, "allow_detector_training", _MISSING)
    if direct is not _MISSING:
        yield "<top-level>", cfg

    for name, value in _iter_items(cfg):
        if isinstance(name, str) and _is_gate_name(name):
            yield name, value


def assert_detector_training_allowed(cfg, entrypoint="tools/train.py"):
    """Fail closed when a config explicitly marks detector training as locked."""
    for gate_name, gate in _iter_candidate_gates(cfg):
        reason = _training_block_reason(gate)
        if reason is not None:
            raise RuntimeError(_format_training_block_error(gate_name, gate, reason, entrypoint))
        reason = _entrypoint_scope_block_reason(gate, entrypoint)
        if reason is not None:
            raise RuntimeError(_format_training_block_error(gate_name, gate, reason, entrypoint))
        reason = _smoke_scope_block_reason(cfg, gate_name, gate, entrypoint)
        if reason is not None:
            raise RuntimeError(_format_training_block_error(gate_name, gate, reason, entrypoint))
        reason = _entrypoint_gate_context_block_reason(gate)
        if reason is not None:
            raise RuntimeError(_format_training_block_error(gate_name, gate, reason, entrypoint))
