from collections.abc import Mapping


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
