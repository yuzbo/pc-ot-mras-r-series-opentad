import json
import re
from pathlib import Path

import numpy as np


INTEGER_TEXT_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)$")


FORBIDDEN_VALUE_TRANSPORT_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_checkpoint",
    "prediction_uses_gt",
)


def _is_true(value):
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _strict_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if INTEGER_TEXT_RE.fullmatch(text):
            return int(text)
    raise ValueError(f"{name} must be an integer")


def validate_value_transport_selection_row(row, line_no=0, require_deployable=True):
    if not isinstance(row, dict):
        raise ValueError(f"line {line_no}: value-transport row must be a JSON object")
    if row.get("selected_positions_unit") != "local_dense_index":
        raise ValueError(f"line {line_no}: selected_positions_unit must be local_dense_index")

    raw_positions = row.get("selected_positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError(f"line {line_no}: selected_positions must be a non-empty list")
    positions = []
    for idx, value in enumerate(raw_positions):
        positions.append(_strict_int(value, f"line {line_no}: selected_positions[{idx}]"))
    if positions != sorted(positions):
        raise ValueError(f"line {line_no}: selected_positions must be sorted")
    if len(set(positions)) != len(positions):
        raise ValueError(f"line {line_no}: selected_positions must be unique")
    if positions[0] < 0:
        raise ValueError(f"line {line_no}: selected_positions must be non-negative")

    selected_count = row.get("selected_count")
    if selected_count is not None and _strict_int(selected_count, f"line {line_no}: selected_count") != len(positions):
        raise ValueError(f"line {line_no}: selected_count does not match selected_positions")
    valid_len = row.get("valid_len")
    dense_len = row.get("dense_len")
    if valid_len is not None and positions[-1] >= _strict_int(valid_len, f"line {line_no}: valid_len"):
        raise ValueError(f"line {line_no}: selected_positions exceed valid_len")
    if dense_len is not None and positions[-1] >= _strict_int(dense_len, f"line {line_no}: dense_len"):
        raise ValueError(f"line {line_no}: selected_positions exceed dense_len")

    for key in FORBIDDEN_VALUE_TRANSPORT_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden deploy-invisible flag {key}=true")
    if require_deployable and not bool(row.get("deploy_selection_ledger", False)):
        raise ValueError(f"line {line_no}: deployable value-transport rows must set deploy_selection_ledger=true")
    if require_deployable and bool(row.get("diagnostic_only", False)):
        raise ValueError(f"line {line_no}: deployable value-transport rows must not be diagnostic_only")
    return np.asarray(positions, dtype=np.int64)


def load_value_transport_selection_ledger(path, require_deployable=True):
    if path is None:
        raise ValueError("value-transport ledger path is required")
    ledger_path = Path(path).expanduser()
    if not ledger_path.is_file():
        raise FileNotFoundError(f"value-transport ledger not found: {ledger_path}")

    rows = {}
    with ledger_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            positions = validate_value_transport_selection_row(
                row,
                line_no=line_no,
                require_deployable=bool(require_deployable),
            )
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"line {line_no}: sample_id must be a non-empty string")
            if sample_id in rows:
                raise ValueError(f"line {line_no}: duplicate sample_id={sample_id}")
            checked = dict(row)
            checked["selected_positions"] = positions
            rows[sample_id] = checked
    if not rows:
        raise ValueError(f"value-transport ledger has no rows: {ledger_path}")
    return rows
