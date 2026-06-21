import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "verify_pc_ot_mras_execution_matrix.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_execution_matrix", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(stage, *, worthy=False, training_status="not_started"):
    return {
        "stage": stage,
        "experiment_model": f"{stage} model",
        "changed_surface": ["unit_surface"],
        "innovation_purpose": f"{stage} innovation purpose",
        "implementation_summary": f"{stage} implementation summary",
        "full_training_worthy_now": worthy,
        "remote_sync_status": "unit_status",
        "training_status": training_status,
        "next_gate": f"{stage} next gate",
    }


def _manifest(tmp_path, *, mutate=None):
    tool = _load_tool()
    models = [
        _model(stage, worthy=stage in {"R17", "R18"}, training_status="completed_formal_train" if stage in {"R17", "R18"} else "not_started")
        for stage in tool.REQUIRED_STAGES
    ]
    payload = {
        "manifest_status": "unit_manifest",
        "analysis_status": {
            "c1_remote_sync_plan": {
                "status": "PLAN_READY_NOT_EXECUTED",
                "remote_sync_allowed": False,
            },
            "c1_reader_disabled_submission_plan": {
                "status": "PLAN_READY_NOT_EXECUTED",
                "slurm_allowed": False,
                "tools_test_allowed": False,
                "detector_map_reporting_allowed": False,
            },
        },
        "active_full_training_jobs": [
            {"stage": "R17", "state": "COMPLETED_TRAINING_OVER"},
            {"stage": "R18", "state": "COMPLETED_TRAINING_OVER"},
        ],
        "models": models,
    }
    if mutate:
        mutate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_execution_matrix_verifier_exports_complete_rows(tmp_path):
    tool = _load_tool()
    result = tool.verify_execution_matrix(_manifest(tmp_path))

    assert result["pass"] is True
    assert result["model_count"] == len(tool.REQUIRED_STAGES)
    assert result["required_stage_count"] == len(tool.REQUIRED_STAGES)
    assert result["full_training_worthy_now"] == ["R17", "R18"]
    assert result["current_execution_conclusion"] == "no_new_full_training_unlocked"
    assert result["c1_control_status"]["remote_sync_plan"] == "PLAN_READY_NOT_EXECUTED"
    assert result["c1_unlock_flags"]["slurm_allowed"] is False
    assert len(result["rows"]) == len(tool.REQUIRED_STAGES)
    assert all(row["innovation_purpose"] for row in result["rows"])
    assert all(row["implementation_summary"] for row in result["rows"])


def test_execution_matrix_verifier_rejects_missing_required_stage(tmp_path):
    tool = _load_tool()

    def mutate(payload):
        payload["models"] = [row for row in payload["models"] if row["stage"] != "R35"]

    result = tool.verify_execution_matrix(_manifest(tmp_path, mutate=mutate))

    assert result["pass"] is False
    assert result["errors"]["missing_stages"] == ["R35"]


def test_execution_matrix_verifier_rejects_unexpected_full_training_worthy(tmp_path):
    tool = _load_tool()

    def mutate(payload):
        for row in payload["models"]:
            if row["stage"] == "R33":
                row["full_training_worthy_now"] = True
                row["training_status"] = "not_started"

    result = tool.verify_execution_matrix(_manifest(tmp_path, mutate=mutate))

    assert result["pass"] is False
    assert result["errors"]["unexpected_full_training_worthy"] == ["R33"]
    assert result["errors"]["unfinished_full_training"] == ["R33"]


def test_execution_matrix_verifier_rejects_empty_required_fields(tmp_path):
    tool = _load_tool()

    def mutate(payload):
        for row in payload["models"]:
            if row["stage"] == "R20":
                row["innovation_purpose"] = ""
                row["changed_surface"] = []

    result = tool.verify_execution_matrix(_manifest(tmp_path, mutate=mutate))

    assert result["pass"] is False
    assert {"stage": "R20", "error": "empty_field", "field": "innovation_purpose"} in result["errors"]["row_errors"]
    assert {"stage": "R20", "error": "invalid_changed_surface"} in result["errors"]["row_errors"]


def test_execution_matrix_verifier_rejects_c1_unexpected_unlock(tmp_path):
    tool = _load_tool()

    def mutate(payload):
        payload["analysis_status"]["c1_reader_disabled_submission_plan"]["tools_test_allowed"] = True

    result = tool.verify_execution_matrix(_manifest(tmp_path, mutate=mutate))

    assert result["pass"] is False
    assert result["errors"]["c1_unexpected_unlocks"] == ["tools_test_allowed"]
