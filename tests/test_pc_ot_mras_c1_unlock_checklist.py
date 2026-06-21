import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "build_pc_ot_mras_c1_unlock_checklist.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_unlock_checklist", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage(name):
    return {"name": name, "executable_now": False}


def _handoff():
    return {
        "schema_version": "pc_ot_mras_c1_execution_handoff_v0",
        "generated_at": "2026-06-22T06:08:51+08:00",
        "status": "HANDOFF_READY_NOT_EXECUTABLE",
        "pass": True,
        "remote_root": "/data/run01/sczc063/yuzibo/OpenTAD_PCOTMRAS_CPU_PRERUN_20260620_1418",
        "execution_allowed_now": False,
        "execution_unlock": {
            "remote_sync_allowed": False,
            "formal_gate_generation_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "runtime_flops_claim_allowed": False,
            "deploy_claim_allowed": False,
        },
        "handoff_stages": [
            _stage("review_replacement_or_user_unlock"),
            _stage("remote_sync_after_unlock"),
            _stage("remote_precheck_after_sync"),
            _stage("formal_gate_generation_after_precheck"),
            _stage("slurm_reader_disabled_eval_submission"),
        ],
    }


def test_unlock_checklist_ready_but_no_execution():
    tool = _load_tool()
    payload = tool.build_unlock_checklist(
        handoff=_handoff(),
        generated_at="2026-06-22T06:20:00+08:00",
    )

    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["status"] == tool.READY
    assert payload["pass"] is True
    assert payload["execution_allowed_now"] is False
    assert payload["current_blocking_item"] == "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate"
    assert payload["all_checklist_items_satisfied"] is False
    assert payload["execution_unlock"]["remote_sync_allowed"] is False
    assert payload["execution_unlock"]["slurm_allowed"] is False
    assert [item["key"] for item in payload["checklist"]][:3] == [
        "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate",
        "tracker_manifest_unlocked_for_c1_remote_sync",
        "post_sync_remote_sha_verification_saved",
    ]


def test_unlock_checklist_rejects_executable_handoff():
    tool = _load_tool()
    handoff = _handoff()
    handoff["execution_allowed_now"] = True

    payload = tool.build_unlock_checklist(handoff=handoff)

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "handoff_ready_not_executable" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["remote_sync_allowed"] is False


def test_unlock_checklist_rejects_unlocked_flag():
    tool = _load_tool()
    handoff = _handoff()
    handoff["execution_unlock"]["tools_test_allowed"] = True

    payload = tool.build_unlock_checklist(handoff=handoff)

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "all_execution_unlocks_false" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["tools_test_allowed"] is False


def test_unlock_checklist_rejects_wrong_stage_order():
    tool = _load_tool()
    handoff = _handoff()
    handoff["handoff_stages"] = list(reversed(handoff["handoff_stages"]))

    payload = tool.build_unlock_checklist(handoff=handoff)

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "expected_stage_sequence_present" for item in payload["failed_checks"])


def test_unlock_checklist_rejects_executable_stage():
    tool = _load_tool()
    handoff = _handoff()
    handoff["handoff_stages"][1]["executable_now"] = True

    payload = tool.build_unlock_checklist(handoff=handoff)

    assert payload["status"] == tool.FAILED
    failed = next(item for item in payload["failed_checks"] if item["name"] == "all_handoff_stages_non_executable")
    assert failed["detail"]["executable_stage_offenders"] == ["remote_sync_after_unlock"]
