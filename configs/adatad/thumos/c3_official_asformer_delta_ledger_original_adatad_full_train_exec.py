_base_ = ["./c3_official_asformer_delta_ledger_original_adatad_full_train.py"]


c3_asformer_delta_ledger_full_train_gate = dict(
    launch_gate_passed=True,
    reviewed_execution_config=True,
    reviewed_execution_reason="remote PRECHECK_ONLY plus final read-only review blocker fixes",
)
