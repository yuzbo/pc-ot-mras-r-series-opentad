from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"

CASES = (
    dict(
        label="R17_C2",
        config="ctf_bdi_pc_ot_mras_r17_c2_selected_times_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py",
        metadata_position_source="selected_times",
        metadata_position_repair="sort_jitter",
        no_gate_scale=False,
    ),
    dict(
        label="R18_C2",
        config="ctf_bdi_pc_ot_mras_r18_c2_selected_times_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r18_post_train_eval_local_lowmem_candidate.py",
        metadata_position_source="selected_times",
        metadata_position_repair="sort_jitter",
        no_gate_scale=False,
    ),
    dict(
        label="R17_C3",
        config="ctf_bdi_pc_ot_mras_r17_c3_no_gate_scale_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py",
        metadata_position_source=None,
        metadata_position_repair=None,
        no_gate_scale=True,
    ),
    dict(
        label="R18_C3",
        config="ctf_bdi_pc_ot_mras_r18_c3_no_gate_scale_eval_local_lowmem_candidate.py",
        parent="ctf_bdi_pc_ot_mras_r18_post_train_eval_local_lowmem_candidate.py",
        metadata_position_source=None,
        metadata_position_repair=None,
        no_gate_scale=True,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=[case["label"] for case in CASES])
def test_c2_c3_eval_ablation_configs_are_local_lowmem_thin_entries(case):
    mmengine_config = pytest.importorskip("mmengine.config")
    config_path = CONFIG_DIR / case["config"]
    parent_path = CONFIG_DIR / case["parent"]

    text = config_path.read_text(encoding="utf-8")
    assert f'_base_ = ["{case["parent"]}"]' in text

    cfg = mmengine_config.Config.fromfile(str(config_path))
    parent = mmengine_config.Config.fromfile(str(parent_path))

    assert cfg.solver.test.batch_size == 1
    assert cfg.solver.test.num_workers == 0
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.neck.allocation_key == "acquisition_matrix"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.work_dir != parent.work_dir

    if case["metadata_position_source"] is None:
        assert "metadata_position_source" not in cfg.model.neck
    else:
        assert cfg.model.neck.metadata_position_source == case["metadata_position_source"]

    if case["metadata_position_repair"] is None:
        assert "metadata_position_repair" not in cfg.model.neck
    else:
        assert cfg.model.neck.metadata_position_repair == case["metadata_position_repair"]

    if case["no_gate_scale"]:
        assert cfg.model.neck.no_gate_scale is True
    else:
        assert "no_gate_scale" not in cfg.model.neck
