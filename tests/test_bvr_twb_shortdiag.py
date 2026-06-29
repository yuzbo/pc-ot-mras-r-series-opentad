import json
from pathlib import Path

import pytest

from tools.bvr_twb.validate_bvr_twb_shortdiag import (
    EXPECTED_COMMIT,
    REQUIRED_PRETRAIN_PATH,
    ShortDiagValidationError,
    validate_shortdiag,
    validate_train_log,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py"
ROUTE_LABEL = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"


def _fake_repo_root_with_pretrain(tmp_path):
    pretrain = tmp_path / REQUIRED_PRETRAIN_PATH
    pretrain.parent.mkdir(parents=True, exist_ok=True)
    pretrain.write_bytes(b"shortdiag fake pretrain existence marker")
    return tmp_path


def test_shortdiag_config_inherits_bvr_route_and_locks_diagnostic_only(tmp_path):
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.route_label == ROUTE_LABEL
    assert cfg.diagnostic_only is True
    assert cfg.shortdiag_expected_base_commit == EXPECTED_COMMIT
    assert cfg.full_train_unlocked is False
    assert cfg.metric_claim is False
    assert cfg.sparse_compute_claim is False
    assert cfg.dataset.train.pipeline[2].method == "bvr_twb_dynamic_subsample"
    assert cfg.dataset.train.pipeline[2].bvr_twb_adapter_bridge_mode == "adapter_fixed_length_padded_bridge"
    assert cfg.dataset.train.pipeline[2].remap_gt_to_selected_axis is False
    assert cfg.model.backbone.custom.pretrain == REQUIRED_PRETRAIN_PATH
    assert cfg.model.rpn_head.type == "IrregularActionFormerHeadV3"
    assert cfg.model.rpn_head.regression_head_fp32 is True
    assert cfg.model.rpn_head.regression_loss_fp32 is True
    assert cfg.model.rpn_head.filter_invalid_regression_samples is False
    assert float(cfg.model.rpn_head.max_reg_log_distance) == pytest.approx(6.0)
    assert float(cfg.model.rpn_head.min_regression_segment_length) == pytest.approx(1e-6)
    assert cfg.solver.amp is False
    assert cfg.solver.fp16_compress is False

    result = validate_shortdiag(CONFIG, repo_root=_fake_repo_root_with_pretrain(tmp_path))
    assert result["gate_pass"] if "gate_pass" in result else result["config_valid"]
    assert result["full_train_unlocked"] is False
    assert result["metric_claim"] is False
    assert result["sparse_compute_claim"] is False


def test_shortdiag_config_one_epoch_no_checkpoint_no_eval():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert int(cfg.workflow.end_epoch) == 1
    assert cfg.workflow.disable_checkpoint is True
    assert int(cfg.workflow.val_start_epoch) > int(cfg.workflow.end_epoch)
    assert int(cfg.workflow.val_loss_interval) == -1
    assert int(cfg.workflow.val_eval_interval) == -1
    assert int(cfg.workflow.logging_interval) <= 5
    assert int(cfg.workflow.runtime_debug_interval) <= 5
    assert "diagnostic_only" in cfg.work_dir


def test_validator_catches_missing_finite_loss(tmp_path):
    log = tmp_path / "missing_loss.log"
    log.write_text(
        "\n".join(
            [
                ROUTE_LABEL,
                "bvr_twb_dynamic_subsample",
                "adapter_fixed_length_padded_bridge",
                REQUIRED_PRETRAIN_PATH,
                "Loads checkpoint by local backend from path: " + REQUIRED_PRETRAIN_PATH,
                "Training Starts...",
                "Training Over...",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ShortDiagValidationError, match="no finite Loss"):
        validate_train_log(log)


def test_validator_catches_missing_pretrain_warning_and_nan(tmp_path):
    bad_warning = tmp_path / "bad_warning.log"
    bad_warning.write_text(
        "\n".join(
            [
                ROUTE_LABEL,
                "bvr_twb_dynamic_subsample",
                "adapter_fixed_length_padded_bridge",
                "Warning: no pretrain path is provided",
                "[Train]: [000][00001/00002] Loss=1.2345",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ShortDiagValidationError, match="missing_pretrain_warning"):
        validate_train_log(bad_warning)

    bad_nan = tmp_path / "bad_nan.log"
    bad_nan.write_text(
        "\n".join(
            [
                ROUTE_LABEL,
                "bvr_twb_dynamic_subsample",
                "adapter_fixed_length_padded_bridge",
                REQUIRED_PRETRAIN_PATH,
                "Loads checkpoint by local backend from path: " + REQUIRED_PRETRAIN_PATH,
                "[Train]: [000][00001/00002] Loss=nan",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ShortDiagValidationError, match="nan_marker"):
        validate_train_log(bad_nan)


def test_validator_json_contract_stays_claim_locked(tmp_path):
    repo_root = _fake_repo_root_with_pretrain(tmp_path)
    good_log = tmp_path / "good.log"
    good_log.write_text(
        "\n".join(
            [
                ROUTE_LABEL,
                "bvr_twb_dynamic_subsample",
                "adapter_fixed_length_padded_bridge",
                REQUIRED_PRETRAIN_PATH,
                "Loads checkpoint by local backend from path: " + REQUIRED_PRETRAIN_PATH,
                "[Train]: [000][00001/00002] Loss=2.5000 cls_loss=1.0 reg_loss=1.5",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_shortdiag(CONFIG, train_log=good_log, repo_root=repo_root)
    assert result["full_train_unlocked"] is False
    assert result["metric_claim"] is False
    assert result["sparse_compute_claim"] is False
    assert result["train_log_valid"] is True
    json.dumps(result)
