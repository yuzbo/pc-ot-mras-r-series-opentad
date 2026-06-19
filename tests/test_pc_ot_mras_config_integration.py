import json
import os
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_synthetic_local.py"


def _load_config_namespace():
    assert CONFIG.exists()
    return runpy.run_path(str(CONFIG))


def test_pc_ot_mras_synthetic_config_declares_local_only_contract():
    cfg = _load_config_namespace()
    text = CONFIG.read_text(encoding="utf-8")
    lower = text.lower()

    assert cfg["local_synthetic_only"] is True
    safety = cfg["safety_boundary"]
    for key in (
        "local",
        "synthetic",
        "no_map",
        "no_remote",
        "no_slurm",
        "no_gpu",
        "no_training",
        "no_dataset",
        "no_checkpoint",
    ):
        assert safety[key] is True

    smoke = cfg["synthetic_smoke_contract"]
    assert "synthetic_tensor_smoke" in smoke["allowed_checks"]
    assert "hard_export_diagnostic_no_grad" in smoke["allowed_checks"]
    for forbidden in (
        "dataset_access",
        "checkpoint_access",
        "tools_test",
        "detector_mAP",
        "remote_sync",
        "Slurm",
        "GPU_run",
        "training",
    ):
        assert forbidden in smoke["forbidden_checks"]

    reader_cfg = cfg["pc_ot_mras_reader"]
    bridge_cfg = cfg["pc_ot_mras_bridge"]
    hard_export = cfg["hard_export_diagnostic"]
    assert reader_cfg["type"] == "PCOTMRASReader"
    assert reader_cfg["hidden_dim"] == bridge_cfg["in_channels"]
    assert bridge_cfg["type"] == "PCOTMRASDetectorBridge"
    assert bridge_cfg["allocation_key"] == "acquisition_matrix"
    assert hard_export["schema_version"] == "pc_ot_mras_hard_positions_v0"
    assert hard_export["diagnostic_or_deploy_only"] is True
    assert hard_export["training_backprop_allowed"] is False
    assert hard_export["detached_reader_tensors"] is True

    forbidden_top_level = (
        "dataset",
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
        "optimizer",
        "scheduler",
        "workflow",
        "work_dir",
        "launcher",
        "load_from",
        "resume_from",
    )
    for name in forbidden_top_level:
        assert name not in cfg

    for literal in (
        "tools/train.py",
        "tools/test.py",
        "#SBATCH",
        "sbatch",
        "srun ",
        "ssh ",
        "scp ",
        "/root/",
        "thumos_14_anno",
        "result_detection.json",
        "load_from =",
        "resume_from =",
    ):
        assert literal.lower() not in lower


def test_pc_ot_mras_config_registry_bridge_and_hard_export_round_trip():
    code = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import runpy
        import sys
        import types
        from pathlib import Path

        import torch

        root = Path(r"{ROOT}")
        cfg = runpy.run_path(str(root / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_synthetic_local.py"))

        def ensure_package(name, path):
            module = sys.modules.get(name)
            if module is None:
                module = types.ModuleType(name)
                module.__path__ = [str(path)]
                sys.modules[name] = module
            return module

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        ensure_package("opentad", root / "opentad")
        ensure_package("opentad.models", root / "opentad" / "models")
        ensure_package("opentad.models.selectors", root / "opentad" / "models" / "selectors")
        ensure_package("opentad.models.necks", root / "opentad" / "models" / "necks")

        backbones = types.ModuleType("opentad.models.backbones")
        class BackboneWrapper:
            pass
        backbones.BackboneWrapper = BackboneWrapper
        sys.modules["opentad.models.backbones"] = backbones

        builder = load_module("opentad.models.builder", root / "opentad" / "models" / "builder.py")
        load_module("opentad.ctf_bdi_role_constants", root / "opentad" / "ctf_bdi_role_constants.py")
        load_module(
            "opentad.models.selectors.lowcost_acquisition_browser",
            root / "opentad" / "models" / "selectors" / "lowcost_acquisition_browser.py",
        )
        reader_module = load_module(
            "opentad.models.selectors.pc_ot_mras_reader",
            root / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py",
        )
        bridge_module = load_module(
            "opentad.models.necks.pc_ot_mras_detector_bridge",
            root / "opentad" / "models" / "necks" / "pc_ot_mras_detector_bridge.py",
        )
        from tools.bata.export_pc_ot_mras_hard_positions import resolve_pc_ot_mras_hard_positions

        assert builder.SELECTORS is builder.MODELS
        assert builder.NECKS is builder.MODELS
        assert builder.MODELS.get("PCOTMRASReader") is reader_module.PCOTMRASReader
        assert builder.MODELS.get("PCOTMRASDetectorBridge") is bridge_module.PCOTMRASDetectorBridge
        reader = builder.build_selector(dict(cfg["pc_ot_mras_reader"]))
        bridge = builder.build_neck(dict(cfg["pc_ot_mras_bridge"]))
        assert isinstance(reader, reader_module.PCOTMRASReader)
        assert isinstance(bridge, bridge_module.PCOTMRASDetectorBridge)
        assert reader.cfg.in_dim == 512
        assert reader.cfg.hidden_dim == 96
        assert reader.cfg.num_slots == 384
        assert bridge.in_channels == reader.cfg.hidden_dim
        assert bridge.out_channels == reader.cfg.in_dim
        assert bridge.allocation_key == "acquisition_matrix"
        assert bridge.time_feature_dim == 4

        torch.manual_seed(20260618)
        batch, time, dim = 2, 16, cfg["pc_ot_mras_reader"]["in_dim"]
        features = torch.randn(batch, time, dim, requires_grad=True)
        valid = torch.ones(batch, time, dtype=torch.bool)
        valid[1, 11:] = False
        coords = torch.linspace(0.0, 1.0, steps=time).unsqueeze(0).repeat(batch, 1)
        coords[1, 11:] = 0.0

        reader_out = reader(features, valid, coords)
        feats, masks, aux = bridge(
            source_tokens=reader_out["browser_memory"],
            reader_outputs=reader_out,
            return_aux=True,
        )
        valid_dense = reader_out["valid_mask"]
        pair_valid = reader_out["pair_valid_mask"]
        detector_like_loss = feats[0].square().mean() + aux["selected_times"].mean()
        dense_aux_loss = (
            reader_out["process_logits"][valid_dense].square().mean()
            + reader_out["boundary_logits"][valid_dense].square().mean()
        )
        pair_aux_loss = reader_out["pair_logits"][pair_valid].square().mean()
        loss = detector_like_loss + 0.01 * dense_aux_loss + 0.01 * pair_aux_loss + reader_out["regularizers"]["total_regularizer"]
        loss.backward()
        expected_recomputed = torch.bmm(reader_out["acquisition_matrix"], reader_out["browser_memory"])

        rows = resolve_pc_ot_mras_hard_positions(
            reader_out,
            budget=8,
            sample_ids=["synthetic_cfg|0", "synthetic_cfg|1"],
        )

        payload = {{
            "reader_type": type(reader).__name__,
            "bridge_type": type(bridge).__name__,
            "feature_shape": list(feats[0].shape),
            "mask_shape": list(masks[0].shape),
            "aux_schema": aux["schema_version"],
            "continuous_axis": bool(aux["continuous_axis"]),
            "uses_hard_gather": bool(aux["uses_hard_gather"]),
            "allocation_shape": list(reader_out["allocation"].shape),
            "selected_token_shape": list(reader_out["selected_tokens"].shape),
            "process_logit_shape": list(reader_out["process_logits"].shape),
            "pair_prob_shape": list(reader_out["pair_prob"].shape),
            "invalid_tail_zero": bool((reader_out["allocation"][1, :, 11:] == 0).all().item()),
            "allocation_rows_sum_one": bool(torch.allclose(reader_out["allocation"].sum(dim=-1), torch.ones(batch, 384), atol=1e-5)),
            "bridge_recomputed_tokens_match_matrix": bool(torch.allclose(aux["selected_tokens"], expected_recomputed, atol=1e-5)),
            "aux_has_acquisition_matrix": "acquisition_matrix" in aux,
            "aux_has_pair_prob": "pair_prob" in aux,
            "feature_grad_finite": bool(features.grad is not None and torch.isfinite(features.grad).all().item()),
            "allocation_grad_finite": bool(reader.key_proj.weight.grad is not None and torch.isfinite(reader.key_proj.weight.grad).all().item()),
            "reader_grad_keys": {{
                "query_embed": reader.query_embed.grad,
                "key_proj": reader.key_proj.weight.grad,
                "center_inc_head": reader.center_inc_head.weight.grad,
                "width_head": reader.width_head.weight.grad,
                "gate_head": reader.gate_head.weight.grad,
                "process_head": reader.process_head.weight.grad,
                "boundary_head": reader.boundary_head.weight.grad,
                "pair_scorer": reader.pair_scorer[-1].weight.grad,
            }},
            "bridge_grad_keys": {{
                "token_proj": bridge.token_proj.weight.grad,
                "time_proj": bridge.time_proj.weight.grad,
            }},
            "row_count": len(rows),
            "row_schemas": [row["schema_version"] for row in rows],
            "row_budgets": [row["budget"] for row in rows],
            "row_selected_lens": [len(row["selected_positions"]) for row in rows],
            "row_selected_unique": [len(set(row["selected_positions"])) == len(row["selected_positions"]) for row in rows],
            "row_generation_no_grad": [
                row["resolver_generation"]["training_backprop_allowed"] is False
                and row["resolver_generation"]["detached_reader_tensors"] is True
                for row in rows
            ],
            "row_valid_bounds": [
                all(0 <= pos < row["valid_len"] for pos in row["selected_positions"])
                for row in rows
            ],
        }}
        payload["reader_grad_ok"] = {{
            key: bool(value is not None and torch.isfinite(value).all().item() and value.abs().sum().item() > 0)
            for key, value in payload.pop("reader_grad_keys").items()
        }}
        payload["bridge_grad_ok"] = {{
            key: bool(value is not None and torch.isfinite(value).all().item() and value.abs().sum().item() > 0)
            for key, value in payload.pop("bridge_grad_keys").items()
        }}
        print(json.dumps(payload, sort_keys=True))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["reader_type"] == "PCOTMRASReader"
    assert payload["bridge_type"] == "PCOTMRASDetectorBridge"
    assert payload["feature_shape"] == [2, 512, 384]
    assert payload["mask_shape"] == [2, 384]
    assert payload["allocation_shape"] == [2, 384, 16]
    assert payload["selected_token_shape"] == [2, 384, 96]
    assert payload["process_logit_shape"] == [2, 16, 7]
    assert payload["pair_prob_shape"] == [2, 16, 16]
    assert payload["invalid_tail_zero"] is True
    assert payload["allocation_rows_sum_one"] is True
    assert payload["bridge_recomputed_tokens_match_matrix"] is True
    assert payload["aux_schema"] == "pc_ot_mras_continuous_bridge_v0"
    assert payload["continuous_axis"] is True
    assert payload["uses_hard_gather"] is False
    assert payload["aux_has_acquisition_matrix"] is True
    assert payload["aux_has_pair_prob"] is True
    assert payload["feature_grad_finite"] is True
    assert payload["allocation_grad_finite"] is True
    assert all(payload["reader_grad_ok"].values()), payload["reader_grad_ok"]
    assert all(payload["bridge_grad_ok"].values()), payload["bridge_grad_ok"]
    assert payload["row_count"] == 2
    assert payload["row_schemas"] == ["pc_ot_mras_hard_positions_v0", "pc_ot_mras_hard_positions_v0"]
    assert payload["row_budgets"] == [8, 8]
    assert payload["row_selected_lens"] == [8, 8]
    assert payload["row_selected_unique"] == [True, True]
    assert payload["row_generation_no_grad"] == [True, True]
    assert payload["row_valid_bounds"] == [True, True]
