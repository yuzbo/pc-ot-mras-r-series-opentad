from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reader_class():
    """Load PCOTMRASReader without importing the full OpenTAD registry stack."""
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    sys.modules["opentad.models.builder"] = builder

    _load_module(
        "opentad.ctf_bdi_role_constants",
        ROOT / "opentad" / "ctf_bdi_role_constants.py",
    )
    _load_module(
        "opentad.models.selectors.lowcost_acquisition_browser",
        ROOT / "opentad" / "models" / "selectors" / "lowcost_acquisition_browser.py",
    )
    reader_module = _load_module(
        "opentad.models.selectors.pc_ot_mras_reader",
        ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py",
    )
    return reader_module.PCOTMRASReader


def _mb(num_bytes: int) -> float:
    return round(float(num_bytes) / (1024.0 * 1024.0), 3)


def _tensor_shape(value: Any) -> list[int] | None:
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None


def _regularizers_are_finite(regularizers: Any, torch_module) -> bool:
    if not isinstance(regularizers, dict) or not regularizers:
        return False
    for value in regularizers.values():
        if not torch_module.is_tensor(value):
            return False
        if not bool(torch_module.isfinite(value).all().item()):
            return False
    return True


def profile_reader(args: argparse.Namespace) -> dict[str, Any]:
    emit_pair = bool(args.emit_pair_distribution)
    device_name = str(args.device)
    payload: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "time": int(args.time),
        "hidden_dim": int(args.hidden_dim),
        "num_slots": int(args.num_slots),
        "emit_pair_distribution": emit_pair,
        "device": device_name,
        "amp": bool(args.amp),
        "success": False,
        "error": "",
        "cuda_max_memory_allocated_mb": 0.0,
        "cuda_max_memory_reserved_mb": 0.0,
        "cuda_memory_allocated_before_mb": 0.0,
        "cuda_memory_allocated_after_mb": 0.0,
        "cuda_peak_extra_allocated_mb": 0.0,
        "elapsed_sec": 0.0,
        "output_keys": [],
        "output_shapes": {},
        "pair_shape": None,
        "regularizers_finite": False,
    }

    output: dict[str, Any] | None = None
    torch = None
    start_time = time.perf_counter()
    try:
        import torch as torch_module

        torch = torch_module
        if int(args.batch_size) <= 0:
            raise ValueError("--batch-size must be positive")
        if int(args.time) <= 0:
            raise ValueError("--time must be positive")
        if int(args.hidden_dim) <= 0:
            raise ValueError("--hidden-dim must be positive")
        if int(args.num_slots) <= 0:
            raise ValueError("--num-slots must be positive")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

        device = torch.device(device_name)
        torch.manual_seed(20260622)
        reader_cls = load_reader_class()
        reader = reader_cls(
            in_dim=int(args.hidden_dim),
            hidden_dim=int(args.hidden_dim),
            num_slots=int(args.num_slots),
            num_blocks=1,
            dropout=0.0,
            emit_pair_distribution=emit_pair,
        ).to(device=device)
        reader.eval()

        lowcost_features = torch.randn(
            int(args.batch_size),
            int(args.time),
            int(args.hidden_dim),
            device=device,
        )
        valid_mask = torch.ones(int(args.batch_size), int(args.time), dtype=torch.bool, device=device)
        if int(args.time) == 1:
            coords_1d = torch.zeros(1, device=device, dtype=lowcost_features.dtype)
        else:
            coords_1d = torch.linspace(0.0, 1.0, steps=int(args.time), device=device, dtype=lowcost_features.dtype)
        time_coords = coords_1d.unsqueeze(0).expand(int(args.batch_size), -1).contiguous()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            before_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            payload["cuda_memory_allocated_before_mb"] = _mb(before_allocated)

        amp_active = bool(args.amp) and device.type == "cuda"
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=amp_active):
                output = reader(lowcost_features, valid_mask, time_coords)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        payload["output_keys"] = sorted(output.keys())
        payload["output_shapes"] = {
            key: _tensor_shape(value)
            for key, value in output.items()
            if _tensor_shape(value) is not None
        }
        payload["regularizers_finite"] = _regularizers_are_finite(output.get("regularizers"), torch)

        pair_keys = ("pair_logits", "pair_prob", "pair_valid_mask")
        if emit_pair:
            missing = [key for key in pair_keys if key not in output]
            if missing:
                raise AssertionError(f"emit_pair_distribution=1 but missing pair outputs: {missing}")
            pair_shape = _tensor_shape(output["pair_logits"])
            if pair_shape != _tensor_shape(output["pair_prob"]) or pair_shape != _tensor_shape(output["pair_valid_mask"]):
                raise AssertionError("pair_logits, pair_prob, and pair_valid_mask shapes differ")
            payload["pair_shape"] = pair_shape
        else:
            present = [key for key in pair_keys if key in output]
            if present:
                raise AssertionError(f"emit_pair_distribution=0 but pair outputs were returned: {present}")
            if not payload["regularizers_finite"]:
                raise AssertionError("regularizers must exist and be finite when pair distribution is disabled")

        if not payload["regularizers_finite"]:
            raise AssertionError("regularizers must exist and be finite")

        payload["success"] = True
    except BaseException as exc:  # Keep CUDA OOM and validation failures JSON-visible.
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        payload["elapsed_sec"] = round(time.perf_counter() - start_time, 6)
        if torch is not None:
            if torch.cuda.is_available() and device_name == "cuda":
                device = torch.device(device_name)
                try:
                    allocated_after = torch.cuda.memory_allocated(device)
                    max_allocated = torch.cuda.max_memory_allocated(device)
                    payload["cuda_memory_allocated_after_mb"] = _mb(allocated_after)
                    payload["cuda_max_memory_allocated_mb"] = _mb(max_allocated)
                    payload["cuda_max_memory_reserved_mb"] = _mb(torch.cuda.max_memory_reserved(device))
                    before = payload.get("cuda_memory_allocated_before_mb", 0.0)
                    payload["cuda_peak_extra_allocated_mb"] = round(
                        max(0.0, float(payload["cuda_max_memory_allocated_mb"]) - float(before)),
                        3,
                    )
                except Exception:
                    pass
                if output is None:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile synthetic PCOTMRASReader memory with or without pair distribution outputs."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--emit-pair-distribution", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = profile_reader(args)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
