import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.sparse_forward_audit import build_fake_sparse_forward_ledger
from opentad.acquisition.bvr_twb.types import ROUTE_LABEL
from opentad.acquisition.bvr_twb.validators import (
    SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS,
    validate_sparse_forward_ledger,
)


def safe_prepare_output_dir(out_dir, overwrite=False, root=ROOT):
    out = Path(out_dir)
    resolved_root = Path(root).resolve()
    resolved_out = out.resolve()
    try:
        resolved_out.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"refusing output path outside worktree: {resolved_out}") from exc

    name = resolved_out.name.lower()
    if overwrite and not (name.startswith(".tmp_bvr_twb") or "bvr_twb" in name):
        raise ValueError(
            "refusing overwrite for output directory without bvr_twb marker: "
            f"{resolved_out}"
        )
    if resolved_out.exists() and overwrite:
        shutil.rmtree(resolved_out)
    resolved_out.mkdir(parents=True, exist_ok=True)
    return resolved_out


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_precheck_ledgers(mode):
    if mode not in {"fake_raw", "module_fake_forward"}:
        raise ValueError(f"unsupported mode: {mode}")
    module_forward = mode == "module_fake_forward"
    specs = [
        ("fake_sparse_background", "fake_sparse_background_0000", 96, [0, 9, 23, 41, 59, 76, 95], 128),
        ("fake_sparse_short_action", "fake_sparse_short_action_0000", 96, [0, 22, 41, 45, 49, 68, 95], 128),
        ("fake_sparse_boundary_pair", "fake_sparse_boundary_pair_0000", 96, [0, 14, 27, 34, 48, 62, 79, 95], 128),
    ]
    return [
        build_fake_sparse_forward_ledger(
            video_name=video_name,
            window_id=window_id,
            dense_T=dense_T,
            selected_positions=positions,
            detector_pad_len=pad_len,
            module_forward=module_forward,
        )
        for video_name, window_id, dense_T, positions, pad_len in specs
    ]


def run_precheck(mode, out_dir, overwrite=False, root=ROOT):
    out = safe_prepare_output_dir(out_dir, overwrite=overwrite, root=root)
    ledgers = build_precheck_ledgers(mode)
    validation_rows = [validate_sparse_forward_ledger(row) for row in ledgers]
    write_jsonl(out / "bvr_twb_sparse_forward_ledgers.jsonl", ledgers)
    summary = {
        "route_label": ROUTE_LABEL,
        "mode": mode,
        "num_ledgers": len(ledgers),
        "claim_status": SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS,
        "sparse_compute_claim": False,
        "no_metric_claim": True,
        "no_runtime_or_flops_claim": True,
        "no_deploy_claim": True,
        "verdict_counts": dict(Counter(row["verdict"] for row in validation_rows)),
        "valid_k_values": [row["valid_k"] for row in validation_rows],
        "dense_T_values": [row["dense_T"] for row in validation_rows],
        "ledger_path": str((out / "bvr_twb_sparse_forward_ledgers.jsonl").resolve()),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run BVR-TWB sparse-forward local precheck audit.")
    parser.add_argument("--mode", choices=("fake_raw", "module_fake_forward"), default="fake_raw")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_precheck(args.mode, args.out_dir, overwrite=args.overwrite)
    print(
        "BVR-TWB sparse-forward precheck: "
        f"mode={summary['mode']} "
        f"ledgers={summary['num_ledgers']} "
        f"verdicts={summary['verdict_counts']} "
        f"claim_status={summary['claim_status']} "
        f"sparse_compute_claim={summary['sparse_compute_claim']}"
    )


if __name__ == "__main__":
    main()
