import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb import (
    DynamicBudgetController,
    PacketValuePredictor,
    build_matched_controls,
    build_scaffold_packets,
    build_scout_from_actionness,
    build_witness_packets,
    estimate_boundary_beliefs,
)
from opentad.acquisition.bvr_twb.types import BudgetConfig, ROUTE_LABEL
from opentad.acquisition.bvr_twb.validators import validate_dynamicity_and_uniform_mimicry


def _gaussian(length, center, width, height=1.0):
    x = np.arange(int(length), dtype=np.float64)
    return float(height) * np.exp(-((x - float(center)) ** 2) / (2.0 * float(width) ** 2))


def synthetic_cases():
    cases = []
    dense_T = 96
    x = np.arange(dense_T, dtype=np.float64)
    cases.append(("pure_background", np.clip(0.08 + 0.02 * np.sin(x / 8.0), 0.0, 1.0), None))
    single = np.zeros(dense_T, dtype=np.float64) + 0.07
    single[28:62] = 0.78
    single += _gaussian(dense_T, 28, 2, 0.12) + _gaussian(dense_T, 62, 2, 0.12)
    cases.append(("single_clear_action", np.clip(single, 0.0, 1.0), _gaussian(dense_T, 28, 3) + _gaussian(dense_T, 62, 3)))
    short = np.zeros(dense_T, dtype=np.float64) + 0.06
    short[44:50] = 0.90
    cases.append(("short_action", short, _gaussian(dense_T, 47, 2)))
    close = np.zeros(dense_T, dtype=np.float64) + 0.06
    close[20:34] = 0.74
    close[40:55] = 0.80
    cases.append(("two_close_instances", close, _gaussian(dense_T, 20, 2) + _gaussian(dense_T, 40, 2) + _gaussian(dense_T, 55, 2)))
    long_core = np.zeros(dense_T, dtype=np.float64) + 0.05
    long_core[12:83] = 0.70
    long_core[38:56] = 0.55
    cases.append(("long_stable_core", long_core, _gaussian(dense_T, 12, 3) + _gaussian(dense_T, 83, 3)))
    ambiguous = np.zeros(dense_T, dtype=np.float64) + 0.15
    ambiguous += _gaussian(dense_T, 30, 8, 0.42) + _gaussian(dense_T, 66, 10, 0.33)
    cases.append(("ambiguous_island", np.clip(ambiguous, 0.0, 1.0), _gaussian(dense_T, 48, 12, 0.60)))
    gap = np.zeros(dense_T, dtype=np.float64) + 0.05
    gap[8:16] = 0.55
    gap[76:90] = 0.88
    cases.append(("high_gap_stress", gap, _gaussian(dense_T, 8, 3) + _gaussian(dense_T, 86, 3)))
    return cases


def run_bvr_case(name, p_action, motion_signal=None, fps=30.0):
    dense_T = int(len(p_action))
    scout = build_scout_from_actionness(p_action, motion_signal=motion_signal)
    scaffold = build_scaffold_packets(
        dense_T=dense_T,
        scaffold_k=4,
        max_gap=22,
        video_id=name,
        window_id=0,
        split="synthetic",
    )
    scaffold_positions = [pos for packet in scaffold for pos in packet.positions]
    brackets = estimate_boundary_beliefs(scout, video_id=name, split="synthetic", max_brackets=5)
    candidates = build_witness_packets(
        scout,
        brackets,
        scaffold_positions=scaffold_positions,
        video_id=name,
        split="synthetic",
        start_packet_id=100,
    )
    predictor = PacketValuePredictor(mode="heuristic_fallback")
    predictor.score_packets(scaffold + candidates)
    predictor.assert_not_actionness_only(candidates)
    max_k = 12 + int(np.clip(np.mean(scout.uncertainty) * 8.0 + np.max(scout.transition_score) * 4.0, 0, 8))
    min_k = 5 if np.mean(scout.p_action) < 0.18 else 7
    config = BudgetConfig(min_k=min_k, max_k=max_k, max_gap=22, min_marginal_value=0.56)
    dense_inputs = np.stack([np.arange(dense_T, dtype=np.float64), p_action.astype(np.float64)], axis=1)
    result = DynamicBudgetController(config).select(
        scaffold,
        candidates,
        brackets,
        dense_T=dense_T,
        fps=fps,
        video_id=name,
        split="synthetic",
        dense_inputs=dense_inputs,
    )
    return result, candidates, dense_inputs


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def build_ledgers(out_dir, overwrite=False):
    out = safe_prepare_output_dir(out_dir, overwrite=overwrite)

    bvr_ledgers = []
    decision_rows = []
    candidate_rows = []
    candidates_by_video = {}
    dense_inputs_by_video = {}
    for name, p_action, motion in synthetic_cases():
        result, candidates, dense_inputs = run_bvr_case(name, p_action, motion)
        bvr_ledgers.append(result.deploy_ledger)
        decision_rows.extend(result.ledger_rows)
        candidate_rows.extend([packet.to_ledger_dict() for packet in candidates])
        candidates_by_video[name] = candidates
        dense_inputs_by_video[name] = dense_inputs

    controls = build_matched_controls(
        bvr_ledgers,
        candidate_packets_by_video=candidates_by_video,
        dense_inputs_by_video=dense_inputs_by_video,
        fps=30.0,
        scaffold_k=4,
        max_gap=22,
    )
    dynamic_diag = validate_dynamicity_and_uniform_mimicry(bvr_ledgers, dynamic_enabled=True)

    write_jsonl(out / "bvr_twb_deploy_ledgers.jsonl", bvr_ledgers)
    write_jsonl(out / "bvr_twb_selection_rows.jsonl", decision_rows)
    write_jsonl(out / "bvr_twb_candidate_packets.jsonl", candidate_rows)
    write_jsonl(out / "bvr_twb_matched_controls.jsonl", controls)
    summary = {
        "route_label": ROUTE_LABEL,
        "num_cases": len(bvr_ledgers),
        "k_values": [row["valid_k"] for row in bvr_ledgers],
        "mean_k": float(np.mean([row["valid_k"] for row in bvr_ledgers])),
        "stop_reasons": dict(Counter(row["budget_stop_reason"] for row in bvr_ledgers)),
        "dynamicity": dynamic_diag,
        "controls": dict(Counter(row["control_name"] for row in controls)),
        "claim_status": "local_gather_smoke_only_no_sparse_compute_or_metric_claim",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build local BVR-TWB synthetic ledgers.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_ledgers(args.out_dir, overwrite=args.overwrite)
    print(
        "BVR-TWB synthetic ledgers: "
        f"cases={summary['num_cases']} "
        f"k={summary['k_values']} "
        f"stops={summary['stop_reasons']} "
        f"claim_status={summary['claim_status']}"
    )


if __name__ == "__main__":
    main()
