import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.rba_rbr import (  # noqa: E402
    DynamicBudgetController,
    RbaRbrBudgetConfig,
    build_probe_candidates,
    build_rba_rbr_open_tad_selection,
    build_risk_map,
    build_scaffold_positions,
    build_soft_brackets,
)
from opentad.acquisition.rba_rbr.types import ROUTE_LABEL  # noqa: E402
from opentad.acquisition.rba_rbr.validators import validate_dynamic_precheck_ledgers, validate_rba_rbr_ledger  # noqa: E402


def _gaussian(length, center, width, height=1.0):
    x = np.arange(int(length), dtype=np.float64)
    return float(height) * np.exp(-((x - float(center)) ** 2) / (2.0 * float(width) ** 2))


def recovery_case_curves():
    dense_T = 80
    actionness = np.zeros(dense_T, dtype=np.float64) + 0.05
    actionness[18:31] = 0.82
    uncertainty = np.zeros(dense_T, dtype=np.float64) + 0.08
    transition = np.zeros(dense_T, dtype=np.float64) + 0.04
    transition[17:20] = 0.55
    transition[30:33] = 0.45
    transition[55:58] = 0.95
    uncertainty[54:59] = 0.92
    return actionness, uncertainty, transition


def synthetic_cases():
    cases = []
    dense_T = 96
    x = np.arange(dense_T, dtype=np.float64)
    actionness = np.zeros(dense_T, dtype=np.float64) + 0.06
    actionness[24:48] = 0.74
    cases.append(("single_soft_action", actionness, 0.12 + _gaussian(dense_T, 50, 4, 0.65), _gaussian(dense_T, 24, 3, 0.8)))
    ambiguous = np.zeros(dense_T, dtype=np.float64) + 0.09
    ambiguous[20:34] = 0.58
    ambiguous[60:72] = 0.64
    cases.append(("ambiguous_two_regions", ambiguous, 0.18 + _gaussian(dense_T, 42, 7, 0.72), _gaussian(dense_T, 62, 3, 0.75)))
    rescue = np.zeros(dense_T, dtype=np.float64) + 0.05
    rescue[12:24] = 0.80
    cases.append(("missed_late_boundary", rescue, 0.10 + _gaussian(dense_T, 76, 3, 0.90), _gaussian(dense_T, 77, 2, 0.95)))
    background = np.clip(0.08 + 0.02 * np.sin(x / 8.0), 0.0, 1.0)
    cases.append(("low_risk_background", background, 0.10 + 0.02 * np.cos(x / 7.0), 0.08 + 0.02 * np.sin(x / 9.0)))
    return [(name, np.clip(a, 0.0, 1.0), np.clip(u, 0.0, 1.0), np.clip(t, 0.0, 1.0)) for name, a, u, t in cases]


def run_recovery_case():
    actionness, uncertainty, transition = recovery_case_curves()
    dense_T = int(actionness.size)
    scaffold = [0, 20, 40, dense_T - 1]
    risk_map = build_risk_map(actionness, uncertainty=uncertainty, transition=transition, observed_positions=scaffold)
    brackets = build_soft_brackets(risk_map, action_threshold=0.55, rescue_threshold=0.70)
    candidates = build_probe_candidates(
        risk_map,
        brackets,
        scaffold_positions=scaffold,
        video_id="synthetic_recovery",
        split="synthetic",
    )
    result = DynamicBudgetController(RbaRbrBudgetConfig(min_k=5, max_k=10, scaffold_k=4)).select(
        risk_map,
        brackets,
        candidates,
        scaffold_positions=scaffold,
        video_id="synthetic_recovery",
        split="synthetic",
    )
    hard_positions = set()
    for bracket in brackets:
        if bracket.hard_bracket:
            hard_positions.update(range(int(bracket.hard_left), int(bracket.hard_right) + 1))
    missed_boundary = 56
    diagnostic = {
        "missed_boundary_position": missed_boundary,
        "hard_bracket_positions": sorted(int(pos) for pos in hard_positions),
        "rescue_candidate_positions": sorted(int(p.center_pos) for p in candidates if p.stage == "rescue"),
        "selected_positions": [int(pos) for pos in result.selected_positions],
        "recovered_boundary_by_rescue": bool(any(abs(int(pos) - missed_boundary) <= 1 for pos in result.selected_positions)),
        "hard_bracket_would_miss_boundary": bool(missed_boundary not in hard_positions),
    }
    return result, candidates, diagnostic


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_prepare_output_dir(out_dir, overwrite=False, root=ROOT):
    out = Path(out_dir)
    resolved_root = Path(root).resolve()
    resolved_out = out.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        resolved_out.relative_to(resolved_root)
    except ValueError as exc:
        try:
            resolved_out.relative_to(temp_root)
        except ValueError as temp_exc:
            raise ValueError(f"refusing output path outside worktree/root or temp dir: {resolved_out}") from temp_exc
    if overwrite and "rba_rbr" not in resolved_out.name.lower():
        raise ValueError(f"refusing overwrite for output dir without rba_rbr marker: {resolved_out}")
    if resolved_out.exists() and overwrite:
        shutil.rmtree(resolved_out)
    resolved_out.mkdir(parents=True, exist_ok=True)
    return resolved_out


def build_ledgers(out_dir, overwrite=False, root=ROOT):
    out = safe_prepare_output_dir(out_dir, overwrite=overwrite, root=root)
    ledgers = []
    selection_rows = []
    probe_rows = []
    for name, actionness, uncertainty, transition in synthetic_cases():
        bridge = build_rba_rbr_open_tad_selection(
            {
                "video_name": name,
                "rba_rbr_preview_actionness": actionness,
                "rba_rbr_preview_uncertainty": uncertainty,
                "rba_rbr_preview_transition": transition,
            },
            dense_window=np.arange(int(actionness.size), dtype=np.int64),
            target_frame_num=16,
            split="synthetic",
            min_keep=5,
            max_keep=16,
            scaffold_k=4,
            train_value_labels=False,
            allow_diagnostic_preview_fallback=False,
        )
        ledger = bridge["ledger"]
        validate_rba_rbr_ledger(ledger)
        ledgers.append(ledger)
        selection_rows.extend(bridge["selection_result"].selection_rows)
        probe_rows.extend([probe.to_ledger_dict() for probe in bridge["candidate_probes"]])

    recovery_result, recovery_candidates, recovery_diagnostic = run_recovery_case()
    validate_rba_rbr_ledger(recovery_result.deploy_ledger)
    dynamic = validate_dynamic_precheck_ledgers(ledgers)
    write_jsonl(out / "rba_rbr_deploy_ledgers.jsonl", ledgers)
    write_jsonl(out / "rba_rbr_selection_rows.jsonl", selection_rows)
    write_jsonl(out / "rba_rbr_candidate_probes.jsonl", probe_rows)
    summary = {
        "route_label": ROUTE_LABEL,
        "all_validated": True,
        "num_cases": int(len(ledgers)),
        "k_values": [int(row["valid_k"]) for row in ledgers],
        "stop_reasons": dict(Counter(row["budget_stop_reason"] for row in ledgers)),
        "dynamicity": dynamic,
        "recovery_diagnostic": recovery_diagnostic,
        "rescue_outside_hard_bracket_total": int(sum(row["rescue_outside_hard_bracket_count"] for row in ledgers)),
        "claim_status": "rba_rbr_local_precheck_only_no_metric_runtime_deploy_or_paper_claim",
        "no_training": True,
        "no_metric_claim": True,
        "no_runtime_claim": True,
        "no_deploy_claim": True,
        "no_paper_claim": True,
        "full_train_unlocked": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build local RBA-RBR synthetic ledgers.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_ledgers(args.out_dir, overwrite=args.overwrite)
    print(
        "RBA-RBR synthetic ledgers: "
        f"cases={summary['num_cases']} "
        f"k={summary['k_values']} "
        f"stops={summary['stop_reasons']} "
        f"recovery={summary['recovery_diagnostic']['recovered_boundary_by_rescue']}"
    )


if __name__ == "__main__":
    main()
