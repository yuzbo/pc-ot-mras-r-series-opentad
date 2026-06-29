from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.mdl_knot import MDL_KNOT_ROUTE_LABEL, MDLKnotConfig, build_synthetic_scout_curve, greedy_mdl_knot_select


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic MDL-Knot ledger for offline inspection.")
    parser.add_argument("--pattern", default="short_islands", choices=["stable_background", "sharp_transition", "two_islands", "short_islands"])
    parser.add_argument("--dense-t", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    curve = build_synthetic_scout_curve(args.pattern, dense_t=args.dense_t)
    ledger = greedy_mdl_knot_select(curve, MDLKnotConfig(route_label=MDL_KNOT_ROUTE_LABEL), video_id=args.pattern)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ledger.to_dict(), indent=2), encoding="utf-8")
    print(f"WROTE_SYNTHETIC_LEDGER={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

