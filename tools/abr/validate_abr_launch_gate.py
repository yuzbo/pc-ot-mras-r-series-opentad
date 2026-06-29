from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.acquisition.abr.validators import ABRValidationError, validate_launch_gate_payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck-json", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.precheck_json).read_text(encoding="utf-8"))
    try:
        decision = validate_launch_gate_payload(payload)
    except ABRValidationError as exc:
        print(f"LOCKED: {exc}")
        return 1
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

