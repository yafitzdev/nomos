"""Evaluate router.v2 cohorts, baselines, ablations and invariance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.evaluation_v2 import evaluate_router_v2_report
from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.router_v2 import load_router_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    return parser


def _read_states(path: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            state = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        report = validate_decision_state_v2(state)
        if report.valid:
            states.append(state)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    if errors:
        raise SystemExit(json.dumps({"invalid_rows": len(errors), "examples": errors[:5]}, indent=2))
    return states


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, metadata = load_router_v2(str(args.artifact))
    report = evaluate_router_v2_report(model, metadata, _read_states(args.input))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
