"""Validate NInfer router.v2 proposals and a deterministic grounding sample."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.pilot_v2 import load_pilot_source_cards
from tools.generate_ninfer_router_v2_slice import validate_proposal


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cards = load_pilot_source_cards(PROJECT_ROOT / "tests" / "fixtures" / "pilot_v2_corpus")
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict):
            parse_errors.append({"line": line_number, "error": "row is not an object"})
            continue
        rows.append(value)
    structural_errors: list[dict[str, Any]] = []
    type_signatures: list[str] = []
    instance_signatures: list[str] = []
    for index, row in enumerate(rows, start=1):
        errors = validate_proposal(row, cards=cards)
        if errors:
            structural_errors.append({"row": index, "errors": errors})
        if isinstance(row.get("type_signature"), str):
            type_signatures.append(row["type_signature"])
        if isinstance(row.get("instance_signature"), str):
            instance_signatures.append(row["instance_signature"])
    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(range(len(rows)), min(args.sample_size, len(rows)))) if rows else []
    sample_errors = [
        {"row": index + 1, "errors": validate_proposal(rows[index], cards=cards)}
        for index in sample_indices
        if validate_proposal(rows[index], cards=cards)
    ]
    report = {
        "valid": (
            not parse_errors
            and len(rows) == args.expected_count
            and not structural_errors
            and len(type_signatures) == len(set(type_signatures))
            and len(instance_signatures) == len(set(instance_signatures))
            and not sample_errors
        ),
        "rows": len(rows),
        "expected_count": args.expected_count,
        "parse_errors": parse_errors,
        "structural_error_count": len(structural_errors),
        "structural_error_examples": structural_errors[:5],
        "sample_size": len(sample_indices),
        "sample_rows": [index + 1 for index in sample_indices],
        "sample_errors": sample_errors,
        "duplicate_type_signatures": len(type_signatures) - len(set(type_signatures)),
        "duplicate_instance_signatures": len(instance_signatures) - len(set(instance_signatures)),
        "target_capability_counts": dict(
            sorted(Counter(str(row.get("target_capability")) for row in rows).items())
        ),
        "source_kind_counts": dict(
            sorted(Counter(str(row.get("source_kind")) for row in rows).items())
        ),
        "execution_status_counts": dict(
            sorted(Counter(str(row.get("execution_status")) for row in rows).items())
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
