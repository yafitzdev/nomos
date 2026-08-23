"""Validate provenance, legality, uniqueness and coverage of a router.v2 pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.pilot_v2 import validate_pilot_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--min-per-target", type=int, default=200)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict):
            errors.append({"line": line_number, "error": "row must be an object"})
            continue
        report = validate_pilot_state(value)
        if report.valid:
            rows.append(value)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    return rows, errors


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, errors = _read_rows(args.input)
    type_counts = Counter(row.get("type_signature") for row in rows)
    instance_counts = Counter(row.get("instance_signature") for row in rows)
    target_counts = Counter(
        (row.get("sampling_context") or {}).get("target_capability") for row in rows
    )
    report = {
        "valid": (
            not errors
            and len(rows) == args.expected_count
            and max(type_counts.values(), default=0) == 1
            and max(instance_counts.values(), default=0) == 1
            and all(count >= args.min_per_target for count in target_counts.values())
        ),
        "rows": len(rows),
        "expected_rows": args.expected_count,
        "invalid_rows": len(errors),
        "duplicate_type_signatures": sorted(
            signature for signature, count in type_counts.items() if count > 1
        ),
        "duplicate_instance_signatures": sorted(
            signature for signature, count in instance_counts.items() if count > 1
        ),
        "target_capability_counts": dict(sorted(target_counts.items())),
        "cohort_counts": dict(sorted(Counter(row.get("evaluation_cohort") for row in rows).items())),
        "source_kind_counts": dict(sorted(Counter(row.get("source_kind") for row in rows).items())),
        "errors": errors[:10],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
