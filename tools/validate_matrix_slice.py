"""Validate a materialized matrix-cell JSONL slice and report coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.matrix import DIMENSION_NAMES, load_matrix_spec, make_cell, validate_matrix_cell


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--min-per-value", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_matrix_spec()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(row, dict):
            errors.append({"line": line_number, "error": "row must be an object"})
            continue
        rows.append(row)
        if row.get("matrix_version") != "matrix.v1":
            errors.append({"line": line_number, "error": "wrong matrix_version"})
        cell_values = {name: row.get(name) for name in DIMENSION_NAMES}
        cell_errors = validate_matrix_cell(cell_values, spec)
        errors.extend({"line": line_number, "error": error} for error in cell_errors)
        try:
            expected_id = make_cell(cell_values).cell_id
            if row.get("cell_id") != expected_id:
                errors.append({"line": line_number, "error": "cell_id does not match canonical cell values"})
        except ValueError:
            pass

    ids = [str(row.get("cell_id")) for row in rows]
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append({"error": "duplicate_cell_ids", "values": duplicates})

    coverage: dict[str, Counter[str]] = {name: Counter() for name in DIMENSION_NAMES}
    for row in rows:
        for name in DIMENSION_NAMES:
            coverage[name][str(row.get(name))] += 1
    for name in DIMENSION_NAMES:
        for value in spec["dimensions"][name]:
            if coverage[name][value] < args.min_per_value:
                errors.append(
                    {
                        "error": "coverage_below_minimum",
                        "dimension": name,
                        "value": value,
                        "count": coverage[name][value],
                        "minimum": args.min_per_value,
                    }
                )

    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "errors": len(errors),
        "coverage": {name: dict(counts) for name, counts in sorted(coverage.items())},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        print(json.dumps({"validation_errors": errors[:20]}, indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
