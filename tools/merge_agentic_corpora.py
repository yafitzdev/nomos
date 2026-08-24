"""Merge validated decision-state corpora while enforcing unique identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.generic_contracts import validate_decision_state_v2


IDENTITY_FIELDS = ("decision_state_id", "matrix_cell_id", "source_row_hash")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seen: dict[str, set[str]] = {field: set() for field in IDENTITY_FIELDS}
    seen["question"] = set()
    rows: list[dict[str, Any]] = []
    input_counts = {}
    dropped_duplicate_questions = 0
    for path in args.input:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                report = validate_decision_state_v2(row)
                if not report.valid:
                    raise ValueError(f"invalid row {path}:{line_number}: {report.as_dict()}")
                values = {
                    field: str(row.get(field) or "").casefold()
                    for field in ("decision_state_id", "matrix_cell_id", "question")
                }
                values["source_row_hash"] = str(
                    (row.get("provenance") or {}).get("source_row_hash") or ""
                )
                for field in IDENTITY_FIELDS:
                    value = values[field]
                    if not value or value in seen[field]:
                        raise ValueError(f"duplicate or empty {field} at {path}:{line_number}")
                if not values["question"] or values["question"] in seen["question"]:
                    dropped_duplicate_questions += 1
                    continue
                for field, value in values.items():
                    seen[field].add(value)
                rows.append(row)
                count += 1
        input_counts[str(path)] = count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "count": len(rows),
        "inputs": input_counts,
        "unique_counts": {field: len(values) for field, values in seen.items()},
        "dropped_duplicate_questions": dropped_duplicate_questions,
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
