"""Merge validated generic JSONL corpora without losing uniqueness guarantees."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.generic_pilot_v3 import GENERIC_DATASET_VERSION, validate_generic_state
from fitz_tool.router_v2 import FEATURE_VERSION


UNIQUE_FIELDS = ("decision_state_id", "matrix_cell_id", "type_signature", "instance_signature", "question")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="inputs", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = [path.resolve() for path in args.inputs]
    output_path = args.output.resolve()
    if output_path in input_paths:
        raise SystemExit("output must be different from every input")
    seen = {field: set[str]() for field in UNIQUE_FIELDS}
    cohort_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    row_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for path in input_paths:
            if not path.exists():
                raise SystemExit(f"input does not exist: {path}")
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    row: dict[str, Any] = json.loads(line)
                    report = validate_generic_state(row)
                    if not report.valid:
                        raise RuntimeError(
                            f"invalid row in {path}:{line_number}: {json.dumps(report.as_dict(), sort_keys=True)}"
                        )
                    for field in UNIQUE_FIELDS:
                        value = str(row.get(field) or "")
                        if not value:
                            raise RuntimeError(f"missing {field} in {path}:{line_number}")
                        if value in seen[field]:
                            raise RuntimeError(f"duplicate {field} in merged corpora: {value}")
                        seen[field].add(value)
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    row_count += 1
                    cohort_counts[str(row.get("evaluation_cohort"))] += 1
                    teacher_counts[str((row.get("provenance") or {}).get("teacher"))] += 1

    manifest = {
        "dataset_version": GENERIC_DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "count": row_count,
        "inputs": [str(path) for path in input_paths],
        "output": str(args.output),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "teacher_counts": dict(sorted(teacher_counts.items())),
        "teacher": next(iter(teacher_counts)) if len(teacher_counts) == 1 else "mixed",
        "unique_fields": list(UNIQUE_FIELDS),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
