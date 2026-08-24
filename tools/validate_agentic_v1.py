"""Validate an agentic v1 JSONL corpus and audit a deterministic sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.agentic_pilot import validate_agentic_state


UNIQUE_FIELDS = (
    "decision_state_id",
    "matrix_cell_id",
    "type_signature",
    "instance_signature",
    "question",
    "teacher_paraphrase",
)

# decision_state_id is intentionally cohort-local in older generated pilots.
# Semantic signatures are the cross-cohort ledger keys: they are what prevent
# the same matrix/type/instance from being silently generated twice. Wording
# overlap is reported as a warning because distinct states can legitimately
# share a short natural-language surface.
CROSS_COHORT_FIELDS = (
    "matrix_cell_id",
    "type_signature",
    "instance_signature",
)
CROSS_COHORT_WORDING_FIELDS = (
    "question",
    "teacher_paraphrase",
)


def _sample_indices(count: int, sample_size: int, seed: int) -> list[int]:
    if count <= sample_size:
        return list(range(count))
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return sorted(indices[:sample_size])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="Existing JSONL cohort(s) whose semantic signatures must not overlap.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_size < 1:
        raise SystemExit("sample-size must be positive")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen = {field: set[str]() for field in UNIQUE_FIELDS}
    excluded = {field: set[str]() for field in CROSS_COHORT_FIELDS}
    excluded_wording = {field: set[str]() for field in CROSS_COHORT_WORDING_FIELDS}
    excluded_counts: Counter[str] = Counter()
    wording_overlap_counts: Counter[str] = Counter()
    excluded_errors: list[dict[str, Any]] = []
    for excluded_path in args.exclude:
        try:
            with excluded_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        excluded_row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        excluded_errors.append(
                            {"input": str(excluded_path), "line": line_number, "error": str(exc)}
                        )
                        continue
                    if not isinstance(excluded_row, dict):
                        excluded_errors.append(
                            {"input": str(excluded_path), "line": line_number, "error": "row must be an object"}
                        )
                        continue
                    for field in CROSS_COHORT_FIELDS:
                        value = " ".join(str(excluded_row.get(field) or "").casefold().split())
                        if value:
                            excluded[field].add(value)
                    for field in CROSS_COHORT_WORDING_FIELDS:
                        value = " ".join(str(excluded_row.get(field) or "").casefold().split())
                        if value:
                            excluded_wording[field].add(value)
        except OSError as exc:
            excluded_errors.append({"input": str(excluded_path), "error": str(exc)})
    dimensions: Counter[str] = Counter()
    teachers: Counter[str] = Counter()
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append({"line": line_number, "error": "blank line"})
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
            report = validate_agentic_state(row)
            if not report.valid:
                errors.append({"line": line_number, "validation": report.as_dict()})
            for field in UNIQUE_FIELDS:
                value = " ".join(str(row.get(field) or "").casefold().split())
                if not value:
                    errors.append({"line": line_number, "error": f"missing {field}"})
                elif value in seen[field]:
                    errors.append({"line": line_number, "error": f"duplicate {field}"})
                seen[field].add(value)
            for field in CROSS_COHORT_FIELDS:
                value = " ".join(str(row.get(field) or "").casefold().split())
                if value and value in excluded[field]:
                    excluded_counts[field] += 1
                    errors.append({"line": line_number, "error": f"overlap with excluded cohort: {field}"})
            for field in CROSS_COHORT_WORDING_FIELDS:
                value = " ".join(str(row.get(field) or "").casefold().split())
                if value and value in excluded_wording[field]:
                    wording_overlap_counts[field] += 1
            cell = row.get("matrix_cell") or {}
            for key in (
                "task_kind",
                "candidate_pool_size",
                "top_k",
                "validation_case",
                "expansion_trigger",
                "recovery_round",
                "unseen_axis",
            ):
                dimensions[f"{key}={cell.get(key)}"] += 1
            teachers[str((row.get("provenance") or {}).get("teacher"))] += 1
    sample_rows: list[dict[str, Any]] = []
    sample_failures: list[dict[str, Any]] = []
    for index in _sample_indices(len(rows), args.sample_size, args.seed):
        row = rows[index]
        report = validate_agentic_state(row)
        sample_rows.append(
            {
                "line": index + 1,
                "decision_state_id": row.get("decision_state_id"),
                "task_kind": row.get("task_kind"),
                "question": row.get("question"),
                "validation_case": (row.get("matrix_cell") or {}).get("validation_case"),
                "valid": report.valid,
                "row_sha256": hashlib.sha256(
                    json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        )
        if not report.valid:
            sample_failures.append({"line": index + 1, "report": report.as_dict()})
    result = {
        "input": str(args.input),
        "excluded_inputs": [str(path) for path in args.exclude],
        "excluded_errors": excluded_errors[:50],
        "cross_cohort_overlap_counts": dict(sorted(excluded_counts.items())),
        "cross_cohort_wording_overlap_counts": dict(sorted(wording_overlap_counts.items())),
        "rows": len(rows),
        "expected_count": args.expected_count,
        "count_match": args.expected_count is None or len(rows) == args.expected_count,
        "invalid_rows": len(errors),
        "errors": errors[:50],
        "unique_counts": {field: len(values) for field, values in seen.items()},
        "teachers": dict(sorted(teachers.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "sample_size": len(sample_rows),
        "sample_failures": sample_failures,
        "sample": sample_rows,
        "valid": not errors and not excluded_errors and not sample_failures and (
            args.expected_count is None or len(rows) == args.expected_count
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
