"""Audit a deterministic sample of accepted generic NInfer rows."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.generic_pilot_v3 import validate_generic_state


PROJECT_MARKER_RE = re.compile(r"(?<![a-z0-9_])(fitz|sage|bm25)(?![a-z0-9_])", re.IGNORECASE)


def _reservoir_sample(
    path: Path, size: int, seed: int
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    total = 0
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue
            total += 1
            if len(sample) < size:
                sample.append(row)
            else:
                replacement = rng.randrange(total)
                if replacement < size:
                    sample[replacement] = row
    return sample, total, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be positive")
    sample, total, errors = _reservoir_sample(args.input, args.sample_size, args.seed)
    row_errors: list[dict[str, Any]] = []
    question_lengths: list[int] = []
    paraphrase_lengths: list[int] = []
    for row in sample:
        report = validate_generic_state(row)
        if not report.valid:
            row_errors.append(
                {"decision_state_id": row.get("decision_state_id"), "validation": report.as_dict()}
            )
        question = str(row.get("question") or "")
        paraphrase = str(row.get("teacher_paraphrase") or "")
        question_lengths.append(len(question))
        paraphrase_lengths.append(len(paraphrase))
        for field, value in (("question", question), ("teacher_paraphrase", paraphrase)):
            if not 20 <= len(value) <= 320:
                row_errors.append(
                    {
                        "decision_state_id": row.get("decision_state_id"),
                        "field": field,
                        "error": "length outside 20..320",
                    }
                )
            if PROJECT_MARKER_RE.search(value):
                row_errors.append(
                    {
                        "decision_state_id": row.get("decision_state_id"),
                        "field": field,
                        "error": "project-specific marker",
                    }
                )
        provenance = row.get("provenance") or {}
        expected_teacher = {
            "ninfer_generic_teacher": "ninfer",
            "deepseek_generic_teacher": "deepseek",
        }.get(str(row.get("source_kind")))
        if expected_teacher is None or provenance.get("teacher") != expected_teacher:
            row_errors.append(
                {
                    "decision_state_id": row.get("decision_state_id"),
                    "error": "row is not marked as an approved external-teacher generation",
                }
            )
    report = {
        "valid": not errors and not row_errors and len(sample) == min(args.sample_size, total),
        "input": str(args.input),
        "seed": args.seed,
        "total_valid_json_rows": total,
        "sample_size": len(sample),
        "json_errors": errors[:20],
        "row_errors": row_errors[:20],
        "cohort_counts": dict(
            sorted(Counter(str(row.get("evaluation_cohort")) for row in sample).items())
        ),
        "question_length_range": [min(question_lengths), max(question_lengths)]
        if question_lengths
        else [],
        "paraphrase_length_range": [min(paraphrase_lengths), max(paraphrase_lengths)]
        if paraphrase_lengths
        else [],
        "sample_ids": [row.get("decision_state_id") for row in sample],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
