"""Validate the derived question-generalization benchmark and freeze boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.pilot_v2 import validate_pilot_state
from fitz_tool.question_generalization_v2 import (
    QUESTION_GENERALIZATION_HOLDOUT_TEMPLATE_IDS,
    QUESTION_GENERALIZATION_TRAIN_TEMPLATE_IDS,
    canonical_question_leakage_markers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--derived-input", type=Path, required=True)
    parser.add_argument("--training-input", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=5000)
    return parser


def _read(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


def _structural_projection(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "matrix_version",
        "matrix_cell_id",
        "matrix_cell",
        "tool_registry",
        "legal_candidate_ids",
        "label",
        "accepted",
        "source_card_ids",
        "source_kind",
        "evaluation_cohort",
        "evaluation_partition",
        "sampling_context",
        "observed_evidence",
        "history",
        "plan",
        "governance",
        "resource_state",
        "source_state",
        "query_state",
        "step",
    )
    projection = {key: copy.deepcopy(row.get(key)) for key in keys}
    query_state = projection.get("query_state")
    if isinstance(query_state, dict):
        query_state.pop("query_terms", None)
    return projection


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frozen, frozen_errors = _read(args.frozen_input)
    derived, derived_errors = _read(args.derived_input)
    errors = [
        {"dataset": "frozen", **error} for error in frozen_errors
    ] + [{"dataset": "derived", **error} for error in derived_errors]
    structural_preservation_errors: list[dict[str, Any]] = []
    question_errors: list[dict[str, Any]] = []
    leakage_errors: list[dict[str, Any]] = []
    if len(frozen) == len(derived):
        for index, (original, candidate) in enumerate(zip(frozen, derived)):
            if _structural_projection(original) != _structural_projection(candidate):
                structural_preservation_errors.append({"index": index})
            cohort = str(original.get("evaluation_cohort"))
            transformed = cohort in {"train", "heldout_questions"}
            if transformed:
                if original.get("question") == candidate.get("question"):
                    question_errors.append({"index": index, "error": "question was not transformed"})
                template_id = str(candidate.get("question_template_id"))
                allowed = (
                    QUESTION_GENERALIZATION_TRAIN_TEMPLATE_IDS
                    if cohort == "train"
                    else QUESTION_GENERALIZATION_HOLDOUT_TEMPLATE_IDS
                )
                if template_id not in allowed:
                    question_errors.append(
                        {"index": index, "error": "unexpected template", "template_id": template_id}
                    )
                target = str((candidate.get("sampling_context") or {}).get("target_capability"))
                markers = canonical_question_leakage_markers(str(candidate.get("question", "")), target)
                if markers:
                    leakage_errors.append(
                        {"index": index, "target": target, "markers": markers}
                    )
            elif original != candidate:
                question_errors.append(
                    {"index": index, "error": "non-question-generalization row changed"}
                )
    train_templates = {
        str(row.get("question_template_id"))
        for row in derived
        if row.get("evaluation_cohort") == "train"
    }
    holdout_templates = {
        str(row.get("question_template_id"))
        for row in derived
        if row.get("evaluation_cohort") == "heldout_questions"
    }
    target_counts = Counter(
        str((row.get("sampling_context") or {}).get("target_capability"))
        for row in derived
    )
    type_counts = Counter(row.get("type_signature") for row in derived)
    instance_counts = Counter(row.get("instance_signature") for row in derived)
    derived_partition_counts = Counter(row.get("evaluation_partition") for row in derived)
    training_view: dict[str, Any] | None = None
    training_view_errors: list[dict[str, Any]] = []
    if args.training_input is not None:
        training_rows, training_view_errors = _read(args.training_input)
        training_partition_counts = Counter(
            row.get("evaluation_partition") for row in training_rows
        )
        training_template_counts = Counter(
            row.get("question_template_id") for row in training_rows
            if row.get("evaluation_partition") == "train"
        )
        training_type_counts = Counter(row.get("type_signature") for row in training_rows)
        training_instance_counts = Counter(
            row.get("instance_signature") for row in training_rows
        )
        training_view = {
            "rows": len(training_rows),
            "errors": len(training_view_errors),
            "partition_counts": dict(sorted(training_partition_counts.items())),
            "template_counts": dict(sorted(training_template_counts.items())),
            "unique_type_signatures": len(training_type_counts),
            "unique_instance_signatures": len(training_instance_counts),
            "valid": (
                not training_view_errors
                and training_partition_counts.get("train", 0)
                == derived_partition_counts.get("train", 0) * 2
                and training_partition_counts.get("validation", 0)
                == derived_partition_counts.get("validation", 0)
                and training_partition_counts.get("test", 0)
                == derived_partition_counts.get("test", 0)
                and max(training_type_counts.values(), default=0) == 1
                and max(training_instance_counts.values(), default=0) == 1
            ),
        }
    report = {
        "valid": (
            not errors
            and len(frozen) == args.expected_count
            and len(derived) == args.expected_count
            and not structural_preservation_errors
            and not question_errors
            and not leakage_errors
            and not (train_templates & holdout_templates)
            and max(type_counts.values(), default=0) == 1
            and max(instance_counts.values(), default=0) == 1
            and (training_view is None or training_view["valid"])
        ),
        "frozen_rows": len(frozen),
        "derived_rows": len(derived),
        "frozen_input_sha256": hashlib.sha256(args.frozen_input.read_bytes()).hexdigest(),
        "derived_input_sha256": hashlib.sha256(args.derived_input.read_bytes()).hexdigest(),
        "invalid_rows": len(errors),
        "errors": errors[:10],
        "structural_preservation_errors": structural_preservation_errors[:10],
        "question_errors": question_errors[:10],
        "leakage_errors": leakage_errors[:10],
        "target_capability_counts": dict(sorted(target_counts.items())),
        "cohort_counts": dict(sorted(Counter(row.get("evaluation_cohort") for row in derived).items())),
        "training_template_counts": dict(
            sorted(Counter(row.get("question_template_id") for row in derived if row.get("evaluation_cohort") == "train").items())
        ),
        "heldout_template_counts": dict(
            sorted(Counter(row.get("question_template_id") for row in derived if row.get("evaluation_cohort") == "heldout_questions").items())
        ),
        "train_holdout_template_overlap": sorted(train_templates & holdout_templates),
        "unique_type_signatures": len(type_counts),
        "unique_instance_signatures": len(instance_counts),
        "frozen_benchmark_preserved_except_questions": not structural_preservation_errors,
        "training_view": training_view,
        "training_view_errors": training_view_errors[:10],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
