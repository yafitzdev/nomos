"""Validate generic Nomos decision-state data and its holdout boundaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.generic_pilot_v3 import (
    GENERIC_COHORT_COUNTS,
    GENERIC_DATASET_VERSION,
    TARGET_CAPABILITIES,
    validate_generic_state,
)
from fitz_tool.tool_registry import ToolRegistry


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        report = validate_generic_state(row)
        if report.valid:
            rows.append(row)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    return rows, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=sum(GENERIC_COHORT_COUNTS.values()))
    parser.add_argument(
        "--require-external-teacher",
        "--require-ninfer-teacher",
        dest="require_external_teacher",
        action="store_true",
        help="Require every accepted row to contain approved NInfer or DeepSeek provenance.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, errors = _read_rows(args.input)
    cohort_counts = Counter(str(row.get("evaluation_cohort")) for row in rows)
    target_counts = Counter(str((row.get("sampling_context") or {}).get("target_capability")) for row in rows)
    template_counts = Counter(str(row.get("question_template_id")) for row in rows)
    type_signatures = [str(row.get("type_signature")) for row in rows]
    instance_signatures = [str(row.get("instance_signature")) for row in rows]
    matrix_cells = [str(row.get("matrix_cell_id")) for row in rows]
    questions = [str(row.get("question")) for row in rows]
    registry_fingerprints = Counter(str(row["tool_registry"]["registry_fingerprint"]) for row in rows)
    training_families = {
        tool.tool_family
        for row in rows
        if row.get("evaluation_cohort") == "train"
        for tool in ToolRegistry.from_dict(row["tool_registry"]).tools
    }
    heldout_family_families = {
        tool.tool_family
        for row in rows
        if row.get("evaluation_cohort") == "heldout_family"
        for tool in ToolRegistry.from_dict(row["tool_registry"]).tools
    }
    train_registry_fingerprints = {
        str(row["tool_registry"]["registry_fingerprint"])
        for row in rows
        if row.get("evaluation_cohort") == "train"
    }
    heldout_registry_fingerprints = {
        str(row["tool_registry"]["registry_fingerprint"])
        for row in rows
        if row.get("evaluation_cohort") in {"heldout_family", "alternate_registry"}
    }
    train_sources = {
        source_id
        for row in rows
        if row.get("evaluation_cohort") == "train"
        for source_id in row.get("source_card_ids") or []
    }
    heldout_sources = {
        source_id
        for row in rows
        if row.get("evaluation_cohort") == "heldout_sources"
        for source_id in row.get("source_card_ids") or []
    }
    question_train = {
        str(row.get("question_template_id"))
        for row in rows
        if row.get("evaluation_cohort") != "heldout_questions"
    }
    question_holdout = {
        str(row.get("question_template_id"))
        for row in rows
        if row.get("evaluation_cohort") == "heldout_questions"
    }
    teacher_pairs = {
        "ninfer_generic_teacher": "ninfer",
        "deepseek_generic_teacher": "deepseek",
    }
    teacher_errors = [
        str(row.get("decision_state_id"))
        for row in rows
        if row.get("source_kind") not in teacher_pairs
        or (row.get("provenance") or {}).get("teacher") != teacher_pairs[row.get("source_kind")]
    ]
    manifest_errors: list[str] = []
    manifest: dict[str, Any] | None = None
    if args.manifest:
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            manifest_errors.append(f"could not read manifest: {exc}")
        else:
            if manifest.get("count") != len(rows):
                manifest_errors.append(
                    f"manifest count {manifest.get('count')!r} does not match {len(rows)} rows"
                )
            if manifest.get("dataset_version") != GENERIC_DATASET_VERSION:
                manifest_errors.append("manifest dataset_version does not match generic dataset")
            if args.require_external_teacher and manifest.get("teacher") not in {
                "ninfer",
                "deepseek",
                "mixed",
            }:
                manifest_errors.append("manifest teacher must identify an approved external teacher")
    report = {
        "valid": not errors
        and len(rows) == args.expected_count
        and cohort_counts == Counter(GENERIC_COHORT_COUNTS)
        and len(type_signatures) == len(set(type_signatures)) == len(rows)
        and len(instance_signatures) == len(set(instance_signatures)) == len(rows)
        and len(matrix_cells) == len(set(matrix_cells)) == len(rows)
        and len(questions) == len(set(questions)) == len(rows)
        and set(target_counts) == set(TARGET_CAPABILITIES)
        and not (training_families & heldout_family_families)
        and not (train_registry_fingerprints & heldout_registry_fingerprints)
        and not (train_sources & heldout_sources)
        and not (question_train & question_holdout)
        and (not args.require_external_teacher or not teacher_errors)
        and not manifest_errors,
        "dataset_version": GENERIC_DATASET_VERSION,
        "rows": len(rows),
        "expected_rows": args.expected_count,
        "errors": errors[:20],
        "invalid_rows": len(errors),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "target_capability_counts": dict(sorted(target_counts.items())),
        "question_template_counts": dict(sorted(template_counts.items())),
        "unique_matrix_cells": len(set(matrix_cells)),
        "unique_type_signatures": len(set(type_signatures)),
        "unique_instance_signatures": len(set(instance_signatures)),
        "unique_questions": len(set(questions)),
        "duplicate_questions": len(questions) - len(set(questions)),
        "unique_registries": len(registry_fingerprints),
        "training_tool_families": sorted(training_families),
        "heldout_family_tool_families": sorted(heldout_family_families),
        "heldout_family_overlap": sorted(training_families & heldout_family_families),
        "training_registry_count": len(train_registry_fingerprints),
        "heldout_registry_overlap": sorted(train_registry_fingerprints & heldout_registry_fingerprints),
        "training_source_ids": sorted(train_sources),
        "heldout_source_overlap": sorted(train_sources & heldout_sources),
        "training_question_templates": sorted(question_train),
        "heldout_question_templates": sorted(question_holdout),
        "question_template_overlap": sorted(question_train & question_holdout),
        "teacher_errors": teacher_errors[:20],
        "teacher_error_count": len(teacher_errors),
        "manifest": manifest,
        "manifest_errors": manifest_errors,
        "project_specific_markers": ["fitz", "sage", "bm25"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
