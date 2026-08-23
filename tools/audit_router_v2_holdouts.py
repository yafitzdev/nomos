"""Audit frozen pilot cohort and descriptor/source/template holdout isolation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.tool_registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _set(rows: list[dict[str, Any]], cohort: str, field: str) -> set[str]:
    return {
        str(row.get(field))
        for row in rows
        if row.get("evaluation_cohort") == cohort and row.get(field) is not None
    }


def _source_ids(rows: list[dict[str, Any]], cohort: str) -> set[str]:
    return {
        str(source_id)
        for row in rows
        if row.get("evaluation_cohort") == cohort
        for source_id in row.get("source_card_ids") or []
    }


def _tool_ids(rows: list[dict[str, Any]], cohort: str) -> set[str]:
    return {
        str(tool_id)
        for row in rows
        if row.get("evaluation_cohort") == cohort
        for tool_id in row.get("legal_candidate_ids") or []
    }


def _families(rows: list[dict[str, Any]], cohort: str) -> set[str]:
    output: set[str] = set()
    for row in rows:
        if row.get("evaluation_cohort") != cohort:
            continue
        registry = ToolRegistry.from_dict(row["tool_registry"])
        output.update(registry.require(tool_id).tool_family for tool_id in row["legal_candidate_ids"])
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _read(args.input)
    training = [row for row in rows if row.get("evaluation_partition") == "train"]
    training_sources = _source_ids(rows, "train")
    training_templates = _set(rows, "train", "question_template_id")
    training_tools = _tool_ids(rows, "train")
    training_families = _families(rows, "train")
    holdout_source_overlap = sorted(training_sources & _source_ids(rows, "heldout_sources"))
    holdout_question_overlap = sorted(training_templates & _set(rows, "heldout_questions", "question_template_id"))
    heldout_family_overlap = sorted(training_families & _families(rows, "heldout_family"))
    unseen_id_overlap = sorted(training_tools & _tool_ids(rows, "unseen_tool_ids"))
    rename_overlap = sorted(training_tools & _tool_ids(rows, "id_renames"))
    all_cells = [str(row.get("matrix_cell_id")) for row in rows]
    all_types = [str(row.get("type_signature")) for row in rows]
    all_instances = [str(row.get("instance_signature")) for row in rows]
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_groups[str(row.get("evaluation_partition"))].add(str(row.get("split_group_id")))
    train_test_group_overlap = sorted(split_groups["train"] & split_groups["test"])
    report = {
        "valid": (
            not holdout_source_overlap
            and not holdout_question_overlap
            and not heldout_family_overlap
            and not unseen_id_overlap
            and not rename_overlap
            and len(all_cells) == len(set(all_cells))
            and len(all_types) == len(set(all_types))
            and len(all_instances) == len(set(all_instances))
            and not train_test_group_overlap
        ),
        "rows": len(rows),
        "training_rows": len(training),
        "training_sources": sorted(training_sources),
        "holdout_source_overlap": holdout_source_overlap,
        "holdout_question_overlap": holdout_question_overlap,
        "heldout_family_overlap": heldout_family_overlap,
        "unseen_tool_id_overlap": unseen_id_overlap,
        "id_rename_overlap_with_training": rename_overlap,
        "train_test_split_group_overlap": train_test_group_overlap,
        "unique_matrix_cells": len(set(all_cells)),
        "unique_type_signatures": len(set(all_types)),
        "unique_instance_signatures": len(set(all_instances)),
        "training_tool_families": sorted(training_families),
        "heldout_tool_families": sorted(_families(rows, "heldout_family")),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
