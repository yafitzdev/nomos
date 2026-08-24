"""Independently validate the complete accepted scaling cohort and holdout boundary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.scaling_matrix_v1 import (
    DATASET_VERSION,
    MATRIX_VERSION,
    PROJECT_MARKER,
    load_scaling_matrix,
    normalized_question,
    semantic_signature,
)


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_scaling_matrix()
    holdout = list(_iter_rows(args.holdout))
    errors: list[dict[str, Any]] = []
    identities: dict[str, list[str]] = {
        "decision_state_id": [],
        "matrix_cell_id": [],
        "question": [],
        "semantic_signature": [],
        "registry_fingerprint": [],
        "source_row_hash": [],
        "question_template_id": [],
        "scenario_id": [],
    }
    actual: dict[str, Counter[str]] = {"scenario_family": Counter()}
    actual.update({name: Counter() for name in spec["dimension_counts"]})
    row_count = 0
    fallback_count = 0
    for line_number, row in enumerate(_iter_rows(args.input), start=1):
        row_count += 1
        report = validate_decision_state_v2(row)
        if not report.valid:
            errors.append({"line": line_number, "reason": "invalid_contract", "detail": report.as_dict()})
        if row.get("dataset_version") != DATASET_VERSION or row.get("matrix_cell", {}).get("matrix_version") != MATRIX_VERSION:
            errors.append({"line": line_number, "reason": "wrong_version"})
        if row.get("evaluation_partition") != "train" or row.get("accepted") is not True:
            errors.append({"line": line_number, "reason": "wrong_partition_or_acceptance"})
        if (row.get("provenance") or {}).get("teacher_fallback_used") is not False:
            errors.append({"line": line_number, "reason": "teacher_fallback"})
            fallback_count += 1
        text = f"{row.get('question', '')}\n{row.get('teacher_paraphrase', '')}"
        if PROJECT_MARKER.search(text):
            errors.append({"line": line_number, "reason": "project_language"})
        if any(
            str(tool.get("tool_id") or "").casefold() in text.casefold()
            for tool in row.get("tool_registry", {}).get("tools", [])
            if isinstance(tool, dict) and tool.get("tool_id")
        ):
            errors.append({"line": line_number, "reason": "tool_id_leakage"})
        cell = row["matrix_cell"]
        for name in actual:
            actual[name][str(cell[name])] += 1
        provenance = row.get("provenance") or {}
        identities["decision_state_id"].append(str(row.get("decision_state_id") or ""))
        identities["matrix_cell_id"].append(str(row.get("matrix_cell_id") or ""))
        identities["question"].append(normalized_question(str(row.get("question") or "")))
        identities["semantic_signature"].append(semantic_signature(str(row.get("question") or ""), str(row.get("teacher_paraphrase") or "")))
        identities["registry_fingerprint"].append(str(provenance.get("registry_fingerprint") or ""))
        identities["source_row_hash"].append(str(provenance.get("source_row_hash") or ""))
        identities["question_template_id"].append(str(row.get("question_template_id") or ""))
        identities["scenario_id"].append(str(row.get("scenario_id") or ""))
    target = int(spec["target_accepted_rows"])
    if row_count != target:
        errors.append({"reason": "wrong_row_count", "expected": target, "actual": row_count})
    unique_counts = {name: len(set(values)) for name, values in identities.items()}
    for name in ("decision_state_id", "matrix_cell_id", "question", "semantic_signature", "registry_fingerprint", "source_row_hash"):
        if unique_counts[name] != target:
            errors.append({"reason": f"non_unique_{name}", "unique": unique_counts[name]})
    expected = {"scenario_family": spec["scenario_family_counts"], **spec["dimension_counts"]}
    for name, counts in actual.items():
        wanted = {str(key): int(value) for key, value in expected[name].items()}
        if dict(counts) != wanted:
            errors.append({"reason": f"distribution_mismatch_{name}", "actual": dict(counts), "expected": wanted})
    holdout_sets = {
        "question": {normalized_question(str(row.get("question") or "")) for row in holdout},
        "registry_fingerprint": {str((row.get("provenance") or {}).get("registry_fingerprint") or "") for row in holdout},
        "source_row_hash": {str((row.get("provenance") or {}).get("source_row_hash") or "") for row in holdout},
        "question_template_id": {str(row.get("question_template_id") or "") for row in holdout},
        "scenario_id": {str(row.get("scenario_id") or "") for row in holdout},
    }
    overlap = {
        name: sorted(set(identities[name]) & values)
        for name, values in holdout_sets.items()
    }
    if any(overlap.values()):
        errors.append({"reason": "frozen_holdout_overlap", "counts": {name: len(values) for name, values in overlap.items()}})
    report = {
        "input": str(args.input),
        "holdout": str(args.holdout),
        "valid": not errors,
        "row_count": row_count,
        "contract_valid_rows": row_count - sum(value.get("reason") == "invalid_contract" for value in errors),
        "unique_counts": unique_counts,
        "dimension_counts": {name: dict(sorted(values.items())) for name, values in actual.items()},
        "holdout_overlap_counts": {name: len(values) for name, values in overlap.items()},
        "teacher_fallback_rows": fallback_count,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"dimension_counts", "errors"}}, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
