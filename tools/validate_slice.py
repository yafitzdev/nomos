"""Validate a scenario JSONL slice and emit a stratified audit manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.audit import select_stratified_sample
from fitz_tool.contracts import validate_scenario, validate_source_card
from fitz_tool.uniqueness import duplicate_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--audit-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--source-card",
        action="append",
        type=Path,
        default=[],
        help="Source-card JSON object; repeat for multi-source rows.",
    )
    parser.add_argument(
        "--source-card-manifest",
        action="append",
        type=Path,
        default=[],
        help="JSONL source-card manifest; repeat for additional manifests.",
    )
    return parser


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row must be a JSON object")
            rows.append(value)
        except (json.JSONDecodeError, ValueError) as exc:
            parse_errors.append({"line": line_number, "error": str(exc)})
    return rows, parse_errors


def _load_cards(paths: Sequence[Path], manifests: Sequence[Path]) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}

    def add(card: Mapping[str, Any], source: str) -> None:
        report = validate_source_card(card)
        if not report.valid:
            raise ValueError(f"invalid source card {source}: {report.as_dict()}")
        source_id = str(card["source_id"])
        if source_id in cards:
            raise ValueError(f"duplicate source card ID: {source_id}")
        cards[source_id] = dict(card)

    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"source card is not an object: {path}")
        add(value, str(path))
    for path in manifests:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"source-card manifest row is not an object: {path}:{line_number}")
            add(value, f"{path}:{line_number}")
    return cards


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, parse_errors = _read_jsonl(args.input)
    invalid_rows: list[dict[str, Any]] = list(parse_errors)
    source_cards: dict[str, dict[str, Any]] = {}
    if args.source_card or args.source_card_manifest:
        try:
            source_cards = _load_cards(args.source_card, args.source_card_manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid_rows.append({"source_cards": str(exc)})
    for index, row in enumerate(rows):
        report = validate_scenario(row)
        if not report.valid:
            invalid_rows.append({"row": index, **report.as_dict()})
        if source_cards:
            row_source_ids = {
                str(source_id)
                for source_id in row.get("source_card_ids", [])
                if isinstance(source_id, str)
            }
            missing_cards = sorted(row_source_ids - set(source_cards))
            if missing_cards:
                invalid_rows.append({"row": index, "error": f"unknown source cards: {missing_cards}"})
            row_modality = row.get("matrix_cell", {}).get("source_modality")
            hashes = {str(value) for value in row.get("provenance", {}).get("source_card_hashes", [])}
            fact_index = {
                (source_id, str(fact.get("fact_id")))
                for source_id in row_source_ids
                for fact in source_cards.get(source_id, {}).get("facts", [])
                if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
            }
            for source_id in sorted(row_source_ids):
                card = source_cards.get(source_id)
                if card is None:
                    continue
                if row_modality != "mixed" and card.get("modality") != row_modality:
                    invalid_rows.append(
                        {
                            "row": index,
                            "error": f"source card {source_id!r} modality {card.get('modality')!r} does not match cell modality {row_modality!r}",
                        }
                    )
                valid_hashes = {str(card.get("content_sha256"))}
                if card.get("normalized_content_sha256"):
                    valid_hashes.add(str(card["normalized_content_sha256"]))
                if not hashes & valid_hashes:
                    invalid_rows.append({"row": index, "error": f"source-card hash missing for {source_id}"})
            for fact in row.get("expected_facts", []):
                if isinstance(fact, dict):
                    key = (str(fact.get("source_id")), str(fact.get("fact_id")))
                    if key not in fact_index:
                        invalid_rows.append({"row": index, "error": f"unknown source-card fact: {key}"})

    duplicate_types = sorted(duplicate_values(rows, "type_signature"))
    duplicate_instances = sorted(duplicate_values(rows, "instance_signature"))
    if duplicate_types:
        invalid_rows.append({"error": "duplicate_type_signatures", "values": duplicate_types})
    if duplicate_instances:
        invalid_rows.append({"error": "duplicate_instance_signatures", "values": duplicate_instances})

    sample_indices = select_stratified_sample(rows, args.audit_size, seed=args.seed)
    audit_manifest = {
        "manifest_version": "audit-manifest.v1",
        "source": str(args.input),
        "seed": args.seed,
        "sample_size": len(sample_indices),
        "status": "pending_external_validation",
        "rows": [
            {
                "row_index": index,
                "scenario_id": rows[index].get("scenario_id"),
                "matrix_cell": rows[index].get("matrix_cell"),
            }
            for index in sample_indices
        ],
    }
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit_manifest, indent=2, sort_keys=True), encoding="utf-8")

    coverage: dict[str, Counter[str]] = {}
    for row in rows:
        cell = row.get("matrix_cell", {})
        for axis, value in cell.items():
            if axis == "cell_id":
                continue
            coverage.setdefault(axis, Counter())[str(value)] += 1
    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "valid_rows": len(rows) - sum(1 for item in invalid_rows if "row" in item),
        "parse_errors": len(parse_errors),
        "invalid_items": len(invalid_rows),
        "duplicate_type_signatures": len(duplicate_types),
        "duplicate_instance_signatures": len(duplicate_instances),
        "audit_manifest": str(args.audit_output) if args.audit_output else None,
        "source_card_count": len(source_cards),
        "audit_sample_size": len(sample_indices),
        "coverage": {axis: dict(counts) for axis, counts in sorted(coverage.items())},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if invalid_rows:
        print(json.dumps({"validation_errors": invalid_rows[:10]}, indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
