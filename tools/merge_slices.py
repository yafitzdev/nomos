"""Merge scenario slices, retaining only valid and unique rows up to a target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.contracts import validate_scenario, validate_source_card


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--source-card", type=Path)
    return parser


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_count < 1:
        raise SystemExit("target-count must be positive")
    source_card = None
    fact_ids: set[str] = set()
    if args.source_card:
        source_card = json.loads(args.source_card.read_text(encoding="utf-8"))
        report = validate_source_card(source_card)
        if not report.valid:
            raise SystemExit(json.dumps(report.as_dict(), indent=2))
        fact_ids = {
            fact["fact_id"]
            for fact in source_card["facts"]
            if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
        }

    accepted: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    seen_instances: set[str] = set()
    rejected = 0
    for path in args.inputs:
        for row in _rows(path):
            report = validate_scenario(row)
            if source_card is not None:
                if source_card["source_id"] not in row.get("source_card_ids", []):
                    report.add("source_card_ids", "row does not reference supplied source card")
                if (
                    source_card["modality"] != "mixed"
                    and row.get("matrix_cell", {}).get("source_modality") != source_card["modality"]
                ):
                    report.add("matrix_cell.source_modality", "does not match supplied source card")
                if source_card["content_sha256"] not in row.get("provenance", {}).get("source_card_hashes", []):
                    report.add("provenance.source_card_hashes", "missing supplied source-card hash")
                for fact in row.get("expected_facts", []):
                    if fact.get("fact_id") not in fact_ids:
                        report.add("expected_facts", "unknown source-card fact")
            type_id = row.get("type_signature")
            instance_id = row.get("instance_signature")
            if type_id in seen_types:
                report.add("type_signature", "duplicate type signature")
            if instance_id in seen_instances:
                report.add("instance_signature", "duplicate instance signature")
            if report.valid and len(accepted) < args.target_count:
                accepted.append(row)
                seen_types.add(type_id)
                seen_instances.add(instance_id)
            else:
                rejected += 1
            if len(accepted) == args.target_count:
                break
        if len(accepted) == args.target_count:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "target_count": args.target_count,
                "rejected_or_skipped": rejected,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if len(accepted) == args.target_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
