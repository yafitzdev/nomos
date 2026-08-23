"""Canonical identifiers and duplicate/type-signature checks."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def type_signature(scenario: dict[str, Any]) -> str:
    """Identify the semantic matrix/source type, independent of wording."""

    return stable_hash(
        {
            "matrix_cell": scenario.get("matrix_cell"),
            "source_card_ids": sorted(scenario.get("source_card_ids", [])),
            "expected_facts": sorted(
                scenario.get("expected_facts", []),
                key=lambda item: (item.get("source_id", ""), item.get("fact_id", "")),
            ),
            "expected_terminal_state": scenario.get("expected_terminal_state"),
        }
    )


def instance_signature(scenario: dict[str, Any]) -> str:
    """Identify an exact generated instance, including wording variants."""

    return stable_hash(
        {
            "type_signature": type_signature(scenario),
            "question": normalize_text(str(scenario.get("question", ""))),
            "difficult_paraphrase": normalize_text(
                str(scenario.get("difficult_paraphrase", ""))
            ),
        }
    )


def annotate_signatures(scenario: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(scenario)
    annotated["type_signature"] = type_signature(annotated)
    annotated["instance_signature"] = instance_signature(annotated)
    return annotated


def duplicate_values(items: Iterable[dict[str, Any]], field: str) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        value = item.get(field)
        if not isinstance(value, str):
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
