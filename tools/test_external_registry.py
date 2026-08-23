"""Evaluate a trained router against a hand-authored unseen tool registry."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.router_v2 import load_router_v2, rank_tools_v2
from fitz_tool.tool_registry import ToolRegistry


DEFAULT_ARTIFACT = Path("artifacts/nomos_generic_ninfer_full.pt")
DEFAULT_INPUT = Path("data/generated/nomos_generic_ninfer_50000.jsonl")
DEFAULT_REGISTRY_ID = "external_spectrum_registry"

TARGET_CAPABILITIES = (
    "plan_retrieval",
    "list_sources",
    "search_content",
    "exact_pattern_search",
    "search_metadata",
    "inspect_structured_schema",
    "search_structured_records",
    "inspect_document_structure",
    "search_document_pages",
    "read_content",
    "inspect_code_structure",
    "inspect_evidence",
    "expand_context",
    "compare_evidence",
    "update_requirements",
    "assess_evidence",
    "finalize_selection",
)

TOOL_IDS = (
    "lumen_route",
    "quartz_catalog",
    "ember_probe",
    "velvet_match",
    "orbit_index",
    "harbor_schema",
    "cinder_rows",
    "atlas_outline",
    "ripple_pages",
    "meadow_reader",
    "forge_symbols",
    "mosaic_evidence",
    "canyon_context",
    "prism_compare",
    "ledger_requirements",
    "signal_assess",
    "northstar_commit",
)

DESCRIPTIONS = {
    "plan_retrieval": "Plan a staged retrieval route from the current research state.",
    "list_sources": "Enumerate the sources currently available for inspection.",
    "search_content": "Locate relevant passages in the available source content.",
    "exact_pattern_search": "Find exact identifiers, phrases, or literal patterns.",
    "search_metadata": "Filter and search source metadata and catalog fields.",
    "inspect_structured_schema": "Inspect fields, types, and structure of tabular data.",
    "search_structured_records": "Filter and retrieve matching structured records.",
    "inspect_document_structure": "Inspect headings, sections, and document organization.",
    "search_document_pages": "Locate relevant pages or page ranges in a document.",
    "read_content": "Read the full content of a selected source.",
    "inspect_code_structure": "Inspect symbols, definitions, and code relationships.",
    "inspect_evidence": "Inspect evidence support, provenance, and verification details.",
    "expand_context": "Expand a partial or ambiguous result with surrounding context.",
    "compare_evidence": "Compare conflicting evidence and identify meaningful differences.",
    "update_requirements": "Update requirement coverage and remaining obligations.",
    "assess_evidence": "Assess whether the available evidence is sufficient.",
    "finalize_selection": "Finalize the best-supported selection when requirements are met.",
}


def _external_registry() -> ToolRegistry:
    tools: list[dict[str, Any]] = []
    for tool_id, capability in zip(TOOL_IDS, TARGET_CAPABILITIES):
        tools.append(
            {
                "tool_id": tool_id,
                "tool_family": f"external_{capability}",
                "description": DESCRIPTIONS[capability],
                "capabilities": [capability],
                "input_modalities": ["text"],
                "output_modalities": [
                    "records"
                    if "structured" in capability
                    else "passages"
                    if capability not in {"list_sources", "plan_retrieval", "update_requirements"}
                    else "structured_summary"
                ],
                "evidence_roles": [
                    "planning"
                    if capability == "plan_retrieval"
                    else "selection"
                    if capability == "finalize_selection"
                    else "observation"
                ],
                "side_effect_class": "none",
                "argument_schema": {
                    "type": "object",
                    "properties": {"request": {"type": "string"}},
                    "required": ["request"],
                },
                "constraints": ["read_only"],
                "prerequisites": ["none"],
            }
        )
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": DEFAULT_REGISTRY_ID,
            "tools": tools,
        }
    )


def _read_external_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evaluation_cohort") != "heldout_questions":
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("input does not contain heldout_questions rows")
    return rows


def _external_state(
    row: dict[str, Any],
    registry: ToolRegistry,
    *,
    seed: int,
    index: int,
) -> tuple[dict[str, Any], str]:
    target = str((row.get("sampling_context") or {}).get("target_capability"))
    if target not in TARGET_CAPABILITIES:
        raise ValueError(f"unsupported target capability in row: {target}")
    target_id = TOOL_IDS[TARGET_CAPABILITIES.index(target)]
    distractors = [tool.tool_id for tool in registry.tools if tool.tool_id != target_id]
    random.Random(seed + index * 7919).shuffle(distractors)
    legal_ids = [target_id, *distractors[:6]]
    random.Random(seed + index * 104729).shuffle(legal_ids)
    output = copy.deepcopy(row)
    output["tool_registry"] = registry.as_dict()
    output["legal_candidate_ids"] = legal_ids
    output["accepted"] = True
    output["evaluation_cohort"] = "external_registry"
    output["evaluation_partition"] = "test"
    output["label"] = {
        "acceptable_tools": [target_id],
        "hard_negative_tools": [tool_id for tool_id in legal_ids if tool_id != target_id],
        "ranked_tools": [target_id, *[tool_id for tool_id in legal_ids if tool_id != target_id]],
        "label_source": "external_registry_test_oracle",
    }
    return output, target_id


def _metrics(rows: list[dict[str, Any]], model: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    recall_1 = recall_3 = 0
    reciprocal_rank = 0.0
    invalid = 0
    for row in rows:
        legal = set(row["legal_candidate_ids"])
        ranked = [item["tool_id"] for item in rank_tools_v2(model, metadata, row, top_k=len(legal))]
        invalid += int(not set(ranked).issubset(legal))
        target = set(row["label"]["acceptable_tools"])
        recall_1 += int(bool(set(ranked[:1]) & target))
        recall_3 += int(bool(set(ranked[:3]) & target))
        first = next((position for position, tool_id in enumerate(ranked, 1) if tool_id in target), None)
        if first is not None:
            reciprocal_rank += 1.0 / first
    count = len(rows)
    return {
        "states": count,
        "recall_at_1": recall_1 / count,
        "recall_at_3": recall_3 / count,
        "mrr": reciprocal_rank / count,
        "invalid_candidate_rate": invalid / count,
    }


def _candidate_order_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recall_1 = recall_3 = 0
    reciprocal_rank = 0.0
    for row in rows:
        ranked = list(row["legal_candidate_ids"])
        target = set(row["label"]["acceptable_tools"])
        recall_1 += int(bool(set(ranked[:1]) & target))
        recall_3 += int(bool(set(ranked[:3]) & target))
        first = next((position for position, tool_id in enumerate(ranked, 1) if tool_id in target), None)
        if first is not None:
            reciprocal_rank += 1.0 / first
    count = len(rows)
    return {
        "states": count,
        "recall_at_1": recall_1 / count,
        "recall_at_3": recall_3 / count,
        "mrr": reciprocal_rank / count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    model, metadata = load_router_v2(str(args.artifact))
    registry = _external_registry()
    source_rows = _read_external_rows(args.input, args.limit)
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows):
        state, _ = _external_state(source_row, registry, seed=args.seed, index=index)
        rows.append(state)
    report: dict[str, Any] = {
        "artifact": str(args.artifact),
        "registry_id": registry.registry_id,
        "registry_fingerprint": registry.fingerprint,
        "registry_tool_ids_are_unseen": True,
        "source_cohort": "heldout_questions",
        "candidate_order_baseline": _candidate_order_metrics(rows),
        "metrics": _metrics(rows, model, metadata),
        "by_target_capability": {},
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = str((row.get("sampling_context") or {}).get("target_capability"))
        groups[target].append(row)
    for target, target_rows in sorted(groups.items()):
        report["by_target_capability"][target] = _metrics(target_rows, model, metadata)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["metrics"]["invalid_candidate_rate"] == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
