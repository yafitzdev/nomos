"""Emit deterministic bootstrap trajectories for validated matrix scenarios.

This runner is intentionally not a Fitz-Sage V2 execution trace. It is a
development oracle for the first encoder: the matrix assigns the target tool,
while this process verifies scenario legality, source-card identity, evidence
references, governance state, and terminal consistency before emitting one
accepted synthetic decision state per scenario. V2-runner labels remain a
separate higher-fidelity gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.contracts import (
    CODE_TOOLS,
    PDF_TOOLS,
    TABLE_TOOLS,
    TOOL_NAMES,
    validate_scenario,
    validate_source_card,
)
from fitz_tool.uniqueness import stable_hash


ORACLE_VERSION = "matrix-oracle.v1"

STATE_TOOL_FAMILIES: dict[str, tuple[str, ...]] = {
    "initial": (
        "set_retrieval_plan",
        "search_bm25",
        "grep_search",
        "search_metadata",
        "list_sources",
        "list_tabular_sources",
        "list_pdf_sources",
        "read_file",
        "inspect_code",
    ),
    "no_hits": (
        "set_retrieval_plan",
        "search_bm25",
        "grep_search",
        "search_metadata",
        "inspect_evidence",
        "expand_context",
        "finalize_document_selection",
    ),
    "noisy_hits": (
        "search_bm25",
        "grep_search",
        "search_metadata",
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "assess_evidence",
    ),
    "partial_evidence": (
        "search_bm25",
        "grep_search",
        "search_metadata",
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "update_requirement_progress",
        "assess_evidence",
    ),
    "expansion_needed": (
        "search_bm25",
        "grep_search",
        "search_metadata",
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "update_requirement_progress",
        "assess_evidence",
    ),
    "contradiction": (
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "update_requirement_progress",
        "assess_evidence",
        "finalize_document_selection",
    ),
    "insufficient": (
        "search_bm25",
        "grep_search",
        "search_metadata",
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "update_requirement_progress",
        "assess_evidence",
        "finalize_document_selection",
    ),
    "disputed": (
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "update_requirement_progress",
        "assess_evidence",
        "finalize_document_selection",
    ),
    "fresh_sufficient": (
        "inspect_evidence",
        "update_requirement_progress",
        "assess_evidence",
        "finalize_document_selection",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-card",
        action="append",
        type=Path,
        default=[],
        help="One or more source-card JSON objects, repeated for mixed corpora.",
    )
    parser.add_argument(
        "--source-card-manifest",
        action="append",
        type=Path,
        default=[],
        help="JSONL source-card manifest; repeat for additional manifests.",
    )
    return parser


def _load_cards(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for path in paths:
        card = json.loads(path.read_text(encoding="utf-8"))
        report = validate_source_card(card)
        if not report.valid:
            raise ValueError(f"invalid source card {path}: {report.as_dict()}")
        source_id = str(card["source_id"])
        if source_id in cards:
            raise ValueError(f"duplicate source card ID: {source_id}")
        cards[source_id] = card
    return cards


def _load_manifest_cards(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            card = json.loads(line)
            if not isinstance(card, dict):
                raise ValueError(f"source-card manifest row is not an object: {path}:{line_number}")
            report = validate_source_card(card)
            if not report.valid:
                raise ValueError(f"invalid source card {path}:{line_number}: {report.as_dict()}")
            source_id = str(card["source_id"])
            if source_id in cards:
                raise ValueError(f"duplicate source card ID: {source_id}")
            cards[source_id] = card
    return cards


def _compatible(tool: str, modality: str) -> bool:
    if tool in PDF_TOOLS:
        return modality in {"pdf", "mixed"}
    if tool in TABLE_TOOLS:
        return modality in {"csv", "excel", "sqlite", "mixed"}
    if tool in CODE_TOOLS:
        return modality in {"code", "mixed"}
    return True


def legal_tools_for_cell(cell: Mapping[str, Any]) -> list[str]:
    """Return the deterministic bootstrap candidate set for a matrix state."""

    state = str(cell["agent_state"])
    modality = str(cell["source_modality"])
    candidates = [
        tool
        for tool in STATE_TOOL_FAMILIES[state]
        if tool in TOOL_NAMES and _compatible(tool, modality)
    ]
    target = str(cell["next_tool_target"])
    if target not in candidates:
        candidates.insert(0, target)
    return list(dict.fromkeys(candidates))


def _fact_index(cards: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (source_id, str(fact["fact_id"])): fact
        for source_id, card in cards.items()
        for fact in card["facts"]
    }


def _validate_grounding(scenario: Mapping[str, Any], cards: Mapping[str, Mapping[str, Any]]) -> None:
    report = validate_scenario(scenario)
    if not report.valid:
        raise ValueError(report.as_dict())
    source_ids = {str(value) for value in scenario["source_card_ids"]}
    missing_cards = sorted(source_ids - set(cards))
    if missing_cards:
        raise ValueError(f"scenario references missing source cards: {missing_cards}")
    hashes = {str(value) for value in scenario["provenance"]["source_card_hashes"]}
    for source_id in source_ids:
        card = cards[source_id]
        valid_hashes = {str(card["content_sha256"])}
        if card.get("normalized_content_sha256"):
            valid_hashes.add(str(card["normalized_content_sha256"]))
        if not hashes & valid_hashes:
            raise ValueError(f"scenario provenance does not identify source card {source_id}")
    facts = _fact_index(cards)
    for reference in scenario.get("expected_facts", []):
        key = (str(reference["source_id"]), str(reference["fact_id"]))
        if key not in facts:
            raise ValueError(f"scenario references missing fact {key}")
    setup = scenario["state_setup"]
    for evidence in setup.get("observed_evidence", []):
        source_id = str(evidence.get("source_id"))
        for fact_id in evidence.get("fact_ids", []):
            if (source_id, str(fact_id)) not in facts:
                raise ValueError(f"state setup references missing fact {(source_id, fact_id)}")


def oracle_trajectory(
    scenario: Mapping[str, Any],
    cards: Mapping[str, Mapping[str, Any]],
    *,
    captured_at: str,
) -> dict[str, Any]:
    _validate_grounding(scenario, cards)
    cell = scenario["matrix_cell"]
    legal_tools = legal_tools_for_cell(cell)
    target = str(cell["next_tool_target"])
    if target not in legal_tools:
        raise ValueError(f"oracle target is not legal: {target}")
    hard_negatives = [tool for tool in legal_tools if tool != target]
    setup = scenario["state_setup"]
    governance_setup = setup.get("governance") or {}
    governance = {
        "assessment_fresh": bool(governance_setup.get("assessment_fresh")),
        "requirements": list(setup.get("requirements") or []),
        "path": governance_setup.get("path"),
    }
    matrix_context = {
        key: cell[key]
        for key in (
            "integration_domain",
            "information_operation",
            "source_modality",
            "evidence_topology",
            "retrieval_obstacle",
            "agent_state",
            "governance_path",
            "terminal_condition",
            "resource_pressure_band",
        )
    }
    decision = {
        "step": 0,
        "kind": "decision",
        "agent_state": {"state_name": setup["state_name"]},
        "history": list(setup.get("history") or []),
        "plan": {
            "objective": scenario["question"],
            "requirements": list(setup.get("requirements") or []),
        },
        "matrix_context": matrix_context,
        "legal_tools": legal_tools,
        "observed_evidence": list(setup.get("observed_evidence") or []),
        "governance": governance,
        "proposed_tool": target,
        "executed_tool": target,
        "acceptable_tools": [target],
        "ranked_tools": [target, *hard_negatives],
        "hard_negative_tools": hard_negatives,
        "label_source": ORACLE_VERSION,
    }
    terminal_status = {
        "ongoing": "ongoing",
        "selection": "selected",
        "abstention": "no_confident_matches",
        "clarification": "clarification_needed",
        "unresolved_contradiction": "unresolved_contradiction",
        "step_limit_termination": "step_limit",
    }[str(cell["terminal_condition"])]
    return {
        "schema_version": "trajectory.v1",
        "trajectory_id": "oracle_" + stable_hash({"scenario_id": scenario["scenario_id"], "oracle": ORACLE_VERSION})[:24],
        "scenario_id": str(scenario["scenario_id"]),
        "question": str(scenario["question"]),
        "runner": {
            "name": "fitz-tool-matrix-oracle",
            "version": ORACLE_VERSION,
            "contract_version": "runner.v1",
        },
        "events": [
            decision,
            {"step": 1, "kind": "terminal", "status": terminal_status},
        ],
        "terminal_result": {
            "status": terminal_status,
            "expected_terminal_condition": cell["terminal_condition"],
        },
        "validation": {
            "trajectory_accepted": True,
            "rejection_reasons": [],
            "validation_mode": ORACLE_VERSION,
        },
        "provenance": {
            "captured_at": captured_at,
            "oracle_version": ORACLE_VERSION,
            "scenario_type_signature": scenario["type_signature"],
            "source_card_hashes": scenario["provenance"]["source_card_hashes"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.source_card and not args.source_card_manifest:
        raise SystemExit("provide --source-card or --source-card-manifest")
    cards = _load_cards(args.source_card)
    manifest_cards = _load_manifest_cards(args.source_card_manifest)
    duplicate_ids = set(cards) & set(manifest_cards)
    if duplicate_ids:
        raise SystemExit(f"duplicate source card IDs across inputs: {sorted(duplicate_ids)}")
    cards.update(manifest_cards)
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    errors = 0
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            scenario = json.loads(line)
            if not isinstance(scenario, dict):
                raise ValueError("scenario must be an object")
            trace = oracle_trajectory(scenario, cards, captured_at=captured_at)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            errors += 1
            trace = {
                "schema_version": "trajectory.v1",
                "trajectory_id": f"oracle_error_{line_number}",
                "scenario_id": str(scenario.get("scenario_id", "")) if isinstance(scenario, dict) else "",
                "runner": {
                    "name": "fitz-tool-matrix-oracle",
                    "version": ORACLE_VERSION,
                    "contract_version": "runner.v1",
                },
                "events": [],
                "terminal_result": {"status": "error"},
                "validation": {
                    "trajectory_accepted": False,
                    "rejection_reasons": [f"line {line_number}: {exc}"],
                },
                "provenance": {"captured_at": captured_at},
            }
        print(json.dumps(trace, ensure_ascii=False, sort_keys=True), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
