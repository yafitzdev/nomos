"""Convert the external V2 router-export JSONL into Fitz-Tool rows."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .contracts import validate_decision_state
from .uniqueness import stable_hash


FRESH_RE = re.compile(r"fresh=(true|false)", re.IGNORECASE)


def _state_name(row: Mapping[str, Any]) -> str:
    state = row.get("state") or {}
    history = state.get("history") or []
    evidence = state.get("accumulated_evidence") or []
    guidance = str(row.get("state_guidance") or "").casefold()
    if not history:
        return "initial"
    if any(item.get("tool") == "compare_evidence" for item in history if isinstance(item, Mapping)):
        return "disputed"
    if "no available evidence" in guidance or (not evidence and any(item.get("tool", "").startswith("search_") for item in history if isinstance(item, Mapping))):
        return "no_hits"
    if evidence and "requirements_not_yet_finalized" in guidance:
        return "partial_evidence"
    if evidence:
        return "fresh_sufficient" if "complete=True" in guidance else "partial_evidence"
    return "noisy_hits"


def _governance(row: Mapping[str, Any]) -> dict[str, Any]:
    guidance = str(row.get("state_guidance") or "")
    match = FRESH_RE.search(guidance)
    state = row.get("state") or {}
    requirements = state.get("requirement_progress") or (state.get("plan") or {}).get("requirements") or []
    return {
        "assessment_fresh": bool(match and match.group(1).casefold() == "true"),
        "requirements": requirements if isinstance(requirements, list) else [],
    }


def _evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in state.get("accumulated_evidence") or []:
        if not isinstance(item, Mapping):
            continue
        evidence_id = item.get("evidence_id") or item.get("display_id")
        if evidence_id:
            output.append({"evidence_id": str(evidence_id), **dict(item)})
    return output


def import_exported_rows(
    rows: list[Mapping[str, Any]],
    *,
    include_unvalidated_hard_negatives: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return validated positive rows and explicit import skips."""

    output: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        candidate_tools = [str(tool) for tool in row.get("candidate_tools") or []]
        accepted_tools = [str(tool) for tool in row.get("accepted_tools") or [] if str(tool) in candidate_tools]
        if accepted_tools:
            hard_negatives = [tool for tool in candidate_tools if tool not in accepted_tools]
            accepted = True
        elif include_unvalidated_hard_negatives:
            chosen = str(row.get("chosen_tool") or "")
            hard_negatives = [chosen] if chosen in candidate_tools else []
            accepted = False
        else:
            skips.append({"row": index, "reason": "missing deterministic accepted_tools"})
            continue
        state = row.get("state") or {}
        run_id = str(row.get("run_id") or "")
        turn_index = int(row.get("turn_index") or 1)
        trajectory_hash = stable_hash({"run_id": run_id, "turn_index": turn_index, "state": state})
        decision = {
            "schema_version": "decision-state.v1",
            "decision_state_id": str(row.get("state_id") or "decision_" + trajectory_hash[:24]),
            "trajectory_id": run_id,
            "scenario_id": run_id,
            "step": max(0, turn_index - 1),
            "question": str(state.get("question") or ""),
            "agent_state": {"state_name": _state_name(row)},
            "history": list(state.get("history") or []),
            "plan": dict(state.get("plan") or {}) if isinstance(state.get("plan"), Mapping) else {},
            "matrix_context": (
                dict(state.get("matrix_context") or {})
                if isinstance(state.get("matrix_context"), Mapping)
                else {}
            ),
            "legal_tools": candidate_tools,
            "observed_evidence": _evidence(state),
            "governance": _governance(row),
            "label": {
                "acceptable_tools": accepted_tools,
                "ranked_tools": accepted_tools or hard_negatives,
                "hard_negative_tools": hard_negatives,
                "label_source": "deterministic_execution",
            },
            "accepted": accepted,
            "provenance": {
                "trajectory_hash": trajectory_hash,
                "validator_version": "v2-router-import.v1",
            },
        }
        report = validate_decision_state(decision)
        if report.valid:
            output.append(decision)
        else:
            skips.append({"row": index, "reason": report.as_dict()})
    return output, skips
