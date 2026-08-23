"""Fitz-Sage V2 mappings for the generic registry and runner.v2 contracts.

This module intentionally imports no Fitz-Sage package. It translates explicit
JSON-compatible state exported through the external runner boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..generic_contracts import validate_runner_request_v2
from ..router_v2 import FEATURE_VERSION
from ..tool_registry import ToolRegistry, load_tool_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "tool_registry.fitz_sage_v2.json"

PRESSURE_TO_STEPS = {"high": 1, "medium": 4, "low": 8}
PHASE_BY_STATE = {
    "initial": "planning",
    "no_hits": "retrieval",
    "noisy_hits": "retrieval",
    "partial_evidence": "inspection",
    "expansion_needed": "inspection",
    "contradiction": "synthesis",
    "insufficient": "governance",
    "disputed": "governance",
    "fresh_sufficient": "terminal_readiness",
}
INVENTORY_BY_STATE = {
    "initial": "unknown",
    "no_hits": "partial",
    "noisy_hits": "partial",
    "partial_evidence": "known",
    "expansion_needed": "known",
    "contradiction": "known",
    "insufficient": "known",
    "disputed": "known",
    "fresh_sufficient": "known",
}
INSPECTION_BY_STATE = {
    "initial": "none",
    "no_hits": "none",
    "noisy_hits": "snippets_only",
    "partial_evidence": "partial_context",
    "expansion_needed": "partial_context",
    "contradiction": "multi_source_inspected",
    "insufficient": "full_context",
    "disputed": "multi_source_inspected",
    "fresh_sufficient": "full_context",
}


def load_fitz_sage_v2_registry(
    path: Path | str = DEFAULT_REGISTRY_PATH,
) -> ToolRegistry:
    return load_tool_registry(path)


def _matrix_hash(matrix_context: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(matrix_context), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _available_modalities(modality: str) -> list[str]:
    if modality == "mixed":
        return ["text", "pdf", "csv", "excel", "sqlite", "code", "metadata"]
    return [modality, "metadata"]


def adapt_v1_decision_state(
    state: Mapping[str, Any],
    *,
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Translate one V1 decision state into the generic decision-state.v2 shape."""

    registry = registry or load_fitz_sage_v2_registry()
    matrix_context = state.get("matrix_context") or {}
    if not isinstance(matrix_context, Mapping):
        matrix_context = {}
    raw_agent_state = state.get("agent_state") or {}
    if not isinstance(raw_agent_state, Mapping):
        raw_agent_state = {}
    state_name = str(raw_agent_state.get("state_name", "initial"))
    modality = str(matrix_context.get("source_modality", "text"))
    pressure = str(matrix_context.get("resource_pressure_band", "medium"))
    governance = dict(state.get("governance") or {})
    governance.setdefault(
        "allowed_side_effect_classes", ["none", "read", "local_state_write"]
    )
    observed_evidence = list(state.get("observed_evidence") or [])
    requirements = list(governance.get("requirements") or [])
    unresolved = sum(
        1
        for requirement in requirements
        if not isinstance(requirement, Mapping)
        or requirement.get("status") not in {"complete", "satisfied"}
    )
    provenance = dict(state.get("provenance") or {})
    trajectory_hash = provenance.get("trajectory_hash")
    if not isinstance(trajectory_hash, str) or len(trajectory_hash) != 64:
        trajectory_hash = hashlib.sha256(
            str(state.get("trajectory_id", state.get("decision_state_id", "unknown"))).encode(
                "utf-8"
            )
        ).hexdigest()

    output = {
        "schema_version": "decision-state.v2",
        "decision_state_id": str(state.get("decision_state_id", "unknown")),
        "trajectory_id": str(state.get("trajectory_id", "unknown")),
        "scenario_id": str(state.get("scenario_id", "unknown")),
        "step": int(state.get("step", 0)),
        "question": str(state.get("question", "")),
        "agent_state": {
            **dict(raw_agent_state),
            "phase": PHASE_BY_STATE.get(state_name, "retrieval"),
        },
        "history": list(state.get("history") or []),
        "plan": dict(state.get("plan") or {}),
        "observed_evidence": observed_evidence,
        "governance": governance,
        "resource_state": {
            "remaining_steps": PRESSURE_TO_STEPS.get(pressure, 4),
            "unresolved_requirement_count": unresolved,
            "observed_evidence_count": len(observed_evidence),
            "distractor_count": int(raw_agent_state.get("distractor_count", 0)),
            "prior_search_count": int(raw_agent_state.get("retrieval_passes", 0)),
            "derived_from_v1": True,
        },
        "source_state": {
            "available_modalities": _available_modalities(modality),
            "inventory_state": INVENTORY_BY_STATE.get(state_name, "partial"),
            "inspection_state": INSPECTION_BY_STATE.get(state_name, "none"),
        },
        "query_state": {
            "operation": str(matrix_context.get("information_operation", "lookup_semantic")),
            "specificity": str(raw_agent_state.get("query_specificity", "broad")),
            "match_strategy": str(raw_agent_state.get("match_strategy", "hybrid")),
        },
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": [str(tool) for tool in state.get("legal_tools") or []],
        "label": dict(state.get("label") or {}),
        "accepted": bool(state.get("accepted", False)),
        "sampling_context": {
            "source_matrix_version": "matrix.v1",
            "matrix_cell": dict(matrix_context),
        },
        "provenance": {
            **provenance,
            "trajectory_hash": trajectory_hash,
            "registry_fingerprint": registry.fingerprint,
            "matrix_cell_id": _matrix_hash(matrix_context),
            "feature_version": FEATURE_VERSION,
            "validator_version": str(
                provenance.get("validator_version", "v1-adapter.unvalidated")
            ),
        },
    }
    return output


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content or "")


def _tool_call_name(value: Mapping[str, Any]) -> str | None:
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        return str(name) if name else None
    name = value.get("name")
    return str(name) if name else None


def _message_tool_names(message: Mapping[str, Any]) -> list[str]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    return [
        name
        for item in raw_calls
        if isinstance(item, Mapping)
        for name in [_tool_call_name(item)]
        if name
    ]


def _json_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _evidence_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key in ("evidence_id", "display_id"):
            item = value.get(key)
            if isinstance(item, str) and item:
                found.append(item)
        for key in ("evidence_ids", "selected_evidence_ids"):
            items = value.get(key)
            if isinstance(items, list):
                found.extend(str(item) for item in items if isinstance(item, (str, int)))
        for item in value.values():
            found.extend(_evidence_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_evidence_ids(item))
    return list(dict.fromkeys(found))


def _tool_result_payload(message: Mapping[str, Any]) -> Any:
    return _json_content(_message_text(message))


def _question_from_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    questions = [
        _message_text(message).strip()
        for message in messages
        if message.get("role") == "user" and _message_text(message).strip()
    ]
    return questions[-1] if questions else "External agent research objective"


def _question_length_band(question: str) -> str:
    word_count = len(question.split())
    if word_count <= 12:
        return "short"
    if word_count <= 32:
        return "medium"
    return "long"


def _state_after_action(last_action: str | None, result: Any) -> str:
    if not last_action:
        return "initial"
    if last_action in {"search_bm25", "grep_search", "search_metadata", "search_table_rows", "search_pdf_pages"}:
        return "no_hits" if not _evidence_ids(result) else "noisy_hits"
    if last_action in {"inspect_evidence", "expand_context", "read_file", "inspect_code"}:
        return "partial_evidence"
    if last_action == "assess_evidence":
        return "fresh_sufficient"
    if last_action == "update_requirement_progress":
        return "partial_evidence"
    if last_action == "compare_evidence":
        return "contradiction"
    return "partial_evidence"


def _phase_for_state(state_name: str) -> str:
    return PHASE_BY_STATE.get(state_name, "retrieval")


def _operation_for_action(last_action: str | None) -> str:
    if not last_action:
        return "lookup_semantic"
    if last_action in {"search_bm25", "grep_search", "search_metadata", "search_table_rows", "search_pdf_pages"}:
        return "search"
    if last_action in {"inspect_evidence", "expand_context", "read_file", "inspect_code"}:
        return "inspect"
    if last_action == "compare_evidence":
        return "contradiction"
    if last_action in {"update_requirement_progress", "assess_evidence"}:
        return "assessment"
    return "lookup_semantic"


def build_runner_request_from_openai(
    body: Mapping[str, Any],
    *,
    registry: ToolRegistry | None = None,
    max_steps: int = 8,
    source_modality: str | None = None,
) -> dict[str, Any]:
    """Translate one V2 OpenAI request into the generic runner contract.

    This is the V2 adapter boundary. It only consumes the visible question,
    transcript, tool schemas and current legal tool list; it does not create
    labels or use future outcomes.
    """

    registry = registry or load_fitz_sage_v2_registry()
    raw_messages = body.get("messages")
    messages = [item for item in raw_messages if isinstance(item, Mapping)] if isinstance(raw_messages, list) else []
    legal_ids: list[str] = []
    raw_tools = body.get("tools")
    if isinstance(raw_tools, list):
        for item in raw_tools:
            if not isinstance(item, Mapping):
                continue
            function = item.get("function")
            if isinstance(function, Mapping) and isinstance(function.get("name"), str):
                legal_ids.append(str(function["name"]))
    legal_ids = list(dict.fromkeys(legal_ids))
    if not legal_ids:
        raise ValueError("OpenAI request does not contain legal function candidates")
    unknown = sorted(set(legal_ids) - set(registry.by_id))
    if unknown:
        raise ValueError("V2 adapter received unknown Fitz-Sage tools: " + ", ".join(unknown))

    prior_actions: list[str] = []
    history: list[dict[str, Any]] = []
    observed_evidence: list[dict[str, Any]] = []
    plan: dict[str, Any] = {}
    last_action: str | None = None
    last_result: Any = None
    for message in messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            for action in _message_tool_names(message):
                prior_actions.append(action)
                last_action = action
                history.append({"action_family": "tool_use", "tool": action, "status": "ok"})
        elif role == "tool":
            last_result = _tool_result_payload(message)
            evidence = _evidence_ids(last_result)
            observed_evidence.extend(
                {
                    "evidence_id": evidence_id,
                    "inspection_status": "inspected" if "inspect" in str(message.get("name", "")) else "candidate",
                    "source_id": "external_source",
                }
                for evidence_id in evidence
            )
            if str(message.get("name") or "") == "set_retrieval_plan" and isinstance(last_result, Mapping):
                plan = dict(last_result)
            history.append(
                {
                    "action_family": "tool_result",
                    "tool": str(message.get("name") or "unknown"),
                    "status": "ok",
                    "evidence_count": len(evidence),
                }
            )
    state_name = _state_after_action(last_action, last_result)
    question = _question_from_messages(messages)
    remaining_steps = max(0, int(max_steps) - len(prior_actions))
    if source_modality == "mixed":
        available_modalities = {"text", "pdf", "csv", "excel", "sqlite", "code", "metadata"}
    elif source_modality:
        available_modalities = {source_modality, "metadata"}
    else:
        available_modalities = {"text", "metadata"}
    governance = {
        "assessment_fresh": last_action == "assess_evidence",
        "allowed_side_effect_classes": ["none", "read", "local_state_write"],
        "requirements": list(plan.get("evidence_requirements") or plan.get("requirements") or []),
    }
    request_id = hashlib.sha256(
        json.dumps(dict(body), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]
    unique_evidence: list[dict[str, Any]] = []
    seen_evidence: set[tuple[str, str]] = set()
    for item in observed_evidence:
        key = (str(item.get("evidence_id", "")), str(item.get("source_id", "")))
        if key not in seen_evidence:
            seen_evidence.add(key)
            unique_evidence.append(item)
    request = {
        "schema_version": "runner-request.v2",
        "request_id": "fitz-sage-openai-" + request_id,
        "question": question,
        "agent_state": {
            "phase": _phase_for_state(state_name),
            "question_length_band": _question_length_band(question),
            "state_name": state_name,
        },
        "history": history,
        "plan": {
            "active": "finalize_document_selection" not in prior_actions,
            "operation": _operation_for_action(last_action),
            "objective": question,
            "requirements": governance["requirements"],
        },
        "observed_evidence": unique_evidence,
        "governance": governance,
        "resource_state": {
            "remaining_steps": remaining_steps,
            "observed_evidence_count": len(observed_evidence),
            "prior_search_count": sum(
                action in {"search_bm25", "grep_search", "search_metadata", "search_table_rows", "search_pdf_pages"}
                for action in prior_actions
            ),
        },
        "source_state": {
            "available_modalities": sorted(available_modalities),
            "inventory_state": "known" if "list_sources" in prior_actions else "partial",
            "inspection_state": "full_context" if observed_evidence else "partial",
        },
        "query_state": {
            "operation": _operation_for_action(last_action),
            "specificity": "identifier" if any(char.isupper() for char in question) else "semantic",
            "match_strategy": "exact" if last_action == "grep_search" else "hybrid",
        },
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": legal_ids,
    }
    report = validate_runner_request_v2(request)
    if not report.valid:
        raise ValueError(json.dumps(report.as_dict(), sort_keys=True))
    return request
