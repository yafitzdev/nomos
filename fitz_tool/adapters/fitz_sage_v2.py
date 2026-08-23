"""Fitz-Sage V2 mappings for the generic registry and runner.v2 contracts.

This module intentionally imports no Fitz-Sage package. It translates explicit
JSON-compatible state exported through the external runner boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

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
