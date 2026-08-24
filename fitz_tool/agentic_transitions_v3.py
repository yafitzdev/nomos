"""Leakage-safe, training-only state-transition examples for Nomos."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .external_registry_fixtures import build_portability_augmentation_registry
from .generic_contracts import validate_decision_state_v2
from .generic_pilot_v3 import TARGET_CAPABILITIES
from .router_v2 import FEATURE_VERSION
from .tool_registry import SIDE_EFFECT_CLASSES, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "matrix.agentic.v3.json"
DATASET_VERSION = "nomos-agentic-transitions.v3"
VALIDATOR_VERSION = "agentic-transition-validator.v1"

NEXT_STEP_PHRASES = {
    "plan_retrieval": (
        "Lay out the order of information-gathering operations before executing them.",
        "Decide how the investigation should proceed from the current state.",
    ),
    "list_sources": (
        "Surface which resources are available before choosing one to inspect.",
        "Show the material that can currently be addressed.",
    ),
    "search_content": (
        "Locate passages that discuss the unresolved behavior.",
        "Search across source bodies for material bearing on the open question.",
    ),
    "exact_pattern_search": (
        "Match the literal identifier exactly instead of using a broad semantic search.",
        "Find the precise token or phrase supplied by the objective.",
    ),
    "search_metadata": (
        "Narrow the source catalog using descriptive attributes before opening content.",
        "Use resource metadata to identify which material is relevant.",
    ),
    "inspect_structured_schema": (
        "Establish the fields and value types before filtering structured rows.",
        "Inspect the shape of the tabular resource before querying records.",
    ),
    "search_structured_records": (
        "Retrieve the structured entries that satisfy the requested conditions.",
        "Filter the known fields to obtain matching records.",
    ),
    "inspect_document_structure": (
        "Map the document's sections before looking for the relevant range.",
        "Inspect headings and landmarks to understand the document layout.",
    ),
    "search_document_pages": (
        "Locate the document range that covers the requested topic.",
        "Find the relevant pages now that the document layout is known.",
    ),
    "read_content": (
        "Open the selected material and return its content.",
        "Read the chosen source now that it has been identified.",
    ),
    "inspect_code_structure": (
        "Inspect definitions and symbol relationships in the implementation.",
        "Map the program structure before reading the relevant implementation.",
    ),
    "inspect_evidence": (
        "Review the support and origin of the result that was just retrieved.",
        "Check whether the candidate observation has adequate provenance.",
    ),
    "expand_context": (
        "Widen the partial result so its surrounding meaning becomes visible.",
        "Recover the missing context around the incomplete observation.",
    ),
    "compare_evidence": (
        "Put the collected observations side by side and reconcile differences.",
        "Compare the evidence obtained from separate resources.",
    ),
    "update_requirements": (
        "Record which requested conditions are now covered and which remain open.",
        "Refresh requirement coverage after inspecting the new evidence.",
    ),
    "assess_evidence": (
        "Judge whether the accumulated support is sufficient for a conclusion.",
        "Assess if the current observations justify finishing the task.",
    ),
    "finalize_selection": (
        "Commit the supported result now that the decision is ready.",
        "Finish with the best-supported selection after all checks are complete.",
    ),
}

WORKFLOW_SEQUENCES = (
    ("plan_retrieval", "list_sources", "search_metadata", "read_content", "inspect_evidence", "finalize_selection"),
    ("search_content", "inspect_evidence", "update_requirements", "assess_evidence", "finalize_selection"),
    ("exact_pattern_search", "inspect_evidence", "finalize_selection"),
    ("inspect_structured_schema", "search_structured_records", "inspect_evidence", "assess_evidence", "finalize_selection"),
    ("inspect_document_structure", "search_document_pages", "inspect_evidence", "finalize_selection"),
    ("inspect_code_structure", "read_content", "compare_evidence", "assess_evidence", "finalize_selection"),
    ("search_content", "expand_context", "inspect_evidence", "finalize_selection"),
    ("search_content", "inspect_code_structure", "compare_evidence", "update_requirements", "assess_evidence", "finalize_selection"),
)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _tool_for(registry: ToolRegistry, capability: str) -> str:
    return next(tool.tool_id for tool in registry.tools if capability in tool.capabilities)


def _completed_history(completed: tuple[str, ...], transition: str) -> list[dict[str, str]]:
    history = [
        {"completed_step": NEXT_STEP_PHRASES[value][1], "status": "complete"}
        for value in completed
    ]
    if transition == "no_history":
        return []
    if transition == "stale_prior_intent" and completed:
        history.append(
            {
                "prior_intent": NEXT_STEP_PHRASES[completed[0]][0],
                "status": "already_satisfied",
            }
        )
    return history


def generate_transition_states(
    count: int, *, seed: int = 20260828
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate unique train rows without using frozen evaluation registry styles."""

    if count < len(TARGET_CAPABILITIES):
        raise ValueError(f"count must be at least {len(TARGET_CAPABILITIES)}")
    spec = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    dimensions = spec["dimensions"]
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {name: Counter() for name in dimensions}

    stage_positions = [
        (sequence, position)
        for sequence in WORKFLOW_SEQUENCES
        for position in range(len(sequence))
    ]
    for index in range(count):
        sequence, position = stage_positions[index % len(stage_positions)]
        target = sequence[position]
        completed = tuple(sequence[:position])
        transition = dimensions["history_transition"][index % len(dimensions["history_transition"])]
        if transition == "related_completed" and not completed:
            transition = "no_history"
        style = dimensions["registry_semantic_style"][(index // 5) % 3]
        instruction_style = dimensions["instruction_style"][(index // 3) % 4]
        pool_size = int(dimensions["candidate_pool_size"][(index // 7) % 2])
        registry = build_portability_augmentation_registry(style, 100 + index)
        target_id = _tool_for(registry, target)
        all_ids = [tool.tool_id for tool in registry.tools]
        distractors = [tool_id for tool_id in all_ids if tool_id != target_id]
        local_rng = random.Random(seed + index * 104729)
        local_rng.shuffle(distractors)
        legal_ids = [target_id, *distractors[: pool_size - 1]]
        local_rng.shuffle(legal_ids)

        is_recovery = transition == "failed_prior_candidates"
        previous_ids: list[str] = []
        if is_recovery:
            previous_ids = distractors[-3:]
            legal_ids = [value for value in legal_ids if value not in previous_ids]
            effective_pool_size = min(pool_size, len(all_ids) - len(previous_ids))
            while len(legal_ids) < effective_pool_size:
                replacement = next(
                    value
                    for value in all_ids
                    if value not in legal_ids and value not in previous_ids
                )
                legal_ids.append(replacement)

        phrase = NEXT_STEP_PHRASES[target][index % 2]
        objective = (
            "Resolve the technical integration question with traceable evidence."
            if instruction_style in {"direct", "constraint"}
            else "Help me finish the investigation without guessing."
        )
        question = f"Objective: {objective} Current need: {phrase}"
        if instruction_style == "constraint":
            question += " Use only a legal operation that fits the current state."
        elif instruction_style == "conversational":
            question = f"We have made some progress. What should happen next? {phrase}"

        history = _completed_history(completed, transition)
        if transition == "confusable_completed" and completed:
            history.append(
                {
                    "completed_step": NEXT_STEP_PHRASES[completed[-1]][0],
                    "status": "do_not_repeat",
                }
            )
        cell = {
            "matrix_version": "matrix.agentic.v3",
            "history_transition": transition,
            "registry_semantic_style": style,
            "instruction_style": instruction_style,
            "candidate_pool_size": len(legal_ids),
            "session_position": "initial" if not completed else "late" if position == len(sequence) - 1 else "middle",
            "interaction_round": "expanded" if is_recovery else "initial",
            "target_capability": target,
            "workflow_shape": f"sequence_{WORKFLOW_SEQUENCES.index(sequence):02d}",
            "ordinal": index,
        }
        cell_id = _digest(cell)
        row: dict[str, Any] = {
            "schema_version": "decision-state.v2",
            "dataset_version": DATASET_VERSION,
            "decision_state_id": f"agentic-transition-{seed}-{index:07d}",
            "trajectory_id": f"agentic-transition-trajectory-{seed}-{index:07d}",
            "scenario_id": f"agentic-transition-scenario-{seed}-{index:07d}",
            "step": position,
            "question": question,
            "task_kind": "recover" if is_recovery else "route",
            "agent_state": {"state_name": "active", "phase": "execution"},
            "history": history,
            "plan": {"remaining_step": phrase},
            "observed_evidence": [
                {"result_id": f"result_{offset}", "inspection_status": "inspected"}
                for offset in range(len(completed))
            ],
            "governance": {
                "allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
                "call_allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
            },
            "resource_state": {"remaining_steps": max(1, len(sequence) - position)},
            "source_state": {
                "source_ids": ["training_source"],
                "available_modalities": sorted(
                    {value for tool in registry.tools for value in tool.input_modalities}
                ),
                "inventory_state": "known",
                "inspection_state": "partial" if completed else "none",
                "schema_known": "inspect_structured_schema" in completed,
            },
            "query_state": {"query_terms": phrase.lower().split(), "schema_known": "inspect_structured_schema" in completed},
            "previous_candidate_ids": previous_ids,
            "expansion_context": {
                "expansion_allowed": is_recovery,
                "expansion_round": 1 if is_recovery else 0,
                "trigger": "wrong_tool" if is_recovery else "none",
                "prior_candidate_ids": previous_ids,
                "excluded_candidate_ids": previous_ids,
                "unresolved_requirement": phrase,
            },
            "tool_registry": registry.as_dict(),
            "legal_candidate_ids": legal_ids,
            "label": {
                "acceptable_tools": [target_id],
                "ranked_tools": [target_id, *[value for value in legal_ids if value != target_id]],
                "hard_negative_tools": [value for value in legal_ids if value != target_id],
                "label_source": VALIDATOR_VERSION,
            },
            "accepted": True,
            "evaluation_partition": "train",
            "split_group_id": f"agentic-transition-scenario-{seed}-{index:07d}",
            "question_template_id": f"transition-{instruction_style}-{index % 2}",
            "matrix_cell": cell,
            "matrix_cell_id": cell_id,
            "provenance": {
                "corpus": DATASET_VERSION,
                "prompt_version": "deterministic-agentic-transition.v1",
                "model": "deterministic",
                "artifact": "train-only-transition-refinement",
                "seed": seed + index * 104729,
                "validator_version": VALIDATOR_VERSION,
                "feature_version": FEATURE_VERSION,
                "registry_fingerprint": registry.fingerprint,
                "trajectory_hash": _digest({"seed": seed, "index": index}),
                "matrix_cell_id": cell_id,
            },
        }
        report = validate_decision_state_v2(row)
        if not report.valid:
            raise RuntimeError(f"invalid transition state {index}: {report.as_dict()}")
        if target.replace("_", " ") in question.lower():
            raise RuntimeError(f"canonical target leaked into question at row {index}")
        rows.append(row)
        for name in counts:
            counts[name][str(cell[name])] += 1

    rng.shuffle(rows)
    return rows, {
        "dataset_version": DATASET_VERSION,
        "matrix_version": "matrix.agentic.v3",
        "count": len(rows),
        "seed": seed,
        "unique_matrix_cells": len({row["matrix_cell_id"] for row in rows}),
        "unique_registry_fingerprints": len(
            {row["provenance"]["registry_fingerprint"] for row in rows}
        ),
        "dimension_counts": {
            name: dict(sorted(values.items())) for name, values in counts.items()
        },
    }
