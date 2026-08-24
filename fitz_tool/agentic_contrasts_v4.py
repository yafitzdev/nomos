"""Focused train-only contrasts for opaque and semantically confusable tools."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .generic_contracts import validate_decision_state_v2
from .generic_pilot_v3 import TARGET_CAPABILITIES
from .router_v2 import FEATURE_VERSION
from .tool_registry import SIDE_EFFECT_CLASSES, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "matrix.agentic.v4.json"
DATASET_VERSION = "nomos-agentic-contrasts.v4"
VALIDATOR_VERSION = "agentic-contrast-validator.v1"

PURPOSES = {
    "plan_retrieval": "arrange information operations that should happen next without closing the task",
    "list_sources": "enumerate addressable resources and their handles",
    "search_content": "locate relevant passages across resource bodies",
    "exact_pattern_search": "match a literal identifier or token with exactness",
    "search_metadata": "filter the resource catalog using descriptive attributes",
    "inspect_structured_schema": "report fields, columns, and value types before querying data",
    "search_structured_records": "apply conditions to structured input and return matching rows",
    "inspect_document_structure": "map sections, headings, and landmarks in a document",
    "search_document_pages": "find document pages or ranges associated with a topic",
    "read_content": "open one selected resource and return its complete content",
    "inspect_code_structure": "trace symbols, definitions, and relationships in source code",
    "inspect_evidence": "check the support, origin, and verification status of an observation",
    "expand_context": "widen a partial result with the surrounding material needed to interpret it",
    "compare_evidence": "reconcile multiple observations and identify agreement or conflict",
    "update_requirements": "mark decision conditions covered or still outstanding",
    "assess_evidence": "judge whether collected support is sufficient without committing the answer",
    "finalize_selection": "close the task by committing the supported answer",
}

INTENTS = {
    "plan_retrieval": (
        "Work out the order of information operations before doing them.",
        "Choose what the investigation should do next, but do not close it.",
        "We still need a route through the available material.",
    ),
    "list_sources": (
        "Expose every resource handle available at this point.",
        "Select the inventory operation, not a filter over that inventory.",
        "What material can the agent address before it narrows anything?",
    ),
    "search_content": (
        "Find wording relevant to the unresolved behavior across source bodies.",
        "Select useful passages broadly rather than matching one literal token.",
        "Look through the material for discussion of the issue.",
    ),
    "exact_pattern_search": (
        "Match the supplied identifier exactly, without broadening the search.",
        "Select occurrences of one literal token rather than related passages.",
        "We need the verbatim label, not semantically similar wording.",
    ),
    "search_metadata": (
        "Narrow the resource catalog using its descriptive attributes.",
        "Select catalog entries by metadata instead of opening their bodies.",
        "Use labels and resource properties to find the right source.",
    ),
    "inspect_structured_schema": (
        "Establish the structured fields and value types before filtering rows.",
        "Select the operation that reveals columns, not the records themselves.",
        "First learn the shape of the machine-readable input.",
    ),
    "search_structured_records": (
        "Retrieve the structured entries that satisfy the active conditions.",
        "Select qualifying rows now that their fields are known.",
        "Pull matching items from the machine-readable input.",
    ),
    "inspect_document_structure": (
        "Map the document's headings and landmarks before locating a passage.",
        "Select the operation that reveals organization, not page contents.",
        "Show how the manual is laid out before opening a range.",
    ),
    "search_document_pages": (
        "Locate the document range associated with the unresolved topic.",
        "Select the relevant pages now that the manual's layout is known.",
        "Find where in the document this issue is covered.",
    ),
    "read_content": (
        "Open the complete content of the resource already selected.",
        "Select the read operation rather than searching across many sources.",
        "The source is known; inspect its full body now.",
    ),
    "inspect_code_structure": (
        "Trace definitions and symbol relationships in the implementation.",
        "Select structural code inspection before opening a function body.",
        "Map the program elements that implement the behavior.",
    ),
    "inspect_evidence": (
        "Check the observation's support, origin, and verification status.",
        "Select provenance inspection, not a search for more surrounding text.",
        "Review whether this candidate result is properly supported.",
    ),
    "expand_context": (
        "Widen the partial observation with enough neighboring material to interpret it.",
        "Select more context around this hit rather than starting a new search.",
        "The current snippet is incomplete; recover what surrounds it.",
    ),
    "compare_evidence": (
        "Place the observations side by side and reconcile differences.",
        "Select comparison rather than deciding whether the task is finished.",
        "Work out where the collected sources agree or conflict.",
    ),
    "update_requirements": (
        "Record which decision conditions are covered and which remain open.",
        "Select coverage bookkeeping, not the final answer operation.",
        "Refresh the obligation state after the latest observation.",
    ),
    "assess_evidence": (
        "Judge whether the accumulated support permits a conclusion, without committing it.",
        "Select the sufficiency check rather than the final commit.",
        "Decide whether the evidence is ready for the finishing step.",
    ),
    "finalize_selection": (
        "Commit the supported answer now that all information work is complete.",
        "Select and close the decision rather than planning or gathering more.",
        "The checks have passed; finish with the justified result.",
    ),
}

HARD_NEGATIVE = {
    "plan_retrieval": "finalize_selection",
    "list_sources": "search_metadata",
    "search_content": "read_content",
    "exact_pattern_search": "search_content",
    "search_metadata": "list_sources",
    "inspect_structured_schema": "search_structured_records",
    "search_structured_records": "inspect_structured_schema",
    "inspect_document_structure": "search_document_pages",
    "search_document_pages": "inspect_document_structure",
    "read_content": "search_content",
    "inspect_code_structure": "read_content",
    "inspect_evidence": "expand_context",
    "expand_context": "inspect_evidence",
    "compare_evidence": "assess_evidence",
    "update_requirements": "finalize_selection",
    "assess_evidence": "finalize_selection",
    "finalize_selection": "plan_retrieval",
}

CONTRAST_FAMILY = {
    frozenset(("plan_retrieval", "finalize_selection")): "plan_vs_finish",
    frozenset(("list_sources", "search_metadata")): "inventory_vs_metadata",
    frozenset(("search_content", "exact_pattern_search")): "broad_vs_exact_search",
    frozenset(("inspect_structured_schema", "search_structured_records")): "schema_vs_records",
    frozenset(("inspect_document_structure", "search_document_pages")): "document_map_vs_pages",
    frozenset(("search_content", "read_content")): "search_vs_open",
    frozenset(("inspect_code_structure", "read_content")): "code_map_vs_open",
    frozenset(("inspect_evidence", "expand_context")): "inspect_vs_expand",
    frozenset(("compare_evidence", "assess_evidence")): "compare_vs_assess",
    frozenset(("update_requirements", "finalize_selection")): "requirements_vs_finish",
    frozenset(("assess_evidence", "finalize_selection")): "requirements_vs_finish",
}


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _registry(style: str, registry_index: int) -> tuple[ToolRegistry, dict[str, str]]:
    rng = random.Random(91009 + registry_index * 7919)
    aliases = list(range(17))
    rng.shuffle(aliases)
    argument_names = ("request", "needle", "subject", "expression")
    tools: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    for slot, capability in enumerate(TARGET_CAPABILITIES):
        tool_id = f"{style[0]}x{registry_index:03d}_{(slot * 41 + 17) % 257:03d}"
        ids[capability] = tool_id
        purpose = PURPOSES[capability]
        if style == "foundry":
            description = f"Read-only operation used to {purpose}."
        elif style == "delta":
            description = f"Invoke this endpoint when the current job must {purpose}."
        else:
            description = f"Local service whose responsibility is to {purpose}."
        if capability in {"inspect_structured_schema", "search_structured_records"}:
            input_modality, output_modality = "structured", "records"
        elif capability in {"inspect_document_structure", "search_document_pages"}:
            input_modality, output_modality = "pdf", "passages"
        elif capability == "inspect_code_structure":
            input_modality, output_modality = "code", "structured_summary"
        else:
            input_modality, output_modality = "text", "evidence"
        argument_name = argument_names[(slot + registry_index) % len(argument_names)]
        tools.append(
            {
                "tool_id": tool_id,
                "tool_family": f"{style}_lane_{(slot * 5 + registry_index) % 11}",
                "description": description,
                "capabilities": [f"adapter.signal_{aliases[slot]:02d}"],
                "input_modalities": [input_modality],
                "output_modalities": [output_modality],
                "evidence_roles": [
                    "selection"
                    if capability == "finalize_selection"
                    else "planning"
                    if capability == "plan_retrieval"
                    else "observation"
                ],
                "side_effect_class": "none",
                "argument_schema": {
                    "type": "object",
                    "properties": {argument_name: {"type": "string"}},
                    "required": [argument_name],
                    "additionalProperties": False,
                },
                "constraints": ["read_only"],
                "prerequisites": ["none"],
            }
        )
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"contrast_{style}_{registry_index:03d}",
            "tools": tools,
        }
    ), ids


def generate_contrast_states(
    count: int, *, seed: int = 20260901
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate balanced opaque-registry hard contrasts for training only."""

    if count < len(TARGET_CAPABILITIES):
        raise ValueError(f"count must be at least {len(TARGET_CAPABILITIES)}")
    spec = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    dimensions = spec["dimensions"]
    styles = dimensions["registry_style"]
    transitions = dimensions["history_transition"]
    instruction_styles = dimensions["instruction_style"]
    rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {name: Counter() for name in dimensions}
    topics = (
        "callback retries",
        "authentication headers",
        "pagination behavior",
        "version compatibility",
        "event field requirements",
        "rate-limit handling",
    )
    for index in range(count):
        target = TARGET_CAPABILITIES[index % len(TARGET_CAPABILITIES)]
        negative = HARD_NEGATIVE[target]
        style = styles[(index // 17) % len(styles)]
        transition = transitions[(index // 7) % len(transitions)]
        instruction_style = instruction_styles[(index // 11) % len(instruction_styles)]
        registry_index = (index // 3) % 96
        registry, tool_ids = _registry(style, registry_index)
        target_id = tool_ids[target]
        negative_id = tool_ids[negative]
        all_ids = [tool.tool_id for tool in registry.tools]
        local_rng = random.Random(seed + index * 104729)
        distractors = [value for value in all_ids if value not in {target_id, negative_id}]
        local_rng.shuffle(distractors)
        pool_size = (10, 17)[(index // 13) % 2]
        legal_ids = [target_id, negative_id, *distractors[: pool_size - 2]]
        local_rng.shuffle(legal_ids)
        previous_ids: list[str] = []
        if transition == "failed_distractor":
            previous_ids = [
                value
                for value in legal_ids
                if value not in {target_id, negative_id}
            ][:3]
            legal_ids = [value for value in legal_ids if value not in previous_ids]
            for replacement in all_ids:
                if len(legal_ids) >= pool_size:
                    break
                if replacement not in legal_ids and replacement not in previous_ids:
                    legal_ids.append(replacement)

        phrase = INTENTS[target][(index // 17) % len(INTENTS[target])]
        objective = (
            f"Resolve the integration question about {topics[(index // 5) % len(topics)]} "
            "and eventually return a supported answer."
        )
        if instruction_style == "conversational":
            question = f"We are partway through this task. {objective} Right now: {phrase}"
        elif instruction_style == "constraint":
            question = f"{objective} Current need: {phrase} Do not repeat completed work."
        elif instruction_style == "ambiguous_verb":
            question = f"{objective} For this step, choose correctly: {phrase}"
        else:
            question = f"Objective: {objective} Current operation: {phrase}"

        history: list[dict[str, str]] = []
        if transition == "completed_neighbor":
            history.append({"completed_step": INTENTS[negative][0], "status": "complete"})
        elif transition == "stale_terminal_intent":
            history.append(
                {
                    "prior_intent": "A supported answer will be committed after the remaining work.",
                    "status": "not_current",
                }
            )
        cell = {
            "matrix_version": "matrix.agentic.v4",
            "contrast_family": CONTRAST_FAMILY.get(
                frozenset((target, negative)), "workflow_neighbor"
            ),
            "history_transition": transition,
            "registry_style": style,
            "instruction_style": instruction_style,
            "capability_visibility": "opaque",
            "candidate_pool_size": len(legal_ids),
            "interaction_round": "recovery" if previous_ids else "initial",
            "target_capability": target,
            "hard_negative_capability": negative,
            "ordinal": index,
        }
        cell_id = _digest(cell)
        row: dict[str, Any] = {
            "schema_version": "decision-state.v2",
            "dataset_version": DATASET_VERSION,
            "decision_state_id": f"agentic-contrast-{seed}-{index:07d}",
            "trajectory_id": f"agentic-contrast-trajectory-{seed}-{index:07d}",
            "scenario_id": f"agentic-contrast-scenario-{seed}-{index:07d}",
            "step": index % 6,
            "question": question,
            "task_kind": "recover" if previous_ids else "route",
            "agent_state": {"state_name": "active", "phase": "execution"},
            "history": history,
            "plan": {"remaining_step": phrase},
            "observed_evidence": [
                {"result_id": "prior_result", "inspection_status": "inspected"}
            ]
            if history
            else [],
            "governance": {
                "allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
                "call_allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
            },
            "resource_state": {"remaining_steps": 4},
            "source_state": {
                "source_ids": ["training_resource"],
                "available_modalities": ["code", "pdf", "structured", "text"],
                "inventory_state": "known",
                "inspection_state": "partial_context",
                "schema_known": target == "search_structured_records",
            },
            "query_state": {"query_terms": phrase.lower().split(), "schema_known": True},
            "previous_candidate_ids": previous_ids,
            "expansion_context": {
                "expansion_allowed": bool(previous_ids),
                "expansion_round": 1 if previous_ids else 0,
                "trigger": "wrong_tool" if previous_ids else "none",
                "prior_candidate_ids": previous_ids,
                "excluded_candidate_ids": previous_ids,
                "unresolved_requirement": phrase,
            },
            "tool_registry": registry.as_dict(),
            "legal_candidate_ids": legal_ids,
            "label": {
                "acceptable_tools": [target_id],
                "ranked_tools": [
                    target_id,
                    *[value for value in legal_ids if value != target_id],
                ],
                "hard_negative_tools": [
                    negative_id,
                    *[
                        value
                        for value in legal_ids
                        if value not in {target_id, negative_id}
                    ],
                ],
                "label_source": VALIDATOR_VERSION,
            },
            "accepted": True,
            "evaluation_partition": "train",
            "split_group_id": f"agentic-contrast-scenario-{seed}-{index:07d}",
            "question_template_id": f"contrast-{instruction_style}-{index % 3}",
            "matrix_cell": cell,
            "matrix_cell_id": cell_id,
            "provenance": {
                "corpus": DATASET_VERSION,
                "prompt_version": "deterministic-agentic-contrast.v1",
                "model": "deterministic",
                "artifact": "train-only-opaque-contrast-refinement",
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
            raise RuntimeError(f"invalid contrast state {index}: {report.as_dict()}")
        if target.replace("_", " ") in question.lower():
            raise RuntimeError(f"canonical target leaked into question at row {index}")
        if any(target in tool.capabilities for tool in registry.tools):
            raise RuntimeError(f"canonical target leaked into registry at row {index}")
        rows.append(row)
        for name in counts:
            counts[name][str(cell[name])] += 1

    random.Random(seed).shuffle(rows)
    return rows, {
        "dataset_version": DATASET_VERSION,
        "matrix_version": "matrix.agentic.v4",
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
