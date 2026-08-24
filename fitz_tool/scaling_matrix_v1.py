"""Fixed-slot matrix and deterministic labels for the first Nomos scaling cohort."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .generic_contracts import validate_decision_state_v2
from .router_v2 import FEATURE_VERSION
from .tool_registry import SIDE_EFFECT_CLASSES, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "matrix.scaling.v1.json"
MATRIX_VERSION = "matrix.scaling.v1"
DATASET_VERSION = "nomos-scaling-targeted.v1"
VALIDATOR_VERSION = "scaling-targeted-validator.v1"
PROMPT_VERSION = "scaling-targeted-deepseek.v2"
PROJECT_MARKER = re.compile(
    r"(?<![a-z0-9_])(fitz|sage|nomos|pyrrho|opsis)(?![a-z0-9_])", re.I
)
RECOVERY_MARKER = re.compile(
    r"\b(again|another|different|earlier|failed|fresh|previous|prior|rejected|avoid)\b", re.I
)


PURPOSES: dict[str, str] = {
    "plan_retrieval": "design the next information-gathering steps without executing or closing the task",
    "list_sources": "enumerate the available resources and their handles without filtering them",
    "search_content": "find conceptually relevant passages across available content",
    "exact_pattern_search": "locate verbatim occurrences of one supplied symbol or literal pattern",
    "search_metadata": "filter resources by catalog attributes without reading their bodies",
    "inspect_structured_schema": "report fields and value types before structured records are queried",
    "search_structured_records": "return structured records that satisfy explicit field conditions",
    "inspect_document_structure": "map document sections and navigation landmarks before reading pages",
    "search_document_pages": "locate document pages or ranges that discuss a requested subject",
    "read_content": "open the complete body of one already selected resource",
    "inspect_code_structure": "trace software symbols, definitions, and their relationships",
    "inspect_evidence": "audit one observation for support, origin, and verification status",
    "expand_context": "extend a partial observation with the surrounding material needed to interpret it",
    "compare_evidence": "reconcile multiple observations and identify agreement or conflict",
    "update_requirements": "record which decision conditions are covered and which remain open",
    "assess_evidence": "judge whether collected support is sufficient without committing an answer",
    "finalize_selection": "commit the supported answer after all decision conditions are satisfied",
    "analyze_image": "inspect visual content for objects, layout, and requested details",
    "transcribe_audio": "convert spoken audio into text with timing information",
    "inspect_provenance": "retrieve the origin and lineage record attached to an existing result",
    "validate_change": "check a proposed change against requirements without applying it",
    "external_web_search": "search remote public sources for current information",
    "write_local_state": "store a reversible update in local task state",
    "publish_external": "send a persistent update to an external service",
    "delete_external": "irreversibly remove an external resource",
    "execute_code": "run a bounded program and return its captured output",
    "list_directory": "enumerate entries in one local directory",
    "compare_values": "compare supplied values and report material differences",
    "validate_schema": "check structured input against a supplied schema",
}


SCENARIOS: dict[str, dict[str, Any]] = {
    "broad_vs_exact_search": {"target": "exact_pattern_search", "negatives": ["search_content", "search_metadata"], "hint": "A literal identifier must be matched exactly."},
    "inventory_vs_metadata": {"target": "list_sources", "negatives": ["search_metadata", "search_content"], "hint": "The complete inventory is needed before any narrowing."},
    "plan_vs_immediate_search": {"target": "plan_retrieval", "negatives": ["search_content", "exact_pattern_search"], "hint": "Choose the sequence of actions, but do not execute the search yet."},
    "plan_vs_final_selection": {"target": "plan_retrieval", "negatives": ["finalize_selection", "assess_evidence"], "hint": "The investigation still needs a plan and cannot be closed."},
    "document_structure_vs_pages": {"target": "inspect_document_structure", "negatives": ["search_document_pages", "read_content"], "hint": "First establish the document map rather than opening pages."},
    "schema_vs_records": {"target": "inspect_structured_schema", "negatives": ["search_structured_records", "read_content"], "hint": "The fields and types are unknown, so records cannot be filtered yet."},
    "code_symbol_vs_full_read": {"target": "inspect_code_structure", "negatives": ["read_content", "exact_pattern_search"], "hint": "Resolve definitions and relationships, not the whole file body."},
    "inspect_vs_compare_evidence": {"target": "inspect_evidence", "negatives": ["compare_evidence", "assess_evidence"], "hint": "One observation needs a provenance and support audit before comparison."},
    "compare_vs_assess_evidence": {"target": "compare_evidence", "negatives": ["assess_evidence", "finalize_selection"], "hint": "Reconcile the collected observations before judging sufficiency."},
    "assess_vs_finalize": {"target": "assess_evidence", "negatives": ["finalize_selection", "update_requirements"], "hint": "Check whether the evidence is sufficient but do not commit the answer."},
    "requirements_vs_assess": {"target": "update_requirements", "negatives": ["assess_evidence", "finalize_selection"], "hint": "Refresh condition coverage before judging the evidence."},
    "requirements_vs_finalize": {"target": "update_requirements", "negatives": ["finalize_selection", "assess_evidence"], "hint": "Outstanding conditions must be recorded before the task can close."},
    "partial_context_vs_new_search": {"target": "expand_context", "negatives": ["search_content", "read_content"], "hint": "A useful hit is clipped; retrieve its surroundings instead of searching again."},
    "search_results_vs_provenance": {"target": "inspect_provenance", "negatives": ["search_content", "inspect_evidence"], "hint": "The result exists; its exact origin and lineage are now required."},
    "search_vs_full_read": {"target": "read_content", "negatives": ["search_content", "search_metadata"], "hint": "The resource is already selected and its complete body is needed."},
    "missing_prerequisite": {"target": "expand_context", "negatives": ["read_content", "search_structured_records"], "hint": "A direct operation lacks its prerequisite; recover the missing context first."},
    "illegal_attractive_candidate": {"target": "search_content", "negatives": ["external_web_search", "read_content"], "illegal": ["external_web_search"], "hint": "Use the legal local search even though a remote search sounds attractive."},
    "no_suitable_tool": {"target": None, "negatives": ["compare_values", "validate_schema", "list_directory"], "hint": "None of the available operations can produce the requested outcome; abstain."},
    "modality_confusion": {"target": "analyze_image", "negatives": ["transcribe_audio", "read_content"], "hint": "The unresolved input is visual rather than audio or ordinary text."},
    "side_effect_policy": {"target": "validate_change", "negatives": ["publish_external", "delete_external"], "illegal": ["publish_external", "delete_external"], "hint": "Only inspect the proposed change; applying or publishing it is forbidden."},
    "recovery_after_rejection": {"target": "search_metadata", "negatives": ["list_sources", "search_content"], "force_recovery": True, "hint": "Earlier candidates failed; return a fresh metadata-based route."},
    "stale_terminal_history": {"target": "search_content", "negatives": ["finalize_selection", "plan_retrieval"], "hint": "Ignore the stale request to finish; a new unresolved search step is active."},
    "conflicting_irrelevant_history": {"target": "inspect_structured_schema", "negatives": ["search_structured_records", "analyze_image"], "hint": "Current structured input needs schema inspection despite irrelevant prior work."}
}


CATALOG = tuple(PURPOSES) + tuple(f"utility_{index:03d}" for index in range(120))
TRAIN_REGISTRY_STYLES = ("procedural", "endpoint", "compact", "responsibility", "contract")
HOLDOUT_REGISTRY_STYLES = ("lattice", "meridian", "prism", "vanguard")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def load_scaling_matrix(path: Path | str = MATRIX_PATH) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("matrix_version") != MATRIX_VERSION:
        raise ValueError(f"expected {MATRIX_VERSION}")
    target = int(spec.get("target_accepted_rows") or 0)
    if target != 25_000:
        raise ValueError("scaling matrix must contain exactly 25,000 target slots")
    if sum(map(int, spec["scenario_family_counts"].values())) != target:
        raise ValueError("scenario family counts do not sum to target_accepted_rows")
    if set(spec["scenario_family_counts"]) != set(SCENARIOS):
        raise ValueError("scenario family specification does not match implementation")
    for name, counts in spec["dimension_counts"].items():
        if sum(map(int, counts.values())) != target:
            raise ValueError(f"dimension {name} does not sum to target_accepted_rows")
    return spec


def _expanded_counts(counts: Mapping[str, int], *, seed: int) -> list[str]:
    values = [str(value) for value, count in counts.items() for _ in range(int(count))]
    random.Random(seed).shuffle(values)
    return values


def materialize_assignments(spec: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Materialize every fixed training slot before any teacher request is made."""

    spec = dict(spec or load_scaling_matrix())
    target = int(spec["target_accepted_rows"])
    seed = int(spec["seed"])
    axes: dict[str, list[str]] = {
        "scenario_family": _expanded_counts(spec["scenario_family_counts"], seed=seed + 11)
    }
    for ordinal, (name, counts) in enumerate(spec["dimension_counts"].items(), start=1):
        if name == "source_modality":
            continue
        axes[name] = _expanded_counts(counts, seed=seed + ordinal * 104729)
    fixed_modality = {
        "code_symbol_vs_full_read": "code",
        "document_structure_vs_pages": "document",
        "schema_vs_records": "structured_data",
        "conflicting_irrelevant_history": "structured_data",
    }
    modalities: list[str | None] = [
        fixed_modality.get(family) for family in axes["scenario_family"]
    ]
    fixed_counts = Counter(value for value in modalities if value)
    modality_counts = {
        str(name): int(count)
        for name, count in spec["dimension_counts"]["source_modality"].items()
    }
    remaining_modalities = [
        modality
        for modality, count in modality_counts.items()
        for _ in range(count - fixed_counts[modality])
    ]
    if len(remaining_modalities) != sum(value is None for value in modalities):
        raise ValueError("fixed modality constraints exceed the balanced allocation")
    random.Random(seed + 999_983).shuffle(remaining_modalities)
    remaining = iter(remaining_modalities)
    axes["source_modality"] = [
        str(value) if value is not None else next(remaining) for value in modalities
    ]
    assignments: list[dict[str, Any]] = []
    for slot in range(target):
        cell = {name: values[slot] for name, values in axes.items()}
        cell["candidate_pool_size"] = int(cell["candidate_pool_size"])
        cell.update({"slot": slot, "matrix_version": MATRIX_VERSION, "replacement_ordinal": 0})
        scenario = SCENARIOS[str(cell["scenario_family"])]
        target = scenario.get("target")
        negatives = list(scenario.get("negatives") or [])
        if cell["scenario_family"] == "modality_confusion":
            modality_routes = {
                "image": ("analyze_image", ["transcribe_audio", "read_content"]),
                "audio": ("transcribe_audio", ["analyze_image", "read_content"]),
                "code": ("inspect_code_structure", ["read_content", "exact_pattern_search"]),
                "document": ("inspect_document_structure", ["search_document_pages", "read_content"]),
                "structured_data": ("inspect_structured_schema", ["search_structured_records", "read_content"]),
                "text": ("read_content", ["analyze_image", "transcribe_audio"]),
                "mixed": ("compare_evidence", ["analyze_image", "transcribe_audio"]),
            }
            target, negatives = modality_routes[str(cell["source_modality"])]
        assignments.append(
            {
                "assignment_id": f"scale-v1-{slot:05d}-r00",
                "slot_id": f"scale-v1-{slot:05d}",
                "slot": slot,
                "replacement_ordinal": 0,
                "seed": seed + slot * 104729,
                "matrix_cell": cell,
                "matrix_cell_id": digest(cell),
                "target_capability": target,
                "hard_negative_capabilities": negatives,
                "illegal_capabilities": list(scenario.get("illegal") or []),
                "routing_hint": str(scenario["hint"]),
            }
        )
    validate_assignments(assignments, spec)
    return assignments


def replacement_assignment(base: Mapping[str, Any], replacement_ordinal: int) -> dict[str, Any]:
    if replacement_ordinal < 1:
        raise ValueError("replacement_ordinal must be positive")
    value = dict(base)
    value["replacement_ordinal"] = replacement_ordinal
    value["assignment_id"] = f"{base['slot_id']}-r{replacement_ordinal:02d}"
    value["seed"] = int(base["seed"]) + replacement_ordinal * 1_000_003
    cell = dict(base["matrix_cell"])
    cell["replacement_ordinal"] = replacement_ordinal
    value["matrix_cell"] = cell
    value["matrix_cell_id"] = digest(cell)
    return value


def validate_assignments(assignments: list[Mapping[str, Any]], spec: Mapping[str, Any]) -> dict[str, Any]:
    target = int(spec["target_accepted_rows"])
    if len(assignments) != target:
        raise ValueError(f"expected {target} assignments, found {len(assignments)}")
    ids = [str(value["assignment_id"]) for value in assignments]
    cells = [str(value["matrix_cell_id"]) for value in assignments]
    if len(set(ids)) != target or len(set(cells)) != target:
        raise ValueError("assignment and matrix-cell identifiers must be unique")
    actual: dict[str, Counter[str]] = {"scenario_family": Counter()}
    actual.update({name: Counter() for name in spec["dimension_counts"]})
    for assignment in assignments:
        cell = assignment["matrix_cell"]
        for name in actual:
            actual[name][str(cell[name])] += 1
    expected = {"scenario_family": spec["scenario_family_counts"], **spec["dimension_counts"]}
    for name, counts in actual.items():
        normalized_expected = {str(key): int(value) for key, value in expected[name].items()}
        if dict(counts) != normalized_expected:
            raise ValueError(f"dimension count mismatch for {name}")
    return {
        "count": target,
        "unique_assignment_ids": len(set(ids)),
        "unique_matrix_cells": len(set(cells)),
        "dimension_counts": {name: dict(sorted(values.items())) for name, values in actual.items()},
    }


def _operation_purpose(capability: str) -> str:
    if capability in PURPOSES:
        return PURPOSES[capability]
    number = int(capability.rsplit("_", 1)[-1])
    return f"perform bounded utility transformation {number} on explicitly supplied input"


def _side_effect(capability: str) -> str:
    if capability == "write_local_state":
        return "local_state_write"
    if capability == "publish_external":
        return "external_write"
    if capability == "delete_external":
        return "irreversible_external_write"
    if capability in {"external_web_search", "read_content", "search_content", "search_metadata"}:
        return "read"
    return "none"


def _modality(capability: str, requested: str) -> str:
    fixed = {
        "analyze_image": "image",
        "transcribe_audio": "audio",
        "inspect_code_structure": "code",
        "inspect_structured_schema": "structured_data",
        "search_structured_records": "structured_data",
        "inspect_document_structure": "document",
        "search_document_pages": "document",
    }
    return fixed.get(capability, requested if requested != "mixed" else "text")


def _schema(style: str, operation_index: int, *, holdout: bool = False) -> dict[str, Any]:
    train_names = ("request", "needle", "subject", "selector", "expression", "scope", "criteria")
    holdout_names = ("work_item", "locator_spec", "intent_packet", "selection_rule")
    names = holdout_names if holdout else train_names
    name = names[operation_index % len(names)]
    if style in {"flat_required", "holdout_tuple"}:
        return {"type": "object", "properties": {name: {"type": "string"}}, "required": [name], "additionalProperties": False}
    if style == "flat_optional":
        return {"type": "object", "properties": {name: {"type": "string"}, "limit": {"type": "integer"}}, "required": [], "additionalProperties": False}
    if style in {"nested_object", "holdout_map"}:
        return {"type": "object", "properties": {name: {"type": "object", "properties": {"value": {"type": "string"}}}}, "required": [name], "additionalProperties": False}
    if style == "array_filter":
        return {"type": "object", "properties": {name: {"type": "array", "items": {"type": "string"}}}, "required": [name], "additionalProperties": False}
    return {"type": "object", "properties": {name: {"type": "string", "enum": ["inspect", "route", "validate"]}}, "required": [name], "additionalProperties": False}


def _description(style: str, purpose: str) -> str:
    if style in {"procedural", "lattice"}:
        return f"Use this operation to {purpose}."
    if style in {"endpoint", "meridian"}:
        return f"Invoke this endpoint when the immediate job is to {purpose}."
    if style in {"compact", "prism"}:
        return f"Bounded interface: {purpose}."
    if style in {"responsibility", "vanguard"}:
        return f"This service is responsible for helping an agent {purpose}."
    return f"Contract outcome: {purpose}; unrelated actions are outside its scope."


def _allowed_effects(policy: str) -> set[str]:
    if policy in {"read_only", "external_read", "destructive_forbidden"}:
        return {"none", "read"}
    if policy == "reversible_write":
        return {"none", "read", "local_state_write", "external_write"}
    return set(SIDE_EFFECT_CLASSES) - {"irreversible_external_write"}


def build_registry(assignment: Mapping[str, Any], *, holdout: bool = False) -> tuple[ToolRegistry, list[str], str | None, list[str], list[str]]:
    """Build the registry and legal set; labels come only from the fixed assignment."""

    cell = assignment["matrix_cell"]
    rng = random.Random(int(assignment["seed"]))
    pool_size = int(cell["candidate_pool_size"])
    target = assignment.get("target_capability")
    negatives = list(assignment.get("hard_negative_capabilities") or [])
    illegal = set(assignment.get("illegal_capabilities") or [])
    required = [value for value in [target, *negatives, *illegal] if value]
    rest = [value for value in CATALOG if value not in set(required)]
    rng.shuffle(rest)
    capabilities = list(dict.fromkeys([*required, *rest]))
    style = str(cell["registry_description_style"])
    schema_style = str(cell["argument_schema_style"])
    visibility = str(cell["capability_visibility"])
    registry_tag = digest({"assignment": assignment["assignment_id"], "holdout": holdout})[:10]
    tools: list[dict[str, Any]] = []
    by_capability: dict[str, str] = {}
    for index, capability in enumerate(capabilities[: max(pool_size + len(illegal) + 8, 108)]):
        tool_id = f"{'h' if holdout else 't'}{registry_tag}_{index:03d}"
        by_capability[capability] = tool_id
        capability_token = (
            f"semantic.{capability}"
            if visibility == "semantic"
            else f"{'hold' if holdout else 'train'}.proto_{(index * 37 + len(style)) % 211:03d}"
        )
        modality = _modality(capability, str(cell["source_modality"]))
        effect = _side_effect(capability)
        family_prefix = f"hold_{style[:2]}" if holdout else f"train_{style[:2]}"
        tools.append(
            {
                "tool_id": tool_id,
                "tool_family": f"{family_prefix}.group_{index % 17:02d}",
                "description": _description(style, _operation_purpose(capability)),
                "capabilities": [capability_token],
                "input_modalities": [modality],
                "output_modalities": ["result"],
                "evidence_roles": ["selection" if capability == "finalize_selection" else "observation"],
                "side_effect_class": effect,
                "argument_schema": _schema(schema_style, index, holdout=holdout),
                "constraints": ["bounded_scope"],
                "prerequisites": ["selected_resource"] if capability in {"read_content", "search_structured_records"} else ["none"],
            }
        )
    registry = ToolRegistry.from_dict(
        {"schema_version": "tool-registry.v2", "registry_id": f"{'hold' if holdout else 'train'}_{registry_tag}", "tools": tools}
    )
    allowed_effects = _allowed_effects(str(cell["side_effect_policy"]))
    legal = [tool.tool_id for tool in registry.tools if tool.side_effect_class in allowed_effects and tool.tool_id not in {by_capability.get(value) for value in illegal}]
    target_id = by_capability.get(str(target)) if target else None
    if target_id and target_id not in legal:
        raise RuntimeError(f"target excluded by policy for {assignment['assignment_id']}")
    previous: list[str] = []
    scenario = SCENARIOS.get(str(cell["scenario_family"]), {})
    recovery = bool(assignment.get("force_recovery")) or bool(scenario.get("force_recovery")) or str(cell["history_transition"]) == "failed_candidates"
    if recovery:
        previous = [tool_id for capability in negatives for tool_id in [by_capability.get(capability)] if tool_id and tool_id in legal][:3]
        if len(previous) < 3:
            additional = [value for value in legal if value != target_id and value not in previous]
            previous.extend(additional[: 3 - len(previous)])
        legal = [value for value in legal if value not in set(previous)]
    if len(legal) < pool_size:
        raise RuntimeError(f"insufficient legal candidates for {assignment['assignment_id']}")
    selected = legal[:pool_size]
    if target_id:
        if target_id not in selected:
            selected[-1] = target_id
        selected.remove(target_id)
        position_name = str(cell["initial_target_position"])
        position = {"first": 0, "second": 1, "third": 2}.get(position_name, min(pool_size - 1, 3 + int(assignment["slot"]) % max(1, pool_size - 3)))
        selected.insert(position, target_id)
    hard_negative_ids = [by_capability[value] for value in negatives if value in by_capability and by_capability[value] in selected]
    hard_negative_ids.extend(value for value in selected if value != target_id and value not in hard_negative_ids)
    return registry, selected, target_id, hard_negative_ids, previous


def teacher_assignment(assignment: Mapping[str, Any]) -> dict[str, Any]:
    cell = assignment["matrix_cell"]
    target = assignment.get("target_capability")
    history_count = {"empty": 0, "short": 1, "long": 3}[str(cell["history_length"])]
    recovery = bool(SCENARIOS[str(cell["scenario_family"])].get("force_recovery")) or str(cell["history_transition"]) == "failed_candidates"
    routing_hint = assignment["routing_hint"] if target else "The user wants a concrete physical-world outcome outside the digital operation catalog."
    mandatory_constraints = [f"Return exactly {history_count} completed_steps strings."]
    if recovery:
        routing_hint = (
            "Earlier candidate choices failed or were rejected. The request must explicitly say this "
            "and require a fresh or different option. " + str(routing_hint)
        )
        mandatory_constraints.append(
            "Both the question and current_step must make prior candidate failure and the need for a different route observable."
        )
    repair_directive = str(assignment.get("_repair_directive") or "").strip()
    if repair_directive:
        mandatory_constraints.append(repair_directive)
    return {
        "assignment_id": assignment["assignment_id"],
        "routing_hint": routing_hint,
        "needed_outcome": _operation_purpose(str(target)) if target else "arrange an in-person physical action in the real world",
        "plausible_but_wrong_outcomes": [_operation_purpose(value) for value in assignment["hard_negative_capabilities"]],
        "source_modality": cell["source_modality"],
        "session_position": cell["session_position"],
        "history_transition": cell["history_transition"],
        "wording_style": cell["wording_style"],
        "completed_step_count": history_count,
        "recovery": recovery,
        "abstain": target is None,
        "revision_attempt": int(assignment.get("replacement_ordinal") or 0),
        "mandatory_constraints": mandatory_constraints,
    }


def generation_prompt(assignments: list[Mapping[str, Any]]) -> str:
    payload = [teacher_assignment(value) for value in assignments]
    return f"""Write natural, project-independent agent requests for deterministic routing assignments.

Ground truth is already fixed. You only provide wording and observable completed history.
For each assignment:
- preserve needed_outcome and routing_hint while making the request concrete and natural;
- make plausible_but_wrong_outcomes clearly wrong for the current state without naming hidden labels;
- obey wording_style, source_modality, session_position, history_transition, and recovery;
- obey every mandatory_constraints entry literally;
- for recovery, say earlier options failed and require a different route;
- for abstain, express the desired real-world outcome naturally; never say that tools are missing or unsuitable and never ask to abstain;
- never mention a tool name, tool ID, assignment ID, registry, benchmark, capability identifier, matrix, label, or this instruction;
- do not claim a completed action that would already satisfy needed_outcome.

Return exactly one JSON object with an items array of exactly {len(assignments)} objects.
Every item must have exactly these fields:
{{"assignment_id":"copied input ID","question":"20-400 chars","current_step":"20-400 chars","completed_steps":["observable past action"]}}
completed_steps must contain exactly completed_step_count strings, each 8-200 characters.
Strict JSON only. No Markdown.

Assignments:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"""


def materialize_row(
    assignment: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    model: str,
    generated_at: str,
    retry_history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine teacher wording with deterministic registry, labels, and state."""

    if str(generated.get("assignment_id")) != str(assignment["assignment_id"]):
        raise ValueError("teacher assignment ID mismatch")
    question = str(generated.get("question") or "").strip()
    current_step = str(generated.get("current_step") or "").strip()
    completed = generated.get("completed_steps")
    expected_history = {"empty": 0, "short": 1, "long": 3}[str(assignment["matrix_cell"]["history_length"])]
    if not 20 <= len(question) <= 400 or not 20 <= len(current_step) <= 400:
        raise ValueError("teacher text length is outside 20-400 characters")
    if not isinstance(completed, list) or len(completed) != expected_history:
        raise ValueError("completed_steps count does not match the assignment")
    if any(not isinstance(value, str) or not 8 <= len(value.strip()) <= 200 for value in completed):
        raise ValueError("completed_steps contains invalid text")
    combined = "\n".join([question, current_step, *map(str, completed)]).casefold()
    if PROJECT_MARKER.search(combined):
        raise ValueError("project-specific marker in teacher text")
    if str(assignment["assignment_id"]).casefold() in combined:
        raise ValueError("assignment identifier leaked into teacher text")
    registry, legal_ids, target_id, hard_negative_ids, previous = build_registry(assignment)
    if any(tool_id.casefold() in combined for tool_id in registry.by_id):
        raise ValueError("tool identifier leaked into teacher text")
    cell = dict(assignment["matrix_cell"])
    task_kind = "recover" if previous else "route"
    if previous and not RECOVERY_MARKER.search(combined):
        raise ValueError("recovery is not observable in teacher wording")
    modality = str(cell["source_modality"])
    available_modalities = [modality] if modality != "mixed" else ["text", "structured_data", "document", "code", "image", "audio"]
    row = {
        "schema_version": "decision-state.v2",
        "dataset_version": DATASET_VERSION,
        "decision_state_id": str(assignment["assignment_id"]),
        "trajectory_id": f"trajectory-{assignment['assignment_id']}",
        "scenario_id": f"scenario-{assignment['slot_id']}",
        "step": {"initial": 0, "middle": 2, "late": 5, "terminal": 7}[str(cell["session_position"])],
        "question": question,
        "task_kind": task_kind,
        "agent_state": {"state_name": "active", "phase": "execution", "session_position": cell["session_position"]},
        "history": [{"completed_step": str(value).strip(), "status": "complete"} for value in completed],
        "plan": {"remaining_step": current_step},
        "observed_evidence": [{"result_id": f"observation_{index}", "inspection_status": "inspected"} for index, _ in enumerate(completed)],
        "governance": {"allowed_side_effect_classes": sorted(_allowed_effects(str(cell["side_effect_policy"]))), "call_allowed_side_effect_classes": sorted(_allowed_effects(str(cell["side_effect_policy"])))},
        "resource_state": {"remaining_steps": max(1, 8 - int({"initial": 0, "middle": 2, "late": 5, "terminal": 7}[str(cell["session_position"])])), "unresolved_requirement_count": 1},
        "source_state": {"source_ids": [f"resource_{assignment['slot'] % 97:02d}"], "available_modalities": available_modalities, "inventory_state": "known", "inspection_state": "partial_context" if str(cell["scenario_family"]) == "partial_context_vs_new_search" else "known", "schema_known": str(cell["scenario_family"]) != "schema_vs_records"},
        "query_state": {"query_terms": current_step.casefold().split()[:24], "schema_known": str(cell["scenario_family"]) != "schema_vs_records"},
        "previous_candidate_ids": previous,
        "expansion_context": {"expansion_allowed": bool(previous), "expansion_round": 1 if previous else 0, "trigger": "wrong_tool" if previous else "none", "prior_candidate_ids": previous, "excluded_candidate_ids": previous, "unresolved_requirement": current_step},
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": legal_ids,
        "label": {"acceptable_tools": [target_id] if target_id else [], "ranked_tools": ([target_id] if target_id else []) + [value for value in legal_ids if value != target_id], "hard_negative_tools": hard_negative_ids if target_id else list(legal_ids), "label_source": VALIDATOR_VERSION},
        "accepted": True,
        "evaluation_partition": "train",
        "split_group_id": f"scenario-{assignment['slot_id']}",
        "question_template_id": f"scaling-train-v1-{cell['scenario_family']}-{cell['wording_style']}",
        "matrix_cell": cell,
        "matrix_cell_id": assignment["matrix_cell_id"],
        "teacher_paraphrase": current_step,
        "provenance": {
            "corpus": DATASET_VERSION,
            "dataset_version": DATASET_VERSION,
            "matrix_version": MATRIX_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "teacher": "deepseek",
            "artifact": "DeepSeek-api",
            "generated_at": generated_at,
            "seed": int(assignment["seed"]),
            "validator_version": VALIDATOR_VERSION,
            "feature_version": FEATURE_VERSION,
            "registry_fingerprint": registry.fingerprint,
            "trajectory_hash": digest({"assignment": assignment, "generated": generated}),
            "matrix_cell_id": assignment["matrix_cell_id"],
            "source_lineage": f"{MATRIX_VERSION}:{assignment['slot_id']}",
            "source_row_hash": digest(assignment),
            "retry_split_history": list(retry_history),
            "teacher_fallback_used": False,
        },
    }
    report = validate_decision_state_v2(row)
    if not report.valid:
        raise ValueError(report.as_dict())
    if target_id and target_id not in legal_ids:
        raise ValueError("positive tool is not legal")
    if set(previous) & set(legal_ids):
        raise ValueError("recovery repeats a previous candidate")
    return row


def normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def semantic_signature(question: str, current_step: str) -> str:
    stop = {"a", "an", "and", "for", "i", "in", "is", "it", "of", "on", "the", "to", "we", "with"}
    tokens = sorted(set(re.findall(r"[a-z0-9]+", f"{question} {current_step}".casefold())) - stop)
    return digest(tokens)
