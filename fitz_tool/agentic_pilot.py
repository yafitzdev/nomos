"""Deterministic skeletons and validators for Nomos agentic v1 data."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .call_validation import validate_tool_call
from .generic_contracts import ContractReport, validate_decision_state_v2
from .generic_pilot_v3 import (
    CAPABILITY_BLUEPRINTS,
    CAPABILITY_FOCUS,
    PHASE_TO_STATE,
    TARGET_CAPABILITIES,
    TASK_FOCUS,
    _schema,
    _source_cards,
    load_generic_matrix_spec,
)
from .router_v2 import FEATURE_VERSION
from .tool_registry import ToolRegistry, ToolSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_MATRIX_PATH = PROJECT_ROOT / "configs" / "matrix.agentic.v1.json"
AGENTIC_DATASET_VERSION = "nomos-agentic.v1"
AGENTIC_PILOT_VERSION = "agentic-pilot.v1"
AGENTIC_GENERATED_AT = "2026-08-24T00:00:00+00:00"
AGENTIC_TARGETS = tuple(TARGET_CAPABILITIES) + ("candidate_expansion",)
PROJECT_MARKER_RE = re.compile(r"(?<![a-z0-9_])(fitz|sage|bm25)(?![a-z0-9_])", re.IGNORECASE)

EXPANSION_BLUEPRINT = {
    "family": "candidate_recovery",
    "inputs": ["agent_state", "text"],
    "outputs": ["candidate_set"],
    "roles": ["candidate_discovery"],
    "side_effect": "none",
    "constraints": ["prior_candidates_exhausted"],
    "prerequisites": ["candidate_set_available"],
    "schema": {
        "reason": "string",
        "excluded_tool_ids": "array",
        "unresolved_requirement": "string",
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_agentic_matrix_spec(path: Path | str = AGENTIC_MATRIX_PATH) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("matrix_version") != "matrix.agentic.v1":
        raise ValueError("expected matrix.agentic.v1")
    for group_name in ("dimensions", "observable_ranges"):
        group = spec.get(group_name)
        if not isinstance(group, Mapping) or not group:
            raise ValueError(f"{group_name} must be a non-empty object")
        for name, values in group.items():
            if not isinstance(values, list) or not values or len(values) != len(set(map(str, values))):
                raise ValueError(f"{group_name}.{name} must contain unique values")
    return spec


def agentic_matrix_cell_id(cell: Mapping[str, Any]) -> str:
    return _digest(dict(cell))


def _agentic_cell_errors(cell: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for group_name in ("dimensions", "observable_ranges"):
        for name, allowed in spec[group_name].items():
            if name not in cell:
                errors.append(f"missing {name}")
            elif cell[name] not in allowed:
                errors.append(f"invalid {name}: {cell[name]!r}")
    if errors:
        return errors
    task_kind = str(cell["task_kind"])
    validation_case = str(cell["validation_case"])
    trigger = str(cell["expansion_trigger"])
    recovery_round = int(cell["recovery_round"])
    history = str(cell["candidate_history_state"])
    terminal = str(cell["terminal_outcome"])
    if task_kind == "route" and (
        validation_case != "not_applicable"
        or trigger != "none"
        or recovery_round != 0
        or history != "none"
        or terminal != "selected"
    ):
        errors.append("route rows must describe an initial selection")
    if task_kind == "recover" and (
        validation_case != "not_applicable"
        or trigger == "none"
        or recovery_round not in {1, 2}
        or history == "none"
        or terminal != "expanded"
    ):
        errors.append("recover rows must describe bounded candidate expansion")
    if task_kind == "verify" and (
        validation_case == "not_applicable"
        or trigger != "none"
        or recovery_round != 0
        or history != "none"
        or terminal not in {"call_accepted", "call_rejected"}
    ):
        errors.append("verify rows must describe one call-validation outcome")
    if int(cell["top_k"]) > int(cell["candidate_pool_size"]):
        errors.append("top_k cannot exceed candidate_pool_size")
    if task_kind != "recover" and int(cell["prior_candidate_count"]) != 0:
        errors.append("non-recovery rows cannot have prior candidates")
    if task_kind == "recover" and int(cell["prior_candidate_count"]) not in {3, 6}:
        errors.append("recovery rows must exclude three or six prior candidates")
    return errors


def _tool_blueprint(capability: str) -> Mapping[str, Any]:
    if capability == "candidate_expansion":
        return EXPANSION_BLUEPRINT
    return CAPABILITY_BLUEPRINTS[capability]


def build_agentic_registry(
    registry_index: int,
    *,
    unseen_axis: str = "familiar",
    variant_count: int = 6,
) -> ToolRegistry:
    """Build a large opaque registry with semantic, not project-specific, tools."""

    tools: list[dict[str, Any]] = []
    for capability_index, capability in enumerate(AGENTIC_TARGETS):
        blueprint = _tool_blueprint(capability)
        for variant in range(variant_count):
            if unseen_axis == "unseen_tool_id":
                tool_id = f"opaque_{registry_index:03d}_{capability_index:02d}_{variant:02d}"
            else:
                tool_id = f"candidate_{registry_index:03d}_{capability_index:02d}_{variant:02d}"
            family = f"family_{blueprint['family']}"
            if unseen_axis == "unseen_tool_family":
                family = f"family_novel_{blueprint['family']}"
            properties = dict(blueprint["schema"])
            if variant % 3 == 1:
                properties[f"context_{(registry_index + capability_index) % 7}"] = "string"
            capabilities = [capability]
            if variant % 3 == 2 and capability in {
                "search_content",
                "search_metadata",
                "inspect_evidence",
                "expand_context",
                "compare_evidence",
            }:
                capabilities.append("candidate_disambiguation")
            focus = CAPABILITY_FOCUS.get(capability, "additional tool candidates")
            descriptions = (
                f"Use this candidate to handle {focus} under the current agent state.",
                f"This operation supports {focus} while preserving the task constraints.",
                f"Apply this semantic operation when the next move depends on {focus}.",
            )
            side_effect = str(blueprint["side_effect"])
            if capability == "candidate_expansion":
                side_effect = "none"
            tools.append(
                {
                    "tool_id": tool_id,
                    "tool_family": family,
                    "description": descriptions[variant % len(descriptions)],
                    "capabilities": capabilities,
                    "input_modalities": list(blueprint["inputs"]),
                    "output_modalities": list(blueprint["outputs"]),
                    "evidence_roles": list(blueprint["roles"]),
                    "side_effect_class": side_effect,
                    "argument_schema": _schema(properties, registry_index + variant),
                    "constraints": list(blueprint["constraints"]),
                    "prerequisites": list(blueprint["prerequisites"]),
                }
            )
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"agentic_registry_{unseen_axis}_{registry_index:04d}",
            "tools": tools,
        }
    )


def _source_state(modality: str, source_ids: list[str]) -> dict[str, Any]:
    if modality == "mixed":
        available = ["text", "pdf", "csv", "excel", "sqlite", "code", "metadata"]
    else:
        available = [modality, "metadata"]
    return {
        "available_modalities": available,
        "inventory_state": "known",
        "inspection_state": "multi_source_inspected",
        "source_ids": source_ids,
        "schema_known": True,
        "evidence_topology_observed": "one_passage",
    }


def _requirements(status: str = "complete") -> list[dict[str, str]]:
    return [
        {"requirement_id": "R1", "status": status},
        {"requirement_id": "R2", "status": status},
    ]


def _base_state(
    cell: Mapping[str, Any],
    question: str,
    source_ids: list[str],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    phase = "inspection" if cell["task_kind"] == "verify" else "retrieval"
    modality = str(cell.get("source_modality", "text"))
    evidence = [
        {
            "evidence_id": f"result-{rng.randrange(10**12):012d}",
            "source_id": source_ids[index % len(source_ids)],
            "modality": modality if modality != "mixed" else "text",
            "inspection_status": "inspected",
            "claim_count": 1,
        }
        for index in range(2)
    ]
    return {
        "question": question,
        "task_kind": cell["task_kind"],
        "candidate_pool_size": cell["candidate_pool_size"],
        "previous_candidate_ids": [],
        "agent_state": {
            "state_name": PHASE_TO_STATE[phase],
            "phase": phase,
            "question_length_band": "long",
        },
        "history": [],
        "plan": {"active": True, "operation": "lookup_semantic"},
        "observed_evidence": evidence,
        "governance": {
            "assessment_fresh": True,
            "requirements": _requirements(),
            "allowed_side_effect_classes": ["none", "read", "local_state_write"],
            "call_allowed_side_effect_classes": ["none", "read", "local_state_write"],
            "canonical_evidence_set": [item["evidence_id"] for item in evidence],
        },
        "resource_state": {
            "remaining_steps": cell["remaining_steps"],
            "unresolved_requirement_count": cell["unresolved_requirement_count"],
            "observed_evidence_count": 2,
            "distractor_count": max(0, int(cell["candidate_pool_size"]) - 3),
            "prior_search_count": 1,
        },
        "source_state": _source_state(modality, source_ids),
        "query_state": {
            "operation": "lookup_semantic",
            "specificity": "entity_bound",
            "match_strategy": "hybrid",
            "query_terms": ["current", "task", "requirements"],
            "schema_known": True,
        },
    }


def _valid_arguments(tool: ToolSpec) -> dict[str, Any]:
    properties = tool.argument_schema.get("properties") or {}
    required = tool.argument_schema.get("required") or list(properties)
    values: dict[str, Any] = {}
    for name in required:
        schema = properties.get(name) if isinstance(properties, Mapping) else {}
        kind = schema.get("type") if isinstance(schema, Mapping) else "string"
        values[str(name)] = {
            "string": "current requirement",
            "integer": 2,
            "number": 2.0,
            "boolean": True,
            "array": ["R1"],
            "object": {"status": "current"},
        }.get(kind, "current requirement")
    return values


def _call_for_case(
    registry: ToolRegistry,
    legal_ids: list[str],
    state: dict[str, Any],
    target: str,
    validation_case: str,
    row_index: int,
) -> tuple[dict[str, Any], str]:
    target_tools = [tool for tool in registry.tools if target in tool.capabilities]
    if not target_tools:
        target_tools = list(registry.tools)
    legal_target_tools = [tool for tool in target_tools if tool.tool_id in set(legal_ids)]
    selection_pool = legal_target_tools or target_tools
    tool = selection_pool[row_index % len(selection_pool)]
    call_tool_id = tool.tool_id
    arguments = _valid_arguments(tool)
    source_state = state["source_state"]
    if validation_case == "valid_call":
        source_state["available_modalities"] = list(tool.input_modalities)
        if "expand_context" in tool.capabilities:
            source_state["inspection_state"] = "partial_context"
    if validation_case == "unknown_tool":
        call_tool_id = f"missing_candidate_{row_index:05d}"
        arguments = {}
    elif validation_case == "illegal_candidate":
        outside = next(candidate for candidate in registry.tools if candidate.tool_id not in legal_ids)
        call_tool_id = outside.tool_id
        arguments = _valid_arguments(outside)
    elif validation_case == "schema_missing_required":
        required = list(tool.argument_schema.get("required") or [])
        if required:
            arguments.pop(str(required[0]), None)
    elif validation_case == "schema_wrong_type":
        properties = tool.argument_schema.get("properties") or {}
        required = list(tool.argument_schema.get("required") or properties)
        if required:
            name = str(required[0])
            schema = properties.get(name) if isinstance(properties, Mapping) else {}
            expected = schema.get("type") if isinstance(schema, Mapping) else "string"
            arguments[name] = {
                "string": 7,
                "integer": "not-an-integer",
                "number": "not-a-number",
                "boolean": "not-a-boolean",
                "array": {},
                "object": "not-an-object",
            }.get(expected, 7)
    elif validation_case == "wrong_modality":
        source_state["available_modalities"] = ["audio"]
    elif validation_case == "stale_state":
        stale_tool = next(
            (candidate for candidate in registry.tools if "source_selected" in candidate.prerequisites),
            tool,
        )
        tool = stale_tool
        call_tool_id = tool.tool_id
        arguments = _valid_arguments(tool)
        source_state["source_ids"] = []
        source_state["inspection_state"] = "none"
        state["observed_evidence"] = []
        state["governance"]["assessment_fresh"] = False
    elif validation_case == "unmet_prerequisite":
        prerequisite_tool = next(
            (
                candidate
                for candidate in registry.tools
                if "schema_known" in candidate.prerequisites
            ),
            tool,
        )
        tool = prerequisite_tool
        call_tool_id = tool.tool_id
        arguments = _valid_arguments(tool)
        state["query_state"]["schema_known"] = False
        state["source_state"]["schema_known"] = False
        state["source_state"]["inspection_state"] = "none"
    elif validation_case == "side_effect_disallowed":
        state["governance"]["call_allowed_side_effect_classes"] = ["none", "read"]
        write_tool = next(
            (candidate for candidate in registry.tools if candidate.side_effect_class == "local_state_write"),
            tool,
        )
        call_tool_id = write_tool.tool_id
        arguments = _valid_arguments(write_tool)
    call = {
        "tool_id": call_tool_id,
        "arguments": arguments,
        "input_modality": (
            "audio" if validation_case == "wrong_modality" else (tool.input_modalities[0] if tool.input_modalities else None)
        ),
    }
    return call, call_tool_id


def _candidate_pool(
    registry: ToolRegistry,
    target: str,
    pool_size: int,
    task_kind: str,
    row_index: int,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    target_tools = [tool for tool in registry.tools if target in tool.capabilities]
    if len(target_tools) < 6:
        raise RuntimeError(f"registry lacks enough target variants for {target}")
    target_tools = target_tools[:6]
    if task_kind == "recover":
        prior_count = 3 if row_index % 2 == 0 else 6
        prior = [tool.tool_id for tool in target_tools[:prior_count]]
        acceptable = [tool.tool_id for tool in target_tools[prior_count : prior_count + 3]]
    else:
        prior = []
        acceptable = [tool.tool_id for tool in target_tools[:3]]
    selected = list(dict.fromkeys([*prior, *acceptable]))
    remaining = [tool.tool_id for tool in registry.tools if tool.tool_id not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: pool_size - len(selected)])
    rng.shuffle(selected)
    return selected, acceptable, prior


def _question_placeholder(cell: Mapping[str, Any], target: str) -> str:
    focus = CAPABILITY_FOCUS.get(target, "the next available alternatives")
    task = TASK_FOCUS.get(str(cell.get("task_domain", "workflow_automation")), "the current task")
    if cell["task_kind"] == "verify":
        return f"Check whether the proposed operation is safe and valid for {task}."
    if cell["task_kind"] == "recover":
        return f"The first candidates did not resolve {task}; what should be considered next for {focus}?"
    return f"Which operation should handle {focus} for {task}?"


def generate_agentic_states(
    count: int = 1000,
    *,
    seed: int = 20260824,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    spec = load_agentic_matrix_spec()
    generic_spec = load_generic_matrix_spec()
    cards = _source_cards()
    rng = random.Random(seed)
    used_cells: set[str] = set()
    used_types: set[str] = set()
    used_instances: set[str] = set()
    id_prefix = f"agentic-v1-{seed}-"
    trajectory_prefix = f"{id_prefix}trajectory-"
    scenario_prefix = f"{id_prefix}scenario-"
    rows: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    for index in range(count):
        task_kind = ("route", "recover", "verify")[index % 3]
        kind_ordinal = kind_counts[task_kind]
        kind_counts[task_kind] += 1
        target = TARGET_CAPABILITIES[index % len(TARGET_CAPABILITIES)]
        pool_size = (10, 30, 100)[index % 3]
        top_k = 3 if index % 4 else 1
        unseen_axis = "familiar"
        if index >= int(count * 0.70) and index % 3 == 0:
            unseen_axis = "unseen_tool_id"
        if index >= int(count * 0.85) and index % 3 == 1:
            unseen_axis = "unseen_tool_family"
        if task_kind == "route":
            validation_case = "not_applicable"
            trigger = "none"
            round_number = 0
            history_state = "none"
            terminal = "selected"
            prior_count = 0
        elif task_kind == "recover":
            validation_case = "not_applicable"
            trigger = (
                "empty_result",
                "partial_result",
                "contradictory_result",
                "tool_error",
                "stale_assessment",
                "budget_pressure",
            )[kind_ordinal % 6]
            round_number = 1 if index % 2 else 2
            history_state = "first_page_failed" if index % 2 else "prior_rejected"
            terminal = "expanded"
            prior_count = 3 if index % 2 else 6
        else:
            validation_cases = (
                "valid_call",
                "unknown_tool",
                "illegal_candidate",
                "schema_missing_required",
                "schema_wrong_type",
                "wrong_modality",
                "stale_state",
                "unmet_prerequisite",
                "side_effect_disallowed",
            )
            validation_case = validation_cases[kind_ordinal % len(validation_cases)]
            trigger = "none"
            round_number = 0
            history_state = "none"
            terminal = "call_accepted" if validation_case == "valid_call" else "call_rejected"
            prior_count = 0
        candidate_budget = min(pool_size, (3, 10, 30, 100)[index % 4])
        base_target = target if target in TARGET_CAPABILITIES else "plan_retrieval"
        base_cell = {
            name: rng.choice(values)
            for name, values in generic_spec["dimensions"].items()
        }
        base_cell.update({name: rng.choice(values) for name, values in spec["observable_ranges"].items()})
        base_cell["target_capability"] = base_target
        base_cell["task_domain"] = base_cell.get("task_domain", "workflow_automation")
        cell: dict[str, Any] = {
            **base_cell,
            "task_kind": task_kind,
            "target_capability": target,
            "candidate_pool_size": pool_size,
            "top_k": top_k,
            "validation_case": validation_case,
            "expansion_trigger": trigger,
            "recovery_round": round_number,
            "unseen_axis": unseen_axis,
            "candidate_history_state": history_state,
            "terminal_outcome": terminal,
            "prior_candidate_count": prior_count,
            "candidate_description_budget": candidate_budget,
        }
        if _agentic_cell_errors(cell, spec):
            raise RuntimeError(f"invalid generated cell {index}: {_agentic_cell_errors(cell, spec)}")
        cell_id = agentic_matrix_cell_id(cell)
        if cell_id in used_cells:
            raise RuntimeError(f"duplicate agentic matrix cell at row {index}")
        used_cells.add(cell_id)
        registry = build_agentic_registry(index % 64, unseen_axis=unseen_axis)
        source_ids = [f"generic_train_source_{index % 32:03d}"]
        question = _question_placeholder(cell, target)
        state = _base_state(cell, question, source_ids, seed + index * 1009)
        legal_ids, acceptable_ids, prior_ids = _candidate_pool(
            registry, target, pool_size, task_kind, index, random.Random(seed + index * 17)
        )
        state["legal_candidate_ids"] = legal_ids
        state["previous_candidate_ids"] = prior_ids
        state["expansion_context"] = {
            "expansion_allowed": task_kind == "recover",
            "expansion_action": "request_more_tool_candidates" if task_kind == "recover" else None,
            "expansion_round": round_number,
            "trigger": trigger,
            "prior_candidate_ids": prior_ids,
            "excluded_candidate_ids": prior_ids,
            "unresolved_requirement": "resolve the remaining task requirement",
        }
        call: dict[str, Any] | None = None
        validation: dict[str, Any] | None = None
        if task_kind == "verify":
            call, call_tool_id = _call_for_case(
                registry, legal_ids, state, target, validation_case, index
            )
            validation_result = validate_tool_call(registry, state, call)
            validation = validation_result.as_dict()
            expected_valid = validation_case == "valid_call"
            if validation_result.valid != expected_valid:
                raise RuntimeError(
                    f"verification oracle mismatch for row {index}: {validation_result.as_dict()}"
                )
            state["proposed_tool_call"] = call
            if expected_valid:
                acceptable_ids = [call_tool_id]
            else:
                acceptable_ids = []
        hard_negatives = [tool_id for tool_id in legal_ids if tool_id not in acceptable_ids]
        state.update(
            {
                "schema_version": "decision-state.v2",
                "dataset_version": AGENTIC_DATASET_VERSION,
                "decision_state_id": f"{id_prefix}{index:05d}",
                "trajectory_id": f"{trajectory_prefix}{index:05d}",
                "scenario_id": f"{scenario_prefix}{index:05d}",
                "step": 0,
                "tool_registry": registry.as_dict(),
                "legal_candidate_ids": legal_ids,
                "label": {
                    "acceptable_tools": acceptable_ids,
                    "ranked_tools": acceptable_ids + hard_negatives,
                    "hard_negative_tools": hard_negatives,
                    "label_source": "agentic_matrix_oracle.v1",
                },
                "accepted": bool(acceptable_ids) if task_kind == "verify" else True,
                "source_card_ids": source_ids,
                "source_kind": "deterministic_agentic_oracle",
                "evaluation_cohort": "agentic_train" if index < int(count * 0.70) else "agentic_holdout",
                "evaluation_partition": (
                    "train" if index < int(count * 0.70) else "validation" if index < int(count * 0.85) else "test"
                ),
                "split_group_id": f"agentic|seed-{seed}|{task_kind}|{unseen_axis}|pool-{pool_size}|row-{index}",
                "question_template_id": (
                    "agentic_verification" if task_kind == "verify" else "agentic_recovery" if task_kind == "recover" else "agentic_route"
                ),
                "matrix_cell": cell,
                "matrix_cell_id": cell_id,
                "sampling_context": {
                    "matrix_version": "matrix.agentic.v1",
                    "target_capability": target,
                    "task_kind": task_kind,
                    "candidate_pool_size": pool_size,
                    "top_k": top_k,
                    "validation_case": validation_case,
                    "expansion_trigger": trigger,
                    "recovery_round": round_number,
                    "unseen_axis": unseen_axis,
                    "candidate_history_state": history_state,
                },
                "provenance": {
                    "corpus": AGENTIC_DATASET_VERSION,
                    "source_card_hashes": [cards[source_id].content_sha256 for source_id in source_ids],
                    "prompt_version": "agentic-deterministic-skeleton.v1",
                    "model": "deterministic_agentic_oracle",
                    "artifact": "agentic-matrix-oracle.v1",
                    "teacher": "deterministic_agentic_oracle",
                    "seed": seed + index * 1009,
                    "validator_version": "agentic-validator.v1",
                    "feature_version": FEATURE_VERSION,
                    "registry_fingerprint": registry.fingerprint,
                    "trajectory_hash": _digest({"trajectory_id": f"{trajectory_prefix}{index:05d}", "seed": seed + index * 1009}),
                    "matrix_cell_id": cell_id,
                    "generated_at": AGENTIC_GENERATED_AT,
                },
            }
        )
        if call is not None:
            state["validation_label"] = validation
        state["type_signature"] = _digest(
            {
                "matrix_cell": cell,
                "candidate_semantics": sorted(
                    registry.by_id[tool_id].semantic_fingerprint for tool_id in legal_ids
                ),
                "previous_candidate_semantics": sorted(
                    registry.by_id[tool_id].semantic_fingerprint for tool_id in prior_ids
                ),
            }
        )
        state["instance_signature"] = _digest(
            {
                "type_signature": state["type_signature"],
                "question": question,
                "teacher_paraphrase": "",
                "proposed_tool_call": call,
                "seed": state["provenance"]["seed"],
            }
        )
        if state["type_signature"] in used_types or state["instance_signature"] in used_instances:
            raise RuntimeError(f"duplicate agentic signature at row {index}")
        used_types.add(state["type_signature"])
        used_instances.add(state["instance_signature"])
        rows.append(state)
    manifest = {
        "dataset_version": AGENTIC_DATASET_VERSION,
        "pilot_version": AGENTIC_PILOT_VERSION,
        "feature_version": FEATURE_VERSION,
        "seed": seed,
        "count": count,
        "task_kind_counts": dict(Counter(str(row["task_kind"]) for row in rows)),
        "partition_counts": dict(Counter(str(row["evaluation_partition"]) for row in rows)),
        "pool_size_counts": dict(Counter(str(row["matrix_cell"]["candidate_pool_size"]) for row in rows)),
        "validation_case_counts": dict(Counter(str(row["matrix_cell"]["validation_case"]) for row in rows)),
        "expansion_trigger_counts": dict(Counter(str(row["matrix_cell"]["expansion_trigger"]) for row in rows)),
        "unseen_axis_counts": dict(Counter(str(row["matrix_cell"]["unseen_axis"]) for row in rows)),
        "matrix_cells": len(used_cells),
        "type_signatures": len(used_types),
        "instance_signatures": len(used_instances),
        "generic_registry_only": True,
    }
    return rows, manifest


def validate_agentic_state(state: Mapping[str, Any]) -> ContractReport:
    report = validate_decision_state_v2(state)
    if state.get("dataset_version") != AGENTIC_DATASET_VERSION:
        report.add("dataset_version", f"must equal {AGENTIC_DATASET_VERSION}")
    required = (
        "task_kind",
        "candidate_pool_size",
        "previous_candidate_ids",
        "expansion_context",
        "matrix_cell",
        "matrix_cell_id",
        "type_signature",
        "instance_signature",
        "validation_label",
    )
    for key in required:
        if key not in state and state.get("task_kind") == "verify":
            report.add(key, "missing agentic dataset field")
    if state.get("source_kind") not in {
        "deterministic_agentic_oracle",
        "ninfer_agentic_teacher",
        "deepseek_agentic_teacher",
    }:
        report.add("source_kind", "must identify an approved agentic source")
    if PROJECT_MARKER_RE.search(_canonical(state)):
        report.add("genericity", "contains a project-specific marker")
    registry_raw = state.get("tool_registry")
    try:
        registry = ToolRegistry.from_dict(registry_raw) if isinstance(registry_raw, Mapping) else None
    except Exception:
        registry = None
    cell = state.get("matrix_cell")
    if isinstance(cell, Mapping):
        if agentic_matrix_cell_id(cell) != state.get("matrix_cell_id"):
            report.add("matrix_cell_id", "does not match agentic matrix cell")
        try:
            errors = _agentic_cell_errors(cell, load_agentic_matrix_spec())
        except Exception as exc:
            errors = [str(exc)]
        for error in errors:
            report.add("matrix_cell", error)
    legal_ids = [str(value) for value in state.get("legal_candidate_ids") or []]
    if state.get("candidate_pool_size") != len(legal_ids):
        report.add("candidate_pool_size", "must equal len(legal_candidate_ids)")
    prior_ids = [str(value) for value in state.get("previous_candidate_ids") or []]
    if len(prior_ids) != len(set(prior_ids)):
        report.add("previous_candidate_ids", "must not contain duplicates")
    if set(prior_ids) & set(state.get("label", {}).get("acceptable_tools") or []):
        report.add("previous_candidate_ids", "must not overlap acceptable recovery tools")
    if registry is not None:
        if not set(legal_ids) <= set(registry.by_id):
            report.add("legal_candidate_ids", "contains IDs absent from registry")
        if state.get("task_kind") == "verify" and isinstance(state.get("proposed_tool_call"), Mapping):
            actual = validate_tool_call(registry, state, state["proposed_tool_call"]).as_dict()
            expected = state.get("validation_label")
            if isinstance(expected, Mapping):
                for key in ("valid", "tool_id", "failure_reasons"):
                    if actual.get(key) != expected.get(key):
                        report.add("validation_label", f"does not match deterministic call validation for {key}")
    return report
