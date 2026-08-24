"""Deterministic frozen decision states for the Nomos agentic v2 line."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any, Mapping

from .agentic_matrix_v2 import generate_matrix_cells
from .call_validation import validate_tool_call
from .generic_contracts import validate_decision_state_v2
from .generic_pilot_v3 import CAPABILITY_BLUEPRINTS, CAPABILITY_FOCUS, _schema
from .router_v2 import FEATURE_VERSION
from .tool_registry import SIDE_EFFECT_CLASSES, ToolRegistry


DATASET_VERSION = "nomos-agentic.v2"
VALIDATOR_VERSION = "agentic-v2-validator.v1"


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _registry(partition: str, profile: str) -> ToolRegistry:
    tools = []
    profile_index = int(profile.rsplit("_", 1)[-1])
    description_templates = {
        "train": (
            "Use this operation to {focus} while respecting the current workflow state.",
            "Handles {focus} from the available technical resources.",
        ),
        "validation": (
            "Suitable when the next step requires {focus}; state constraints still apply.",
            "Provides an interface for {focus} in a technical investigation.",
        ),
        "test": (
            "Choose this interface when the workflow needs {focus}.",
            "Its purpose is {focus}, subject to the supplied state and policy.",
        ),
    }[partition]
    for capability_index, (capability, blueprint) in enumerate(CAPABILITY_BLUEPRINTS.items()):
        for variant in range(8):
            properties = dict(blueprint["schema"])
            if variant % 4 == 3:
                properties[f"context_{(profile_index + capability_index) % 11}"] = "string"
            tools.append(
                {
                    "tool_id": f"{partition[0]}_{profile_index:03d}_{capability_index:02d}_{variant:02d}",
                    "tool_family": f"{partition}_family_{blueprint['family']}",
                    "description": description_templates[variant % 2].format(
                        focus=CAPABILITY_FOCUS[capability]
                    ),
                    "capabilities": [capability],
                    "input_modalities": list(blueprint["inputs"]),
                    "output_modalities": list(blueprint["outputs"]),
                    "evidence_roles": list(blueprint["roles"]),
                    "side_effect_class": blueprint["side_effect"],
                    "argument_schema": {
                        **_schema(properties, variant),
                        "additionalProperties": False,
                    },
                    "constraints": list(blueprint["constraints"]),
                    "prerequisites": list(blueprint["prerequisites"]),
                }
            )
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"{partition}_registry_{profile_index:03d}",
            "tools": tools,
        }
    )


def _valid_arguments(schema: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        definition = properties.get(name) or {}
        value_type = definition.get("type")
        output[name] = {
            "string": "value",
            "integer": 1,
            "number": 1.0,
            "boolean": True,
            "array": ["value"],
            "object": {},
        }.get(value_type)
    return output


def _question(cell: Mapping[str, Any]) -> str:
    focus = CAPABILITY_FOCUS[str(cell["target_capability"])]
    style = cell["language_style"]
    kind = cell["task_kind"]
    if kind == "verify":
        return "Check whether this proposed tool call is valid for the observable state and policy."
    prefix = {
        "direct": "Which available operation should handle",
        "conversational": "Can you pick the best operation for",
        "terse": "Next tool for",
        "technical": "Rank the legal interface that supports",
        "noisy_user": "I tried a few things; what should actually handle",
    }[style]
    if kind == "recover":
        return f"The previous suggestions failed. {prefix} {focus}, without repeating them?"
    return f"{prefix} {focus}?"


def _base_state(cell: Mapping[str, Any], question: str) -> dict[str, Any]:
    return {
        "question": question,
        "task_kind": cell["task_kind"],
        "agent_state": {"state_name": "active", "phase": "retrieval"},
        "history": [],
        "plan": {"objective": "resolve the technical integration question"},
        "observed_evidence": [
            {"result_id": "evidence_1", "inspection_status": "inspected"},
            {"result_id": "evidence_2", "inspection_status": "inspected"},
        ],
        "governance": {
            "assessment_fresh": True,
            "requirements": [{"requirement_id": "r1", "status": "complete"}],
            "canonical_evidence_set": ["evidence_1", "evidence_2"],
            "allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
            "call_allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
        },
        "resource_state": {"remaining_steps": 8},
        "source_state": {
            "source_ids": ["source_1"],
            "available_modalities": [
                "metadata",
                "text",
                "code",
                "pdf",
                "csv",
                "excel",
                "sqlite",
                "mixed",
                "source_list",
                "schema",
                "content",
                "document_structure",
                "evidence",
                "evidence_candidates",
                "evidence_comparison",
                "agent_state",
                "assessment",
                "governance_state",
                "plan",
            ],
            "inventory_state": "known",
            "inspection_state": "full_context",
            "schema_known": True,
        },
        "query_state": {"query_terms": question.lower().split(), "schema_known": True},
        "previous_candidate_ids": [],
        "expansion_context": {
            "expansion_allowed": cell["task_kind"] == "recover",
            "expansion_round": cell["recovery_round"],
            "trigger": cell["recovery_trigger"],
            "prior_candidate_ids": [],
            "excluded_candidate_ids": [],
            "unresolved_requirement": CAPABILITY_FOCUS[str(cell["target_capability"])],
        },
    }


def _candidate_pool(
    registry: ToolRegistry, cell: Mapping[str, Any], *, seed: int
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    target = str(cell["target_capability"])
    target_ids = [tool.tool_id for tool in registry.tools if target in tool.capabilities]
    distractors = [tool.tool_id for tool in registry.tools if target not in tool.capabilities]
    rng.shuffle(target_ids)
    rng.shuffle(distractors)
    previous = target_ids[: min(int(cell["prior_candidate_count"]), 3)] if cell["task_kind"] == "recover" else []
    acceptable = [] if cell["candidate_outcome"] == "no_suitable_candidate" else target_ids[3:4] if previous else target_ids[:1]
    pool_size = int(cell["candidate_pool_size"])
    selected = [*acceptable, *distractors[: max(0, pool_size - len(acceptable))]]
    rng.shuffle(selected)
    return selected, acceptable, previous


def _verification_call(
    registry: ToolRegistry,
    legal_ids: list[str],
    state: dict[str, Any],
    case: str,
) -> dict[str, Any]:
    tool = registry.require(legal_ids[0])
    if case == "unmet_prerequisite":
        tool = next(item for item in registry.tools if "schema_known" in item.prerequisites)
        legal_ids[0] = tool.tool_id
        state["query_state"]["schema_known"] = False
        state["source_state"]["schema_known"] = False
        state["source_state"]["inspection_state"] = "none"
    elif case == "stale_state":
        tool = next(item for item in registry.tools if "source_selected" in item.prerequisites)
        legal_ids[0] = tool.tool_id
        state["source_state"]["source_ids"] = []
        state["governance"]["assessment_fresh"] = False
    elif case == "side_effect_disallowed":
        tool = next(item for item in registry.tools if item.side_effect_class == "local_state_write")
        legal_ids[0] = tool.tool_id
        state["governance"]["call_allowed_side_effect_classes"] = ["none", "read"]
    arguments = _valid_arguments(tool.argument_schema)
    call = {"tool_id": tool.tool_id, "arguments": arguments}
    if case == "unknown_tool":
        call["tool_id"] = "unknown_tool"
    elif case == "illegal_candidate":
        call["tool_id"] = next(item.tool_id for item in registry.tools if item.tool_id not in legal_ids)
    elif case == "schema_missing_required" and tool.argument_schema.get("required"):
        arguments.pop(str(tool.argument_schema["required"][0]), None)
    elif case == "schema_wrong_type" and tool.argument_schema.get("required"):
        name = str(tool.argument_schema["required"][0])
        expected = (tool.argument_schema.get("properties") or {}).get(name, {}).get("type")
        arguments[name] = "wrong" if expected != "string" else 42
    elif case == "schema_extra_property":
        arguments["unexpected"] = "value"
    elif case == "wrong_modality":
        call["input_modality"] = "audio"
    elif case == "repeated_rejected_candidate":
        state["expansion_context"]["excluded_candidate_ids"] = [tool.tool_id]
    unique_ids = list(dict.fromkeys(legal_ids))
    target_size = len(legal_ids)
    unique_ids.extend(
        item.tool_id
        for item in registry.tools
        if item.tool_id not in unique_ids
    )
    legal_ids[:] = unique_ids[:target_size]
    return call


def generate_agentic_v2_states(
    count: int, *, seed: int = 20260827
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells, matrix_report = generate_matrix_cells(count, seed=seed)
    registries: dict[str, ToolRegistry] = {}
    rows = []
    for index, cell in enumerate(cells):
        profile = str(cell["registry_profile"])
        if profile not in registries:
            registries[profile] = _registry(str(cell["evaluation_partition"]), profile)
        registry = registries[profile]
        question = _question(cell)
        state = _base_state(cell, question)
        legal_ids, acceptable, previous = _candidate_pool(
            registry, cell, seed=seed + index * 7919
        )
        legal_ids = list(dict.fromkeys(legal_ids))
        state["legal_candidate_ids"] = legal_ids
        state["previous_candidate_ids"] = previous
        state["expansion_context"]["prior_candidate_ids"] = previous
        state["expansion_context"]["excluded_candidate_ids"] = previous
        validation_label = None
        if cell["task_kind"] == "verify":
            call = _verification_call(
                registry,
                legal_ids,
                state,
                str(cell["validation_case"]),
            )
            state["proposed_tool_call"] = call
            validation_label = validate_tool_call(registry, state, call).as_dict()
            if validation_label["valid"] != (cell["validation_case"] == "valid_call"):
                raise RuntimeError(
                    f"verification oracle mismatch at row {index}: {validation_label}"
                )
            acceptable = []
        hard_negatives = [tool_id for tool_id in legal_ids if tool_id not in acceptable]
        row = {
            **state,
            "schema_version": "decision-state.v2",
            "dataset_version": DATASET_VERSION,
            "decision_state_id": f"agentic-v2-{seed}-{index:07d}",
            "trajectory_id": f"agentic-v2-trajectory-{seed}-{index:07d}",
            "scenario_id": str(cell["scenario_group_id"]),
            "step": 0,
            "tool_registry": registry.as_dict(),
            "label": {
                "acceptable_tools": acceptable,
                "ranked_tools": [*acceptable, *hard_negatives],
                "hard_negative_tools": hard_negatives,
                "label_source": VALIDATOR_VERSION,
            },
            "accepted": bool(acceptable) or cell["task_kind"] == "verify",
            "evaluation_partition": cell["evaluation_partition"],
            "split_group_id": cell["scenario_group_id"],
            "question_template_id": cell["question_template_group"],
            "matrix_cell": cell,
            "matrix_cell_id": cell["matrix_cell_id"],
            "provenance": {
                "corpus": DATASET_VERSION,
                "prompt_version": "deterministic-agentic-v2-eval.v1",
                "model": "deterministic",
                "artifact": "agentic-v2-frozen-suite",
                "seed": seed + index * 7919,
                "validator_version": VALIDATOR_VERSION,
                "feature_version": FEATURE_VERSION,
                "registry_fingerprint": registry.fingerprint,
                "trajectory_hash": _digest({"index": index, "seed": seed}),
                "matrix_cell_id": cell["matrix_cell_id"],
            },
        }
        if validation_label is not None:
            row["validation_label"] = validation_label
        report = validate_decision_state_v2(row)
        if not report.valid:
            raise RuntimeError(f"invalid decision state {index}: {report.as_dict()}")
        rows.append(row)
    manifest = {
        **matrix_report,
        "dataset_version": DATASET_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "action_counts": dict(Counter(str(cell["expected_action"]) for cell in cells)),
        "registry_fingerprints_by_partition": {
            partition: sorted(
                {
                    ToolRegistry.from_dict(row["tool_registry"]).fingerprint
                    for row in rows
                    if row["evaluation_partition"] == partition
                }
            )
            for partition in ("train", "validation", "test")
        },
    }
    return rows, manifest
