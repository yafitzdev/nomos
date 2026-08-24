"""Agent-agnostic deterministic validation for proposed tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .tool_registry import ToolRegistry, ToolSpec


@dataclass(frozen=True)
class ToolCallValidation:
    """The runner-safe result of validating one proposed call."""

    valid: bool
    tool_id: str
    failure_reasons: tuple[str, ...]
    repairable: bool
    checked: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "tool_id": self.tool_id,
            "failure_reasons": list(self.failure_reasons),
            "repairable": self.repairable,
            "checked": list(self.checked),
        }


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _schema_errors(schema: Mapping[str, Any], value: Any, path: str = "arguments") -> list[str]:
    errors: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _type_matches(value, schema_type):
        errors.append(f"schema_type:{path}:{schema_type}")
        return errors
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"schema_enum:{path}")
    if isinstance(value, Mapping):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for name in required:
            if name not in value:
                errors.append(f"schema_missing_required:{path}.{name}")
        if isinstance(properties, Mapping):
            for name, child in properties.items():
                if name in value and isinstance(child, Mapping):
                    errors.extend(_schema_errors(child, value[name], f"{path}.{name}"))
        if schema.get("additionalProperties") is False and isinstance(properties, Mapping):
            for name in value:
                if name not in properties:
                    errors.append(f"schema_additional_property:{path}.{name}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            errors.extend(_schema_errors(schema["items"], item, f"{path}[{index}]"))
    return errors


def _state_mapping(state: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = state.get(name)
    return value if isinstance(value, Mapping) else {}


def _has_inspected_evidence(state: Mapping[str, Any]) -> bool:
    evidence = state.get("observed_evidence")
    return isinstance(evidence, list) and any(
        isinstance(item, Mapping) and item.get("inspection_status") == "inspected"
        for item in evidence
    )


def _prerequisite_met(prerequisite: str, state: Mapping[str, Any]) -> bool:
    governance = _state_mapping(state, "governance")
    source_state = _state_mapping(state, "source_state")
    query_state = _state_mapping(state, "query_state")
    evidence = state.get("observed_evidence")
    evidence_items = evidence if isinstance(evidence, list) else []
    available = set(str(value) for value in source_state.get("available_modalities") or [])
    requirements = governance.get("requirements")
    requirements_list = requirements if isinstance(requirements, list) else []
    statuses = {
        str(item.get("status"))
        for item in requirements_list
        if isinstance(item, Mapping)
    }
    if prerequisite in {"none", "objective_available"}:
        return prerequisite == "none" or bool(str(state.get("question") or "").strip())
    if prerequisite == "query_available":
        return bool(query_state.get("query_terms")) or bool(str(state.get("question") or "").strip())
    if prerequisite == "source_selected":
        return bool(source_state.get("source_ids"))
    if prerequisite == "indexed_sources_required":
        return bool(source_state.get("source_ids")) and source_state.get("inventory_state") in {
            "known",
            "partial",
        }
    if prerequisite == "metadata_catalog_required":
        return "metadata" in available
    if prerequisite == "structured_source_required":
        return bool(available & {"csv", "excel", "sqlite", "structured", "mixed"})
    if prerequisite == "pdf_source_required":
        return bool(available & {"pdf", "mixed"})
    if prerequisite == "code_source_required":
        return bool(available & {"code", "mixed"})
    if prerequisite == "textual_source_required":
        return bool(available & {"text", "pdf", "code", "mixed"})
    if prerequisite == "schema_known":
        return bool(
            source_state.get("schema_known")
            or query_state.get("schema_known")
            or source_state.get("inspection_state") in {"full_context", "multi_source_inspected"}
        )
    if prerequisite == "evidence_candidate_selected":
        return any(
            isinstance(item, Mapping)
            and item.get("inspection_status") in {"candidate", "inspected"}
            for item in evidence_items
        )
    if prerequisite == "incomplete_context_required":
        return source_state.get("inspection_state") in {"snippets_only", "partial_context"}
    if prerequisite == "multiple_evidence_required":
        return len(evidence_items) >= 2
    if prerequisite == "evidence_inspected":
        return _has_inspected_evidence(state)
    if prerequisite == "tracked_requirements_required":
        return bool(requirements_list)
    if prerequisite == "canonical_evidence_set_required":
        return bool(governance.get("canonical_evidence_set")) or _has_inspected_evidence(state)
    if prerequisite == "requirements_updated":
        return bool(requirements_list) and statuses != {"missing"}
    if prerequisite == "requirements_complete":
        return bool(requirements_list) and statuses <= {"complete"} and "complete" in statuses
    if prerequisite == "fresh_sufficient_assessment_required":
        return bool(governance.get("assessment_fresh")) and _prerequisite_met(
            "requirements_complete", state
        )
    if prerequisite == "candidate_set_available":
        return bool(state.get("legal_candidate_ids"))
    if prerequisite == "prior_candidates_exhausted":
        expansion = _state_mapping(state, "expansion_context")
        return bool(expansion.get("prior_candidate_ids")) and bool(
            expansion.get("expansion_allowed")
        )
    # Unknown prerequisites are unsafe to assume satisfied.
    return False


def _repairable(reasons: list[str]) -> bool:
    permanent_prefixes = (
        "unknown_tool",
        "illegal_candidate",
        "side_effect_disallowed",
        "repeated_rejected_candidate",
    )
    return not any(reason.startswith(permanent_prefixes) for reason in reasons)


def validate_tool_call(
    registry: ToolRegistry,
    state: Mapping[str, Any],
    proposed_call: Mapping[str, Any],
) -> ToolCallValidation:
    """Validate identity, legality, schema, modality, prerequisites, and effects.

    The function intentionally validates only deterministic contract properties.
    It does not claim that the tool's eventual result is factually correct.
    """

    tool_id = str(proposed_call.get("tool_id") or "")
    reasons: list[str] = []
    checked = ["tool_identity"]
    tool: ToolSpec | None = registry.by_id.get(tool_id)
    if tool is None:
        reasons.append("unknown_tool")
    else:
        legal_ids = {str(value) for value in state.get("legal_candidate_ids") or []}
        if tool_id not in legal_ids:
            reasons.append("illegal_candidate")

    if tool is not None:
        checked.append("argument_schema")
        arguments = proposed_call.get("arguments")
        if not isinstance(arguments, Mapping):
            reasons.append("schema_arguments_not_object")
        else:
            reasons.extend(_schema_errors(tool.argument_schema, arguments))

        checked.append("modality")
        source_state = _state_mapping(state, "source_state")
        available = set(str(value) for value in source_state.get("available_modalities") or [])
        call_modality = proposed_call.get("input_modality")
        if call_modality is not None and str(call_modality) not in tool.input_modalities:
            reasons.append("wrong_modality:call_input")
        if available and not (available & set(tool.input_modalities)):
            reasons.append("wrong_modality:state")

        checked.append("prerequisites")
        for prerequisite in tool.prerequisites:
            if not _prerequisite_met(prerequisite, state):
                reasons.append(f"unmet_prerequisite:{prerequisite}")

        checked.append("side_effect_policy")
        governance = _state_mapping(state, "governance")
        allowed = governance.get("call_allowed_side_effect_classes")
        if allowed is None:
            allowed = governance.get("allowed_side_effect_classes")
        if isinstance(allowed, list) and tool.side_effect_class not in set(str(value) for value in allowed):
            reasons.append(f"side_effect_disallowed:{tool.side_effect_class}")

        checked.append("candidate_history")
        history = _state_mapping(state, "expansion_context")
        rejected = {str(value) for value in history.get("excluded_candidate_ids") or []}
        if tool_id in rejected:
            reasons.append("repeated_rejected_candidate")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return ToolCallValidation(
        valid=not unique_reasons,
        tool_id=tool_id,
        failure_reasons=unique_reasons,
        repairable=_repairable(list(unique_reasons)) if unique_reasons else False,
        checked=tuple(checked),
    )
