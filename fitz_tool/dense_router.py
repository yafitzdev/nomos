"""Identity-free text views for semantic tool retrieval experiments."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .generic_contracts import observable_router_state
from .tool_registry import ToolRegistry, ToolSpec


DENSE_TEXT_VERSION = "dense-text.v1"


def _words(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip()


def _values(label: str, values: Iterable[str]) -> str | None:
    rendered = ", ".join(_words(str(value)) for value in values if str(value))
    return f"{label}: {rendered}" if rendered else None


def _schema_summary(schema: Mapping[str, Any]) -> str:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not isinstance(properties, Mapping):
        return ""
    fields = []
    for name, definition in sorted(properties.items(), key=lambda item: str(item[0])):
        field_type = "value"
        if isinstance(definition, Mapping):
            field_type = str(definition.get("type") or field_type)
        marker = " required" if name in required else " optional"
        fields.append(f"{_words(str(name))} ({_words(field_type)}{marker})")
    return ", ".join(fields)


def candidate_document(tool: ToolSpec) -> str:
    """Render tool semantics without its concrete identity."""

    parts = [tool.description.strip()]
    for value in (
        _values("Capabilities", tool.capabilities),
        _values("Accepts", tool.input_modalities),
        _values("Returns", tool.output_modalities),
        _values("Evidence role", tool.evidence_roles),
        _values("Prerequisites", tool.prerequisites),
        _values("Constraints", tool.constraints),
        f"Side effects: {_words(tool.side_effect_class)}",
    ):
        if value:
            parts.append(value)
    schema = _schema_summary(tool.argument_schema)
    if schema:
        parts.append(f"Arguments: {schema}")
    return ". ".join(parts)


def query_document(state: Mapping[str, Any]) -> str:
    """Render only information observable to a router at decision time."""

    observable = observable_router_state(state)
    parts = [str(observable.get("question") or "").strip()]
    task_kind = str(observable.get("task_kind") or "route")
    parts.append(f"Decision: {_words(task_kind)}")

    expansion = observable.get("expansion_context") or {}
    if isinstance(expansion, Mapping) and task_kind == "recover":
        unresolved = expansion.get("unresolved_requirement")
        trigger = expansion.get("trigger")
        if unresolved:
            parts.append(f"Still unresolved: {unresolved}")
        if trigger and trigger != "none":
            parts.append(f"Earlier tools failed because: {_words(str(trigger))}")

    for key in ("agent_state", "source_state", "query_state", "governance", "resource_state"):
        value = observable.get(key)
        if isinstance(value, Mapping) and value:
            parts.append(f"{_words(key).title()}: {json.dumps(value, sort_keys=True)}")
    return "\n".join(part for part in parts if part)


def eligible_tools(state: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
    """Resolve legal tools and enforce recovery no-repeat as an invariant."""

    raw_registry = state.get("tool_registry")
    if not isinstance(raw_registry, Mapping):
        raise ValueError("state must embed a tool_registry")
    registry = ToolRegistry.from_dict(raw_registry)
    legal_ids = [str(value) for value in state.get("legal_candidate_ids") or []]
    excluded = (
        {str(value) for value in state.get("previous_candidate_ids") or []}
        if state.get("task_kind") == "recover"
        else set()
    )
    return tuple(tool for tool in registry.resolve(legal_ids) if tool.tool_id not in excluded)
