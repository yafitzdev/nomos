"""Identity-free text views for semantic tool retrieval experiments."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .generic_contracts import observable_router_state
from .tool_registry import ToolRegistry, ToolSpec


DENSE_TEXT_VERSION = "dense-text.v3"


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


def candidate_views(tool: ToolSpec) -> tuple[str, ...]:
    """Render complementary semantic views of a tool without its concrete ID.

    Tool registries vary wildly: some descriptions are concise while others bury
    the useful action among schemas and governance metadata. Late interaction over
    both a complete document and focused fields makes retrieval less sensitive to
    that formatting without teaching Nomos any fixed tool vocabulary.
    """

    views = [candidate_document(tool)]
    description = tool.description.strip()
    if description:
        views.append(f"Tool purpose: {description}")
    capability = _values("Tool capabilities", tool.capabilities)
    inputs = _values("Accepts", tool.input_modalities)
    outputs = _values("Returns", tool.output_modalities)
    if capability:
        views.append(". ".join(value for value in (capability, inputs, outputs) if value))
    evidence = _values("Evidence role", tool.evidence_roles)
    if evidence:
        views.append(
            f"{evidence}. Side effects: {_words(tool.side_effect_class)}"
        )
    return tuple(dict.fromkeys(view for view in views if view))


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

    for key in (
        "history",
        "plan",
        "agent_state",
        "source_state",
        "query_state",
        "governance",
        "resource_state",
    ):
        value = observable.get(key)
        if isinstance(value, (Mapping, list)) and value:
            parts.append(f"{_words(key).title()}: {json.dumps(value, sort_keys=True)}")
    return "\n".join(part for part in parts if part)


def weighted_query_views(state: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    """Return identity-free intent views for robust late interaction scoring.

    A current plan or recovery requirement is often much more specific than the
    overall user objective. Keeping those views separate prevents completed or
    background intent from dominating one pooled query embedding.
    """

    observable = observable_router_state(state)
    question = str(observable.get("question") or "").strip()
    full_state = query_document(state)
    focused: list[str] = []
    plan = observable.get("plan") or {}
    if isinstance(plan, Mapping):
        for key in ("remaining_step", "next_step", "current_step"):
            value = plan.get(key)
            if isinstance(value, str) and value.strip():
                focused.append(f"Next required outcome: {value.strip()}")
                break
    expansion = observable.get("expansion_context") or {}
    if str(observable.get("task_kind") or "route") == "recover" and isinstance(
        expansion, Mapping
    ):
        unresolved = expansion.get("unresolved_requirement")
        if isinstance(unresolved, str) and unresolved.strip():
            focused.append(f"Still unresolved: {unresolved.strip()}")

    raw_views: list[tuple[str, float]]
    if focused:
        focus_weight = 0.65 / len(focused)
        raw_views = [(question, 0.25), (full_state, 0.10)] + [
            (value, focus_weight) for value in focused
        ]
    else:
        raw_views = [(question, 0.80), (full_state, 0.20)]

    combined: dict[str, float] = {}
    for text, weight in raw_views:
        if text:
            combined[text] = combined.get(text, 0.0) + weight
    total = sum(combined.values())
    return tuple((text, weight / total) for text, weight in combined.items())


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
