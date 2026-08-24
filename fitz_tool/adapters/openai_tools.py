"""Project-neutral OpenAI tool-schema adapter for Nomos experiments."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ..generic_contracts import validate_runner_request_v2
from ..tool_registry import SIDE_EFFECT_CLASSES, ToolRegistry


_TOKEN_CHARS = re.compile(r"[^a-z0-9_.:-]+")


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping) and block.get("type") in {None, "text"}
        )
    return ""


def _token(value: Any, fallback: str) -> str:
    normalized = _TOKEN_CHARS.sub("_", str(value or "").strip().casefold()).strip("_.:-")
    if not normalized or not normalized[0].isalpha():
        return fallback
    return normalized[:128]


def _tokens(value: Any, fallback: str) -> list[str]:
    raw = value if isinstance(value, list) else [value] if value else []
    output = [_token(item, fallback) for item in raw]
    return list(dict.fromkeys(output)) or [fallback]


def _extension(item: Mapping[str, Any], function: Mapping[str, Any]) -> Mapping[str, Any]:
    for parent in (function, item):
        value = parent.get("x-nomos", parent.get("x_nomos"))
        if isinstance(value, Mapping):
            return value
    return {}


def _argument_schema(function: Mapping[str, Any]) -> dict[str, Any]:
    raw = function.get("parameters")
    schema = dict(raw) if isinstance(raw, Mapping) else {}
    schema["type"] = "object"
    if not isinstance(schema.get("properties"), Mapping):
        schema["properties"] = {}
    if not isinstance(schema.get("required"), list):
        schema["required"] = []
    schema["required"] = [
        str(name) for name in schema["required"] if str(name) in schema["properties"]
    ]
    return schema


def registry_from_openai_tools(
    tools: Sequence[Mapping[str, Any]], *, registry_id: str | None = None
) -> ToolRegistry:
    """Convert visible OpenAI function definitions into a generic registry.

    Rich metadata can be supplied under ``x-nomos`` on either the tool or its
    ``function`` object. Plain OpenAI schemas remain usable, but necessarily
    provide weaker candidate metadata.
    """

    rows: list[dict[str, Any]] = []
    for item in tools:
        if item.get("type", "function") != "function":
            continue
        function = item.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        metadata = _extension(item, function)
        side_effect = str(metadata.get("side_effect_class", "none"))
        if side_effect not in SIDE_EFFECT_CLASSES:
            side_effect = "none"
        description = str(function.get("description") or "").strip()
        if len(description) < 12:
            description = "OpenAI function described by its supplied argument schema."
        rows.append(
            {
                "tool_id": name,
                "tool_family": _token(metadata.get("tool_family"), "openai_function"),
                "description": description,
                "capabilities": _tokens(metadata.get("capabilities"), "generic_operation"),
                "input_modalities": _tokens(metadata.get("input_modalities"), "text"),
                "output_modalities": _tokens(metadata.get("output_modalities"), "text"),
                "evidence_roles": _tokens(metadata.get("evidence_roles"), "operation"),
                "side_effect_class": side_effect,
                "argument_schema": _argument_schema(function),
                "constraints": _tokens(metadata.get("constraints"), "none"),
                "prerequisites": _tokens(metadata.get("prerequisites"), "none"),
            }
        )
    if not rows:
        raise ValueError("OpenAI request does not contain function tools")
    if registry_id is None:
        digest = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        registry_id = f"openai_registry_{digest}"
    return ToolRegistry.from_dict(
        {"schema_version": "tool-registry.v2", "registry_id": registry_id, "tools": rows}
    )


def _tool_name(item: Mapping[str, Any]) -> str | None:
    function = item.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return str(function["name"])
    return None


def build_runner_request_from_openai(
    body: Mapping[str, Any], *, request_id: str | None = None
) -> dict[str, Any]:
    """Build an observable runner-request.v2 from a generic OpenAI request."""

    raw_tools = body.get("tools")
    tools = [item for item in raw_tools if isinstance(item, Mapping)] if isinstance(raw_tools, list) else []
    registry = registry_from_openai_tools(tools)
    legal_ids = [name for item in tools for name in [_tool_name(item)] if name]

    raw_messages = body.get("messages")
    messages = [item for item in raw_messages if isinstance(item, Mapping)] if isinstance(raw_messages, list) else []
    user_messages = [
        _message_text(message).strip()
        for message in messages
        if message.get("role") == "user" and _message_text(message).strip()
    ]
    question = user_messages[-1] if user_messages else "Select the next useful tool."
    history: list[dict[str, Any]] = []
    prior_tools: list[str] = []
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(calls, list):
            for call in calls:
                if isinstance(call, Mapping):
                    name = _tool_name(call)
                    if name:
                        prior_tools.append(name)
                        history.append(
                            {"action_family": "tool_use", "tool": name, "status": "requested"}
                        )
        elif message.get("role") == "tool":
            history.append(
                {
                    "action_family": "tool_result",
                    "tool": str(message.get("name") or "unknown"),
                    "status": "observed",
                }
            )

    extension = body.get("x-nomos", body.get("x_nomos"))
    extension = dict(extension) if isinstance(extension, Mapping) else {}
    state = extension.get("state")
    state = dict(state) if isinstance(state, Mapping) else {}
    allowed_side_effects = sorted({tool.side_effect_class for tool in registry.tools})
    source_modalities = sorted(
        {modality for tool in registry.tools for modality in tool.input_modalities}
    )
    payload_hash = hashlib.sha256(
        json.dumps(dict(body), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    request = {
        "schema_version": "runner-request.v2",
        "request_id": request_id or f"openai-{payload_hash}",
        "question": question,
        "task_kind": str(state.get("task_kind") or "route"),
        "agent_state": {
            "phase": str(state.get("phase") or ("execution" if prior_tools else "planning")),
            "state_name": str(state.get("state_name") or ("active" if prior_tools else "initial")),
        },
        "history": history,
        "plan": dict(state.get("plan")) if isinstance(state.get("plan"), Mapping) else {},
        "observed_evidence": (
            list(state.get("observed_evidence"))
            if isinstance(state.get("observed_evidence"), list)
            else []
        ),
        "governance": {
            "allowed_side_effect_classes": allowed_side_effects,
            **(
                dict(state.get("governance"))
                if isinstance(state.get("governance"), Mapping)
                else {}
            ),
        },
        "resource_state": {
            "prior_tool_count": len(prior_tools),
            **(
                dict(state.get("resource_state"))
                if isinstance(state.get("resource_state"), Mapping)
                else {}
            ),
        },
        "source_state": {
            "available_modalities": source_modalities,
            **(
                dict(state.get("source_state"))
                if isinstance(state.get("source_state"), Mapping)
                else {}
            ),
        },
        "query_state": (
            dict(state.get("query_state"))
            if isinstance(state.get("query_state"), Mapping)
            else {}
        ),
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": list(dict.fromkeys(legal_ids)),
    }
    if request["task_kind"] == "recover":
        request["previous_candidate_ids"] = list(state.get("previous_candidate_ids") or [])
        request["expansion_context"] = dict(state.get("expansion_context") or {})
    report = validate_runner_request_v2(request)
    if not report.valid:
        raise ValueError(json.dumps(report.as_dict(), sort_keys=True))
    return request


def retain_openai_tools(body: Mapping[str, Any], tool_ids: Sequence[str]) -> dict[str, Any]:
    """Return a request copy containing only the selected function definitions."""

    selected = set(tool_ids)
    output = json.loads(json.dumps(dict(body), ensure_ascii=False))
    raw_tools = output.get("tools")
    if isinstance(raw_tools, list):
        output["tools"] = [
            item
            for item in raw_tools
            if isinstance(item, Mapping) and _tool_name(item) in selected
        ]
    return output


def selected_tool_from_response(payload: Mapping[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], Mapping):
        return None
    return _tool_name(calls[0])
