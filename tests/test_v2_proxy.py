from __future__ import annotations

import json

from fitz_tool.adapters.fitz_sage_v2 import (
    _state_after_action,
    build_runner_request_from_openai,
)
from fitz_tool.generic_contracts import validate_runner_request_v2
from tools.nomos_openai_proxy import (
    _candidate_compatible_with_source,
    parse_tool_text,
    repair_completion_payload,
)


def _tool(name: str) -> dict:
    properties = {}
    required = []
    if name == "search_bm25":
        properties = {"query": {"type": "string"}}
        required = ["query"]
    elif name == "grep_search":
        properties = {"pattern": {"type": "string"}}
        required = ["pattern"]
    elif name == "list_tabular_sources":
        properties = {"scope": {"type": "string"}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call {name} for the current research objective.",
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _body() -> dict:
    return {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [
            {"role": "system", "content": "Use one tool."},
            {"role": "user", "content": "Find the exact AUTH-409 identifier in the API documentation."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "set_retrieval_plan",
                            "arguments": json.dumps({"objective": "Find AUTH-409"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "set_retrieval_plan",
                "content": json.dumps({"requirements": [{"requirement_id": "R1"}]}),
            },
        ],
        "tools": [_tool("search_bm25"), _tool("grep_search")],
        "tool_choice": "required",
        "max_tokens": 192,
    }


def test_v2_openai_adapter_emits_observable_runner_request() -> None:
    request = build_runner_request_from_openai(_body(), source_modality="text")
    report = validate_runner_request_v2(request)
    assert report.valid, report.as_dict()
    assert request["legal_candidate_ids"] == ["search_bm25", "grep_search"]
    assert "label" not in request
    assert "sampling_context" not in request
    assert "matrix_context" not in request
    assert request["history"][0]["tool"] == "set_retrieval_plan"
    assert request["source_state"]["available_modalities"] == ["metadata", "text"]


def test_proxy_parses_complete_qwen_xml_and_repairs_openai_payload() -> None:
    text = (
        "<tool_call>\n<function=grep_search>\n"
        "<parameter=pattern>\"AUTH-409\"</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert parse_tool_text(text, ["grep_search"]) == (
        "grep_search",
        {"pattern": "AUTH-409"},
    )
    repaired, changed = repair_completion_payload(
        {"choices": [{"message": {"role": "assistant", "content": text}}]},
        allowed_names=["grep_search"],
    )
    assert changed is True
    message = repaired["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "grep_search"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "pattern": "AUTH-409"
    }


def test_proxy_enforces_the_visible_legal_tool_when_backend_ignores_tool_choice() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "bad-call",
                            "type": "function",
                            "function": {
                                "name": "search_bm25",
                                "arguments": json.dumps({"query": "AUTH-409"}),
                            },
                        }
                    ],
                },
            }
        ]
    }
    repaired, changed = repair_completion_payload(
        payload,
        allowed_names=["list_tabular_sources"],
        preferred_name="list_tabular_sources",
        request_body={**_body(), "tools": [_tool("list_tabular_sources")]},
    )
    assert changed is True
    call = repaired["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "list_tabular_sources"
    assert json.loads(call["function"]["arguments"]) == {"scope": "AUTH-409"}


def test_progress_update_keeps_router_in_evidence_phase() -> None:
    assert _state_after_action("update_requirement_progress", {}) == "partial_evidence"
    assert _state_after_action("assess_evidence", {}) == "fresh_sufficient"


def test_source_discovery_remains_legal_when_named_source_is_unresolved() -> None:
    body = _body()
    body["tools"] = [_tool("grep_search"), _tool("list_sources")]
    request = build_runner_request_from_openai(body, source_modality="text")
    request["observed_evidence"] = [{"evidence_id": "E1", "source_id": "external"}]
    assert _candidate_compatible_with_source(request, "list_sources", "text") is True
