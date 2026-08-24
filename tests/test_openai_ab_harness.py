from __future__ import annotations

import pytest

from fitz_tool.adapters.openai_tools import (
    build_runner_request_from_openai,
    registry_from_openai_tools,
    retain_openai_tools,
    selected_tool_from_response,
)
from fitz_tool.generic_contracts import validate_runner_request_v2
from tools.benchmark_nomos_openai_ab import summarize


def _tool(name: str, description: str, capability: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "x-nomos": {
                "tool_family": "research",
                "capabilities": [capability],
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "evidence_roles": ["observation"],
                "side_effect_class": "read",
            },
        },
    }


def _body() -> dict:
    return {
        "model": "agent-model",
        "messages": [
            {"role": "user", "content": "Find the implementation of this SDK symbol."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_docs", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": "search_docs", "content": "no exact match"},
        ],
        "tools": [
            _tool("search_docs", "Search documentation for relevant passages.", "semantic_search"),
            _tool("inspect_code", "Inspect source symbols and implementation structure.", "code_inspection"),
            _tool("browse_web", "Search public web pages for current information.", "web_search"),
        ],
    }


def test_generic_openai_adapter_builds_valid_identity_free_request() -> None:
    body = _body()
    registry = registry_from_openai_tools(body["tools"])
    request = build_runner_request_from_openai(body, request_id="case-1")

    assert [tool.tool_id for tool in registry.tools] == [
        "search_docs",
        "inspect_code",
        "browse_web",
    ]
    assert registry.require("inspect_code").capabilities == ("code_inspection",)
    assert request["question"] == "Find the implementation of this SDK symbol."
    assert request["legal_candidate_ids"] == ["search_docs", "inspect_code", "browse_web"]
    assert request["history"][0]["tool"] == "search_docs"
    assert validate_runner_request_v2(request).valid


def test_retain_tools_and_parse_selected_tool() -> None:
    reduced = retain_openai_tools(_body(), ["inspect_code", "browse_web"])
    names = [item["function"]["name"] for item in reduced["tools"]]
    assert names == ["inspect_code", "browse_web"]
    assert len(_body()["tools"]) == 3
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"type": "function", "function": {"name": "inspect_code"}}
                    ]
                }
            }
        ]
    }
    assert selected_tool_from_response(response) == "inspect_code"


def test_summary_reports_token_cost_and_quality_guardrails() -> None:
    rows = [
        {
            "acceptable_tools": ["inspect_code"],
            "ranking_seconds": 0.125,
            "ranked_tool_ids": ["inspect_code", "search_docs", "browse_web"],
            "conditions": {
                "full": {
                    "visible_tools": 10,
                    "request_bytes": 1000,
                    "tool_schema_bytes": 800,
                    "estimated_request_tokens": 500,
                    "estimated_tool_schema_tokens": 400,
                    "prompt_tokens": 600,
                    "completion_tokens": 100,
                    "selected_tool": "inspect_code",
                },
                "nomos": {
                    "visible_tools": 3,
                    "request_bytes": 400,
                    "tool_schema_bytes": 200,
                    "estimated_request_tokens": 250,
                    "estimated_tool_schema_tokens": 100,
                    "prompt_tokens": 300,
                    "completion_tokens": 80,
                    "selected_tool": "search_docs",
                },
            },
        }
    ]
    report = summarize(
        rows,
        top_k=3,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    assert report["nomos_top_k_recall"] == 1.0
    assert report["request_byte_reduction"] == pytest.approx(0.6)
    assert report["tool_schema_byte_reduction"] == pytest.approx(0.75)
    assert report["estimated_request_token_reduction"] == pytest.approx(0.5)
    assert report["estimated_tool_schema_token_reduction"] == pytest.approx(0.75)
    assert report["provider_prompt_token_reduction"] == pytest.approx(0.5)
    assert report["estimated_cost_reduction"] == pytest.approx(0.425)
    assert report["nomos_ranking_p50_seconds"] == pytest.approx(0.125)
    assert report["nomos_ranking_p95_seconds"] == pytest.approx(0.125)
    assert report["full_selected_tool_accuracy"] == 1.0
    assert report["nomos_selected_tool_accuracy"] == 0.0
