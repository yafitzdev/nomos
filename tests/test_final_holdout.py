from __future__ import annotations

import json

from fitz_tool.external_registry_fixtures import EXTERNAL_STYLE_TOOL_IDS
from fitz_tool.final_holdout_fixtures import (
    FINAL_CANONICAL_CAPABILITY_BY_TOOL_ID,
    FINAL_REGISTRY_STYLES,
    build_final_registry,
)
from fitz_tool.generic_pilot_v3 import TARGET_CAPABILITIES
from fitz_tool.promotion_holdout_fixtures import (
    PROMOTION_CANONICAL_CAPABILITY_BY_TOOL_ID,
    PROMOTION_REGISTRY_STYLES,
    build_promotion_registry,
)
from tools.evaluate_real_agent_sessions import WORKFLOWS, _run_session


class ScriptedBackend:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = list(calls)

    def complete(self, _messages: object, *, max_new_tokens: int = 128) -> dict[str, object]:
        del max_new_tokens
        return {
            "text": json.dumps(self.calls.pop(0)),
            "prompt_tokens": 10,
            "completion_tokens": 4,
        }


def test_final_registries_hide_canonical_capability_names() -> None:
    development_ids = {
        tool_id for ids in EXTERNAL_STYLE_TOOL_IDS.values() for tool_id in ids
    }
    for style in FINAL_REGISTRY_STYLES:
        registry = build_final_registry(style)
        assert not ({tool.tool_id for tool in registry.tools} & development_ids)
        assert all(
            capability not in TARGET_CAPABILITIES
            for tool in registry.tools
            for capability in tool.capabilities
        )
        mapped = {
            FINAL_CANONICAL_CAPABILITY_BY_TOOL_ID[tool.tool_id]
            for tool in registry.tools
        }
        assert mapped == set(TARGET_CAPABILITIES)


def test_promotion_registries_double_the_pool_with_unrelated_distractors() -> None:
    for style in PROMOTION_REGISTRY_STYLES:
        registry = build_promotion_registry(style)
        mapped = [
            PROMOTION_CANONICAL_CAPABILITY_BY_TOOL_ID[tool.tool_id]
            for tool in registry.tools
        ]
        assert len(registry.tools) == 34
        assert {value for value in mapped if value is not None} == set(
            TARGET_CAPABILITIES
        )
        assert sum(value is None for value in mapped) == 17
        assert all(
            capability not in TARGET_CAPABILITIES
            for tool in registry.tools
            for capability in tool.capabilities
        )


def test_repairable_schema_error_keeps_correct_tool_available() -> None:
    from fitz_tool.external_registry_fixtures import build_external_registry

    registry = build_external_registry("spectrum")
    workflow = next(item for item in WORKFLOWS if item["name"] == "source_discovery")
    backend = ScriptedBackend(
        [
            {"tool_id": "quartz_catalog", "arguments": {}},
            {"tool_id": "quartz_catalog", "arguments": {"query": "sources"}},
            {"tool_id": "orbit_index", "arguments": {"query": "fields"}},
            {"tool_id": "meadow_reader", "arguments": {"query": "resource"}},
            {"tool_id": "northstar_commit", "arguments": {"query": "finish"}},
        ]
    )

    result = _run_session(
        backend,
        None,
        workflow,
        registry,
        condition="full",
        max_attempts=2,
    )

    assert result["success"] is True
    assert result["events"][0]["repairable"] is True
    assert result["events"][0]["tool_id"] == result["events"][1]["tool_id"]


def test_raw_condition_is_one_shot_and_does_not_expose_repair_guidance() -> None:
    from fitz_tool.external_registry_fixtures import build_external_registry

    registry = build_external_registry("spectrum")
    workflow = next(item for item in WORKFLOWS if item["name"] == "source_discovery")
    backend = ScriptedBackend(
        [
            {"tool_id": "quartz_catalog", "arguments": {}},
            {"tool_id": "quartz_catalog", "arguments": {"query": "sources"}},
        ]
    )

    result = _run_session(
        backend,
        None,
        workflow,
        registry,
        condition="full_raw",
        max_attempts=99,
    )

    assert result["success"] is False
    assert len(result["events"]) == 1
    assert result["events"][0]["repair"] is None
    assert result["events"][0]["selection_correct"] is True
