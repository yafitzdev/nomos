from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import fitz_tool.router_v2 as router_v2
from fitz_tool.adapters.fitz_sage_v2 import (
    adapt_v1_decision_state,
    load_fitz_sage_v2_registry,
)
from fitz_tool.generic_contracts import (
    validate_decision_state_v2,
    validate_runner_request_v2,
)
from fitz_tool.evaluation_v2 import evaluate_router_v2_report
from fitz_tool.matrix_v2 import (
    load_matrix_v2_spec,
    materialize_matrix_v2_cells,
    validate_matrix_v2_cell,
)
from fitz_tool.router_v2 import (
    RouterV2Config,
    featurize_v2,
    rank_tools_v2,
    train_router_v2,
)
from fitz_tool.tool_registry import RegistryValidationError, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]


def _tool(tool_id: str, capability: str, *, family: str = "retrieval") -> dict:
    return {
        "tool_id": tool_id,
        "tool_family": family,
        "description": f"Perform {capability.replace('_', ' ')} for the current objective.",
        "capabilities": [capability],
        "input_modalities": ["text"],
        "output_modalities": ["evidence_candidates"],
        "evidence_roles": ["candidate_discovery"],
        "side_effect_class": "read",
        "argument_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "constraints": ["none"],
        "prerequisites": ["query_available"],
    }


def _registry(*tools: dict, registry_id: str = "fixture_registry") -> ToolRegistry:
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": registry_id,
            "tools": list(tools),
        }
    )


def _state(
    state_id: str,
    registry: ToolRegistry,
    legal: list[str],
    positive: str,
    *,
    split: str = "train",
) -> dict:
    negatives = [tool_id for tool_id in legal if tool_id != positive]
    return {
        "schema_version": "decision-state.v2",
        "decision_state_id": state_id,
        "trajectory_id": "trajectory-" + state_id,
        "scenario_id": "scenario-" + state_id,
        "step": 0,
        "question": "Find the exact identifier in the API documentation.",
        "agent_state": {"state_name": "initial", "phase": "retrieval"},
        "history": [],
        "plan": {},
        "observed_evidence": [],
        "governance": {
            "assessment_fresh": False,
            "requirements": [],
            "allowed_side_effect_classes": ["none", "read"],
        },
        "resource_state": {
            "remaining_steps": 4,
            "unresolved_requirement_count": 1,
            "observed_evidence_count": 0,
            "distractor_count": 2,
            "prior_search_count": 1,
        },
        "source_state": {
            "available_modalities": ["text"],
            "inventory_state": "known",
            "inspection_state": "none",
        },
        "query_state": {
            "operation": "lookup_exact",
            "specificity": "identifier_exact",
            "match_strategy": "exact_identifier",
        },
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": legal,
        "label": {
            "acceptable_tools": [positive],
            "ranked_tools": [positive, *negatives],
            "hard_negative_tools": negatives,
            "label_source": "fixture-validator.v2",
        },
        "accepted": True,
        "evaluation_partition": split,
        "sampling_context": {
            "target_capability": "exact_pattern_search",
            "terminal_outcome": "ongoing",
        },
        "provenance": {
            "trajectory_hash": hashlib.sha256(state_id.encode()).hexdigest(),
            "registry_fingerprint": registry.fingerprint,
            "matrix_cell_id": hashlib.sha256((state_id + "-matrix").encode()).hexdigest(),
            "feature_version": "registry-features.v2",
            "validator_version": "fixture-validator.v2",
        },
    }


def test_fitz_sage_registry_is_valid_and_complete() -> None:
    registry = load_fitz_sage_v2_registry()
    assert len(registry.tools) == 19
    assert registry.require("search_bm25").capabilities
    assert len(registry.fingerprint) == 64


def test_registry_rejects_duplicate_tool_ids() -> None:
    with pytest.raises(RegistryValidationError, match="duplicate tool_id"):
        _registry(_tool("search", "search_content"), _tool("search", "search_metadata"))


def test_runner_contract_rejects_unknown_and_governance_disallowed_candidates() -> None:
    external = _tool("external_write", "publish_result")
    external["side_effect_class"] = "external_write"
    registry = _registry(_tool("search", "search_content"), external)
    state = _state("contract", registry, ["search"], "search")
    request = {
        **state,
        "schema_version": "runner-request.v2",
        "request_id": "request-contract",
        "legal_candidate_ids": ["external_write", "unknown"],
    }
    report = validate_runner_request_v2(request)
    messages = [issue.message for issue in report.issues]
    assert any("absent from the registry" in message for message in messages)
    assert any("disallowed by governance" in message for message in messages)


def test_decision_state_contract_accepts_generic_fixture() -> None:
    registry = _registry(
        _tool("lexical_search", "search_content"),
        _tool("exact_search", "exact_pattern_search"),
    )
    state = _state(
        "valid",
        registry,
        ["lexical_search", "exact_search"],
        "exact_search",
    )
    report = validate_decision_state_v2(state)
    assert report.valid, report.as_dict()


def test_tool_id_rename_and_candidate_order_do_not_change_features() -> None:
    first = _tool("search_alpha", "exact_pattern_search")
    renamed = {**first, "tool_id": "unseen_search_name"}
    negative = _tool("semantic_search", "search_content")
    registry_a = _registry(first, negative, registry_id="registry_a")
    registry_b = _registry(renamed, negative, registry_id="registry_b")
    state_a = _state("rename-a", registry_a, ["search_alpha", "semantic_search"], "search_alpha")
    state_b = _state(
        "rename-b",
        registry_b,
        ["unseen_search_name", "semantic_search"],
        "unseen_search_name",
    )
    spec_a = registry_a.require("search_alpha")
    spec_b = registry_b.require("unseen_search_name")
    features_a = featurize_v2(state_a, spec_a, registry_a.resolve(state_a["legal_candidate_ids"]), 256)
    features_b = featurize_v2(state_b, spec_b, registry_b.resolve(state_b["legal_candidate_ids"]), 256)
    reversed_features = featurize_v2(
        state_a,
        spec_a,
        reversed(registry_a.resolve(state_a["legal_candidate_ids"])),
        256,
    )
    assert features_a == features_b
    assert features_a == reversed_features


def test_sampling_only_oracle_fields_are_not_features() -> None:
    registry = _registry(
        _tool("lexical_search", "search_content"),
        _tool("exact_search", "exact_pattern_search"),
    )
    state = _state("sampling", registry, ["lexical_search", "exact_search"], "exact_search")
    altered = copy.deepcopy(state)
    altered["sampling_context"] = {
        "target_capability": "finalize_selection",
        "terminal_outcome": "selection",
        "future_governance_path": "insufficient_to_sufficient",
    }
    tool = registry.require("exact_search")
    legal = registry.resolve(state["legal_candidate_ids"])
    assert featurize_v2(state, tool, legal, 256) == featurize_v2(
        altered, tool, legal, 256
    )


def test_matrix_v2_materialization_is_legal_unique_and_covers_targets() -> None:
    spec = load_matrix_v2_spec()
    cells = materialize_matrix_v2_cells(250, seed=17, spec=spec)
    assert len({cell.cell_id for cell in cells}) == len(cells)
    assert all(not validate_matrix_v2_cell(cell.values, spec) for cell in cells)
    targets = {cell.values["target_capability"] for cell in cells}
    assert targets == set(spec["dimensions"]["target_capability"])


def test_v1_adapter_produces_valid_generic_state() -> None:
    v1 = json.loads((ROOT / "tests" / "fixtures" / "decision_state_smoke.json").read_text())
    generic = adapt_v1_decision_state(v1)
    report = validate_decision_state_v2(generic)
    assert report.valid, report.as_dict()
    assert generic["sampling_context"]["source_matrix_version"] == "matrix.v1"
    assert "terminal_condition" not in router_v2.observable_router_state(generic)


def test_router_v2_trains_and_never_returns_an_illegal_candidate() -> None:
    registry = _registry(
        _tool("lexical_search", "search_content"),
        _tool("exact_search", "exact_pattern_search"),
        _tool("metadata_search", "search_metadata", family="metadata_retrieval"),
    )
    states = [
        _state(
            f"train-{index}",
            registry,
            ["lexical_search", "exact_search"],
            "exact_search",
        )
        for index in range(6)
    ]
    model, metadata = train_router_v2(
        states,
        config=RouterV2Config(feature_dim=128, hidden_dim=16, epochs=3, seed=5),
    )
    ranked = rank_tools_v2(model, metadata, states[0], top_k=3)
    assert {row["tool_id"] for row in ranked} <= {"lexical_search", "exact_search"}
    assert "metadata_search" not in {row["tool_id"] for row in ranked}
    report = evaluate_router_v2_report(model, metadata, states)
    assert report["invariance"]["passed"] is True
    assert report["overall"]["invalid_candidate_rate"] == 0.0


def test_router_v2_core_has_no_fitz_sage_dependency() -> None:
    source = inspect.getsource(router_v2)
    assert "fitz_sage" not in source.casefold()
