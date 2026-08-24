from __future__ import annotations

from fitz_tool.generic_contracts import validate_runner_request_v2
from tools.evaluate_runner_contract import _build_tasks, _stage_capabilities
from tools.run_router_contract import route_request


def _source_row() -> dict:
    return {
        "question": "Which operation should locate the relevant content in the source?",
        "agent_state": {"state_name": "partial_evidence", "question_length_band": "medium"},
        "history": [],
        "observed_evidence": [],
        "governance": {
            "assessment_fresh": False,
            "requirements": [],
            "allowed_side_effect_classes": ["none", "read", "local_state_write"],
        },
        "source_state": {
            "available_modalities": ["text", "pdf"],
            "inspection_state": "partial",
            "inventory_state": "known",
        },
        "query_state": {
            "operation": "lookup_semantic",
            "match_strategy": "semantic",
            "specificity": "broad",
        },
        "resource_state": {
            "observed_evidence_count": 0,
            "remaining_steps": 4,
            "unresolved_requirement_count": 1,
        },
        "source_card_ids": ["generic_train_source_000"],
        "sampling_context": {"target_capability": "search_content"},
    }


def test_runner_contract_builds_multi_step_requests_without_oracle_fields() -> None:
    tasks = _build_tasks([_source_row()], ["spectrum"], seed=17)
    assert _stage_capabilities("search_content") == [
        "search_content",
        "inspect_evidence",
        "assess_evidence",
        "finalize_selection",
    ]
    assert len(tasks) == 1
    assert len(tasks[0]["steps"]) == 4
    for step in tasks[0]["steps"]:
        request = step["request"]
        assert validate_runner_request_v2(request).valid
        assert "sampling_context" not in request
        assert "label" not in request
        assert "matrix_cell" not in request


def test_router_contract_candidate_order_mode_returns_only_legal_tools() -> None:
    request = _build_tasks([_source_row()], ["spectrum"], seed=17)[0]["steps"][0]["request"]
    response = route_request(
        request, mode="candidate_order", model=None, metadata=None, ranker=None
    )
    assert response["schema_version"] == "router-response.v2"
    assert response["selected_tool"] in request["legal_candidate_ids"]
    assert [item["tool_id"] for item in response["ranked_tools"]] == request["legal_candidate_ids"]
