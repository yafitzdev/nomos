from __future__ import annotations

from copy import deepcopy

from fitz_tool.agentic_pilot import generate_agentic_states
from fitz_tool.coprocessor import coprocessor_response


def _request(row: dict) -> dict:
    request = deepcopy(row)
    request["schema_version"] = "runner-request.v2"
    request["request_id"] = row["decision_state_id"]
    return request


def _ranked(request: dict) -> list[dict]:
    return [
        {"tool_id": tool_id, "score": float(len(request["legal_candidate_ids"]) - index)}
        for index, tool_id in enumerate(request["legal_candidate_ids"])
    ]


def test_recovery_never_recommends_previous_candidates() -> None:
    rows, _manifest = generate_agentic_states(2)
    request = _request(rows[1])
    response = coprocessor_response(
        request,
        _ranked(request),
        router_version="test-router",
    )
    recommended = {item["tool_id"] for item in response["recommendations"]}
    assert response["operation"] == "request_more_tool_candidates"
    assert response["action"] == "recommend_tools"
    assert recommended.isdisjoint(request["previous_candidate_ids"])
    assert response["confidence"]["calibrated"] is False
    assert len(response["recommendations"]) == 3


def test_exhausted_recovery_requests_registry_expansion() -> None:
    rows, _manifest = generate_agentic_states(2)
    request = _request(rows[1])
    request["previous_candidate_ids"] = list(request["legal_candidate_ids"])
    response = coprocessor_response(
        request,
        _ranked(request),
        router_version="test-router",
    )
    assert response["action"] == "request_more_tool_candidates"
    assert response["selected_tool"] is None


def test_verification_is_deterministic_and_does_not_rank() -> None:
    rows, _manifest = generate_agentic_states(3)
    request = _request(rows[2])
    request["operation"] = "verify_tool_call"
    response = coprocessor_response(
        request,
        _ranked(request),
        router_version="test-router",
    )
    assert response["action"] in {"accept_tool_call", "reject_tool_call"}
    assert response["validation"] == request["validation_label"]
    assert response["ranked_tools"] == []
    assert response["confidence"]["method"] == "deterministic_contract_validation"


def test_schema_rejection_returns_non_executable_repair_shape() -> None:
    row = generate_agentic_states(1)[0][0]
    request = _request(row)
    request["operation"] = "verify_tool_call"
    tool_id = request["legal_candidate_ids"][0]
    request["proposed_tool_call"] = {"tool_id": tool_id, "arguments": {}}

    response = coprocessor_response(
        request,
        [],
        router_version="test-router",
    )

    assert response["action"] == "reject_tool_call"
    assert response["repair"]["strategy"] == "repair_same_tool_call"
    assert response["repair"]["tool_id"] == tool_id
    assert response["repair"]["warning"].startswith("Replace every placeholder")


def test_calibrated_low_confidence_abstains() -> None:
    row = generate_agentic_states(1)[0][0]
    request = _request(row)
    response = coprocessor_response(
        request,
        _ranked(request),
        router_version="test-router",
        calibration={
            "method": "logistic.v1",
            "intercept": -20.0,
            "coefficients": {},
            "abstention_threshold": 0.9,
        },
    )
    assert response["action"] == "abstain"
    assert response["selected_tool"] is None
    assert response["confidence"]["calibrated"] is True
