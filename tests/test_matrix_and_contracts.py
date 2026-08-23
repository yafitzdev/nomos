from __future__ import annotations

import hashlib

from fitz_tool.audit import select_stratified_sample
from fitz_tool.contracts import validate_scenario, validate_source_card
from fitz_tool.decision_states import extract_decision_states
from fitz_tool.matrix import materialize_cells, validate_matrix_cell
from fitz_tool.router import RouterConfig, rank_tools, train_router
from fitz_tool.uniqueness import annotate_signatures


def _source_card() -> dict:
    return {
        "schema_version": "source-card.v1",
        "source_id": "fixture-payments-migration",
        "document_id": "payments-migration-guide",
        "title": "Payments API migration guide",
        "modality": "text",
        "content_sha256": hashlib.sha256(b"fixture-payments-migration").hexdigest(),
        "facts": [
            {"fact_id": "oauth-expiry", "statement": "OAuth access tokens expire after 45 minutes."},
            {"fact_id": "refresh-rotation", "statement": "Refresh tokens are single-use and rotate after successful refresh."},
            {"fact_id": "auth-409", "statement": "AUTH-409 means a previously consumed refresh token was reused."},
        ],
    }


def _scenario(cell: dict) -> dict:
    scenario = {
        "schema_version": "scenario.v1",
        "scenario_id": "fixture-0001",
        "matrix_version": "matrix.v1",
        "matrix_cell": cell,
        "source_card_ids": ["fixture-payments-migration"],
        "state_setup": {
            "state_name": cell["agent_state"],
            "history": [],
            "observed_evidence": [],
            "requirements": [{"requirement_id": "R1", "status": "missing"}],
            "governance": {"assessment_fresh": False, "path": cell["governance_path"]},
        },
        "question": "Which evidence explains the refresh-token failure and what should happen next?",
        "difficult_paraphrase": "Why does the token refresh fail after an earlier successful refresh, and what response is required?",
        "expected_facts": [
            {"source_id": "fixture-payments-migration", "fact_id": "refresh-rotation"},
            {"source_id": "fixture-payments-migration", "fact_id": "auth-409"},
        ],
        "expected_tools": [cell["next_tool_target"]],
        "expected_terminal_state": cell["terminal_condition"],
        "provenance": {
            "teacher": "fixture",
            "model": "fixture",
            "prompt_version": "fixture",
            "seed": 1,
            "source_card_hashes": [_source_card()["content_sha256"]],
            "generated_at": "2026-08-23T00:00:00+00:00",
        },
    }
    return annotate_signatures(scenario)


def test_materialized_cells_are_legal_and_unique() -> None:
    cells = materialize_cells(100, seed=7)
    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert all(not validate_matrix_cell(cell.values) for cell in cells)


def test_materialized_cells_cover_reachable_tool_targets() -> None:
    cells = materialize_cells(1000, seed=19)
    targets = {cell.values["next_tool_target"] for cell in cells}
    assert targets == {
        "set_retrieval_plan",
        "search_bm25",
        "grep_search",
        "search_metadata",
        "list_sources",
        "list_tabular_sources",
        "inspect_table_schema",
        "search_table_rows",
        "list_pdf_sources",
        "inspect_pdf_structure",
        "search_pdf_pages",
        "read_file",
        "inspect_code",
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "update_requirement_progress",
        "assess_evidence",
        "finalize_document_selection",
    }


def test_materialized_cells_reserve_prior_slice_cells() -> None:
    first = materialize_cells(40, seed=7)
    second = materialize_cells(
        40,
        seed=7,
        excluded_cell_ids={cell.cell_id for cell in first},
    )
    assert len({cell.cell_id for cell in first} & {cell.cell_id for cell in second}) == 0


def test_matrix_rejects_modality_tool_mismatch() -> None:
    cell = {
        "integration_domain": "payments",
        "information_operation": "lookup",
        "source_modality": "text",
        "evidence_topology": "one_passage",
        "retrieval_obstacle": "none",
        "agent_state": "initial",
        "next_tool_target": "search_pdf_pages",
        "governance_path": "stale_assessment",
        "terminal_condition": "selection",
        "resource_pressure_band": "low",
    }
    assert any("requires pdf" in error for error in validate_matrix_cell(cell))


def test_source_card_and_scenario_validate() -> None:
    source_report = validate_source_card(_source_card())
    assert source_report.valid, source_report.as_dict()
    cell = materialize_cells(1, seed=11)[0].as_dict()
    scenario_report = validate_scenario(_scenario(cell))
    assert scenario_report.valid, scenario_report.as_dict()


def test_absent_evidence_cannot_claim_positive_facts() -> None:
    cell = next(
        candidate.as_dict()
        for candidate in materialize_cells(100, seed=12)
        if candidate.values["evidence_topology"] == "absent"
    )
    report = validate_scenario(_scenario(cell))
    assert any("absent evidence cells" in issue.message for issue in report.issues)


def test_audit_sample_is_reproducible_and_bounded() -> None:
    rows = [{"matrix_cell": {"integration_domain": str(i % 2), "source_modality": str(i % 3)}} for i in range(20)]
    first = select_stratified_sample(rows, 7, seed=3)
    second = select_stratified_sample(rows, 7, seed=3)
    assert first == second
    assert len(first) == 7
    assert len(set(first)) == 7


def test_trajectory_decision_event_extracts_positive_router_row() -> None:
    trajectory = {
        "schema_version": "trajectory.v1",
        "trajectory_id": "trajectory-fixture-1",
        "scenario_id": "scenario-fixture-1",
        "runner": {"name": "fixture-runner", "version": "1", "contract_version": "runner.v1"},
        "events": [
            {
                "step": 0,
                "kind": "decision",
                "agent_state": {"state_name": "initial"},
                "legal_tools": ["search_bm25", "grep_search"],
                "observed_evidence": [],
                "governance": {"assessment_fresh": False, "requirements": []},
                "proposed_tool": "search_bm25",
                "executed_tool": "search_bm25",
                "acceptable_tools": ["search_bm25"],
                "ranked_tools": ["search_bm25", "grep_search"],
                "hard_negative_tools": ["grep_search"],
            }
        ],
        "terminal_result": {"status": "ongoing"},
        "validation": {"trajectory_accepted": True, "rejection_reasons": []},
        "provenance": {"captured_at": "2026-08-23T00:00:00+00:00"},
    }
    states = extract_decision_states(trajectory, question="How should the API be searched?")
    assert len(states) == 1
    assert states[0]["accepted"] is True
    assert states[0]["label"]["acceptable_tools"] == ["search_bm25"]
    assert states[0]["label"]["hard_negative_tools"] == ["grep_search"]


def test_router_trains_and_ranks_only_legal_tools() -> None:
    def state(state_id: str, trajectory_id: str, accepted: bool) -> dict:
        return {
            "schema_version": "decision-state.v1",
            "decision_state_id": state_id,
            "trajectory_id": trajectory_id,
            "scenario_id": "scenario-" + trajectory_id,
            "step": 0,
            "question": "Which retrieval action should happen next?",
            "agent_state": {"state_name": "initial"},
            "legal_tools": ["search_bm25", "grep_search"],
            "observed_evidence": [],
            "governance": {"assessment_fresh": False, "requirements": []},
            "label": {
                "acceptable_tools": ["search_bm25"] if accepted else [],
                "ranked_tools": ["search_bm25"] if accepted else ["grep_search"],
                "hard_negative_tools": ["grep_search"] if accepted else ["grep_search"],
                "label_source": "deterministic_execution",
            },
            "accepted": accepted,
            "provenance": {"trajectory_hash": "a" * 64, "validator_version": "decision-labels.v1"},
        }

    states = [state("state-1", "trajectory-1", True), state("state-2", "trajectory-2", False)]
    model, metadata = train_router(
        states,
        config=RouterConfig(feature_dim=64, hidden_dim=16, epochs=2, seed=3),
    )
    ranked = rank_tools(model, metadata, states[0], top_k=2)
    assert len(ranked) == 2
    assert {item["tool"] for item in ranked} == {"search_bm25", "grep_search"}
    assert metadata["metrics"]["train"]["states"] >= 1
