from __future__ import annotations

from fitz_tool.agentic_pilot import (
    AGENTIC_TARGETS,
    build_agentic_registry,
    generate_agentic_states,
    load_agentic_matrix_spec,
    validate_agentic_state,
)
from fitz_tool.call_validation import validate_tool_call


def test_agentic_matrix_covers_requested_axes() -> None:
    spec = load_agentic_matrix_spec()
    assert spec["dimensions"]["candidate_pool_size"] == [10, 30, 100]
    assert spec["dimensions"]["top_k"] == [1, 3]
    assert set(spec["dimensions"]["task_kind"]) == {"route", "recover", "verify"}
    assert set(spec["dimensions"]["target_capability"]) == set(AGENTIC_TARGETS)


def test_agentic_skeletons_are_unique_and_contract_valid() -> None:
    rows, manifest = generate_agentic_states(60, seed=19)
    assert len(rows) == 60
    assert manifest["matrix_cells"] == 60
    assert manifest["type_signatures"] == 60
    assert manifest["instance_signatures"] == 60
    assert all(validate_agentic_state(row).valid for row in rows)
    assert {len(row["legal_candidate_ids"]) for row in rows} == {10, 30, 100}
    assert any(row["task_kind"] == "recover" for row in rows)
    assert any(row["task_kind"] == "verify" for row in rows)


def test_verification_rows_match_the_deterministic_call_validator() -> None:
    rows, _manifest = generate_agentic_states(60, seed=23)
    verification_rows = [row for row in rows if row["task_kind"] == "verify"]
    assert verification_rows
    for row in verification_rows:
        registry = build_agentic_registry(
            int(row["decision_state_id"].rsplit("-", 1)[-1]) % 64,
            unseen_axis=str(row["matrix_cell"]["unseen_axis"]),
        )
        result = validate_tool_call(registry, row, row["proposed_tool_call"])
        assert result.as_dict() == row["validation_label"]


def test_recovery_rows_exclude_the_failed_candidates() -> None:
    rows, _manifest = generate_agentic_states(30, seed=29)
    recovery_rows = [row for row in rows if row["task_kind"] == "recover"]
    assert recovery_rows
    for row in recovery_rows:
        prior = set(row["previous_candidate_ids"])
        acceptable = set(row["label"]["acceptable_tools"])
        assert prior
        assert not prior & acceptable
        assert row["expansion_context"]["expansion_action"] == "request_more_tool_candidates"


def test_agentic_cohort_ids_include_the_seed() -> None:
    first, _ = generate_agentic_states(1, seed=31)
    second, _ = generate_agentic_states(1, seed=32)
    assert first[0]["decision_state_id"] != second[0]["decision_state_id"]
    assert first[0]["trajectory_id"] != second[0]["trajectory_id"]
    assert first[0]["scenario_id"] != second[0]["scenario_id"]
