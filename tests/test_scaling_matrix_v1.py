from __future__ import annotations

from collections import Counter

import pytest

from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.post_scaling_holdout_v1 import (
    HOLDOUT_POOL_SIZES,
    HOLDOUT_SCHEMA_STYLES,
    HOLDOUT_STYLES,
    generate_post_scaling_holdout,
)
from fitz_tool.scaling_matrix_v1 import (
    PROJECT_MARKER,
    build_registry,
    load_scaling_matrix,
    materialize_assignments,
    materialize_row,
    replacement_assignment,
)


def test_scaling_matrix_materializes_exact_fixed_distribution() -> None:
    spec = load_scaling_matrix()
    assignments = materialize_assignments(spec)

    assert len(assignments) == 25_000
    assert len({value["assignment_id"] for value in assignments}) == 25_000
    assert len({value["matrix_cell_id"] for value in assignments}) == 25_000
    assert Counter(value["matrix_cell"]["scenario_family"] for value in assignments) == Counter(
        spec["scenario_family_counts"]
    )
    assert Counter(value["matrix_cell"]["candidate_pool_size"] for value in assignments) == Counter(
        {int(key): value for key, value in spec["dimension_counts"]["candidate_pool_size"].items()}
    )
    required_modalities = {
        "code_symbol_vs_full_read": "code",
        "document_structure_vs_pages": "document",
        "schema_vs_records": "structured_data",
        "conflicting_irrelevant_history": "structured_data",
    }
    assert all(
        value["matrix_cell"]["source_modality"]
        == required_modalities[value["matrix_cell"]["scenario_family"]]
        for value in assignments
        if value["matrix_cell"]["scenario_family"] in required_modalities
    )


def test_registry_obeys_pool_target_position_and_recovery_no_repeat() -> None:
    assignments = materialize_assignments()
    samples = [
        next(
            value
            for value in assignments
            if value["matrix_cell"]["candidate_pool_size"] == size
            and value["target_capability"]
        )
        for size in (10, 17, 34, 50, 100)
    ]
    for assignment in samples:
        _registry, legal, target, _negatives, previous = build_registry(assignment)
        assert len(legal) == assignment["matrix_cell"]["candidate_pool_size"]
        assert target in legal
        assert set(previous).isdisjoint(legal)


def test_replacement_assignment_preserves_slot_but_changes_identity() -> None:
    base = materialize_assignments()[0]
    replacement = replacement_assignment(base, 1)

    assert replacement["slot_id"] == base["slot_id"]
    assert replacement["assignment_id"] != base["assignment_id"]
    assert replacement["matrix_cell_id"] != base["matrix_cell_id"]
    assert replacement["matrix_cell"]["scenario_family"] == base["matrix_cell"]["scenario_family"]


def test_materialized_row_is_valid_and_has_no_teacher_fallback() -> None:
    assignment = next(value for value in materialize_assignments() if value["matrix_cell"]["history_length"] == "short" and value["target_capability"])
    row = materialize_row(
        assignment,
        {
            "assignment_id": assignment["assignment_id"],
            "question": "Please inspect the currently selected evidence source for the exact unresolved requirement.",
            "current_step": "Choose the bounded operation that directly satisfies the active evidence need.",
            "completed_steps": ["Identified the stable handle for the relevant input."],
        },
        model="deepseek-v4-flash",
        generated_at="2026-08-24T00:00:00+00:00",
        retry_history=[],
    )

    assert validate_decision_state_v2(row).valid
    assert row["provenance"]["teacher_fallback_used"] is False
    assert not PROJECT_MARKER.search(row["question"])


def test_materialized_row_rejects_wrong_history_count() -> None:
    assignment = next(value for value in materialize_assignments() if value["matrix_cell"]["history_length"] == "long")
    with pytest.raises(ValueError, match="completed_steps count"):
        materialize_row(
            assignment,
            {
                "assignment_id": assignment["assignment_id"],
                "question": "Choose the next safe operation for this still unresolved request.",
                "current_step": "Advance the active requirement without repeating a completed action.",
                "completed_steps": [],
            },
            model="deepseek-v4-flash",
            generated_at="2026-08-24T00:00:00+00:00",
            retry_history=[],
        )


def test_post_scaling_holdout_is_frozen_unseen_and_contract_valid() -> None:
    rows, manifest = generate_post_scaling_holdout()

    assert len(rows) == manifest["count"] == 756
    assert manifest["answer_present_count"] == 720
    assert manifest["abstention_count"] == 36
    assert {row["matrix_cell"]["candidate_pool_size"] for row in rows} == set(HOLDOUT_POOL_SIZES)
    assert {row["matrix_cell"]["registry_description_style"] for row in rows} == set(HOLDOUT_STYLES)
    assert {row["matrix_cell"]["argument_schema_style"] for row in rows} == set(HOLDOUT_SCHEMA_STYLES)
    assert all(row["question_template_id"].startswith("postscale-holdout-v1") for row in rows)
    assert all(validate_decision_state_v2(row).valid for row in rows)
    assert all(
        tool["tool_family"].startswith("hold_")
        for row in rows
        for tool in row["tool_registry"]["tools"]
    )
    assert all(set(row["previous_candidate_ids"]).isdisjoint(row["legal_candidate_ids"]) for row in rows)
