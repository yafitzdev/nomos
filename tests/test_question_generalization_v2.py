from __future__ import annotations

from collections import Counter

from fitz_tool.pilot_v2 import (
    COHORT_COUNTS,
    FROZEN_PILOT_FEATURE_VERSION,
    SCALE_COHORT_COUNTS,
    generate_pilot_states,
)
from fitz_tool.question_generalization_v2 import (
    QUESTION_GENERALIZATION_HOLDOUT_TEMPLATE_IDS,
    QUESTION_GENERALIZATION_TRAIN_TEMPLATE_IDS,
    canonical_question_leakage_markers,
    generate_question_generalization_rows,
)


def _projection(row: dict) -> dict:
    query_state = dict(row["query_state"])
    query_state.pop("query_terms", None)
    return {
        key: row[key]
        for key in (
            "matrix_cell_id",
            "matrix_cell",
            "tool_registry",
            "legal_candidate_ids",
            "label",
            "accepted",
            "source_card_ids",
            "source_kind",
            "evaluation_cohort",
            "evaluation_partition",
            "sampling_context",
            "observed_evidence",
            "history",
            "plan",
            "governance",
            "resource_state",
            "source_state",
            "step",
        )
    } | {"query_state": query_state}


def test_question_generalization_preserves_frozen_structure_and_uses_disjoint_templates() -> None:
    frozen, _ = generate_pilot_states()
    derived, manifest = generate_question_generalization_rows(frozen)

    assert len(derived) == sum(COHORT_COUNTS.values()) == 5000
    assert manifest["transformed_train_rows"] == COHORT_COUNTS["train"]
    assert manifest["transformed_heldout_question_rows"] == COHORT_COUNTS["heldout_questions"]
    assert set(manifest["training_template_ids"]) == set(QUESTION_GENERALIZATION_TRAIN_TEMPLATE_IDS)
    assert set(manifest["heldout_template_ids"]) == set(QUESTION_GENERALIZATION_HOLDOUT_TEMPLATE_IDS)

    train_templates = {
        row["question_template_id"] for row in derived if row["evaluation_cohort"] == "train"
    }
    holdout_templates = {
        row["question_template_id"]
        for row in derived
        if row["evaluation_cohort"] == "heldout_questions"
    }
    assert train_templates.isdisjoint(holdout_templates)
    assert all(
        _projection(original) == _projection(candidate)
        for original, candidate in zip(frozen, derived)
    )
    assert all(
        original == candidate
        for original, candidate in zip(frozen, derived)
        if original["evaluation_cohort"] not in {"train", "heldout_questions"}
    )


def test_question_generalization_has_no_canonical_target_phrase_leaks() -> None:
    frozen, _ = generate_pilot_states()
    derived, _ = generate_question_generalization_rows(frozen)
    transformed = [
        row
        for row in derived
        if row["evaluation_cohort"] in {"train", "heldout_questions"}
    ]
    assert transformed
    assert not any(
        canonical_question_leakage_markers(
            row["question"], row["sampling_context"]["target_capability"]
        )
        for row in transformed
    )
    assert len({row["question"] for row in transformed}) > 100
    assert Counter(row["question_template_id"] for row in transformed)


def test_frozen_feature_version_and_post_gate_scale_shape_are_explicit() -> None:
    rows, _ = generate_pilot_states()
    assert rows[0]["provenance"]["feature_version"] == FROZEN_PILOT_FEATURE_VERSION
    assert sum(SCALE_COHORT_COUNTS.values()) == 30000
    assert SCALE_COHORT_COUNTS == {key: value * 6 for key, value in COHORT_COUNTS.items()}
