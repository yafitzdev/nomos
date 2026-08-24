from __future__ import annotations

import re

from fitz_tool.agentic_transitions_v3 import generate_transition_states
from fitz_tool.external_registry_fixtures import (
    DIRECT_DESCRIPTIONS,
    EXTERNAL_REGISTRY_STYLES,
    INDIRECT_DESCRIPTIONS,
)


def test_transition_rows_are_unique_train_only_and_do_not_use_frozen_styles() -> None:
    rows, manifest = generate_transition_states(170)
    assert len(rows) == manifest["unique_matrix_cells"] == 170
    assert {row["evaluation_partition"] for row in rows} == {"train"}
    assert not any(
        style in row["tool_registry"]["registry_id"]
        for row in rows
        for style in EXTERNAL_REGISTRY_STYLES
    )


def test_transition_rows_cover_history_and_recovery_without_target_leakage() -> None:
    rows, manifest = generate_transition_states(340)
    assert set(manifest["dimension_counts"]["history_transition"]) == {
        "confusable_completed",
        "failed_prior_candidates",
        "no_history",
        "related_completed",
        "stale_prior_intent",
    }
    for row in rows:
        target_id = row["label"]["acceptable_tools"][0]
        assert target_id in row["legal_candidate_ids"]
        assert target_id not in row["previous_candidate_ids"]
        target = row["matrix_cell"]["target_capability"]
        assert target.replace("_", " ") not in row["question"].lower()


def test_transition_questions_do_not_copy_frozen_tool_descriptions() -> None:
    rows, _manifest = generate_transition_states(340)

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()

    frozen_descriptions = {
        normalize(value) for value in (*DIRECT_DESCRIPTIONS.values(), *INDIRECT_DESCRIPTIONS.values())
    }
    for row in rows:
        current_need = row["question"].split("Current need: ")[-1].split(" Use only")[0]
        assert normalize(current_need) not in frozen_descriptions
