from __future__ import annotations

from collections import Counter

from fitz_tool.agentic_contrasts_v4 import generate_contrast_states
from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.generic_pilot_v3 import TARGET_CAPABILITIES


def test_contrast_rows_are_balanced_opaque_and_valid() -> None:
    rows, manifest = generate_contrast_states(34, seed=41)

    targets = Counter(row["matrix_cell"]["target_capability"] for row in rows)
    assert set(targets) == set(TARGET_CAPABILITIES)
    assert set(targets.values()) == {2}
    assert manifest["count"] == 34
    assert all(validate_decision_state_v2(row).valid for row in rows)
    for row in rows:
        target = row["matrix_cell"]["target_capability"]
        assert target.replace("_", " ") not in row["question"].lower()
        assert all(
            target not in tool["capabilities"]
            for tool in row["tool_registry"]["tools"]
        )
        assert row["label"]["acceptable_tools"][0] in row["legal_candidate_ids"]
        assert row["label"]["hard_negative_tools"][0] in row["legal_candidate_ids"]


def test_failed_distractor_rows_exclude_prior_candidates() -> None:
    rows, _manifest = generate_contrast_states(136, seed=43)
    recovery = [row for row in rows if row["task_kind"] == "recover"]

    assert recovery
    for row in recovery:
        assert set(row["previous_candidate_ids"]).isdisjoint(
            row["legal_candidate_ids"]
        )
