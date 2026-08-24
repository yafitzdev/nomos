from __future__ import annotations

import pytest

from fitz_tool.scaling_salvage import canonical_scaling_target, select_balanced_hard_rows
from tools.audit_scaling_salvage import _chunks


def test_balanced_salvage_prefers_mistakes_then_low_margin() -> None:
    scores = [
        {"decision_state_id": "a-easy", "target_capability": "a", "target_rank": 1, "margin": 0.5},
        {"decision_state_id": "a-hard", "target_capability": "a", "target_rank": 1, "margin": 0.01},
        {"decision_state_id": "a-wrong", "target_capability": "a", "target_rank": 3, "margin": -0.2},
        {"decision_state_id": "b-hard", "target_capability": "b", "target_rank": 1, "margin": 0.02},
        {"decision_state_id": "abstain", "target_capability": None, "target_rank": 1, "margin": 0.0},
    ]

    selected, manifest = select_balanced_hard_rows(scores, max_per_capability=2)

    assert selected == {"a-wrong", "a-hard", "b-hard"}
    assert manifest["selected_counts"] == {"a": 2, "b": 1}
    assert manifest["selected_mistake_counts"] == {"a": 1}


def test_balanced_salvage_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_balanced_hard_rows([], max_per_capability=0)


def test_chunks_streams_without_dropping_tail() -> None:
    assert list(_chunks(({"value": index} for index in range(5)), 2)) == [
        [{"value": 0}, {"value": 1}],
        [{"value": 2}, {"value": 3}],
        [{"value": 4}],
    ]
    with pytest.raises(ValueError, match="positive"):
        list(_chunks([], 0))


def test_canonical_target_uses_scenario_not_assignment_identity() -> None:
    row = {
        "decision_state_id": "replacement-id-not-in-original-matrix",
        "matrix_cell": {"scenario_family": "schema_vs_records"},
    }

    assert canonical_scaling_target(row) == "inspect_structured_schema"
    assert canonical_scaling_target(
        {"matrix_cell": {"scenario_family": "no_suitable_tool"}}
    ) is None
