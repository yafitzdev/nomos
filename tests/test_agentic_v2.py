from __future__ import annotations

from fitz_tool.agentic_v2 import generate_agentic_v2_states


def test_agentic_v2_has_disjoint_registry_fingerprints() -> None:
    rows, manifest = generate_agentic_v2_states(220)
    fingerprints = manifest["registry_fingerprints_by_partition"]
    assert set(fingerprints["train"]).isdisjoint(fingerprints["validation"])
    assert set(fingerprints["train"]).isdisjoint(fingerprints["test"])
    assert set(fingerprints["validation"]).isdisjoint(fingerprints["test"])
    assert len(rows) == 220


def test_agentic_v2_represents_routing_abstention_recovery_and_verification() -> None:
    rows, manifest = generate_agentic_v2_states(440)
    assert set(manifest["action_counts"]) == {
        "recommend_tools",
        "abstain",
        "accept_tool_call",
        "reject_tool_call",
    }
    recovery = [row for row in rows if row["task_kind"] == "recover"]
    assert recovery
    assert all(
        set(row["previous_candidate_ids"]).isdisjoint(row["legal_candidate_ids"])
        for row in recovery
    )


def test_agentic_v2_verification_labels_match_expected_actions() -> None:
    rows, _manifest = generate_agentic_v2_states(440)
    verify = [row for row in rows if row["task_kind"] == "verify"]
    assert verify
    for row in verify:
        expected = row["matrix_cell"]["expected_action"]
        assert row["validation_label"]["valid"] is (expected == "accept_tool_call")
