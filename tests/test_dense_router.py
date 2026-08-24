from __future__ import annotations

from fitz_tool.agentic_pilot import generate_agentic_states
from fitz_tool.dense_router import candidate_document, eligible_tools, query_document
from fitz_tool.tool_registry import ToolRegistry


def test_dense_text_does_not_expose_hidden_matrix_labels_or_tool_ids() -> None:
    row = generate_agentic_states(1)[0][0]
    query = query_document(row)
    assert row["matrix_cell"]["target_capability"] not in query
    assert row["legal_candidate_ids"][0] not in query

    registry = ToolRegistry.from_dict(row["tool_registry"])
    tool = registry.require(row["legal_candidate_ids"][0])
    assert tool.tool_id not in candidate_document(tool)


def test_recovery_candidates_deterministically_exclude_previous_tools() -> None:
    rows, _manifest = generate_agentic_states(2)
    recovery = rows[1]
    previous = set(recovery["previous_candidate_ids"])
    eligible = eligible_tools(recovery)
    assert eligible
    assert previous.isdisjoint(tool.tool_id for tool in eligible)
    assert set(recovery["label"]["acceptable_tools"]).issubset(
        tool.tool_id for tool in eligible
    )
