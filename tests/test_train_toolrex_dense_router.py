from __future__ import annotations

import json
from pathlib import Path

from tools.train_toolrex_dense_router import _clean_document, load_training_rows


def test_load_training_rows_is_seeded_and_supports_exclusions(tmp_path: Path) -> None:
    source = tmp_path / "toolrex.json"
    payload = [
        {
            "query": f"query {index}",
            "response": {"description": f"positive {index}"},
            "rejected_response": [f"negative {index}a", f"negative {index}b"],
        }
        for index in range(5)
    ]
    source.write_text(json.dumps(payload), encoding="utf-8")

    rows, counts = load_training_rows(
        source,
        limit=3,
        seed=7,
        negative_count=2,
        excluded_queries={"query 4"},
    )

    assert len(rows) == 3
    assert counts["source_rows"] == 5
    assert all(row["anchor"] != "query 4" for row in rows)
    assert set(rows[0]) == {"anchor", "positive", "negative_1", "negative_2"}


def test_clean_document_keeps_semantics_and_drops_large_response_templates() -> None:
    document = json.dumps(
        {
            "name": "Weather Forecast",
            "description": "Retrieve a city forecast.",
            "template_response": {"hourly": ["unused"] * 100},
            "tool_profile": {
                "function": "Gets weather forecasts",
                "tags": ["weather", "forecast"],
                "when_to_use": "When future weather is requested",
            },
            "parameters": {"city": {"description": "City name"}},
        }
    )

    cleaned = _clean_document(document)

    assert "Weather Forecast" in cleaned
    assert "Gets weather forecasts" in cleaned
    assert "City name" in cleaned
    assert "template_response" not in cleaned
