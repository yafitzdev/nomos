from __future__ import annotations

import json
from pathlib import Path

from tools.train_toolrex_cross_encoder import _parse_pair, load_pairs


def test_parse_pair_extracts_query_document_and_label() -> None:
    row = {
        "input": (
            "Query: Find tomorrow's weather\nTool doc: "
            '{"name":"Forecast","description":"Gets future weather."}'
        ),
        "output": "<think> </think> true",
    }

    query, document, label = _parse_pair(row) or (None, None, None)

    assert query == "Find tomorrow's weather"
    assert "Gets future weather" in str(document)
    assert label == 1.0


def test_load_pairs_reservoir_is_seeded_and_excludes_queries(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "input": f"Query: query {index}\nTool doc: document {index}",
            "output": "true" if index % 2 else "false",
        }
        for index in range(10)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    pairs, counts = load_pairs(
        path, limit=4, seed=11, excluded_queries={"query 3"}
    )

    assert len(pairs) == 4
    assert counts["source_rows"] == 10
    assert all(pair["query"] != "query 3" for pair in pairs)
