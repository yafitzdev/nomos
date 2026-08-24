from __future__ import annotations

import json

from tools.evaluate_toolret import _documentation_views, _label_ids


def test_toolret_label_ids_support_viewer_string_encoding() -> None:
    labels = [
        {"id": "relevant", "relevance": 1},
        {"id": "irrelevant", "relevance": 0},
    ]
    assert _label_ids({"labels": json.dumps(labels)}) == {"relevant"}
    assert _label_ids({"labels": labels}) == {"relevant"}


def test_documentation_views_extracts_clean_semantic_text() -> None:
    documentation = (
        '{"name":"Fetch Weather","description":"Gets a forecast.",'
        '"parameters":[]}'
    )

    assert _documentation_views(documentation) == (
        documentation,
        "Fetch Weather. Gets a forecast.",
        "Gets a forecast.",
        "Fetch Weather",
    )


def test_documentation_views_exposes_expanded_usage_profile() -> None:
    documentation = json.dumps(
        {
            "name": "Fetch Weather",
            "description": "Gets a forecast.",
            "tool_profile": {
                "function": "Retrieve future weather",
                "tags": ["weather", "forecast"],
                "when_to_use": "When a forecast is requested",
            },
        }
    )

    views = _documentation_views(documentation)

    assert any("Retrieve future weather" in view for view in views)
    assert any("When a forecast is requested" in view for view in views)
