from __future__ import annotations

import json

from tools.generate_toolret_agentic_v1 import _materialize, _skeleton


def _doc(name: str, description: str) -> str:
    return json.dumps(
        {
            "name": name,
            "description": description,
            "parameters": {
                "query": {"type": "str", "description": "The requested expression."}
            },
        }
    )


def test_toolret_state_uses_opaque_ids_and_valid_contract() -> None:
    source = {
        "id": "train_test",
        "query": "Find the current weather for Berlin.",
        "prompt": "Retrieve a weather lookup operation.",
        "positive": [_doc("weather_lookup", "Returns current weather for a location.")],
        "negative": [
            _doc("weather_history", "Returns historical weather for a location."),
            _doc("city_lookup", "Returns descriptive information about a city."),
            _doc("timezone_lookup", "Returns the timezone for a location."),
        ],
    }
    skeleton = _skeleton(source, 1, 7)
    row = _materialize(
        skeleton,
        {
            "question": "What operation can give me the current weather conditions in Berlin?",
            "current_step": "Retrieve live weather conditions for the specified city.",
            "completed_steps": ["Confirmed that the location is Berlin."],
        },
        index=1,
        seed=7,
        model="deepseek-v4-flash",
    )
    assert row["evaluation_partition"] == "train"
    assert row["label"]["acceptable_tools"][0].startswith("op_")
    assert "weather_lookup" not in row["label"]["acceptable_tools"][0]
