from __future__ import annotations

import json

import pytest

from fitz_tool.scaling_matrix_v1 import materialize_assignments
from tools import generate_scaling_deepseek_v1 as generator


def _assignment(index: int) -> dict[str, object]:
    return {
        "assignment_id": f"assignment-{index:02d}",
        "slot_id": f"slot-{index:02d}",
        "replacement_ordinal": 0,
    }


def _item(assignment: dict[str, object]) -> dict[str, object]:
    return {
        "assignment_id": assignment["assignment_id"],
        "question": "Choose the operation that handles this concrete unresolved request safely.",
        "current_step": "Advance the immediate outcome without repeating an earlier candidate.",
        "completed_steps": [],
    }


def test_persistent_batch_failure_splits_sixteen_to_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    assignments = [_assignment(index) for index in range(16)]

    def fake_request(batch, **_kwargs):
        if len(batch) == 16:
            raise generator.RequestFailure("bad JSON", "malformed_json")
        return [_item(value) for value in batch], {
            "finish_reason": "stop",
            "returned_items": len(batch),
            "usage": {},
        }

    monkeypatch.setattr(generator, "_request_once", fake_request)
    successes, failures, events = generator._process_batch(
        assignments,
        api_key="not-logged",
        model="deepseek-v4-flash",
        timeout=180,
        max_tokens=8000,
        same_batch_attempts=2,
    )

    assert len(successes) == 16
    assert not failures
    assert [event["child_sizes"] for event in events if event["event"] == "split"] == [[8, 8]]
    assert sum(event["event"] == "request" for event in events) == 4
    assert "not-logged" not in json.dumps(events)


def test_eight_row_failure_splits_to_four(monkeypatch: pytest.MonkeyPatch) -> None:
    assignments = [_assignment(index) for index in range(8)]

    def fake_request(batch, **_kwargs):
        if len(batch) > 4:
            raise generator.RequestFailure("truncated", "finish_reason")
        return [_item(value) for value in batch], {
            "finish_reason": "stop",
            "returned_items": len(batch),
            "usage": {},
        }

    monkeypatch.setattr(generator, "_request_once", fake_request)
    successes, failures, events = generator._process_batch(
        assignments,
        api_key="temporary",
        model="deepseek-v4-flash",
        timeout=180,
        max_tokens=8000,
        same_batch_attempts=2,
    )

    assert len(successes) == 8
    assert not failures
    assert [event["child_sizes"] for event in events if event["event"] == "split"] == [[4, 4]]


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_request_rejects_duplicate_assignment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    assignments = materialize_assignments()[:2]
    duplicate = [_item(assignments[0]), _item(assignments[0])]
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"items": duplicate})},
            }
        ],
        "usage": {},
    }
    monkeypatch.setattr(generator.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(payload))

    with pytest.raises(generator.RequestFailure) as captured:
        generator._request_once(
            assignments,
            api_key="temporary",
            model="deepseek-v4-flash",
            timeout=180,
            max_tokens=8000,
        )

    assert captured.value.category == "duplicate_assignment_id"


def test_canary_selection_covers_every_scenario_family() -> None:
    selected = generator._stratified_canary(materialize_assignments(), 115)

    assert len(selected) == 115
    assert len({value["matrix_cell"]["scenario_family"] for value in selected}) == 23
