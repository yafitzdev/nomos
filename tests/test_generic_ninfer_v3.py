from __future__ import annotations

import json
from pathlib import Path

import tools.generate_generic_ninfer_v3 as ninfer
from fitz_tool.generic_pilot_v3 import validate_generic_state


def test_teacher_row_is_generic_and_contract_valid() -> None:
    state = next(ninfer._skeleton_batches(1, seed=17, batch_size=1))[0]
    row = ninfer._teacher_row(
        state,
        {
            "assignment_id": state["decision_state_id"],
            "question": "Which operation should inspect the relevant information next?",
            "difficult_paraphrase": "Given the current evidence, what kind of next move should examine the needed details?",
        },
        model="test-model",
        seed=17,
    )
    assert row is not None
    assert row["source_kind"] == "ninfer_generic_teacher"
    assert row["provenance"]["teacher"] == "ninfer"
    assert validate_generic_state(row).valid


def test_request_fallback_splits_a_batch(monkeypatch) -> None:
    calls: list[int] = []

    def fake_request(**kwargs):
        batch = kwargs["batch"]
        calls.append(len(batch))
        if len(batch) > 1:
            return [], "synthetic serialization failure"
        return [{"assignment_id": batch[0]["assignment_id"]}], None

    monkeypatch.setattr(ninfer, "_request", fake_request)
    outputs, error = ninfer._request_with_fallback(
        base_url="http://unused",
        model="unused",
        api_key=None,
        batch=[{"assignment_id": f"row-{index}"} for index in range(4)],
        timeout=1,
        retries=0,
        max_tokens=1,
    )
    assert error is None
    assert [item["assignment_id"] for item in outputs] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]
    assert calls == [4, 2, 1, 1, 2, 1, 1]


def test_resume_reader_repairs_invalid_and_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "resume.jsonl"
    first = json.dumps({"decision_state_id": "row-1"})
    second = json.dumps({"decision_state_id": "row-2"})
    path.write_text(f"{first}\ncorrupt-fragment\n{second}\n{first}\n", encoding="utf-8")

    assert ninfer._existing_ids(path) == {"row-1", "row-2"}
    retained = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert retained == [{"decision_state_id": "row-1"}, {"decision_state_id": "row-2"}]
    assert list(tmp_path.glob("resume.resume-backup-*.jsonl"))
